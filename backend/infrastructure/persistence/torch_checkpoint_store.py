"""Checkpoint store backed by ``torch.save``, one directory per experiment.

Two files, never one: ``best.pt`` and ``final.pt`` live at separate paths under the
experiment's own directory, so writing the final checkpoint cannot overwrite the
best one. Under early stopping they are different models, and a store that used a
single ``model.pt`` would leave the report quoting a validation score belonging to
parameters that are no longer on disk — with nothing to indicate it.

Payloads hold tensors and primitives only. That is what lets them be read back with
``weights_only=True``, so loading a checkpoint never executes arbitrary pickled
code. The configuration travels as the plain dictionary
:meth:`TrainingConfig.to_dict` produces and is rebuilt on load, rather than being
pickled as an object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from application.dto.training import TrainingConfig
from application.ports.output.experiment_store import Checkpoint, CheckpointStorePort
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.exceptions.training import (
    CheckpointArchitectureMismatchError,
    CheckpointNotFoundError,
)

logger = logging.getLogger(__name__)

CHECKPOINT_SUFFIX = ".pt"
FORMAT_VERSION = 1
"""Stamped into every payload so a future format change can be detected."""


class TorchCheckpointStore(CheckpointStorePort):
    """Saves and loads checkpoints as ``<root>/<experiment_id>/<kind>.pt``."""

    def __init__(self, root: Path) -> None:
        """Prepare the checkpoint tree.

        Args:
            root: Directory to write experiment subdirectories under.
        """
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        """Directory checkpoints are written under."""
        return self._root

    def path_for(self, experiment_id: str, kind: CheckpointKind) -> Path:
        """Where a given experiment's checkpoint of that kind lives."""
        return self._root / experiment_id / f"{kind.value}{CHECKPOINT_SUFFIX}"

    def save(self, checkpoint: Checkpoint) -> Path:
        """Write a checkpoint, creating its experiment directory if needed."""
        path = self.path_for(checkpoint.experiment_id, checkpoint.kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_payload(checkpoint), path)
        logger.info(
            "Saved the %s checkpoint for '%s' (epoch %d) to %s",
            checkpoint.kind.value,
            checkpoint.experiment_id,
            checkpoint.epoch,
            path,
        )
        return path

    def load(
        self,
        experiment_id: str,
        kind: CheckpointKind,
        *,
        expected_architecture: ModelArchitecture | None = None,
    ) -> Checkpoint:
        """Read a checkpoint back.

        Raises:
            CheckpointNotFoundError: Nothing is stored at that path.
            CheckpointArchitectureMismatchError: The stored architecture differs
                from ``expected_architecture``.
        """
        path = self.path_for(experiment_id, kind)
        if not path.is_file():
            raise CheckpointNotFoundError(path)

        # weights_only=True: a checkpoint is data, and loading data must not be able
        # to execute code. The payload is written to satisfy this deliberately.
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        architecture = ModelArchitecture(str(payload["architecture"]))
        if expected_architecture is not None and architecture is not expected_architecture:
            raise CheckpointArchitectureMismatchError(
                expected_architecture.value, architecture.value, path
            )

        return Checkpoint(
            experiment_id=str(payload["experiment_id"]),
            kind=CheckpointKind(str(payload["kind"])),
            architecture=architecture,
            epoch=int(payload["epoch"]),
            parameters=dict(payload["parameters"]),
            config=TrainingConfig.from_dict(payload["config"]),
            feature_dimension=int(payload["feature_dimension"]),
            threshold=float(payload["threshold"]),
            validation_loss=float(payload["validation_loss"]),
            random_seed=int(payload["random_seed"]),
            optimizer_state=payload.get("optimizer_state"),
            validation_metrics={
                str(name): float(value)
                for name, value in dict(payload.get("validation_metrics", {})).items()
            },
        )


def _payload(checkpoint: Checkpoint) -> dict[str, Any]:
    """Flatten a checkpoint into tensors and primitives only."""
    return {
        "format_version": FORMAT_VERSION,
        "experiment_id": checkpoint.experiment_id,
        "kind": checkpoint.kind.value,
        "architecture": checkpoint.architecture.value,
        "epoch": checkpoint.epoch,
        "parameters": dict(checkpoint.parameters),
        "config": checkpoint.config.to_dict(),
        "feature_dimension": checkpoint.feature_dimension,
        "threshold": checkpoint.threshold,
        "validation_loss": checkpoint.validation_loss,
        "random_seed": checkpoint.random_seed,
        "optimizer_state": (
            dict(checkpoint.optimizer_state)
            if checkpoint.optimizer_state is not None
            else None
        ),
        "validation_metrics": dict(checkpoint.validation_metrics),
    }
