"""Federated checkpoint store organizing global models by algorithm, architecture, and round.

Directory layout:
    <root>/<algorithm>/<architecture>/<experiment_id>/
        round_1.pt
        round_2.pt
        ...
        global_best.pt
        final.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from application.dto.training import TrainingConfig
from application.ports.output.experiment_store import Checkpoint
from application.ports.output.federated_store import FederatedCheckpointStorePort
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.exceptions.training import (
    CheckpointArchitectureMismatchError,
    CheckpointNotFoundError,
)
from infrastructure.persistence.torch_checkpoint_store import (
    CHECKPOINT_SUFFIX,
    _payload,
)

logger = logging.getLogger(__name__)


class FederatedCheckpointStore(FederatedCheckpointStorePort):
    """Saves and loads federated global checkpoints organized by algorithm and architecture."""

    def __init__(self, root: Path, algorithm: str) -> None:
        """Initialize store.

        Args:
            root: Base directory (e.g. models/checkpoints/federated/iid).
            algorithm: Algorithm subdirectory ('fedavg' or 'fedprox').
        """
        self._root = root.resolve() / algorithm.lower()

    @property
    def root(self) -> Path:
        """Root directory for this algorithm's checkpoints."""
        return self._root

    def experiment_dir(self, architecture: ModelArchitecture, experiment_id: str) -> Path:
        """Directory for a specific experiment's checkpoints."""
        return self._root / architecture.value / experiment_id

    def path_for_round(
        self,
        architecture: ModelArchitecture,
        experiment_id: str,
        round_num: int,
    ) -> Path:
        """File path for a specific round's global checkpoint."""
        return (
            self.experiment_dir(architecture, experiment_id)
            / f"round_{round_num}{CHECKPOINT_SUFFIX}"
        )

    def path_for_kind(
        self,
        architecture: ModelArchitecture,
        experiment_id: str,
        kind: CheckpointKind,
    ) -> Path:
        """File path for special checkpoint kinds (best, final)."""
        return (
            self.experiment_dir(architecture, experiment_id)
            / f"{kind.value}{CHECKPOINT_SUFFIX}"
        )

    def save_round(self, checkpoint: Checkpoint, round_num: int) -> Path:
        """Save a global checkpoint for a specific communication round."""
        path = self.path_for_round(
            checkpoint.architecture, checkpoint.experiment_id, round_num
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_payload(checkpoint), path)
        logger.info(
            "Saved global checkpoint for round %d to %s",
            round_num,
            path,
        )
        return path

    def save_best(self, checkpoint: Checkpoint) -> Path:
        """Save the best global checkpoint according to validation metrics."""
        path = self.path_for_kind(
            checkpoint.architecture, checkpoint.experiment_id, CheckpointKind.GLOBAL_BEST
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_payload(checkpoint), path)
        logger.info(
            "Saved best global checkpoint (round %d) to %s",
            checkpoint.epoch,
            path,
        )
        return path

    def save_final(self, checkpoint: Checkpoint) -> Path:
        """Save the final global checkpoint after all rounds complete."""
        path = self.path_for_kind(
            checkpoint.architecture, checkpoint.experiment_id, CheckpointKind.FINAL
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_payload(checkpoint), path)
        logger.info("Saved final global checkpoint to %s", path)
        return path

    def load(
        self,
        experiment_id: str,
        kind: CheckpointKind,
        round_num: int | None = None,
        architecture: ModelArchitecture | None = None,
    ) -> Checkpoint:
        """Load a saved federated checkpoint."""
        if architecture is None:
            # Look across architectures if not specified
            candidates = [
                arch for arch in ModelArchitecture
                if self.experiment_dir(arch, experiment_id).is_dir()
            ]
            if not candidates:
                raise CheckpointNotFoundError(
                    f"No experiment directory found for {experiment_id} under {self._root}"
                )
            arch = candidates[0]
        else:
            arch = architecture

        if round_num is not None:
            path = self.path_for_round(arch, experiment_id, round_num)
        else:
            path = self.path_for_kind(arch, experiment_id, kind)

        if not path.is_file():
            raise CheckpointNotFoundError(path)

        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        stored_arch = ModelArchitecture(str(payload["architecture"]))
        if architecture is not None and stored_arch is not architecture:
            raise CheckpointArchitectureMismatchError(
                architecture.value, stored_arch.value, path
            )

        return Checkpoint(
            experiment_id=str(payload["experiment_id"]),
            kind=CheckpointKind(str(payload["kind"])),
            architecture=stored_arch,
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

