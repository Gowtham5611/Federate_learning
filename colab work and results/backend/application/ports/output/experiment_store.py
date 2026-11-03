"""Outbound ports for persisting checkpoints and experiment results.

Two stores with one thing in common: both are the record that an experiment
happened, and both must make it impossible to confuse one run's output with
another's.

The checkpoint store keeps *best* and *final* strictly apart. Under early stopping
they are different models, and overwriting the best with the final one is a
failure that leaves no trace — the file is still there, still loads, and no longer
holds the model whose validation score was reported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.dto.experiment import ExperimentReport
from application.dto.training import TrainingConfig
from application.ports.output.model_trainer import ParameterState
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A stored model, with enough context to evaluate or resume it.

    Attributes:
        experiment_id: Run that wrote it.
        kind: Best or final.
        architecture: Architecture that produced the parameters. Checked on load
            so GRU weights can never be loaded into an LSTM.
        epoch: Epoch the parameters come from.
        parameters: The parameters themselves.
        config: Configuration that produced it.
        feature_dimension: Input width the parameters expect, checked against the
            data on load.
        threshold: Decision threshold frozen with this model.
        validation_loss: Validation loss at ``epoch``.
        random_seed: Seed the run used.
        optimizer_state: Optimiser state, so training can resume rather than only
            being evaluated. ``None`` when not captured.
        validation_metrics: Validation metrics at ``epoch``.
    """

    experiment_id: str
    kind: CheckpointKind
    architecture: ModelArchitecture
    epoch: int
    parameters: ParameterState
    config: TrainingConfig
    feature_dimension: int
    threshold: float
    validation_loss: float
    random_seed: int
    optimizer_state: ParameterState | None = None
    validation_metrics: dict[str, float] = field(default_factory=dict)


class CheckpointStorePort(ABC):
    """Saves and loads model checkpoints."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Directory checkpoints are written under."""

    @abstractmethod
    def path_for(self, experiment_id: str, kind: CheckpointKind) -> Path:
        """Where a given experiment's checkpoint of that kind lives.

        Distinct paths per kind, so writing the final checkpoint cannot overwrite
        the best one.
        """

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> Path:
        """Write a checkpoint.

        Returns:
            Path written.
        """

    @abstractmethod
    def load(
        self,
        experiment_id: str,
        kind: CheckpointKind,
        *,
        expected_architecture: ModelArchitecture | None = None,
    ) -> Checkpoint:
        """Read a checkpoint back.

        Args:
            experiment_id: Run that wrote it.
            kind: Which checkpoint.
            expected_architecture: When given, the load fails unless the stored
                architecture matches.

        Returns:
            The checkpoint.

        Raises:
            CheckpointNotFoundError: Nothing is stored there.
            CheckpointArchitectureMismatchError: The stored architecture differs
                from ``expected_architecture``.
        """


class ExperimentResultStorePort(ABC):
    """Persists experiment reports and the flat rows a comparison reads."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Directory results are written under."""

    @abstractmethod
    def save_report(self, report: ExperimentReport) -> Path:
        """Write one experiment's full report.

        Returns:
            Path written.
        """

    @abstractmethod
    def append_rows(self, report: ExperimentReport) -> Path:
        """Append the report's result rows to the cross-experiment table.

        The table is the accumulating record every phase adds to, with one row per
        (experiment, split, threshold policy). A later federated run appends rows
        of the same shape, with its client, round and algorithm fields populated;
        that is the whole reason those fields exist now and are ``null``.

        Returns:
            Path appended to.
        """

    @abstractmethod
    def load_rows(self) -> list[dict[str, Any]]:
        """Read the cross-experiment table back, oldest row first."""
