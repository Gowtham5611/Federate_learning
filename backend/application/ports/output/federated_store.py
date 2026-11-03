"""Outbound port for persisting federated checkpoints.

Defines the contract for storing global model states across federated rounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from application.ports.output.experiment_store import Checkpoint
from domain.enums.checkpoint_kind import CheckpointKind


class FederatedCheckpointStorePort(ABC):
    """Saves and loads global model checkpoints for federated learning."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Directory federated checkpoints are written under."""

    @abstractmethod
    def save_round(self, checkpoint: Checkpoint, round_num: int) -> Path:
        """Save a global checkpoint for a specific round."""

    @abstractmethod
    def save_best(self, checkpoint: Checkpoint) -> Path:
        """Save the best global checkpoint across all rounds."""

    @abstractmethod
    def save_final(self, checkpoint: Checkpoint) -> Path:
        """Save the final global checkpoint after the last round."""

    @abstractmethod
    def load(
        self,
        experiment_id: str,
        kind: CheckpointKind,
        round_num: int | None = None,
    ) -> Checkpoint:
        """Load a saved checkpoint."""

