"""Test doubles for the centralized training service.

The service's job is *order*: which split is read when, when the threshold is
frozen, which checkpoints are written. Order is invisible to a test that only
inspects return values, so every double here appends to one shared
:class:`EventLog` and the tests assert on the resulting sequence.

That is what makes the central rule checkable rather than merely intended. "The
test split is not read until the threshold is frozen" becomes a comparison of two
positions in one list.

No PyTorch, no scikit-learn and no file I/O: the doubles return fixed arrays, so
these tests exercise the orchestration and nothing else. Whether the real engine
trains correctly is a separate question, asked in the infrastructure tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from application.dto.experiment import (
    ClassificationMetrics,
    ConfusionCounts,
    DataProvenance,
    ExperimentReport,
    SequenceStrategy,
)
from application.dto.training import EpochRecord, TrainingConfig
from application.ports.output.experiment_store import (
    Checkpoint,
    CheckpointStorePort,
    ExperimentResultStorePort,
)
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    EpochCallback,
    FitOutcome,
    ParameterState,
    SplitPredictions,
    TrainingEnginePort,
)
from application.ports.output.sequence_data import (
    SequenceDataProviderPort,
    SequenceDataset,
)
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.exceptions.training import (
    CheckpointArchitectureMismatchError,
    CheckpointNotFoundError,
)
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
    build_previous_row_index,
)

EventLog = list[str]
"""Ordered record of what the doubles were asked to do."""

FIXTURE_FEATURE_DIMENSION = 4
"""Feature width the doubles agree on, small enough to write tensors by hand."""


def make_dataset(
    split: SplitName,
    *,
    rows: int = 12,
    accounts: int = 3,
    positives: int = 3,
    feature_dimension: int = FIXTURE_FEATURE_DIMENSION,
    sequence_length: int = 3,
    stride: int = 1,
) -> AccountSequenceDataset:
    """Build a small in-memory dataset without touching the filesystem.

    Args:
        split: Which split it claims to be.
        rows: Rows in the split.
        accounts: Distinct sending accounts, dealt round-robin.
        positives: How many rows carry label 1, taken from the front.
        feature_dimension: Feature width.
        sequence_length: Window length.
        stride: Target spacing.

    Returns:
        The dataset.
    """
    features = np.arange(rows * feature_dimension, dtype=np.float32).reshape(
        rows, feature_dimension
    )
    targets = np.zeros(rows, dtype=np.float32)
    targets[:positives] = 1.0
    accounts_codes = np.arange(rows, dtype=np.int64) % accounts
    return AccountSequenceDataset(
        split=split,
        features=features,
        targets=targets,
        previous_row=build_previous_row_index(accounts_codes),
        sequence_length=sequence_length,
        stride=stride,
    )


class RecordingSequenceProvider(SequenceDataProviderPort):
    """Hands out prepared datasets and records the order splits were read in."""

    def __init__(
        self,
        events: EventLog,
        *,
        datasets: dict[SplitName, SequenceDataset] | None = None,
        feature_dimension: int = FIXTURE_FEATURE_DIMENSION,
    ) -> None:
        """Prepare one dataset per split unless explicit ones are supplied."""
        self.events = events
        self.feature_dimension = feature_dimension
        self.loaded: list[SplitName] = []
        self.datasets: dict[SplitName, SequenceDataset] = datasets or {
            split: make_dataset(split, feature_dimension=feature_dimension)
            for split in SplitName
        }

    def load(self, split: SplitName, config: TrainingConfig) -> SequenceDataset:
        """Return that split's prepared dataset, recording the read."""
        self.events.append(f"load:{split.value}")
        self.loaded.append(split)
        return self.datasets[split]

    def describe_features(self, config: TrainingConfig) -> DataProvenance:
        """Report provenance, with counts covering the splits read so far."""
        self.events.append("describe_features")
        return DataProvenance(
            processed_directory="/synthetic/processed",
            preprocessing_version="test-0.0.0",
            feature_columns=tuple(
                f"feature_{index}" for index in range(self.feature_dimension)
            ),
            identifier_columns=("timestamp", "from_account", "to_account"),
            target_column="is_laundering",
            feature_dimension=self.feature_dimension,
            rows_per_split={
                split.value: self.datasets[split].row_count for split in self.loaded
            },
            positives_per_split={
                split.value: self.datasets[split].positive_count
                for split in self.loaded
            },
        )

    def describe_strategy(self, config: TrainingConfig) -> SequenceStrategy:
        """Report a fixed strategy description."""
        return SequenceStrategy(
            grouping_column="from_account",
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            target_assignment="last transaction in the window",
            padding="zero-padded on the right, true lengths supplied",
            boundary_policy="one split's rows only",
            ordering="row order in the processed file",
        )


class StubTrainingEngine(TrainingEnginePort):
    """Returns prepared predictions and records what it was asked to score."""

    def __init__(
        self,
        events: EventLog,
        *,
        epochs: int = 2,
        best_epoch: int = 1,
        early_stopped: bool = False,
        probabilities: dict[SplitName, NDArray[np.float32]] | None = None,
    ) -> None:
        """Prepare a fit outcome without training anything."""
        self.events = events
        self.epoch_count = epochs
        self.best_epoch = best_epoch
        self.early_stopped = early_stopped
        self.probabilities = probabilities or {}
        self.fit_calls: list[tuple[SplitName, ...]] = []
        self.predicted: list[SplitName] = []
        self.best_state: ParameterState = {"marker": "best"}
        self.final_state: ParameterState = {"marker": "final"}

    def fit(
        self,
        config: TrainingConfig,
        train: SequenceDataset,
        validation: SequenceDataset,
        *,
        initial_parameters: ParameterState | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> FitOutcome:
        """Emit ``epoch_count`` prepared epoch records."""
        self.events.append("fit")
        self.fit_calls.append((train.split, validation.split))
        records: list[EpochRecord] = []
        for epoch in range(1, self.epoch_count + 1):
            record = EpochRecord(
                epoch=epoch,
                train_loss=1.0 / epoch,
                validation_loss=2.0 / epoch,
                validation_metrics={"pr_auc": 0.1 * epoch, "recall": 0.2 * epoch},
                seconds=0.5,
                selected=epoch == self.best_epoch,
            )
            records.append(record)
            if on_epoch is not None:
                on_epoch(record)
        return FitOutcome(
            epochs=tuple(records),
            best_epoch=self.best_epoch,
            best_state=self.best_state,
            final_state=self.final_state,
            parameter_count=1234,
            device="cpu",
            early_stopped=self.early_stopped,
            training_seconds=3.5,
            deterministic=False,
            extra={"framework_version": "0.0.0-test"},
        )

    def predict(
        self,
        config: TrainingConfig,
        dataset: SequenceDataset,
        parameters: ParameterState,
    ) -> SplitPredictions:
        """Return prepared probabilities for a split, recording the call."""
        self.events.append(f"predict:{dataset.split.value}")
        self.predicted.append(dataset.split)
        labels = dataset.labels
        scores = self.probabilities.get(dataset.split)
        if scores is None:
            # Perfectly separable by default, so the metrics are predictable and a
            # test asserting on ordering is not distracted by a degenerate score.
            scores = np.where(labels > 0.5, 0.9, 0.1).astype(np.float32)
        return SplitPredictions(
            labels=labels,
            probabilities=scores,
            row_indices=np.arange(labels.shape[0], dtype=np.int64),
            mean_loss=0.25,
            seconds=0.75,
        )

    def count_parameters(self, config: TrainingConfig, feature_dimension: int) -> int:
        """Report a fixed parameter count."""
        return 1234


class RecordingMetricsCalculator(MetricsCalculatorPort):
    """Computes trivial metrics and records every threshold selection."""

    def __init__(self, events: EventLog, *, selected: float = 0.42) -> None:
        """Prepare the threshold this double will select."""
        self.events = events
        self.selected = selected
        self.selection_splits: list[SplitName] = []
        self.evaluated_thresholds: list[float] = []

    def evaluate(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        threshold: float,
        loss: float | None = None,
    ) -> ClassificationMetrics:
        """Count the confusion matrix honestly; leave the ratios as placeholders."""
        self.evaluated_thresholds.append(threshold)
        truth = labels > 0.5
        flagged = probabilities >= threshold
        confusion = ConfusionCounts(
            true_positives=int(np.count_nonzero(truth & flagged)),
            true_negatives=int(np.count_nonzero(~truth & ~flagged)),
            false_positives=int(np.count_nonzero(~truth & flagged)),
            false_negatives=int(np.count_nonzero(truth & ~flagged)),
        )
        return ClassificationMetrics(
            precision=0.5,
            recall=0.5,
            f1=0.5,
            pr_auc=0.5,
            roc_auc=0.5,
            accuracy=0.5,
            confusion=confusion,
            threshold=threshold,
            positive_rate=0.5,
            loss=loss,
        )

    def select_threshold(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        split: SplitName,
    ) -> float:
        """Record which split was offered, then return the prepared threshold."""
        self.events.append(f"select_threshold:{split.value}")
        self.selection_splits.append(split)
        return self.selected


class RecordingCheckpointStore(CheckpointStorePort):
    """Keeps saved checkpoints in memory, one entry per kind."""

    def __init__(self, events: EventLog, root: Path | None = None) -> None:
        """Prepare an in-memory store rooted at a nominal path."""
        self.events = events
        self._root = root or Path("/synthetic/checkpoints")
        self.saved: dict[CheckpointKind, Checkpoint] = {}
        self.save_order: list[CheckpointKind] = []

    @property
    def root(self) -> Path:
        """Nominal checkpoint root."""
        return self._root

    def path_for(self, experiment_id: str, kind: CheckpointKind) -> Path:
        """Distinct path per kind, as the real store guarantees."""
        return self._root / experiment_id / f"{kind.value}.pt"

    def save(self, checkpoint: Checkpoint) -> Path:
        """Record the checkpoint under its kind."""
        self.events.append(f"save_checkpoint:{checkpoint.kind.value}")
        self.saved[checkpoint.kind] = checkpoint
        self.save_order.append(checkpoint.kind)
        return self.path_for(checkpoint.experiment_id, checkpoint.kind)

    def load(
        self,
        experiment_id: str,
        kind: CheckpointKind,
        *,
        expected_architecture: ModelArchitecture | None = None,
    ) -> Checkpoint:
        """Return a previously saved checkpoint.

        Raises:
            CheckpointNotFoundError: Nothing of that kind was saved.
            CheckpointArchitectureMismatchError: The stored architecture differs.
        """
        stored = self.saved.get(kind)
        if stored is None:
            raise CheckpointNotFoundError(self.path_for(experiment_id, kind))
        if (
            expected_architecture is not None
            and stored.architecture is not expected_architecture
        ):
            raise CheckpointArchitectureMismatchError(
                expected_architecture,
                stored.architecture,
                self.path_for(experiment_id, kind),
            )
        return stored


class RecordingResultStore(ExperimentResultStorePort):
    """Keeps reports and rows in memory."""

    def __init__(self, events: EventLog, root: Path | None = None) -> None:
        """Prepare an in-memory result store."""
        self.events = events
        self._root = root or Path("/synthetic/results")
        self.reports: list[ExperimentReport] = []
        self.rows: list[dict[str, Any]] = []

    @property
    def root(self) -> Path:
        """Nominal results root."""
        return self._root

    def save_report(self, report: ExperimentReport) -> Path:
        """Record the report."""
        self.events.append("save_report")
        self.reports.append(report)
        return self._root / "reports" / f"{report.experiment_id}.json"

    def append_rows(self, report: ExperimentReport) -> Path:
        """Record the report's flattened rows."""
        self.events.append("append_rows")
        self.rows.extend(result.to_dict() for result in report.results)
        return self._root / "experiment_results.csv"

    def load_rows(self) -> list[dict[str, Any]]:
        """Return every recorded row."""
        return list(self.rows)
