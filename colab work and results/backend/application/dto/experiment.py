"""Experiment results: the record every phase of this project writes.

The shape here is deliberately wider than Phase 4 needs. A centralized baseline
has no client, no federated round and no proximal term, yet
:class:`ExperimentResult` carries those fields anyway. The reason is that the
project's final comparison is FedAvg against FedProx under IID against
typology-based non-IID for three architectures, and a dashboard reading that
comparison wants one flat table it can filter — not one schema per phase joined
after the fact.

A field is only filled in when this run actually answers it. A centralized run
genuinely has a centralized data distribution and genuinely used no aggregation
algorithm, so those two are recorded as such. It has no client, no round and no
``mu``, so those are ``null``. Nothing here invents a federated value.

A single :class:`ExperimentResult` is one (experiment, split, threshold policy)
row. :class:`ExperimentReport` is the whole run: configuration, training curve,
every result row, and the provenance needed to tell where the numbers came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from application.dto.training import EpochRecord, TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy, TrainingMode

CENTRALIZED_DISTRIBUTION = "centralized"
"""Value of ``data_distribution`` when the data was never partitioned.

The later phases write ``iid`` and the typology-based non-IID label here. Naming
the centralized case explicitly, rather than leaving it empty, is what lets a
comparison filter the table by distribution and get every row back.
"""

NO_FEDERATED_ALGORITHM = "none"
"""Value of ``federated_algorithm`` when no aggregation took place.

``"none"`` rather than ``null``: the question "which aggregation strategy?" has a
definite answer for a centralized run, and it is not missing data.
"""


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """The four outcomes of a binary decision, as counts.

    Held as its own type because the AML reading of these numbers is asymmetric.
    A false negative is a laundering transaction the system passed; a false
    positive is an alert an analyst closes. They are not interchangeable, so
    :attr:`false_negatives` is surfaced everywhere rather than folded into an
    error rate.
    """

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def total(self) -> int:
        """Every decision counted."""
        return (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )

    @property
    def actual_positives(self) -> int:
        """Laundering transactions present, found or missed."""
        return self.true_positives + self.false_negatives

    @property
    def predicted_positives(self) -> int:
        """Transactions the model flagged."""
        return self.true_positives + self.false_positives

    @property
    def missed_positive_rate(self) -> float:
        """Share of laundering transactions the model let through.

        ``1 - recall``, named for what it means operationally. Zero when there
        were no positives to miss.
        """
        actual = self.actual_positives
        return self.false_negatives / actual if actual else 0.0

    def to_dict(self) -> dict[str, int]:
        """Serialise the four counts."""
        return {
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Metrics for one split at one threshold.

    Accuracy is present and is deliberately not first. On a split that is 99.9%
    negative, a model that flags nothing scores 99.9% accuracy while catching no
    laundering at all, so accuracy is recorded for completeness and read last.
    The primary quantities are recall, F1, PR-AUC, precision and the false-negative
    count.

    ``pr_auc`` and ``roc_auc`` are threshold-free: they describe the ranking the
    model produces, so they are identical across the threshold policies applied
    to the same predictions.
    """

    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    accuracy: float
    confusion: ConfusionCounts
    threshold: float
    positive_rate: float
    loss: float | None = None

    @property
    def false_negatives(self) -> int:
        """Laundering transactions missed. Promoted for how often it is read."""
        return self.confusion.false_negatives

    def to_dict(self) -> dict[str, Any]:
        """Serialise, primary metrics first."""
        return {
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "false_negatives": self.confusion.false_negatives,
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            **self.confusion.to_dict(),
            "threshold": self.threshold,
            "positive_rate": self.positive_rate,
            "loss": self.loss,
            "missed_positive_rate": self.confusion.missed_positive_rate,
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One flat, dashboard-ready result row.

    Two of the federated fields are answerable for a centralized run and are
    answered: ``data_distribution`` is ``"centralized"`` and
    ``federated_algorithm`` is ``"none"``. Those are facts about this run, not
    predictions about a later one.

    The other three (``client_id``, ``federated_round``, ``mu``) have no value
    here and are ``None``, written as JSON ``null`` and as an empty CSV cell.
    ``None`` means *not applicable*, which is different from zero: round 0 and
    ``mu`` 0.0 are both meaningful federated settings, and writing either one for
    a centralized baseline would put a fabricated number in the comparison table.

    Attributes:
        experiment_id: Identifies the run that produced this row.
        architecture: Which encoder was trained.
        training_mode: Regime that produced the row.
        split: Which split was measured.
        metrics: The measurements.
        threshold_policy: How the threshold in ``metrics`` was arrived at.
        seed: Seed the run used.
        loss_type: Objective the model was trained with.
        training_time_seconds: Time to fit the model that produced this row.
        evaluation_time_seconds: Time to score this split.
        sequence_count: Sequences scored.
        hyperparameters: Full configuration, for reproduction.
        data_distribution: How data was distributed across clients.
            ``"centralized"`` when there are none.
        federated_algorithm: Aggregation strategy, ``"none"`` when not federated.
        client_id: Owning client. ``None`` when centralized.
        federated_round: Round index. ``None`` when centralized.
        mu: FedProx proximal coefficient. ``None`` when not FedProx.
    """

    experiment_id: str
    architecture: ModelArchitecture
    training_mode: TrainingMode
    split: SplitName
    metrics: ClassificationMetrics
    threshold_policy: ThresholdPolicy
    seed: int
    loss_type: str
    training_time_seconds: float
    evaluation_time_seconds: float
    sequence_count: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    data_distribution: str = CENTRALIZED_DISTRIBUTION
    federated_algorithm: str = NO_FEDERATED_ALGORITHM
    client_id: str | None = None
    federated_round: int | None = None
    mu: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise as one flat row.

        Flat rather than nested: the row is the unit a comparison table filters
        and sorts, and nesting the metrics would make every consumer reach two
        levels deep for the number it came for.
        """
        return {
            "experiment_id": self.experiment_id,
            "model": self.architecture.value,
            "training_mode": self.training_mode.value,
            "data_distribution": self.data_distribution,
            "federated_algorithm": self.federated_algorithm,
            "client_id": self.client_id,
            "round": self.federated_round,
            "mu": self.mu,
            "loss_type": self.loss_type,
            "split": self.split.value,
            "threshold_policy": self.threshold_policy.value,
            **self.metrics.to_dict(),
            "sequence_count": self.sequence_count,
            "training_time": self.training_time_seconds,
            "evaluation_time": self.evaluation_time_seconds,
            "seed": self.seed,
            "hyperparameters": dict(self.hyperparameters),
        }


@dataclass(frozen=True, slots=True)
class DataProvenance:
    """Where the data came from, so a result can be traced back to its input.

    Attributes:
        processed_directory: Directory the splits were read from.
        preprocessing_version: Version recorded by the phase that wrote them.
        feature_columns: Source feature columns, in the order Phase 3 recorded.
        identifier_columns: Traceability columns, excluded from the tensor.
        target_column: Label column.
        feature_dimension: Width of the tensor actually fed to the model, after
            any nominal-code expansion. Differs from ``len(feature_columns)``
            whenever that expansion is on, which is why both are recorded.
        rows_per_split: Rows read from each split.
        positives_per_split: Positive labels in each split.
    """

    processed_directory: str
    preprocessing_version: str
    feature_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    target_column: str
    feature_dimension: int
    rows_per_split: dict[str, int] = field(default_factory=dict)
    positives_per_split: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the provenance block."""
        return {
            "processed_directory": self.processed_directory,
            "preprocessing_version": self.preprocessing_version,
            "source_feature_columns": list(self.feature_columns),
            "source_feature_count": len(self.feature_columns),
            "identifier_columns_excluded_from_model": list(self.identifier_columns),
            "target_column": self.target_column,
            "model_input_dimension": self.feature_dimension,
            "rows_per_split": dict(self.rows_per_split),
            "positives_per_split": dict(self.positives_per_split),
        }


@dataclass(frozen=True, slots=True)
class SequenceStrategy:
    """How rows became sequences, recorded so the tensor shape is explainable.

    Attributes:
        grouping_column: Column whose identity defines one temporal stream.
        sequence_length: Maximum sequence length.
        stride: Spacing between successive targets.
        target_assignment: Which element of the sequence supplies the label.
        padding: How sequences shorter than the maximum are handled.
        boundary_policy: What happens at a split boundary.
        ordering: How chronological order within a sequence is established.
    """

    grouping_column: str
    sequence_length: int
    stride: int
    target_assignment: str
    padding: str
    boundary_policy: str
    ordering: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the strategy block."""
        return {
            "grouping_column": self.grouping_column,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "target_assignment": self.target_assignment,
            "padding": self.padding,
            "boundary_policy": self.boundary_policy,
            "ordering": self.ordering,
        }


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Everything one experiment produced.

    Attributes:
        experiment_id: Identifier shared by every row of this run.
        config: Hyperparameters used.
        provenance: Where the data came from.
        sequences: How rows became sequences.
        epochs: Training curve, one entry per completed epoch.
        results: One row per (split, threshold policy) measured.
        selected_epoch: Epoch whose checkpoint was frozen.
        frozen_threshold: Threshold applied to test, and the split it came from.
        threshold_selected_on: Split the threshold was fitted on, or ``None``
            when it was configured rather than fitted.
        device: Device training ran on.
        parameter_count: Trainable parameters in the model.
        training_seconds: Fitting time, excluding data preparation.
        data_preparation_seconds: Time spent reading splits and building indices.
        total_seconds: Whole run.
        early_stopped: Whether training stopped before the epoch budget.
        torch_version: Recorded because determinism guarantees are version- and
            backend-specific.
        deterministic: Whether bitwise-reproducible kernels were successfully
            requested. ``False`` is reported, not hidden.
    """

    experiment_id: str
    config: TrainingConfig
    provenance: DataProvenance
    sequences: SequenceStrategy
    epochs: tuple[EpochRecord, ...]
    results: tuple[ExperimentResult, ...]
    selected_epoch: int
    frozen_threshold: float
    threshold_selected_on: str | None
    device: str
    parameter_count: int
    training_seconds: float
    data_preparation_seconds: float
    total_seconds: float
    early_stopped: bool
    torch_version: str
    deterministic: bool

    def result_for(
        self, split: SplitName, policy: ThresholdPolicy
    ) -> ExperimentResult | None:
        """Return one result row, or ``None`` if that combination was not measured."""
        for result in self.results:
            if result.split is split and result.threshold_policy is policy:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole report."""
        return {
            "experiment_id": self.experiment_id,
            "model": self.config.architecture.value,
            "training_mode": TrainingMode.CENTRALIZED.value,
            "configuration": self.config.to_dict(),
            "data_provenance": self.provenance.to_dict(),
            "sequence_strategy": self.sequences.to_dict(),
            "training_curve": [record.to_dict() for record in self.epochs],
            "results": [result.to_dict() for result in self.results],
            "selected_epoch": self.selected_epoch,
            "frozen_threshold": self.frozen_threshold,
            "threshold_selected_on": self.threshold_selected_on,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "timing": {
                "data_preparation_seconds": self.data_preparation_seconds,
                "training_seconds": self.training_seconds,
                "total_seconds": self.total_seconds,
            },
            "early_stopped": self.early_stopped,
            "torch_version": self.torch_version,
            "deterministic_kernels_requested": self.deterministic,
        }
