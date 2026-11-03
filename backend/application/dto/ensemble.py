"""DTOs for Phase 7 IID + Non-IID ensemble modeling.

Immutable dataclasses describing candidate model selection, alpha grid tuning,
validation threshold freezing, test evaluation, and ensemble reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from application.dto.experiment import ClassificationMetrics, ExperimentResult
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy


@dataclass(frozen=True, slots=True)
class ModelSelectionRecord:
    """Identity and validation score of a candidate model selected for the ensemble.

    Attributes:
        distribution: Source distribution ('iid' or 'non_iid').
        algorithm: Federated algorithm ('fedavg' or 'fedprox').
        model: Recurrent architecture name ('gru', 'lstm', or 'bilstm').
        experiment_id: Full experiment identifier.
        checkpoint_kind: Checkpoint kind used ('global_best').
        checkpoint_path: Path to the selected checkpoint on disk.
        selection_metric: Validation metric name ('pr_auc').
        validation_score: Validation metric value achieved.
    """

    distribution: str
    algorithm: str
    model: str
    experiment_id: str
    checkpoint_kind: str
    checkpoint_path: str
    selection_metric: str
    validation_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize selection record to dictionary."""
        return {
            "distribution": self.distribution,
            "algorithm": self.algorithm,
            "model": self.model,
            "experiment_id": self.experiment_id,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_path": self.checkpoint_path,
            "selection_metric": self.selection_metric,
            "validation_score": self.validation_score,
        }


@dataclass(frozen=True, slots=True)
class AlphaCandidateRecord:
    """Validation evaluation metrics for one candidate alpha value.

    Attributes:
        alpha: Weight given to the IID model predictions (1 - alpha for Non-IID).
        pr_auc: Validation PR-AUC achieved by the ensemble at this alpha.
        roc_auc: Validation ROC-AUC achieved.
        f1: Validation F1 score at default threshold (0.5).
        recall: Validation Recall at default threshold (0.5).
        precision: Validation Precision at default threshold (0.5).
        accuracy: Validation Accuracy at default threshold (0.5).
        selected: True if this candidate was selected as the optimal alpha.
    """

    alpha: float
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    accuracy: float
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize alpha candidate to dictionary."""
        return {
            "alpha": self.alpha,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Configuration for Phase 7 ensemble tuning and evaluation.

    Attributes:
        alpha_step: Grid step size for alpha search (default: 0.05).
        seed: Random seed for orchestration and kernel determinism.
        selection_metric: Metric used to select best alpha on validation.
        threshold_policy: Policy used to choose decision threshold on validation.
        batch_size: Batch size for prediction inference passes.
        row_limit: Optional row limit for smoke tests / reduced runs.
        device: PyTorch device ('cpu', 'cuda', etc.).
    """

    alpha_step: float = 0.05
    seed: int = 42
    selection_metric: SelectionMetric = SelectionMetric.PR_AUC
    threshold_policy: ThresholdPolicy = ThresholdPolicy.VALIDATION_F1
    batch_size: int = 1024
    row_limit: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        if not (0.0 < self.alpha_step <= 0.5):
            raise ValueError(f"alpha_step must be in (0.0, 0.5]; got {self.alpha_step!r}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {self.batch_size!r}")

    @property
    def alpha_grid(self) -> list[float]:
        """Generate deterministic alpha search grid from 0.0 to 1.0 inclusive."""
        grid: list[float] = []
        val = 0.0
        while val <= 1.000001:
            grid.append(round(val, 6))
            val += self.alpha_step
        if abs(grid[-1] - 1.0) > 1e-5:
            grid.append(1.0)
        return grid

    def to_dict(self) -> dict[str, Any]:
        """Serialize ensemble configuration to dictionary."""
        return {
            "alpha_step": self.alpha_step,
            "alpha_grid": self.alpha_grid,
            "seed": self.seed,
            "selection_metric": self.selection_metric.value,
            "threshold_policy": self.threshold_policy.value,
            "batch_size": self.batch_size,
            "row_limit": self.row_limit,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class EnsembleExperimentReport:
    """Comprehensive artifact summarizing an ensemble experiment."""

    experiment_id: str
    timestamp: str
    config: EnsembleConfig
    selected_iid: ModelSelectionRecord
    selected_non_iid: ModelSelectionRecord
    alpha_search: list[AlphaCandidateRecord]
    best_alpha: float
    frozen_threshold: float
    validation_metrics_at_frozen_threshold: ClassificationMetrics
    validation_metrics_at_fixed_threshold: ClassificationMetrics
    test_metrics_at_frozen_threshold: ClassificationMetrics
    test_metrics_at_fixed_threshold: ClassificationMetrics
    timing: dict[str, float]
    results: list[ExperimentResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete ensemble report to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "selected_iid": self.selected_iid.to_dict(),
            "selected_non_iid": self.selected_non_iid.to_dict(),
            "alpha_search": [a.to_dict() for a in self.alpha_search],
            "best_alpha": self.best_alpha,
            "frozen_threshold": self.frozen_threshold,
            "validation_metrics_at_frozen_threshold": (
                self.validation_metrics_at_frozen_threshold.to_dict()
            ),
            "validation_metrics_at_fixed_threshold": (
                self.validation_metrics_at_fixed_threshold.to_dict()
            ),
            "test_metrics_at_frozen_threshold": (
                self.test_metrics_at_frozen_threshold.to_dict()
            ),
            "test_metrics_at_fixed_threshold": (
                self.test_metrics_at_fixed_threshold.to_dict()
            ),
            "timing": self.timing,
            "results": [r.to_dict() for r in self.results],
        }

