"""Data transfer objects for Phase 10 Probability Calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from domain.entities.calibration import CalibrationMethod, CalibrationParameters


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for Phase 10 probability calibration experiment."""

    seed: int = 42
    batch_size: int = 1024
    device: str | None = None
    row_limit: int | None = None
    n_bins_ece: int = 15
    ece_strategy: str = "uniform"  # "uniform" or "quantile"
    historical_threshold: float = 0.13
    threshold_policy: str = "validation_f1"
    selection_metric: str = "log_loss"  # Predefined validation rule: log_loss, ece, brier
    methods: tuple[CalibrationMethod, ...] = (
        CalibrationMethod.NONE,
        CalibrationMethod.TEMPERATURE,
        CalibrationMethod.PLATT,
        CalibrationMethod.ISOTONIC,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "device": self.device,
            "row_limit": self.row_limit,
            "n_bins_ece": self.n_bins_ece,
            "ece_strategy": self.ece_strategy,
            "historical_threshold": self.historical_threshold,
            "threshold_policy": self.threshold_policy,
            "selection_metric": self.selection_metric,
            "methods": [m.value for m in self.methods],
        }


@dataclass(frozen=True)
class CalibrationEvaluationMetrics:
    """Comprehensive evaluation metrics for probability quality and classification."""

    log_loss: float
    brier_score: float
    ece: float
    mce: float
    pr_auc: float
    roc_auc: float
    # Metrics at optimal validation threshold
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    # Metrics at historical threshold (0.13)
    historical_threshold: float
    historical_f1: float
    historical_precision: float
    historical_recall: float
    historical_accuracy: float
    historical_tp: int
    historical_fp: int
    historical_tn: int
    historical_fn: int
    # Data profile
    positive_count: int
    total_count: int
    prevalence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMethodResult:
    """Results for a single calibration method on a given split."""

    method: CalibrationMethod
    split: str
    parameters: CalibrationParameters
    metrics: CalibrationEvaluationMetrics
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "method": self.method.value,
            "split": self.split,
            "parameters": self.parameters.to_dict(),
            "metrics": self.metrics.to_dict(),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class CalibrationExperimentReport:
    """Full experiment report for Phase 10 Probability Calibration."""

    experiment_id: str
    timestamp: str
    source_model: dict[str, Any]
    config: CalibrationConfig
    validation_results: list[CalibrationMethodResult]
    selected_method: CalibrationMethod
    selected_parameters: CalibrationParameters
    frozen_threshold: float
    test_baseline: CalibrationMethodResult
    test_calibrated: CalibrationMethodResult
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "source_model": self.source_model,
            "config": self.config.to_dict(),
            "validation_results": [r.to_dict() for r in self.validation_results],
            "selected_method": self.selected_method.value,
            "selected_parameters": self.selected_parameters.to_dict(),
            "frozen_threshold": self.frozen_threshold,
            "test_baseline": self.test_baseline.to_dict(),
            "test_calibrated": self.test_calibrated.to_dict(),
            "timing": self.timing,
        }
