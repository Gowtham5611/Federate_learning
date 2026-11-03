"""Data Transfer Objects for Phase 8 Performance Analysis and Ablation Study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CentralizedVsFederatedRecord:
    """Comparison record across Centralized, IID Federated, and Non-IID Federated paradigms."""

    paradigm: str
    model: str
    algorithm: str
    experiment_id: str
    split: str
    threshold_policy: str
    threshold: float
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class ModelComparisonRecord:
    """Architectural comparison record comparing GRU, LSTM, and BiLSTM."""

    paradigm: str
    algorithm: str
    model: str
    split: str
    threshold_policy: str
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    accuracy: float
    experiment_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class FedAvgVsFedProxRecord:
    """Comparison record between FedAvg and FedProx optimization."""

    distribution: str
    model: str
    fedavg_exp_id: str | None
    fedprox_exp_id: str | None
    fedavg_val_pr_auc: float | None
    fedprox_val_pr_auc: float | None
    val_pr_auc_delta: float | None
    fedavg_test_pr_auc: float | None
    fedprox_test_pr_auc: float | None
    fedavg_test_f1: float | None
    fedprox_test_f1: float | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class IIDVsNonIIDRecord:
    """Comparison record examining heterogeneity effects between IID and Non-IID partitions."""

    model: str
    algorithm: str
    iid_exp_id: str | None
    non_iid_exp_id: str | None
    iid_val_pr_auc: float | None
    non_iid_val_pr_auc: float | None
    val_pr_auc_degradation: float | None
    iid_test_pr_auc: float | None
    non_iid_test_pr_auc: float | None
    iid_test_f1: float | None
    non_iid_test_f1: float | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class ClientDistributionRecord:
    """Distribution profile of a typology-based Non-IID banking client partition."""

    client_id: str
    typology: str
    transaction_count: int
    percentage_of_total: float
    positive_count: int
    negative_count: int
    positive_rate: float
    account_count: int
    retail_share: float | None = None
    crypto_wire_fx_share: float | None = None
    cu_share: float | None = None
    same_bank_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class ConvergenceMetricRecord:
    """Round-level convergence metric tracking performance over federated communication rounds."""

    experiment_id: str
    distribution: str
    algorithm: str
    model: str
    round: int
    val_pr_auc: float
    val_roc_auc: float
    val_f1: float
    val_recall: float
    val_precision: float
    val_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class AlphaAnalysisRecord:
    """Evaluation metrics for a candidate alpha weight during Phase 7 ensemble grid search."""

    alpha: float
    iid_weight: float
    non_iid_weight: float
    val_pr_auc: float
    val_roc_auc: float
    val_f1: float
    val_recall: float
    val_precision: float
    val_accuracy: float
    is_selected: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class ThresholdAnalysisRecord:
    """Metrics comparing decision thresholds and precision-recall trade-offs."""

    split: str
    threshold_policy: str
    threshold: float
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class AblationRecord:
    """Stage-by-stage ablation assessing architectural and methodological contributions."""

    stage: str
    description: str
    experiment_id: str
    val_pr_auc: float
    test_pr_auc: float
    test_roc_auc: float
    test_f1: float
    test_recall: float
    test_precision: float
    test_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    delta_test_pr_auc_vs_baseline: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class Phase8Report:
    """Composite experiment analysis report summarizing Phase 8 performance findings."""

    analysis_id: str
    timestamp: str
    seed: int
    total_experiments_analyzed: int
    centralized_records_count: int
    iid_records_count: int
    non_iid_records_count: int
    ensemble_records_count: int
    centralized_vs_federated: list[CentralizedVsFederatedRecord]
    model_comparison: list[ModelComparisonRecord]
    fedavg_vs_fedprox: list[FedAvgVsFedProxRecord]
    iid_vs_non_iid: list[IIDVsNonIIDRecord]
    client_distribution: list[ClientDistributionRecord]
    convergence_metrics: list[ConvergenceMetricRecord]
    alpha_analysis: list[AlphaAnalysisRecord]
    threshold_analysis: list[ThresholdAnalysisRecord]
    ablation_study: list[AblationRecord]
    frozen_ensemble_metrics: dict[str, Any]
    key_findings: list[str]
    limitations: list[str]
    artifacts_generated: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "total_experiments_analyzed": self.total_experiments_analyzed,
            "centralized_records_count": self.centralized_records_count,
            "iid_records_count": self.iid_records_count,
            "non_iid_records_count": self.non_iid_records_count,
            "ensemble_records_count": self.ensemble_records_count,
            "centralized_vs_federated": [r.to_dict() for r in self.centralized_vs_federated],
            "model_comparison": [r.to_dict() for r in self.model_comparison],
            "fedavg_vs_fedprox": [r.to_dict() for r in self.fedavg_vs_fedprox],
            "iid_vs_non_iid": [r.to_dict() for r in self.iid_vs_non_iid],
            "client_distribution": [r.to_dict() for r in self.client_distribution],
            "convergence_metrics": [r.to_dict() for r in self.convergence_metrics],
            "alpha_analysis": [r.to_dict() for r in self.alpha_analysis],
            "threshold_analysis": [r.to_dict() for r in self.threshold_analysis],
            "ablation_study": [r.to_dict() for r in self.ablation_study],
            "frozen_ensemble_metrics": self.frozen_ensemble_metrics,
            "key_findings": self.key_findings,
            "limitations": self.limitations,
            "artifacts_generated": self.artifacts_generated,
        }
