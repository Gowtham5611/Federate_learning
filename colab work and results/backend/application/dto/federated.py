"""DTOs for federated learning configurations, partition reports, and experiment records.

Immutable dataclasses describing the inputs, intermediate rounds, and outcomes
of federated training experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.dto.experiment import ClassificationMetrics, ExperimentResult
from application.dto.training import TrainingConfig
from domain.enums.training_mode import TrainingMode


@dataclass(frozen=True, slots=True)
class FederatedConfig:
    """Hyperparameters and settings for a federated experiment.

    Attributes:
        training_config: Base configuration for model architecture and local training.
        n_clients: Number of participating clients.
        rounds: Number of federated communication rounds.
        local_epochs: Number of local training epochs each client executes per round.
        distribution: Data distribution across clients (e.g., 'iid').
        algorithm: Federated aggregation/optimization algorithm ('fedavg' or 'fedprox').
        mu: FedProx proximal coefficient, or None for FedAvg.
        seed: Random seed for partitioning and orchestration reproducibility.
    """

    training_config: TrainingConfig
    n_clients: int = 3
    rounds: int = 5
    local_epochs: int = 1
    distribution: str = "iid"
    algorithm: str = "fedavg"
    mu: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate federated settings."""
        if self.n_clients <= 0:
            raise ValueError(f"n_clients must be positive; got {self.n_clients!r}")
        if self.rounds <= 0:
            raise ValueError(f"rounds must be positive; got {self.rounds!r}")
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive; got {self.local_epochs!r}")
        dist = self.distribution.lower().replace("-", "_")
        if dist not in {"iid", "non_iid"}:
            raise ValueError(
                f"unsupported distribution {self.distribution!r}; supported: 'iid', 'non_iid'"
            )
        algo = self.algorithm.lower()
        if algo not in {"fedavg", "fedprox", "scaffold", "fedprox_scaffold"}:
            raise ValueError(
                f"unsupported algorithm {self.algorithm!r}; supported: "
                "'fedavg', 'fedprox', 'scaffold', 'fedprox_scaffold'"
            )
        if algo in {"fedavg", "scaffold"} and self.mu is not None:
            raise ValueError(
                f"{self.algorithm} does not use proximal penalty; mu must be None, got {self.mu!r}"
            )
        if algo in {"fedprox", "fedprox_scaffold"} and (self.mu is None or self.mu < 0.0):
            raise ValueError(
                f"{self.algorithm} requires a non-negative mu coefficient; got {self.mu!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "n_clients": self.n_clients,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "distribution": self.distribution,
            "algorithm": self.algorithm,
            "mu": self.mu,
            "seed": self.seed,
            "training_config": self.training_config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ClientPartitionStats:
    """Class balance and size statistics for one client's partition.

    Attributes:
        client_id: Unique client identifier (e.g., 'client_1').
        sample_count: Total training samples on this client.
        positive_count: Number of positive (laundering) samples.
        negative_count: Number of negative samples.
        positive_rate: Proportion of positive samples (positive_count / sample_count).
        distribution: Partitioning methodology ('iid').
        seed: Random seed used to create the partition.
    """

    client_id: str
    sample_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    distribution: str
    seed: int
    typology: str | None = None
    account_count: int | None = None
    feature_distributions: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to dictionary."""
        d: dict[str, Any] = {
            "client_id": self.client_id,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_rate": self.positive_rate,
            "distribution": self.distribution,
            "seed": self.seed,
        }
        if self.typology is not None:
            d["typology"] = self.typology
        if self.account_count is not None:
            d["account_count"] = self.account_count
        if self.feature_distributions is not None:
            d["feature_distributions"] = self.feature_distributions
        return d


@dataclass(frozen=True, slots=True)
class IIDPartitionReport:
    """Overall report describing the IID data partitioning across clients.

    Attributes:
        clients: Tuple of per-client partition statistics.
        total_samples: Total samples across all clients.
        total_positives: Total positive samples across all clients.
        overall_positive_rate: Overall positive rate across all clients.
        seed: Random seed used for the partition.
    """

    clients: tuple[ClientPartitionStats, ...]
    total_samples: int
    total_positives: int
    overall_positive_rate: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "clients": [c.to_dict() for c in self.clients],
            "total_samples": self.total_samples,
            "total_positives": self.total_positives,
            "overall_positive_rate": self.overall_positive_rate,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class NonIIDPartitionReport:
    """Overall report describing the typology-based Non-IID data partitioning across clients.

    Attributes:
        clients: Tuple of per-client partition statistics.
        total_samples: Total samples across all clients.
        total_positives: Total positive samples across all clients.
        overall_positive_rate: Overall positive rate across all clients.
        seed: Random seed used for the partition.
        partition_method: Identification of the partition rule applied.
    """

    clients: tuple[ClientPartitionStats, ...]
    total_samples: int
    total_positives: int
    overall_positive_rate: float
    seed: int
    partition_method: str = "typology_rule_v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "clients": [c.to_dict() for c in self.clients],
            "total_samples": self.total_samples,
            "total_positives": self.total_positives,
            "overall_positive_rate": self.overall_positive_rate,
            "seed": self.seed,
            "partition_method": self.partition_method,
        }


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """Metrics and state recorded for one completed federated communication round.

    Attributes:
        round: 1-indexed federated round number.
        aggregation_weights: Client aggregation weights used in FedAvg.
        client_train_losses: Average local training loss per client.
        client_sample_counts: Number of samples per client used for aggregation.
        validation_loss: Loss of the aggregated global model on centralized validation.
        validation_metrics: Full classification metrics on centralized validation.
        round_seconds: Wall-clock duration of the round in seconds.
        checkpoint_path: Path to the saved global checkpoint, if any.
        selected: Whether this round's global model was selected as the best checkpoint.
    """

    round: int
    aggregation_weights: dict[str, float]
    client_train_losses: dict[str, float]
    client_sample_counts: dict[str, int]
    validation_loss: float
    validation_metrics: ClassificationMetrics
    round_seconds: float
    checkpoint_path: str | None = None
    selected: bool = False
    drift_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize round record to dictionary."""
        d: dict[str, Any] = {
            "round": self.round,
            "aggregation_weights": dict(self.aggregation_weights),
            "client_train_losses": dict(self.client_train_losses),
            "client_sample_counts": dict(self.client_sample_counts),
            "validation_loss": self.validation_loss,
            "validation_metrics": self.validation_metrics.to_dict(),
            "round_seconds": self.round_seconds,
            "checkpoint_path": self.checkpoint_path,
            "selected": self.selected,
        }
        if self.drift_metrics is not None:
            d["drift_metrics"] = dict(self.drift_metrics)
        return d


@dataclass(frozen=True, slots=True)
class FederatedExperimentReport:
    """Complete summary of a federated experiment.

    Attributes:
        experiment_id: Unique experiment identifier.
        config: Federated experiment configuration.
        partition_report: Detailed client partitioning statistics.
        rounds: Tuple of records for every completed round.
        results: Results rows for validation (and final test, if evaluated).
        selected_round: Round whose global model achieved the best validation score.
        frozen_threshold: Decision threshold frozen on validation.
        threshold_selected_on: Which split selected the threshold ('validation').
        device: Device where models ran.
        parameter_count: Number of trainable parameters in the model.
        training_seconds: Time spent in federated rounds.
        data_preparation_seconds: Time spent partitioning and loading datasets.
        total_seconds: Total wall-clock time for the experiment.
        torch_version: PyTorch version.
        deterministic: Whether deterministic kernels were requested.
    """

    experiment_id: str
    config: FederatedConfig
    partition_report: IIDPartitionReport | NonIIDPartitionReport
    rounds: tuple[RoundRecord, ...]
    results: tuple[ExperimentResult, ...]
    selected_round: int
    frozen_threshold: float
    threshold_selected_on: str | None
    device: str
    parameter_count: int
    training_seconds: float
    data_preparation_seconds: float
    total_seconds: float
    torch_version: str
    deterministic: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize experiment report to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model": self.config.training_config.architecture.value,
            "training_mode": TrainingMode.FEDERATED.value,
            "data_distribution": self.config.distribution,
            "federated_algorithm": self.config.algorithm,
            "mu": self.config.mu,
            "configuration": self.config.to_dict(),
            "partition_report": self.partition_report.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "results": [res.to_dict() for res in self.results],
            "selected_round": self.selected_round,
            "frozen_threshold": self.frozen_threshold,
            "threshold_selected_on": self.threshold_selected_on,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "timing": {
                "data_preparation_seconds": self.data_preparation_seconds,
                "training_seconds": self.training_seconds,
                "total_seconds": self.total_seconds,
            },
            "torch_version": self.torch_version,
            "deterministic_kernels_requested": self.deterministic,
        }
