"""Federated parameter aggregation and proximal regularizers."""

from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import (
    FedProxTrainingEngine,
    compute_proximal_loss,
)
from infrastructure.federated.aggregation.scaffold import (
    SCAFFOLDTrainingEngine,
    aggregate_scaffold,
    init_control_variate,
)

__all__ = [
    "FedProxTrainingEngine",
    "SCAFFOLDTrainingEngine",
    "aggregate_fedavg",
    "aggregate_scaffold",
    "compute_aggregation_weights",
    "compute_proximal_loss",
    "init_control_variate",
]
