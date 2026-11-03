"""Federated client abstraction for local model training.

Each client holds its local training partition, receives the global model
parameters from the server at each round, trains locally using the configured
engine (BCEWithLogitsLoss for FedAvg, BCEWithLogitsLoss + proximal penalty for FedProx),
and returns updated parameters and sample counts for aggregation.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn

from application.dto.training import TrainingConfig
from application.ports.output.model_trainer import ParameterState, TrainingEnginePort
from application.ports.output.sequence_data import SequenceDataset
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.aggregation.scaffold import SCAFFOLDTrainingEngine
from infrastructure.models.pytorch.reproducibility import resolve_device, seed_everything
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

logger = logging.getLogger(__name__)


class FederatedClient:
    """Represents a single federated learning client."""

    def __init__(
        self,
        client_id: str,
        dataset: SequenceDataset,
        engine: TrainingEnginePort,
        config: TrainingConfig,
        *,
        client_index: int = 0,
    ) -> None:
        """Initialize client.

        Args:
            client_id: Unique identifier (e.g. 'client_1').
            dataset: Local sequence dataset.
            engine: Training engine instance (TorchTrainingEngine or FedProxTrainingEngine).
            config: Model and training hyperparameters.
            client_index: 0-indexed position of client among all clients.
        """
        self._client_id = client_id
        self._dataset = dataset
        self._engine = engine
        self._config = config
        self._client_index = client_index

    @property
    def client_id(self) -> str:
        """Client identifier."""
        return self._client_id

    @property
    def sample_count(self) -> int:
        """Total local training sequences."""
        return self._dataset.sequence_count

    @property
    def positive_count(self) -> int:
        """Total local positive sequences."""
        return self._dataset.positive_count

    @property
    def engine(self) -> TrainingEnginePort:
        """Client training engine."""
        return self._engine

    @property
    def control_variate_delta(self) -> dict[str, torch.Tensor]:
        """SCAFFOLD control variate update delta if SCAFFOLD engine is used."""
        if isinstance(self._engine, SCAFFOLDTrainingEngine):
            return self._engine.last_delta_c
        return {}

    def train_round(
        self,
        global_parameters: ParameterState,
        *,
        round_num: int,
        local_epochs: int = 1,
    ) -> tuple[ParameterState, float, int, float]:
        """Perform local training from the global model state.

        Args:
            global_parameters: Server global model parameters to start from.
            round_num: Current 1-indexed communication round.
            local_epochs: Number of local training passes over the local data.

        Returns:
            Tuple of (updated_parameter_state, average_train_loss, sample_count, train_seconds).
        """
        started = time.perf_counter()

        # Seed locally based on base seed, round, and client index for full reproducibility
        seed = self._config.random_seed + (round_num * 1000) + self._client_index
        seed_everything(seed)
        device = resolve_device(self._config.device)

        model = build_classifier(self._config, self._dataset.feature_dimension)
        if isinstance(self._engine, TorchTrainingEngine):
            self._engine._load_into(
                model, global_parameters, self._dataset.feature_dimension
            )
        model.to(device)

        # If FedProx engine, ensure global parameters are set for proximal penalty
        if isinstance(self._engine, FedProxTrainingEngine):
            self._engine._global_parameters = global_parameters
        elif isinstance(self._engine, SCAFFOLDTrainingEngine):
            self._engine._initial_global_parameters = global_parameters
            self._engine._total_steps_taken = 0

        criterion = nn.BCEWithLogitsLoss()
        optimizer: torch.optim.Optimizer
        if isinstance(self._engine, TorchTrainingEngine):
            optimizer = self._engine._build_optimizer(self._config, model)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self._config.learning_rate
            )

        logger.info(
            "Client %s starting local training for round %d: %d samples, %d local epoch(s), lr=%g",
            self._client_id,
            round_num,
            self._dataset.sequence_count,
            local_epochs,
            self._config.learning_rate,
        )

        total_loss = 0.0
        for ep in range(1, local_epochs + 1):
            if isinstance(self._engine, TorchTrainingEngine):
                ep_loss = self._engine._train_one_epoch(
                    self._config, model, self._dataset, criterion, optimizer, ep
                )
            else:
                raise TypeError(f"Unsupported engine type: {type(self._engine)}")
            total_loss += ep_loss

        avg_loss = total_loss / local_epochs if local_epochs > 0 else 0.0

        if isinstance(self._engine, TorchTrainingEngine):
            final_state = self._engine._snapshot(model)
        else:
            final_state = {
                k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
            }

        if isinstance(self._engine, SCAFFOLDTrainingEngine):
            self._engine.compute_control_variate_update(
                final_state, self._config.learning_rate
            )

        elapsed = time.perf_counter() - started
        logger.info(
            "Client %s completed round %d: train_loss=%.6f, time=%.1fs",
            self._client_id,
            round_num,
            avg_loss,
            elapsed,
        )

        return final_state, avg_loss, self._dataset.sequence_count, elapsed

