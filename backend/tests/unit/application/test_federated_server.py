"""Unit tests for FederatedServer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from application.dto.federated import FederatedConfig
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
)
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.server.federated_checkpoint_store import (
    FederatedCheckpointStore,
)
from infrastructure.federated.server.federated_server import FederatedServer
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine


@pytest.fixture
def dummy_dataset() -> AccountSequenceDataset:
    """Create a small dummy dataset."""
    n_rows = 60
    n_features = 4
    rng = np.random.default_rng(42)

    features = rng.standard_normal((n_rows, n_features), dtype=np.float32)
    targets = np.zeros(n_rows, dtype=np.float32)
    targets[::6] = 1.0
    previous_row = np.full(n_rows, -1, dtype=np.int64)

    return AccountSequenceDataset(
        split=SplitName.VALIDATION,
        features=features,
        targets=targets,
        previous_row=previous_row,
        sequence_length=4,
        stride=1,
    )


def test_server_multi_round_updates_global_model(
    tmp_path: Path, dummy_dataset: AccountSequenceDataset
) -> None:
    metrics = SklearnMetricsCalculator()
    engine = TorchTrainingEngine(metrics)
    store = FederatedCheckpointStore(tmp_path, algorithm="fedavg")

    t_cfg = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=8,
        batch_size=16,
        learning_rate=0.05,
    )
    fed_cfg = FederatedConfig(
        training_config=t_cfg,
        n_clients=2,
        rounds=2,
        local_epochs=1,
    )

    client1 = FederatedClient("client_1", dummy_dataset, engine, t_cfg, client_index=0)
    client2 = FederatedClient("client_2", dummy_dataset, engine, t_cfg, client_index=1)

    server = FederatedServer(
        engine=engine,
        metrics=metrics,
        checkpoint_store=store,
    )

    initial_state = server.initialize_global_model(fed_cfg, dummy_dataset.feature_dimension)

    records, best_round, best_state, final_state, val_rows, _ = server.run_rounds(
        experiment_id="test_exp",
        fed_config=fed_cfg,
        clients=[client1, client2],
        validation_dataset=dummy_dataset,
        initial_state=initial_state,
    )

    assert len(records) == 2
    assert len(val_rows) == 2
    assert best_round in (1, 2)

    # Verify global parameters changed from initial state
    diff = False
    for k in initial_state:
        if not torch.allclose(initial_state[k], final_state[k]):
            diff = True
            break
    assert diff, "Final global state should differ from initial state after federated rounds"

