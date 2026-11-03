"""Unit tests for FederatedClient."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
)
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine


@pytest.fixture
def dummy_client_dataset() -> AccountSequenceDataset:
    """Create a small deterministic AccountSequenceDataset."""
    n_rows = 100
    n_features = 8
    rng = np.random.default_rng(42)

    features = rng.standard_normal((n_rows, n_features), dtype=np.float32)
    targets = np.zeros(n_rows, dtype=np.float32)
    targets[::10] = 1.0  # 10% positives
    previous_row = np.full(n_rows, -1, dtype=np.int64)
    for i in range(1, n_rows):
        if i % 5 != 0:
            previous_row[i] = i - 1

    return AccountSequenceDataset(
        split=SplitName.TRAIN,
        features=features,
        targets=targets,
        previous_row=previous_row,
        sequence_length=5,
        stride=1,
    )


def test_client_local_training_updates_parameters(
    dummy_client_dataset: AccountSequenceDataset,
) -> None:
    metrics = SklearnMetricsCalculator()
    engine = TorchTrainingEngine(metrics)
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=16,
        batch_size=16,
        learning_rate=0.01,
        epochs=1,
    )

    client = FederatedClient(
        client_id="client_1",
        dataset=dummy_client_dataset,
        engine=engine,
        config=config,
    )

    initial_model = build_classifier(config, dummy_client_dataset.feature_dimension)
    initial_state = {
        k: v.detach().clone() for k, v in initial_model.state_dict().items()
    }

    updated_state, train_loss, count, elapsed = client.train_round(
        initial_state, round_num=1, local_epochs=1
    )

    assert count == dummy_client_dataset.sequence_count
    assert train_loss > 0.0
    assert elapsed > 0.0

    # Model parameters should have been updated by gradient descent
    changed = False
    for k in initial_state:
        if not torch.allclose(initial_state[k], updated_state[k]):
            changed = True
            break
    assert changed, "Client training should update model parameters"

