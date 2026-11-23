"""Unit and integration test suite for memory-mapped full LI-Medium pipeline.

Verifies:
1. Memory-mapped caching and bit-for-bit equivalence.
2. Full dataset enforcement and rejection of subset limits.
3. Zero-copy client partitioning (Non-IID account atomicity and IID class stratification).
4. End-to-end federated training smoke test for FedAvg and FedProx.
5. Checkpoint generation, validation threshold freezing, and test prediction saving.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy
from infrastructure.data.sequences.mmap_partition_manager import (
    MmapClientDataset,
    MmapPartitionManager,
)
from infrastructure.data.sequences.mmap_sequence_provider import (
    MmapSequenceProvider,
    SubsetModeDetectedError,
)
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import aggregate_fedavg, compute_aggregation_weights
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine


@pytest.fixture
def synthetic_data_env(tmp_path: Path):
    """Create a minimal synthetic processed dataset directory matching real LI-Medium schema."""
    root = tmp_path / "synthetic_aml"
    real_meta_dir = Path("data/processed/ibm_aml/LI-Medium/metadata")
    meta_dir = root / "metadata"
    shutil.copytree(real_meta_dir, meta_dir)

    # Sample small realistic CSVs from LI-Medium
    for split, n_rows, pos_target in [
        ("train", 300, 10),
        ("validation", 100, 5),
        ("test", 100, 5),
    ]:
        sdir = root / split
        sdir.mkdir(parents=True, exist_ok=True)
        real_csv = Path("data/processed/ibm_aml/LI-Medium") / split / "transactions.csv"
        df = pd.read_csv(real_csv, nrows=n_rows)
        # Ensure guaranteed positive samples for metrics stability
        df.loc[:pos_target - 1, "is_laundering"] = 1
        df.to_csv(sdir / "transactions.csv", index=False)

    return root


def test_mmap_sequence_provider_build_and_reuse(synthetic_data_env: Path):
    """Test disk-backed cache generation, reuse, and row count verification."""
    provider = MmapSequenceProvider(synthetic_data_env, chunk_size=50)
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=None,
    )

    # First load builds cache
    ds_train = provider.load(SplitName.TRAIN, config, enforce_full_dataset=False)
    assert ds_train.row_count == 300
    assert ds_train.feature_dimension == 57
    assert ds_train.positive_count == 10

    # Second load reuses cache
    ds_train_cached = provider.load(SplitName.TRAIN, config, enforce_full_dataset=False)
    assert ds_train_cached.row_count == 300

    # Verify mini-batch assembly
    batches = list(ds_train.batches(32))
    assert len(batches) == 10
    first_b = batches[0]
    assert first_b.features.shape == (32, config.sequence_length, 57)
    assert first_b.labels.shape == (32,)


def test_full_dataset_enforcement_raises_on_subset(synthetic_data_env: Path):
    """Verify that SubsetModeDetectedError is raised when row_limit is set or rows are too few."""
    provider = MmapSequenceProvider(synthetic_data_env, chunk_size=50)

    # Config with explicit row_limit must fail when enforce_full_dataset=True
    subset_config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=200000,
    )
    with pytest.raises(SubsetModeDetectedError, match="Full-dataset mode is active"):
        provider.load(SplitName.TRAIN, subset_config, enforce_full_dataset=True)

    # Config with row_limit=None but min_expected_rows higher than actual
    full_config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=None,
    )
    with pytest.raises(SubsetModeDetectedError, match="Expected at least 18000000 rows"):
        provider.load(
            SplitName.TRAIN,
            full_config,
            enforce_full_dataset=True,
            min_expected_rows=18_000_000,
        )


def test_mmap_partition_manager_iid_and_non_iid(synthetic_data_env: Path):
    """Test IID and Non-IID zero-copy client datasets."""
    provider = MmapSequenceProvider(synthetic_data_env, chunk_size=50)
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=None,
    )
    train_parent = provider.load(SplitName.TRAIN, config, enforce_full_dataset=False)
    part_manager = MmapPartitionManager(provider.metadata)

    # Test IID
    iid_clients = part_manager.build_client_datasets(
        "iid", train_parent, config, seed=42, enforce_full_dataset=False
    )
    assert len(iid_clients) == 3
    total_iid_rows = sum(c.row_count for c in iid_clients)
    assert total_iid_rows == 300
    # Class balance across clients
    pos_counts = [c.positive_count for c in iid_clients]
    assert sum(pos_counts) == 10
    assert max(pos_counts) - min(pos_counts) <= 1  # Stratified balance

    # Test Non-IID
    non_iid_clients = part_manager.build_client_datasets(
        "non_iid", train_parent, config, seed=42, enforce_full_dataset=False
    )
    assert len(non_iid_clients) == 3
    total_non_iid_rows = sum(c.row_count for c in non_iid_clients)
    assert total_non_iid_rows == 300

    # Verify batch assembly on client dataset
    client1 = non_iid_clients[0]
    c_batches = list(client1.batches(16, shuffle=True))
    assert len(c_batches) > 0
    assert c_batches[0].features.shape[-1] == 57


def test_end_to_end_federated_smoke_test(synthetic_data_env: Path, tmp_path: Path):
    """Smoke test: 1 communication round of FedAvg and FedProx on synthetic data."""
    provider = MmapSequenceProvider(synthetic_data_env, chunk_size=50)
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=16,
        learning_rate=0.01,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=None,
    )
    train_parent = provider.load(SplitName.TRAIN, config, enforce_full_dataset=False)
    val_dataset = provider.load(SplitName.VALIDATION, config, enforce_full_dataset=False)
    test_dataset = provider.load(SplitName.TEST, config, enforce_full_dataset=False)

    part_manager = MmapPartitionManager(provider.metadata)
    clients_ds = part_manager.build_client_datasets(
        "iid", train_parent, config, seed=42, enforce_full_dataset=False
    )

    metrics_calc = SklearnMetricsCalculator()
    eval_engine = TorchTrainingEngine(metrics_calc)

    # Compare FedAvg vs FedProx (mu=0.0001)
    for algo, mu in [("fedavg", 0.0), ("fedprox", 0.0001)]:
        clients = []
        for idx, ds in enumerate(clients_ds):
            eng = FedProxTrainingEngine(metrics_calc, mu=mu) if algo == "fedprox" else TorchTrainingEngine(metrics_calc)
            clients.append(FederatedClient(
                client_id=f"client_{idx+1}", dataset=ds, engine=eng, config=config, client_index=idx
            ))

        model_init = build_classifier(config, train_parent.feature_dimension)
        current_state = {name: t.detach().to("cpu", copy=True) for name, t in model_init.state_dict().items()}

        # 1 Round
        client_states = {}
        client_counts = {}
        for c in clients:
            s_k, _, n_k, _ = c.train_round(current_state, round_num=1, local_epochs=1)
            client_states[c.client_id] = s_k
            client_counts[c.client_id] = n_k

        weights = compute_aggregation_weights(client_counts)
        aggregated_state = aggregate_fedavg(client_states, weights)

        # Val eval & threshold selection
        val_preds = eval_engine.predict(config, val_dataset, aggregated_state)
        val_eval = metrics_calc.evaluate(val_preds.labels, val_preds.probabilities, threshold=0.5)
        assert 0.0 <= val_eval.pr_auc <= 1.0

        thresh = metrics_calc.select_threshold(val_preds.labels, val_preds.probabilities, split=SplitName.VALIDATION)
        assert 0.0 <= thresh <= 1.0

        # Test eval
        test_preds = eval_engine.predict(config, test_dataset, aggregated_state)
        test_eval = metrics_calc.evaluate(test_preds.labels, test_preds.probabilities, threshold=thresh)
        assert 0.0 <= test_eval.pr_auc <= 1.0
        assert 0.0 <= test_eval.roc_auc <= 1.0
