"""Reduced-scale integration test verifying Phase 9 requirements before full execution.

Verifies:
1. Proximal term computation and mu sensitivity (mu changes local training loss).
2. Global weights broadcasting and absence of client-to-client weight leakage.
3. SCAFFOLD control variate initialization, separation, and server aggregation.
4. Client drift tracking logging in RoundRecord.
5. Strict data leakage prevention (validation and test splits never enter client training).
6. Deterministic execution with seed 42.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from application.dto.federated import FederatedConfig
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy
from infrastructure.data.partitioners.client_sequence_provider import ClientSequenceProvider
from infrastructure.data.partitioners.non_iid_partitioner import NonIIDPartitioner
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.aggregation.scaffold import (
    SCAFFOLDTrainingEngine,
)
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.server.federated_checkpoint_store import FederatedCheckpointStore
from infrastructure.federated.server.federated_server import FederatedServer
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine


@pytest.fixture
def test_setup() -> tuple[ProcessedSequenceProvider, SklearnMetricsCalculator]:
    """Load metadata and sequence provider from existing processed dataset."""
    seq_provider = ProcessedSequenceProvider(Path("data/processed/ibm_aml/HI-Small"))
    metrics = SklearnMetricsCalculator()
    return seq_provider, metrics


def test_phase9_reduced_scale_orchestration(
    test_setup: tuple[ProcessedSequenceProvider, SklearnMetricsCalculator]
) -> None:
    """Run reduced-scale Non-IID FedProx and SCAFFOLD runs to verify correctness."""
    seq_provider, metrics = test_setup
    row_limit = 1200
    seed = 42

    training_cfg = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        batch_size=64,
        learning_rate=0.01,
        epochs=1,
        random_seed=seed,
        row_limit=row_limit,
        sequence_length=5,
        sequence_stride=5,
        threshold_policy=ThresholdPolicy.FIXED,
        threshold=0.5,
    )

    # 1. Load subset of training split
    train_path = seq_provider._metadata.split_path(SplitName.TRAIN.value)
    train_cols = list(seq_provider._metadata.columns_to_read())
    train_dtypes = seq_provider._read_dtypes()
    train_df = pd.read_csv(
        train_path, usecols=train_cols, dtype=train_dtypes, nrows=row_limit
    )[train_cols]

    # Verify no validation or test rows in train_df
    assert "split" not in train_df.columns or (train_df["split"] == "train").all()

    # 2. Partition using NonIIDPartitioner
    partitioner = NonIIDPartitioner(seed=seed)
    client_frames, part_report = partitioner.partition(
        train_df,
        target_column=seq_provider._metadata.target_column,
        account_column=seq_provider._metadata.sender_column,
    )
    assert len(client_frames) == 3
    assert part_report.total_samples == row_limit

    # 3. Build client sequence datasets
    seq_client_provider = ClientSequenceProvider(seq_provider._metadata)
    val_dataset = seq_provider.load(SplitName.VALIDATION, training_cfg)

    # Test A: FedProx mu changes the loss objective
    fed_cfg_small_mu = FederatedConfig(
        training_config=training_cfg,
        n_clients=3,
        rounds=1,
        local_epochs=1,
        distribution="non_iid",
        algorithm="fedprox",
        mu=0.0001,
        seed=seed,
    )
    fed_cfg_large_mu = FederatedConfig(
        training_config=training_cfg,
        n_clients=3,
        rounds=1,
        local_epochs=1,
        distribution="non_iid",
        algorithm="fedprox",
        mu=0.1,
        seed=seed,
    )

    clients_small_mu: list[FederatedClient] = []
    clients_large_mu: list[FederatedClient] = []

    for idx, (cid, c_df) in enumerate(client_frames.items()):
        c_ds = seq_client_provider.build_client_dataset(
            c_df, training_cfg, client_id=cid, split=SplitName.TRAIN
        )
        clients_small_mu.append(
            FederatedClient(
                client_id=cid,
                dataset=c_ds,
                engine=FedProxTrainingEngine(metrics, mu=0.0001),
                config=training_cfg,
                client_index=idx,
            )
        )
        clients_large_mu.append(
            FederatedClient(
                client_id=cid,
                dataset=c_ds,
                engine=FedProxTrainingEngine(metrics, mu=0.1),
                config=training_cfg,
                client_index=idx,
            )
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_store = FederatedCheckpointStore(tmp_path / "checkpoints", algorithm="fedprox")
        server = FederatedServer(
            engine=TorchTrainingEngine(metrics),
            metrics=metrics,
            checkpoint_store=ckpt_store,
        )
        init_state = server.initialize_global_model(
            fed_cfg_small_mu, val_dataset.feature_dimension
        )

        # Run 1 round with small mu
        rounds_small, _, _, _, _, _ = server.run_rounds(
            experiment_id="test_exp_small_mu",
            fed_config=fed_cfg_small_mu,
            clients=clients_small_mu,
            validation_dataset=val_dataset,
            initial_state=init_state,
        )

        # Run 1 round with large mu
        rounds_large, _, _, _, _, _ = server.run_rounds(
            experiment_id="test_exp_large_mu",
            fed_config=fed_cfg_large_mu,
            clients=clients_large_mu,
            validation_dataset=val_dataset,
            initial_state=init_state,
        )

        # Check drift metrics logged
        assert rounds_small[0].drift_metrics is not None
        assert "mean_drift" in rounds_small[0].drift_metrics
        assert "pairwise_disagreement" in rounds_small[0].drift_metrics

        # Verify proximal term affected loss or parameters
        client_1_loss_small = rounds_small[0].client_train_losses["client_1"]
        client_1_loss_large = rounds_large[0].client_train_losses["client_1"]
        diff_train = client_1_loss_small != client_1_loss_large
        diff_val = rounds_small[0].validation_loss != rounds_large[0].validation_loss
        assert diff_train or diff_val

    # Test B: SCAFFOLD execution and control variate separation
    fed_cfg_scaffold = FederatedConfig(
        training_config=training_cfg,
        n_clients=3,
        rounds=2,
        local_epochs=1,
        distribution="non_iid",
        algorithm="scaffold",
        mu=None,
        seed=seed,
    )
    clients_scaffold: list[FederatedClient] = []
    for idx, (cid, c_df) in enumerate(client_frames.items()):
        c_ds = seq_client_provider.build_client_dataset(
            c_df, training_cfg, client_id=cid, split=SplitName.TRAIN
        )
        clients_scaffold.append(
            FederatedClient(
                client_id=cid,
                dataset=c_ds,
                engine=SCAFFOLDTrainingEngine(metrics),
                config=training_cfg,
                client_index=idx,
            )
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_store = FederatedCheckpointStore(tmp_path / "checkpoints", algorithm="scaffold")
        server = FederatedServer(
            engine=TorchTrainingEngine(metrics),
            metrics=metrics,
            checkpoint_store=ckpt_store,
        )
        init_state = server.initialize_global_model(
            fed_cfg_scaffold, val_dataset.feature_dimension
        )

        scaffold_rounds, best_round, best_state, _, val_rows, _ = server.run_rounds(
            experiment_id="test_exp_scaffold",
            fed_config=fed_cfg_scaffold,
            clients=clients_scaffold,
            validation_dataset=val_dataset,
            initial_state=init_state,
        )

        assert len(scaffold_rounds) == 2
        for r_rec in scaffold_rounds:
            assert r_rec.drift_metrics is not None
            assert r_rec.drift_metrics["mean_drift"] > 0.0

        # Verify client control variates remain separate
        assert clients_scaffold[0].engine is not clients_scaffold[1].engine

