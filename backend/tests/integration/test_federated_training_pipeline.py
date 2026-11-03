"""Integration tests for the Phase 5 IID Federated Learning pipeline.

Tests end-to-end execution of FedAvg and FedProx with GRU, LSTM, and BiLSTM
architectures on reduced datasets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.dto.federated import FederatedConfig
from application.dto.training import TrainingConfig
from application.services.federated_training_service import FederatedTrainingService
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.training_mode import TrainingMode
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.server.federated_checkpoint_store import (
    FederatedCheckpointStore,
)
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine
from infrastructure.persistence.csv_experiment_result_store import (
    CsvExperimentResultStore,
)
from tests.fixtures.sequence_fixtures import write_processed_tree


@pytest.fixture
def dummy_processed_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal synthetic processed data directory with metadata and splits."""
    processed_dir = write_processed_tree(tmp_path / "data" / "processed" / "ibm_aml" / "HI-Small")
    checkpoints_dir = tmp_path / "models" / "checkpoints"
    results_dir = tmp_path / "experiments" / "results"
    return processed_dir, checkpoints_dir, results_dir


@pytest.mark.parametrize(
    "architecture",
    [ModelArchitecture.GRU, ModelArchitecture.LSTM, ModelArchitecture.BILSTM],
)
def test_federated_pipeline_architectures_fedavg(
    dummy_processed_environment: tuple[Path, Path, Path],
    architecture: ModelArchitecture,
) -> None:
    processed_dir, checkpoints_dir, results_dir = dummy_processed_environment

    metrics = SklearnMetricsCalculator()
    service = FederatedTrainingService(
        sequence_provider=ProcessedSequenceProvider(processed_dir),
        metrics=metrics,
        checkpoint_store=FederatedCheckpointStore(
            checkpoints_dir / "federated" / "iid", algorithm="fedavg"
        ),
        result_store=CsvExperimentResultStore(results_dir),
        training_engine=TorchTrainingEngine(metrics),
    )

    t_cfg = TrainingConfig(
        architecture=architecture,
        sequence_length=3,
        hidden_size=8,
        batch_size=16,
        learning_rate=0.01,
        epochs=1,
    )
    fed_cfg = FederatedConfig(
        training_config=t_cfg,
        n_clients=3,
        rounds=2,
        local_epochs=1,
        algorithm="fedavg",
    )

    report = service.run(fed_cfg)

    assert report.config.algorithm == "fedavg"
    assert len(report.rounds) == 2
    assert len(report.partition_report.clients) == 3
    assert report.selected_round in (1, 2)
    assert report.parameter_count > 0

    # Verify result rows
    for r in report.results:
        assert r.training_mode == TrainingMode.FEDERATED
        assert r.data_distribution == "iid"
        assert r.federated_algorithm == "fedavg"
        assert r.mu is None

    # Checkpoint directory verified
    ckpt_dir = (
        checkpoints_dir
        / "federated"
        / "iid"
        / "fedavg"
        / architecture.value
        / report.experiment_id
    )
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "round_1.pt").is_file()
    assert (ckpt_dir / "round_2.pt").is_file()
    assert (ckpt_dir / "final.pt").is_file()


def test_federated_pipeline_fedprox(
    dummy_processed_environment: tuple[Path, Path, Path],
) -> None:
    processed_dir, checkpoints_dir, results_dir = dummy_processed_environment

    metrics = SklearnMetricsCalculator()
    service = FederatedTrainingService(
        sequence_provider=ProcessedSequenceProvider(processed_dir),
        metrics=metrics,
        checkpoint_store=FederatedCheckpointStore(
            checkpoints_dir / "federated" / "iid", algorithm="fedprox"
        ),
        result_store=CsvExperimentResultStore(results_dir),
        training_engine=TorchTrainingEngine(metrics),
    )

    t_cfg = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        sequence_length=3,
        hidden_size=8,
        batch_size=16,
        learning_rate=0.01,
        epochs=1,
    )
    fed_cfg = FederatedConfig(
        training_config=t_cfg,
        n_clients=3,
        rounds=2,
        local_epochs=1,
        algorithm="fedprox",
        mu=0.02,
    )

    report = service.run(fed_cfg)

    assert report.config.algorithm == "fedprox"
    assert report.config.mu == 0.02
    assert len(report.rounds) == 2

    for r in report.results:
        assert r.training_mode == TrainingMode.FEDERATED
        assert r.data_distribution == "iid"
        assert r.federated_algorithm == "fedprox"
        assert r.mu == 0.02

    ckpt_dir = checkpoints_dir / "federated" / "iid" / "fedprox" / "gru" / report.experiment_id
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "round_1.pt").is_file()
    assert (ckpt_dir / "round_2.pt").is_file()
    assert (ckpt_dir / "final.pt").is_file()
