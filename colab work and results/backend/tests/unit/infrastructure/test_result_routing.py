"""Tests for experiment result routing and category separation (Phase 5).

Verifies the 10 required routing, separation, and cross-contamination guarantees:
- TEST 1: Centralized results written only to centralized_results.csv.
- TEST 2: IID FedAvg results written only to iid_fedavg_results.csv.
- TEST 3: IID FedProx results written only to iid_fedprox_results.csv.
- TEST 4: FedAvg records never appear in FedProx CSV.
- TEST 5: FedProx records never appear in FedAvg CSV.
- TEST 6: Centralized records never appear in either federated CSV.
- TEST 7: Existing experiment records preserved without loss or duplication.
- TEST 8: Small/reduced runs write to correct category.
- TEST 9: Phase 4 centralized session factory wiring.
- TEST 10: Phase 5 federated session factory wiring.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from application.dto.experiment import (
    CENTRALIZED_DISTRIBUTION,
    NO_FEDERATED_ALGORITHM,
    ClassificationMetrics,
    ConfusionCounts,
    DataProvenance,
    ExperimentReport,
    ExperimentResult,
    SequenceStrategy,
)
from application.dto.federated import (
    ClientPartitionStats,
    FederatedConfig,
    FederatedExperimentReport,
    IIDPartitionReport,
    RoundRecord,
)
from application.dto.training import EpochRecord, TrainingConfig
from composition_root import build_federated_session, build_training_session
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy, TrainingMode
from infrastructure.persistence.csv_experiment_result_store import (
    CENTRALIZED_RESULTS_TABLE,
    IID_FEDAVG_RESULTS_TABLE,
    IID_FEDPROX_RESULTS_TABLE,
    CategoryMismatchError,
    CsvExperimentResultStore,
)

pytestmark = pytest.mark.unit


def _make_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        precision=0.8,
        recall=0.5,
        f1=0.615,
        pr_auc=0.45,
        roc_auc=0.92,
        accuracy=0.998,
        confusion=ConfusionCounts(
            true_positives=5,
            true_negatives=990,
            false_positives=1,
            false_negatives=5,
        ),
        threshold=0.5,
        positive_rate=0.01,
        loss=0.02,
    )


def make_centralized_report(experiment_id: str = "centralized_exp_1") -> ExperimentReport:
    """Build a synthetic centralized experiment report."""
    cfg = TrainingConfig(architecture=ModelArchitecture.GRU, hidden_size=16)
    result = ExperimentResult(
        experiment_id=experiment_id,
        architecture=ModelArchitecture.GRU,
        training_mode=TrainingMode.CENTRALIZED,
        split=SplitName.TEST,
        metrics=_make_metrics(),
        threshold_policy=ThresholdPolicy.FIXED,
        seed=42,
        data_distribution=CENTRALIZED_DISTRIBUTION,
        federated_algorithm=NO_FEDERATED_ALGORITHM,
        client_id=None,
        federated_round=None,
        mu=None,
        loss_type="bce_with_logits",
        training_time_seconds=2.5,
        evaluation_time_seconds=0.5,
        sequence_count=100,
        hyperparameters={"hidden_size": 16},
    )
    return ExperimentReport(
        experiment_id=experiment_id,
        config=cfg,
        provenance=DataProvenance(
            processed_directory="data/processed/ibm_aml/HI-Small",
            preprocessing_version="0.1.0",
            feature_columns=("f1", "f2"),
            identifier_columns=("ts", "acc1", "acc2"),
            target_column="is_laundering",
            feature_dimension=2,
            rows_per_split={"train": 100},
            positives_per_split={"train": 2},
        ),
        sequences=SequenceStrategy(
            grouping_column="acc1",
            sequence_length=10,
            stride=1,
            target_assignment="newest",
            padding="right-padded",
            boundary_policy="no-cross",
            ordering="ts",
        ),
        epochs=(
            EpochRecord(
                epoch=1,
                train_loss=0.1,
                validation_loss=0.1,
                validation_metrics={"f1": 0.6},
                seconds=1.0,
                selected=True,
            ),
        ),
        results=(result,),
        selected_epoch=1,
        frozen_threshold=0.5,
        threshold_selected_on=SplitName.VALIDATION.value,
        device="cpu",
        parameter_count=500,
        training_seconds=2.5,
        data_preparation_seconds=0.5,
        total_seconds=3.0,
        early_stopped=False,
        torch_version="2.2.0",
        deterministic=True,
    )


def make_federated_report(
    experiment_id: str,
    algorithm: str = "fedavg",
    distribution: str = "iid",
    mu: float | None = None,
) -> FederatedExperimentReport:
    """Build a synthetic federated experiment report."""
    t_cfg = TrainingConfig(architecture=ModelArchitecture.GRU, hidden_size=16)
    fed_cfg = FederatedConfig(
        training_config=t_cfg,
        n_clients=3,
        rounds=2,
        local_epochs=1,
        distribution=distribution,
        algorithm=algorithm,
        mu=mu,
        seed=42,
    )
    loss_type = "fedprox" if algorithm == "fedprox" else "bce_with_logits"
    result = ExperimentResult(
        experiment_id=experiment_id,
        architecture=ModelArchitecture.GRU,
        training_mode=TrainingMode.FEDERATED,
        split=SplitName.TEST,
        metrics=_make_metrics(),
        threshold_policy=ThresholdPolicy.FIXED,
        seed=42,
        data_distribution=distribution,
        federated_algorithm=algorithm,
        client_id=None,
        federated_round=2,
        mu=mu,
        loss_type=loss_type,
        training_time_seconds=4.0,
        evaluation_time_seconds=0.5,
        sequence_count=100,
        hyperparameters={"algorithm": algorithm, "mu": mu},
    )
    client_rec = ClientPartitionStats(
        client_id="client_0",
        sample_count=50,
        positive_count=1,
        negative_count=49,
        positive_rate=0.02,
        distribution=distribution,
        seed=42,
    )
    part_rep = IIDPartitionReport(
        clients=(client_rec,),
        total_samples=50,
        total_positives=1,
        overall_positive_rate=0.02,
        seed=42,
    )
    round_rec = RoundRecord(
        round=1,
        aggregation_weights={"client_0": 1.0},
        client_train_losses={"client_0": 0.1},
        client_sample_counts={"client_0": 50},
        validation_loss=0.1,
        validation_metrics=_make_metrics(),
        round_seconds=1.0,
        checkpoint_path=None,
        selected=True,
    )
    return FederatedExperimentReport(
        experiment_id=experiment_id,
        config=fed_cfg,
        partition_report=part_rep,
        rounds=(round_rec,),
        results=(result,),
        selected_round=2,
        frozen_threshold=0.5,
        threshold_selected_on="validation",
        device="cpu",
        parameter_count=500,
        training_seconds=4.0,
        data_preparation_seconds=0.5,
        total_seconds=4.5,
        torch_version="2.2.0",
        deterministic=True,
    )


def test_1_centralized_result_written_only_to_centralized_csv(tmp_path: Path) -> None:
    """TEST 1: A centralized result is written only to centralized_results.csv."""
    root = tmp_path / "results"
    store = CsvExperimentResultStore.for_centralized(root)
    rep = make_centralized_report("centralized_run_1")

    written = store.append_rows(rep)
    expected = root / CENTRALIZED_RESULTS_TABLE

    assert written == expected
    assert expected.is_file()

    # Verify no federated paths exist
    assert not (root / "federated").exists()

    rows = store.load_rows()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "centralized_run_1"
    assert rows[0]["training_mode"] == "centralized"


def test_2_iid_fedavg_result_written_only_to_iid_fedavg_csv(tmp_path: Path) -> None:
    """TEST 2: An IID FedAvg result is written only to iid_fedavg_results.csv."""
    root = tmp_path / "results"
    store = CsvExperimentResultStore.for_federated(root, distribution="iid", algorithm="fedavg")
    rep = make_federated_report("fed_iid_fedavg_1", algorithm="fedavg", distribution="iid")

    written = store.append_rows(rep)  # type: ignore[arg-type]
    expected = root / IID_FEDAVG_RESULTS_TABLE

    assert written == expected
    assert expected.is_file()

    # Verify centralized and fedprox paths do not exist
    assert not (root / "centralized").exists()
    assert not (root / "federated" / "iid" / "fedprox").exists()

    rows = store.load_rows()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "fed_iid_fedavg_1"
    assert rows[0]["federated_algorithm"] == "fedavg"


def test_3_iid_fedprox_result_written_only_to_iid_fedprox_csv(tmp_path: Path) -> None:
    """TEST 3: An IID FedProx result is written only to iid_fedprox_results.csv."""
    root = tmp_path / "results"
    store = CsvExperimentResultStore.for_federated(root, distribution="iid", algorithm="fedprox")
    rep = make_federated_report(
        "fed_iid_fedprox_1", algorithm="fedprox", distribution="iid", mu=0.01
    )

    written = store.append_rows(rep)  # type: ignore[arg-type]
    expected = root / IID_FEDPROX_RESULTS_TABLE

    assert written == expected
    assert expected.is_file()

    # Verify centralized and fedavg paths do not exist
    assert not (root / "centralized").exists()
    assert not (root / "federated" / "iid" / "fedavg").exists()

    rows = store.load_rows()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "fed_iid_fedprox_1"
    assert rows[0]["federated_algorithm"] == "fedprox"
    assert float(rows[0]["mu"]) == 0.01


def test_4_fedavg_records_never_appear_in_fedprox_csv(tmp_path: Path) -> None:
    """TEST 4: FedAvg records never appear in the FedProx CSV."""
    root = tmp_path / "results"

    # Category store strictly rejects mismatched records
    fedprox_store = CsvExperimentResultStore.for_federated(
        root, distribution="iid", algorithm="fedprox"
    )
    fedavg_rep = make_federated_report("fedavg_run", algorithm="fedavg", distribution="iid")

    with pytest.raises(CategoryMismatchError):
        fedprox_store.append_rows(fedavg_rep)  # type: ignore[arg-type]

    # Auto-routing store separates them cleanly
    auto_store = CsvExperimentResultStore(root)
    fedprox_rep = make_federated_report(
        "fedprox_run", algorithm="fedprox", distribution="iid", mu=0.01
    )

    auto_store.append_rows(fedavg_rep)  # type: ignore[arg-type]
    auto_store.append_rows(fedprox_rep)  # type: ignore[arg-type]

    fedprox_rows = auto_store.load_rows(category="iid_fedprox")
    assert len(fedprox_rows) == 1
    for r in fedprox_rows:
        assert r["federated_algorithm"] == "fedprox"
        assert r["federated_algorithm"] != "fedavg"


def test_5_fedprox_records_never_appear_in_fedavg_csv(tmp_path: Path) -> None:
    """TEST 5: FedProx records never appear in the FedAvg CSV."""
    root = tmp_path / "results"

    # Category store strictly rejects mismatched records
    fedavg_store = CsvExperimentResultStore.for_federated(
        root, distribution="iid", algorithm="fedavg"
    )
    fedprox_rep = make_federated_report(
        "fedprox_run", algorithm="fedprox", distribution="iid", mu=0.01
    )

    with pytest.raises(CategoryMismatchError):
        fedavg_store.append_rows(fedprox_rep)  # type: ignore[arg-type]

    # Auto-routing store separates them cleanly
    auto_store = CsvExperimentResultStore(root)
    fedavg_rep = make_federated_report("fedavg_run", algorithm="fedavg", distribution="iid")

    auto_store.append_rows(fedavg_rep)  # type: ignore[arg-type]
    auto_store.append_rows(fedprox_rep)  # type: ignore[arg-type]

    fedavg_rows = auto_store.load_rows(category="iid_fedavg")
    assert len(fedavg_rows) == 1
    for r in fedavg_rows:
        assert r["federated_algorithm"] == "fedavg"
        assert r["federated_algorithm"] != "fedprox"


def test_6_centralized_records_never_appear_in_either_federated_csv(tmp_path: Path) -> None:
    """TEST 6: Centralized records never appear in either federated CSV."""
    root = tmp_path / "results"

    fedavg_store = CsvExperimentResultStore.for_federated(
        root, distribution="iid", algorithm="fedavg"
    )
    fedprox_store = CsvExperimentResultStore.for_federated(
        root, distribution="iid", algorithm="fedprox"
    )
    c_rep = make_centralized_report("c_run")

    with pytest.raises(CategoryMismatchError):
        fedavg_store.append_rows(c_rep)
    with pytest.raises(CategoryMismatchError):
        fedprox_store.append_rows(c_rep)

    auto_store = CsvExperimentResultStore(root)
    auto_store.append_rows(c_rep)

    assert not (root / IID_FEDAVG_RESULTS_TABLE).exists()
    assert not (root / IID_FEDPROX_RESULTS_TABLE).exists()
    assert (root / CENTRALIZED_RESULTS_TABLE).is_file()


def test_7_existing_experiment_records_preserved_without_loss_or_duplicates() -> None:
    """TEST 7: Existing experiment records are not lost during migration/separation."""
    c_path = Path("experiments/results/centralized/centralized_results.csv")
    fa_path = Path("experiments/results/federated/iid/fedavg/iid_fedavg_results.csv")
    fp_path = Path("experiments/results/federated/iid/fedprox/iid_fedprox_results.csv")
    bak_path = Path("experiments/results/experiment_results.csv.backup")

    assert c_path.is_file(), "Centralized CSV missing"
    assert fa_path.is_file(), "IID FedAvg CSV missing"
    assert fp_path.is_file(), "IID FedProx CSV missing"
    assert bak_path.is_file(), "Backup CSV missing"

    def read_csv(p: Path) -> list[dict[str, str]]:
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    c_rows = read_csv(c_path)
    fa_rows = read_csv(fa_path)
    fp_rows = read_csv(fp_path)
    bak_rows = read_csv(bak_path)

    assert len(c_rows) == 24
    assert len(fa_rows) == 21
    assert len(fp_rows) == 13
    assert len(bak_rows) == 58
    assert len(c_rows) + len(fa_rows) + len(fp_rows) == len(bak_rows)

    # Invariants in separated files
    for r in c_rows:
        assert r["training_mode"] == "centralized"
    for r in fa_rows:
        assert r["training_mode"] == "federated"
        assert r["data_distribution"] == "iid"
        assert r["federated_algorithm"] == "fedavg"
    for r in fp_rows:
        assert r["training_mode"] == "federated"
        assert r["data_distribution"] == "iid"
        assert r["federated_algorithm"] == "fedprox"

    # All experiment IDs in backup are accounted for
    orig_exp_ids = {r["experiment_id"] for r in bak_rows}
    separated_exp_ids = (
        {r["experiment_id"] for r in c_rows}
        | {r["experiment_id"] for r in fa_rows}
        | {r["experiment_id"] for r in fp_rows}
    )
    assert orig_exp_ids == separated_exp_ids


def test_8_running_small_reduced_experiment_writes_to_correct_category(tmp_path: Path) -> None:
    """TEST 8: Running a new small/reduced experiment writes to the correct category."""
    root = tmp_path / "results"
    store = CsvExperimentResultStore(root)

    c_rep = make_centralized_report("reduced_c")
    fa_rep = make_federated_report("reduced_fa", algorithm="fedavg", distribution="iid")
    fp_rep = make_federated_report("reduced_fp", algorithm="fedprox", distribution="iid", mu=0.01)

    store.append_rows(c_rep)
    store.append_rows(fa_rep)  # type: ignore[arg-type]
    store.append_rows(fp_rep)  # type: ignore[arg-type]

    assert (root / CENTRALIZED_RESULTS_TABLE).is_file()
    assert (root / IID_FEDAVG_RESULTS_TABLE).is_file()
    assert (root / IID_FEDPROX_RESULTS_TABLE).is_file()

    c_rows = store.load_rows(category="centralized")
    fa_rows = store.load_rows(category="iid_fedavg")
    fp_rows = store.load_rows(category="iid_fedprox")

    assert len(c_rows) == 1
    assert len(fa_rows) == 1
    assert len(fp_rows) == 1

    assert c_rows[0]["experiment_id"] == "reduced_c"
    assert fa_rows[0]["experiment_id"] == "reduced_fa"
    assert fp_rows[0]["experiment_id"] == "reduced_fp"


def test_9_centralized_pipeline_session_wiring(tmp_path: Path) -> None:
    """TEST 9: The existing Phase 4 centralized pipeline factory wires the correct result store."""
    config = {
        "paths": {"experiment_results": str(tmp_path / "results")},
        "training": {"architecture": "gru"},
        "reproducibility": {"random_seed": 42},
    }
    service, t_cfg = build_training_session(config)

    # Access the private _results store on service
    result_store = service._results
    assert isinstance(result_store, CsvExperimentResultStore)
    assert result_store.table_path == tmp_path / "results" / CENTRALIZED_RESULTS_TABLE


def test_10_federated_pipeline_session_wiring(tmp_path: Path) -> None:
    """TEST 10: The existing Phase 5 federated pipeline factory wires the correct result store."""
    config = {
        "paths": {"experiment_results": str(tmp_path / "results")},
        "training": {"architecture": "gru"},
        "federated": {
            "distribution": "iid",
            "algorithm": "fedavg",
            "n_clients": 3,
            "rounds": 2,
        },
        "reproducibility": {"random_seed": 42},
    }
    service, fed_cfg = build_federated_session(config)

    result_store = service._results
    assert isinstance(result_store, CsvExperimentResultStore)
    assert result_store.table_path == tmp_path / "results" / IID_FEDAVG_RESULTS_TABLE

    # Now with FedProx
    fed_prox_config = {
        "paths": {"experiment_results": str(tmp_path / "results")},
        "training": {"architecture": "gru"},
        "federated": {
            "distribution": "iid",
            "algorithm": "fedprox",
            "mu": 0.05,
            "n_clients": 3,
            "rounds": 2,
        },
        "reproducibility": {"random_seed": 42},
    }
    service_prox, _ = build_federated_session(fed_prox_config)
    result_store_prox = service_prox._results
    assert isinstance(result_store_prox, CsvExperimentResultStore)
    assert result_store_prox.table_path == tmp_path / "results" / IID_FEDPROX_RESULTS_TABLE


def test_11_non_iid_fedavg_result_routing(tmp_path: Path) -> None:
    """TEST 11: Non-IID FedAvg results are written strictly to per-experiment results.csv."""
    store = CsvExperimentResultStore.for_federated(
        tmp_path / "results", distribution="non_iid", algorithm="fedavg"
    )
    report = make_federated_report(
        "fed_non_iid_fedavg_exp", algorithm="fedavg", distribution="non_iid"
    )

    written_path = store.append_rows(report)  # type: ignore[arg-type]
    expected_path = (
        tmp_path
        / "results"
        / "federated"
        / "non_iid"
        / "fedavg"
        / "gru"
        / "fed_non_iid_fedavg_exp"
        / "results.csv"
    )
    assert written_path == expected_path
    assert written_path.is_file()

    # Load back using category
    rows = store.load_rows(category="non_iid_fedavg")
    assert len(rows) == 1
    assert rows[0]["data_distribution"] == "non_iid"
    assert rows[0]["federated_algorithm"] == "fedavg"


def test_12_non_iid_fedprox_result_routing(tmp_path: Path) -> None:
    """TEST 12: Non-IID FedProx results are written strictly to per-experiment results.csv."""
    store = CsvExperimentResultStore.for_federated(
        tmp_path / "results", distribution="non_iid", algorithm="fedprox"
    )
    report = make_federated_report(
        "fed_non_iid_fedprox_exp", algorithm="fedprox", distribution="non_iid", mu=0.01
    )

    written_path = store.append_rows(report)  # type: ignore[arg-type]
    expected_path = (
        tmp_path
        / "results"
        / "federated"
        / "non_iid"
        / "fedprox"
        / "gru"
        / "fed_non_iid_fedprox_exp"
        / "results.csv"
    )
    assert written_path == expected_path
    assert written_path.is_file()

    # Load back using category
    rows = store.load_rows(category="non_iid_fedprox")
    assert len(rows) == 1
    assert rows[0]["data_distribution"] == "non_iid"
    assert rows[0]["federated_algorithm"] == "fedprox"
    assert float(rows[0]["mu"]) == 0.01


def test_13_non_iid_cannot_contaminate_iid_or_centralized(tmp_path: Path) -> None:
    """TEST 13: Non-IID results cannot contaminate IID or centralized result stores."""
    iid_store = CsvExperimentResultStore.for_federated(
        tmp_path / "results", distribution="iid", algorithm="fedavg"
    )
    cent_store = CsvExperimentResultStore.for_centralized(tmp_path / "results")
    non_iid_report = make_federated_report(
        "fed_non_iid_exp", algorithm="fedavg", distribution="non_iid"
    )

    with pytest.raises(CategoryMismatchError, match="Refusing to append"):
        iid_store.append_rows(non_iid_report)  # type: ignore[arg-type]

    with pytest.raises(CategoryMismatchError, match="Refusing to append"):
        cent_store.append_rows(non_iid_report)  # type: ignore[arg-type]


def test_14_non_iid_pipeline_session_wiring(tmp_path: Path) -> None:
    """TEST 14: build_federated_session wires non_iid result store and checkpoint store."""
    config_avg = {
        "paths": {
            "experiment_results": str(tmp_path / "results"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "training": {"architecture": "gru"},
        "federated": {
            "distribution": "non-iid",
            "algorithm": "fedavg",
            "n_clients": 3,
            "rounds": 2,
        },
        "reproducibility": {"random_seed": 42},
    }
    service_avg, _ = build_federated_session(config_avg)
    res_store_avg = service_avg._results
    ckpt_store_avg = service_avg._checkpoints

    assert isinstance(res_store_avg, CsvExperimentResultStore)
    assert "non_iid" in str(res_store_avg.table_path)
    assert "non_iid" in str(ckpt_store_avg.root)

    config_prox = {
        "paths": {
            "experiment_results": str(tmp_path / "results"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "training": {"architecture": "gru"},
        "federated": {
            "distribution": "non_iid",
            "algorithm": "fedprox",
            "mu": 0.01,
            "n_clients": 3,
            "rounds": 2,
        },
        "reproducibility": {"random_seed": 42},
    }
    service_prox, _ = build_federated_session(config_prox)
    res_store_prox = service_prox._results
    ckpt_store_prox = service_prox._checkpoints

    assert isinstance(res_store_prox, CsvExperimentResultStore)
    assert "non_iid" in str(res_store_prox.table_path)
    assert "non_iid" in str(ckpt_store_prox.root)

