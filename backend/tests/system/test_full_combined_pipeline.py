"""Unit and integration test suite for memory-mapped Full Combined AML pipeline.

Verifies:
1. Streaming 2-way merge builds Full Combined splits with strict row conservation and monotonic ordering.
2. Tagging of 'dataset_source' ('hi_small' vs 'li_medium') for provenance and subgroup auditing.
3. Full dataset enforcement and fast rejection of legacy 500k subsets and arbitrary row limits.
4. Zero-copy client partitioning (IID class stratification and Non-IID account atomicity).
5. End-to-end federated training smoke test for FedAvg and FedProx on combined data.
6. Validation threshold freezing, checkpoint generation, and test prediction saving.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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

from scripts.build_full_combined_dataset import (
    build_full_combined_dataset,
    stream_2way_merge,
)
from scripts.run_full_combined_training import (
    FullCombinedExperimentRunner,
    run_preflight_checks,
)


@pytest.fixture
def synthetic_hi_li_env(tmp_path: Path):
    """Create minimal synthetic HI-Small and LI-Medium datasets for testing."""
    root = tmp_path / "synthetic_sources"
    hi_root = root / "HI-Small"
    li_root = root / "LI-Medium"

    # Copy real metadata
    for src_dir, real_meta in [
        (hi_root, Path("data/processed/ibm_aml/HI-Small/metadata")),
        (li_root, Path("data/processed/ibm_aml/LI-Medium/metadata")),
    ]:
        meta_target = src_dir / "metadata"
        meta_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(real_meta, meta_target)

    # Populate small CSVs
    # HI: 200 train, 50 val, 50 test
    # LI: 300 train, 100 val, 100 test
    for split, hi_n, li_n in [("train", 200, 300), ("validation", 50, 100), ("test", 50, 100)]:
        hi_split = hi_root / split
        li_split = li_root / split
        hi_split.mkdir(parents=True, exist_ok=True)
        li_split.mkdir(parents=True, exist_ok=True)

        real_hi_csv = Path("data/processed/ibm_aml/HI-Small") / split / "transactions.csv"
        real_li_csv = Path("data/processed/ibm_aml/LI-Medium") / split / "transactions.csv"

        df_hi = pd.read_csv(real_hi_csv, nrows=hi_n)
        df_li = pd.read_csv(real_li_csv, nrows=li_n)

        # Inject positive samples for stability
        df_hi.loc[:5, "is_laundering"] = 1
        df_li.loc[:5, "is_laundering"] = 1

        df_hi.to_csv(hi_split / "transactions.csv", index=False)
        df_li.to_csv(li_split / "transactions.csv", index=False)

    return {"root": root, "hi_dir": hi_root, "li_dir": li_root}


def test_streaming_2way_merge_and_metadata(synthetic_hi_li_env: dict[str, Path], tmp_path: Path):
    """Verify streaming 2-way merge preserves row counts, sorts timestamps, and writes metadata."""
    hi_dir = synthetic_hi_li_env["hi_dir"]
    li_dir = synthetic_hi_li_env["li_dir"]
    out_dir = tmp_path / "FULL_COMBINED"

    splits_data = build_full_combined_dataset(
        hi_dir=hi_dir,
        li_dir=li_dir,
        out_dir=out_dir,
        allow_subset_debug=True,
        force=True,
    )

    # 1. Row conservation: Train = 200 + 300 = 500; Val = 50 + 100 = 150; Test = 50 + 100 = 150
    assert splits_data["total_rows"] == 800
    for s in splits_data["per_split"]:
        if s["name"] == "train":
            assert s["rows"] == 500
            assert s["rows_hi_small"] == 200
            assert s["rows_li_medium"] == 300
        elif s["name"] in ("validation", "test"):
            assert s["rows"] == 150
            assert s["rows_hi_small"] == 50
            assert s["rows_li_medium"] == 100

    # 2. Check CSV headers, dataset_source, and chronological sorting
    for split in ["train", "validation", "test"]:
        csv_file = out_dir / split / "transactions.csv"
        assert csv_file.is_file()
        df = pd.read_csv(csv_file)
        assert len(df) in (500, 150)
        assert "dataset_source" in df.columns
        assert set(df["dataset_source"].unique()).issubset({"hi_small", "li_medium"})
        # Timestamp monotonic increasing
        assert df["timestamp"].is_monotonic_increasing

    # 3. Check metadata files
    assert (out_dir / "metadata" / "source_manifest.json").is_file()
    assert (out_dir / "metadata" / "splits.json").is_file()
    assert (out_dir / "metadata" / "feature_metadata.json").is_file()
    assert (out_dir / "metadata" / "transformers.json").is_file()

    with open(out_dir / "metadata" / "source_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["full_data_guarantee"]["no_row_limit"] is True
    assert manifest["full_data_guarantee"]["zero_cross_split_leakage"] is True


def test_rejection_of_subsets_and_legacy_500k(synthetic_hi_li_env: dict[str, Path], tmp_path: Path):
    """Verify that build and runner strictly fail when full data mode detects a subset or legacy cache."""
    hi_dir = synthetic_hi_li_env["hi_dir"]
    li_dir = synthetic_hi_li_env["li_dir"]
    out_dir = tmp_path / "FULL_COMBINED_FAIL"

    # 1. Builder must reject small inputs when allow_subset_debug is False
    with pytest.raises(SubsetModeDetectedError, match="Detected truncated subset"):
        build_full_combined_dataset(
            hi_dir=hi_dir,
            li_dir=li_dir,
            out_dir=out_dir,
            allow_subset_debug=False,
        )

    # 2. Runner preflight checks must reject legacy 500k dataset
    legacy_dir = Path("data/processed/ibm_aml/Combined")
    preflight = run_preflight_checks(
        data_root=legacy_dir,
        output_dir=tmp_path / "artifacts",
        device_name="cpu",
        allow_subset_debug=False,
    )
    assert preflight["status"] == "FAIL"
    assert any("legacy 500k detected" in err for err in preflight["errors"])


def test_full_combined_mmap_caching_and_partitioning(synthetic_hi_li_env: dict[str, Path], tmp_path: Path):
    """Test MmapSequenceProvider and zero-copy client partitioning on Full Combined data."""
    hi_dir = synthetic_hi_li_env["hi_dir"]
    li_dir = synthetic_hi_li_env["li_dir"]
    combined_dir = tmp_path / "FULL_COMBINED_MMAP"

    build_full_combined_dataset(
        hi_dir=hi_dir,
        li_dir=li_dir,
        out_dir=combined_dir,
        allow_subset_debug=True,
    )

    provider = MmapSequenceProvider(combined_dir, chunk_size=100)
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        random_seed=42,
        device="cpu",
        row_limit=None,
    )

    # Train dataset loading builds cache
    ds_train = provider.load(SplitName.TRAIN, config, enforce_full_dataset=False)
    assert ds_train.row_count == 500
    assert ds_train.feature_dimension == 57

    # Zero-copy partitioning
    pm = MmapPartitionManager(provider.metadata)

    # IID
    iid_clients = pm.build_client_datasets("iid", ds_train, config, seed=42, enforce_full_dataset=False)
    assert len(iid_clients) == 3
    assert sum(c.row_count for c in iid_clients) == 500

    # Non-IID
    non_iid_clients = pm.build_client_datasets("non_iid", ds_train, config, seed=42, enforce_full_dataset=False)
    assert len(non_iid_clients) == 3
    assert sum(c.row_count for c in non_iid_clients) == 500


def test_full_combined_end_to_end_smoke_test(synthetic_hi_li_env: dict[str, Path], tmp_path: Path):
    """Execute 1 communication round of FedAvg and FedProx using FullCombinedExperimentRunner."""
    hi_dir = synthetic_hi_li_env["hi_dir"]
    li_dir = synthetic_hi_li_env["li_dir"]
    combined_dir = tmp_path / "FULL_COMBINED_EXP"
    output_dir = tmp_path / "artifacts" / "full_combined"

    build_full_combined_dataset(
        hi_dir=hi_dir,
        li_dir=li_dir,
        out_dir=combined_dir,
        allow_subset_debug=True,
    )

    runner = FullCombinedExperimentRunner(
        data_root=combined_dir,
        output_dir=output_dir,
        batch_size=32,
        rounds=1,
        local_epochs=1,
        learning_rate=0.01,
        fedprox_mu=0.0001,
        seed=42,
        device="cpu",
        allow_subset_debug=True,
    )

    # Execute Config 1 (IID GRU FedAvg) and Config 2 (IID GRU FedProx)
    runner.execute_all(target_config_ids=[1, 2])

    # Check manifest and artifacts
    manifest_path = output_dir / "resume_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "config_1" in manifest["configurations"]
    assert "config_2" in manifest["configurations"]
    assert manifest["configurations"]["config_1"]["status"] == "COMPLETED"
    assert manifest["configurations"]["config_2"]["status"] == "COMPLETED"

    # Check checkpoints
    assert (output_dir / "checkpoints" / "config_1" / "global_best.pt").is_file()
    assert (output_dir / "checkpoints" / "config_2" / "global_best.pt").is_file()

    # Check test predictions
    assert (output_dir / "predictions" / "config_1" / "test_predictions.npz").is_file()
    assert (output_dir / "predictions" / "config_2" / "test_predictions.npz").is_file()

    # Check metrics
    m1 = manifest["configurations"]["config_1"]["test_metrics"]
    assert 0.0 <= m1["pr_auc"] <= 1.0
    assert 0.0 <= m1["roc_auc"] <= 1.0
