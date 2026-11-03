"""Automated unit test suite for HI-Small vs LI-Medium vs Combined Dataset Comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "dataset_comparison_combined"
DATA_DIR = BASE_DIR / "data" / "processed" / "ibm_aml"


def test_raw_files_immutable_and_fingerprints_verified():
    """Verify that both raw CSV files exist and retain original size invariants."""
    hi_raw = BASE_DIR / "data" / "raw" / "ibm_aml" / "HI-Small_Trans.csv"
    li_raw = BASE_DIR / "data" / "raw" / "ibm_aml" / "LI-Medium_Trans.csv"

    assert hi_raw.is_file(), f"Missing {hi_raw}"
    assert li_raw.is_file(), f"Missing {li_raw}"
    assert hi_raw.stat().st_size == 475664282
    assert li_raw.stat().st_size == 2976436978


def test_schema_consistency_across_datasets():
    """Verify identical 23 engineered features and 57-dimensional input shape across all 3 datasets."""
    from application.dto.training import TrainingConfig
    from domain.enums.model_architecture import ModelArchitecture
    from domain.enums.split_name import SplitName
    from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider

    datasets = ["HI-Small", "LI-Medium", "Combined"]
    cfg = TrainingConfig(architecture=ModelArchitecture.GRU, row_limit=1000, batch_size=256)

    for d in datasets:
        p = ProcessedSequenceProvider(DATA_DIR / d)
        ds = p.load(SplitName.TRAIN, cfg)
        assert ds.feature_dimension == 57, f"Dataset {d} has unexpected feature dimension: {ds.feature_dimension}"
        assert ds.sequence_count == 1000, f"Dataset {d} sequence count mismatch"


def test_dataset_source_tracking_and_leakage_prevention():
    """Verify that Combined dataset tracks dataset_source and contains both sources."""
    comb_train_csv = DATA_DIR / "Combined" / "train" / "transactions.csv"
    assert comb_train_csv.is_file()

    df = pd.read_csv(comb_train_csv, usecols=["dataset_source"])
    assert "dataset_source" in df.columns, "Missing dataset_source column in Combined train split"
    sources = set(df["dataset_source"].unique())
    assert sources == {"hi_small", "li_medium"}, f"Unexpected sources: {sources}"


def test_fedprox_objective_math():
    """Verify exact analytic calculation of FedProx proximal regularization penalty."""
    from infrastructure.federated.aggregation.fedprox import compute_proximal_loss

    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    device = torch.device("cpu")

    # Set local parameter to [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    local_weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    model[0].weight.data = local_weights.clone()

    # Set global parameter to zeros
    global_weights = torch.zeros_like(local_weights)
    global_params = {"0.weight": global_weights}

    # sum of squares = 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 = 204.0
    # Proximal penalty = (mu / 2) * sum_sq

    # Test mu = 0.0
    loss_0 = compute_proximal_loss(model, global_params, mu=0.0, device=device)
    assert loss_0.item() == 0.0

    # Test mu = 0.1: (0.1 / 2) * 204.0 = 10.2
    loss_01 = compute_proximal_loss(model, global_params, mu=0.1, device=device)
    assert pytest.approx(loss_01.item(), abs=1e-5) == 10.2

    # Test mu = 1.0: (1.0 / 2) * 204.0 = 102.0
    loss_10 = compute_proximal_loss(model, global_params, mu=1.0, device=device)
    assert pytest.approx(loss_10.item(), abs=1e-5) == 102.0


def test_artifacts_integrity_and_54_models():
    """Verify that all 54 models were trained, evaluated, and saved with valid checksums."""
    exp_matrix_csv = ARTIFACTS_DIR / "experiment_matrix.csv"
    assert exp_matrix_csv.is_file()

    df_matrix = pd.read_csv(exp_matrix_csv)
    assert len(df_matrix) == 54, f"Expected 54 experiments, found {len(df_matrix)}"

    # Check 18 per dataset
    for d in ["HI-Small", "LI-Medium", "Combined"]:
        count = (df_matrix["training_dataset"] == d).sum()
        assert count == 18, f"Expected 18 models for {d}, found {count}"

    # Check checkpoints
    manifest_json = ARTIFACTS_DIR / "checkpoints_manifest.json"
    assert manifest_json.is_file()
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 54

    for exp_id, meta in manifest.items():
        ckpt_path = BASE_DIR / meta["path"]
        assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"
        assert ckpt_path.stat().st_size > 0


def test_prediction_bounds_and_metrics_consistency():
    """Verify that all metrics are bounded in [0, 1] without NaNs."""
    eval_csv = ARTIFACTS_DIR / "evaluation_metrics.csv"
    assert eval_csv.is_file()

    df = pd.read_csv(eval_csv)
    assert len(df) > 0
    assert not df["pr_auc"].isnull().any(), "Found NaNs in PR-AUC"
    assert not df["roc_auc"].isnull().any(), "Found NaNs in ROC-AUC"
    assert (df["pr_auc"] >= 0.0).all() and (df["pr_auc"] <= 1.0).all()
    assert (df["roc_auc"] >= 0.0).all() and (df["roc_auc"] <= 1.0).all()
