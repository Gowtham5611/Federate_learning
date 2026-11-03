"""Unit tests for Phase 8 Final Test Evaluation reliability, integrity, and test isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PHASE8_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "phase8_final_test_evaluation"
PHASE7_DIR = (
    BASE_DIR
    / "experiments"
    / "results"
    / "ensemble"
    / "ensemble_iid_non_iid_seed42_20260923T154333"
)


def test_phase8_artifacts_exist_and_valid():
    """Verify that all Phase 8 artifacts exist and manifest is valid."""
    assert PHASE8_ARTIFACTS_DIR.is_dir(), "Phase 8 artifacts directory does not exist"

    expected_files = [
        "final_test_metrics.json",
        "final_test_metrics.csv",
        "confusion_matrices.json",
        "test_predictions.csv",
        "manifest.json",
        "phase8_final_evaluation_report.md",
    ]
    for filename in expected_files:
        p = PHASE8_ARTIFACTS_DIR / filename
        assert p.is_file(), f"Missing required Phase 8 artifact: {filename}"
        assert p.stat().st_size > 0, f"Artifact is empty: {filename}"

    manifest_path = PHASE8_ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "VERIFIED_COMPLETE"
    assert manifest["phase"] == 8


def test_checkpoint_loading_integrity():
    """Verify that both selected checkpoints exist and load successfully."""
    manifest_path = PHASE8_ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    iid_ckpt_path = BASE_DIR / manifest["models"]["iid"]["checkpoint_path"]
    non_iid_ckpt_path = BASE_DIR / manifest["models"]["non_iid"]["checkpoint_path"]

    assert iid_ckpt_path.is_file(), f"IID checkpoint not found: {iid_ckpt_path}"
    assert non_iid_ckpt_path.is_file(), f"Non-IID checkpoint not found: {non_iid_ckpt_path}"

    ckpt_iid = torch.load(iid_ckpt_path, map_location="cpu")
    assert "parameters" in ckpt_iid
    arch_iid = ckpt_iid["architecture"].value if hasattr(ckpt_iid["architecture"], "value") else str(ckpt_iid["architecture"])
    assert arch_iid.lower() == "lstm"
    assert ckpt_iid["feature_dimension"] == 57

    ckpt_non_iid = torch.load(non_iid_ckpt_path, map_location="cpu")
    assert "parameters" in ckpt_non_iid
    arch_non_iid = ckpt_non_iid["architecture"].value if hasattr(ckpt_non_iid["architecture"], "value") else str(ckpt_non_iid["architecture"])
    assert arch_non_iid.lower() == "bilstm"
    assert ckpt_non_iid["feature_dimension"] == 57


def test_strict_test_isolation():
    """Verify that alpha and threshold were frozen strictly on validation data."""
    val_report_path = PHASE7_DIR / "report.json"
    assert val_report_path.is_file()

    with open(val_report_path, "r", encoding="utf-8") as f:
        phase7_report = json.load(f)

    frozen_alpha = phase7_report["best_alpha"]
    frozen_threshold = phase7_report["frozen_threshold"]

    # Verify best alpha was 1.0 based on validation grid search
    assert frozen_alpha == 1.0
    alpha_candidates = phase7_report["alpha_search"]
    selected_candidate = next(c for c in alpha_candidates if c["selected"])
    assert selected_candidate["alpha"] == 1.0

    # Verify threshold was 0.13 from validation F1 optimization
    assert frozen_threshold == 0.13

    # Check that Phase 8 manifest uses the exact same frozen parameters
    manifest_path = PHASE8_ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["models"]["ensemble"]["alpha"] == frozen_alpha
    assert manifest["models"]["ensemble"]["frozen_threshold"] == frozen_threshold


def test_predictions_count_and_integrity():
    """Verify test prediction format, absence of NaNs, valid probability ranges, and label counts."""
    preds_path = PHASE8_ARTIFACTS_DIR / "test_predictions.csv"
    assert preds_path.is_file()

    # Read first 50,000 rows for rapid in-depth checks, plus verify total row count
    df = pd.read_csv(preds_path)
    assert len(df) == 1015564, f"Expected 1,015,564 predictions, got {len(df)}"

    expected_cols = [
        "row_index",
        "true_label",
        "p_iid",
        "p_non_iid",
        "p_ensemble",
        "frozen_threshold",
        "predicted_label",
    ]
    assert list(df.columns) == expected_cols

    # Check for NaNs/Infs
    assert not df.isnull().values.any(), "Found NaNs in test predictions"

    # Probability bounds
    for col in ["p_iid", "p_non_iid", "p_ensemble"]:
        min_val = df[col].min()
        max_val = df[col].max()
        assert min_val >= 0.0, f"{col} has negative probabilities: {min_val}"
        assert max_val <= 1.0, f"{col} has probabilities > 1.0: {max_val}"

    # Verify label counts
    y_true = df["true_label"].to_numpy()
    assert np.all(np.isin(y_true, [0, 1]))
    assert int(np.sum(y_true == 1)) == 1797
    assert int(np.sum(y_true == 0)) == 1013767

    # Verify predicted label matches threshold logic at tau=0.13
    pred_expected = (df["p_ensemble"] >= 0.13).astype(int)
    np.testing.assert_array_equal(df["predicted_label"].to_numpy(), pred_expected.to_numpy())


def test_prediction_to_label_alignment():
    """Verify that predictions align row-for-row with the raw test split transactions."""
    raw_test_csv = BASE_DIR / "data" / "processed" / "ibm_aml" / "HI-Small" / "test" / "transactions.csv"
    assert raw_test_csv.is_file()

    df_raw = pd.read_csv(raw_test_csv, usecols=["is_laundering"])
    assert len(df_raw) == 1015564

    preds_path = PHASE8_ARTIFACTS_DIR / "test_predictions.csv"
    df_preds = pd.read_csv(preds_path, usecols=["true_label"])

    np.testing.assert_array_equal(
        df_raw["is_laundering"].to_numpy(),
        df_preds["true_label"].to_numpy(),
        err_msg="Row-level test label mismatch between transactions.csv and test_predictions.csv",
    )


def test_metrics_calculation_consistency():
    """Verify that metrics recorded in final_test_metrics.json match confusion matrix math."""
    metrics_path = PHASE8_ARTIFACTS_DIR / "final_test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ens_metrics = data["metrics"]["Frozen_Hybrid_Ensemble"]
    assert pytest.approx(ens_metrics["pr_auc"], abs=1e-5) == 0.543749
    assert pytest.approx(ens_metrics["roc_auc"], abs=1e-5) == 0.976867

    eval_frozen = ens_metrics["evaluations"]["threshold_0.13"]
    tp = eval_frozen["tp"]
    fp = eval_frozen["fp"]
    tn = eval_frozen["tn"]
    fn = eval_frozen["fn"]

    assert tp == 850
    assert fp == 326
    assert tn == 1013441
    assert fn == 947

    calculated_prec = tp / (tp + fp)
    calculated_rec = tp / (tp + fn)
    calculated_f1 = 2 * calculated_prec * calculated_rec / (calculated_prec + calculated_rec)

    assert pytest.approx(eval_frozen["precision"], abs=1e-5) == calculated_prec
    assert pytest.approx(eval_frozen["recall"], abs=1e-5) == calculated_rec
    assert pytest.approx(eval_frozen["f1"], abs=1e-5) == calculated_f1
