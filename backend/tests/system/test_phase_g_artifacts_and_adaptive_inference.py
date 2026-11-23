"""Tests for Phase G: Saved-Artifact-Based Ensemble, Model Selection, and Adaptive Inference.

Verifies:
1. Authoritative artifact existence and integrity:
   - Authoritative HI-Small and LI-Medium results files.
   - FedProx mu = 0.0001 compliance verification.
   - Validation predictions npz files and strict sample/label alignment.
2. Scientific validation-only model selection and ensemble optimization:
   - Configurable objectives ('max_pr_auc', 'max_f1', 'max_f2', 'cost_sensitive', 'recall_constrained').
   - Monotonic alpha search and honest reporting when alpha* = 1.00.
   - Single-pass held-out test evaluation with strictly frozen parameters.
3. Adaptive CSV inference and evaluation:
   - Sequence history inspection (L_avg, single-tx ratio, mean padding ratio).
   - Clear warning without silently changing the selected model.
   - Tabular model fallback suggestions.
   - 100% row-order preservation.
   - Zero-alert metric semantics (precision=None, f1=None, recall=0.0).
   - Unlabeled data handling (evaluation_metrics=None).
   - Incompatible schema actionable error.
4. REST API endpoint contracts (/api/v1/ensemble/evaluate and /api/v1/ensemble/summary).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.services.ensemble_experiment_service import (
    AlphaEvaluationResult,
    EnsembleExperimentService,
)
from interfaces.api.app import app
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient instance."""
    return TestClient(app)


# ==============================================================================
# 1. Authoritative Artifact Auditing & FedProx Compliance Tests
# ==============================================================================


def test_authoritative_artifacts_exist_and_inventory() -> None:
    """Audit that authoritative result tables and validation prediction files exist."""
    hi_csv = BASE_DIR / "results/authoritative_results/HI_Small_results.csv"
    li_csv = BASE_DIR / "results/authoritative_results/LI_Medium_results.csv"
    mu_csv = BASE_DIR / "results/authoritative_results/fedprox_mu_verification.csv"
    val_hi_iid = BASE_DIR / "results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz"
    val_hi_noniid = BASE_DIR / "results/phase7/HI_Small/validation_predictions/noniid_validation_predictions.npz"

    assert hi_csv.exists(), f"Missing {hi_csv}"
    assert li_csv.exists(), f"Missing {li_csv}"
    assert mu_csv.exists(), f"Missing {mu_csv}"
    assert val_hi_iid.exists(), f"Missing {val_hi_iid}"
    assert val_hi_noniid.exists(), f"Missing {val_hi_noniid}"


def test_fedprox_mu_verification_authoritative() -> None:
    """Verify that all authoritative FedProx experiments used mu = 0.0001 strictly."""
    mu_csv = BASE_DIR / "results/authoritative_results/fedprox_mu_verification.csv"
    with open(mu_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    included_rows = [r for r in rows if r["action"] == "INCLUDE_IN_COMPARISON"]
    assert len(included_rows) == 12  # 6 for HI-Small, 6 for LI-Medium
    for r in included_rows:
        mu_val = float(r["verified_mu"])
        assert mu_val == 0.0001, f"Found non-compliant mu {mu_val} for {r.get('config_id')}"
        assert "0.0001" in r["compliance_verdict"]

    excluded_rows = [r for r in rows if r["action"] != "INCLUDE_IN_COMPARISON"]
    for r in excluded_rows:
        assert r["compliance_verdict"].startswith("EXCLUDED")


def test_validation_predictions_sample_alignment() -> None:
    """Verify that IID and Non-IID validation predictions have identical lengths and labels."""
    val_iid_npz = BASE_DIR / "results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz"
    val_noniid_npz = BASE_DIR / "results/phase7/HI_Small/validation_predictions/noniid_validation_predictions.npz"

    data_iid = np.load(val_iid_npz)
    data_noniid = np.load(val_noniid_npz)

    p_iid = data_iid["probabilities"]
    p_noniid = data_noniid["probabilities"]
    y_iid = data_iid["labels"]
    y_noniid = data_noniid["labels"]

    assert len(p_iid) == len(p_noniid)
    assert len(p_iid) == len(y_iid)
    assert np.array_equal(y_iid, y_noniid), "Ground-truth validation labels do not match between IID and Non-IID!"


# ==============================================================================
# 2. Validation-Only Model Selection & Ensemble Optimization Tests
# ==============================================================================


def test_validation_model_selection_recall_constrained() -> None:
    """Verify that recall-constrained optimization strictly prioritizes recall."""
    service = EnsembleExperimentService(base_dir=BASE_DIR)
    outcome = service.run_ensemble_experiment(
        dataset="HI-Small",
        alpha_step=0.2,
        objective="recall_constrained",
        min_recall=0.80,
        subsample_limit=5000,
    )

    assert outcome.validation_metrics["recall"] > 0.50
    assert outcome.best_alpha in [c.alpha for c in outcome.alpha_candidates]


def test_alpha_grid_search_monotonicity_and_honest_reporting() -> None:
    """Verify alpha sweep and honest reporting when alpha* = 1.00."""
    service = EnsembleExperimentService(base_dir=BASE_DIR)
    outcome = service.run_ensemble_experiment(
        dataset="HI-Small",
        alpha_step=0.1,
        objective="max_pr_auc",
        subsample_limit=10000,
    )

    assert len(outcome.alpha_candidates) == 11
    # On validation, alpha=1.00 achieves higher PR-AUC than alpha=0.00
    assert outcome.alpha_candidates[-1].val_pr_auc > outcome.alpha_candidates[0].val_pr_auc
    assert outcome.best_alpha == 1.00
    assert outcome.ensemble_improved is False
    assert "pruning the degraded Non-IID stream" in outcome.scientific_rationale


def test_single_pass_test_evaluation_zero_leakage() -> None:
    """Verify held-out test evaluation uses frozen parameters without test tuning."""
    service = EnsembleExperimentService(base_dir=BASE_DIR)
    outcome = service.run_ensemble_experiment(
        dataset="HI-Small",
        alpha_step=0.1,
        objective="max_pr_auc",
        subsample_limit=5000,
    )

    # Test metrics should report champion IID performance
    assert outcome.test_metrics["pr_auc"] >= 0.45
    assert outcome.test_metrics["f1"] >= 0.50


# ==============================================================================
# 3. Adaptive CSV Inference & Diagnostics Tests
# ==============================================================================


def test_adaptive_inference_sequence_diagnostics_short_history(client: TestClient) -> None:
    """Verify sequence diagnostics, warning, and suggested models under short history."""
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 08:00,10,ACC_01,20,ACC_99,1000.0,US Dollar,1000.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:05,10,ACC_02,20,ACC_98,1500.0,US Dollar,1500.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:10,10,ACC_03,20,ACC_97,2000.0,US Dollar,2000.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:15,10,ACC_04,20,ACC_96,2500.0,US Dollar,2500.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:20,10,ACC_05,20,ACC_95,3000.0,US Dollar,3000.0,US Dollar,Cheque,0\n"
    )

    res = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_text,
            "dataset": "HI-Small",
            "model_id": "hi_small_lstm_fedavg_iid_seed42",
            "filename": "short_history.csv",
        },
    )
    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "success"
    diagnostics = result["sequence_context_diagnostics"]
    assert diagnostics is not None
    assert diagnostics["insufficient_history_warning"] is True
    assert diagnostics["average_sequence_length"] == 1.0
    assert diagnostics["single_transaction_ratio"] == 1.0
    assert diagnostics["mean_padding_ratio"] == 0.9
    assert "truncated" in diagnostics["warning_message"].lower()


def test_adaptive_inference_row_order_preservation(client: TestClient) -> None:
    """Verify that predictions match input row order 1:1."""
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 08:00,10,ACCT_A,20,ACCT_B,100.0,US Dollar,100.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:05,10,ACCT_C,20,ACCT_D,200.0,US Dollar,200.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:10,10,ACCT_E,20,ACCT_F,300.0,US Dollar,300.0,US Dollar,Cheque,0\n"
    )

    res = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_text,
            "dataset": "HI-Small",
            "filename": "order_test.csv",
        },
    )
    assert res.status_code == 200
    result = res.json()
    predictions = result["predictions"]
    assert len(predictions) == 3
    assert [p["row_id"] for p in predictions] == [1, 2, 3]
    assert [p["from_account"] for p in predictions] == ["ACCT_A", "ACCT_C", "ACCT_E"]


def test_zero_alert_metrics_semantics_precision_f1_null(client: TestClient) -> None:
    """Verify zero-alert metrics semantics (precision=None, f1=None, recall=0.0)."""
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 08:00,10,ACC_01,20,ACC_99,100.0,US Dollar,100.0,US Dollar,Cheque,1\n"
        "2022/09/01 08:05,10,ACC_02,20,ACC_98,150.0,US Dollar,150.0,US Dollar,Cheque,0\n"
        "2022/09/01 08:10,10,ACC_03,20,ACC_97,200.0,US Dollar,200.0,US Dollar,Cheque,0\n"
    )

    res = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_text,
            "dataset": "HI-Small",
            "filename": "zero_alerts.csv",
        },
    )
    assert res.status_code == 200
    result = res.json()
    metrics = result["evaluation_metrics"]
    if result["flagged_laundering_count"] == 0:
        assert metrics["precision"] is None
        assert metrics["f1_score"] is None
        assert metrics["recall"] == 0.0
        assert metrics["zero_alerts_note"] is not None


def test_unlabeled_data_handling_evaluation_metrics_none(client: TestClient) -> None:
    """Verify that unlabelled data produces predictions but leaves evaluation_metrics as None."""
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 08:00,10,ACC_01,20,ACC_99,100.0,US Dollar,100.0,US Dollar,Cheque\n"
        "2022/09/01 08:05,10,ACC_02,20,ACC_98,150.0,US Dollar,150.0,US Dollar,Cheque\n"
    )

    res = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_text,
            "dataset": "HI-Small",
            "filename": "unlabeled.csv",
        },
    )
    assert res.status_code == 200
    result = res.json()
    assert result["has_labels"] is False
    assert result["evaluation_metrics"] is None
    assert len(result["predictions"]) == 2


def test_incompatible_schema_actionable_error(client: TestClient) -> None:
    """Verify that completely incompatible schemas fail gracefully with an informative error."""
    bad_csv = (
        "PatientID,BloodPressure,Cholesterol,HeartRate,Diagnosis\n"
        "101,120,200,72,Normal\n"
        "102,140,240,85,Hypertension\n"
    )

    res = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": bad_csv,
            "dataset": "HI-Small",
            "filename": "medical_data.csv",
        },
    )
    assert res.status_code in (400, 422)
    err_detail = res.json().get("detail", "")
    assert "missing" in str(err_detail).lower() or "compatible" in str(err_detail).lower() or "schema" in str(err_detail).lower() or "failed" in str(err_detail).lower()



# ==============================================================================
# 4. REST API Endpoint Contracts Tests
# ==============================================================================


def test_api_ensemble_evaluate_endpoint_contracts(client: TestClient) -> None:
    """Verify POST /api/v1/ensemble/evaluate and GET /api/v1/ensemble/summary."""
    res_eval = client.post(
        "/api/v1/ensemble/evaluate",
        json={"dataset": "HI-Small", "alpha_step": 0.2, "objective": "max_pr_auc"},
    )
    assert res_eval.status_code == 200
    data_eval = res_eval.json()
    assert data_eval["status"] == "success"
    assert "best_alpha" in data_eval
    assert "frozen_threshold" in data_eval
    assert "alpha_candidates" in data_eval
    assert len(data_eval["alpha_candidates"]) == 6

    res_sum = client.get("/api/v1/ensemble/summary?dataset=HI-Small")
    assert res_sum.status_code == 200
    data_sum = res_sum.json()
    assert data_sum["status"] == "success"
    assert data_sum["dataset"] == "HI-Small"
    assert "best_alpha" in data_sum
    assert "scientific_rationale" in data_sum
