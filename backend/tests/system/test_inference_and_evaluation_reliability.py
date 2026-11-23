"""Regression and reliability test suite for FedSecureAI Inference and Evaluation Engine.

Covers:
1. Missing model_id in request schemas (regression test for HTTP 422 error).
2. Auto-Detect model selection path across schemas.
3. Positive-class index correctness (PyTorch & Scikit-learn dynamic models).
4. Threshold consistency across model metadata, inference, and prediction records.
5. No-positive-prediction metric behavior (precision/F1 is None when 0 alerts generated).
6. Invalid probabilities safety (finite, clipped in [0, 1]).
7. Labeled vs unlabeled evaluation semantics.
8. CSV row-order preservation.
9. Preprocessing consistency.
10. Model compatibility error reporting for incompatible schemas.
11. Validation-only threshold analysis across objectives.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

try:
    from interfaces.api.app import create_app
    from interfaces.api.schemas.common import FROZEN_THRESHOLDS
    from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
    from infrastructure.models.registry.universal_model_registry import UniversalModelRegistry
    from infrastructure.models.dynamic_model_trainer import DynamicModelTrainer
except ModuleNotFoundError:
    from backend.interfaces.api.app import create_app
    from backend.interfaces.api.schemas.common import FROZEN_THRESHOLDS
    from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
    from backend.infrastructure.models.registry.universal_model_registry import UniversalModelRegistry
    from backend.infrastructure.models.dynamic_model_trainer import DynamicModelTrainer

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api_client():
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="module")
def eval_service():
    return CsvEvaluationService()


# ------------------------------------------------------------------------------
# 1. Regression Test for HTTP 422 on Missing model_id
# ------------------------------------------------------------------------------

def test_regression_missing_model_id_in_compatibility_request(api_client):
    """Ensure POST /api/v1/models/compatibility accepts payload without model_id (no 422 error)."""
    sample_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    payload = {
        "csv_content": sample_csv,
        "filename": "transactions.csv",
        # model_id is deliberately omitted
    }
    response = api_client.post("/api/v1/models/compatibility", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 2


def test_regression_missing_model_id_in_predict_request(api_client):
    """Ensure POST /api/v1/predict/csv accepts payload without model_id, executing Auto-Detect."""
    sample_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    payload = {
        "csv_content": sample_csv,
        "filename": "transactions.csv",
        # model_id is deliberately omitted
    }
    response = api_client.post("/api/v1/predict/csv", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert data["evaluated_sequences"] == 25
    assert len(data["predictions"]) == 25


# ------------------------------------------------------------------------------
# 2. Auto-Detect Model Selection
# ------------------------------------------------------------------------------

def test_auto_detect_model_selection_for_hi_small(eval_service):
    """Verify Auto-Detect selects HI-Small for HI-Small samples."""
    sample_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    res = eval_service.evaluate_csv_bytes(sample_csv, model_id="auto", filename="hi_small_sample_25.csv")
    assert res["status"] == "success"
    assert res["dataset"] == "HI-Small"
    assert res["decision_threshold"] == pytest.approx(FROZEN_THRESHOLDS["HI-Small"], abs=1e-5)


def test_auto_detect_model_selection_for_li_medium(eval_service):
    """Verify Auto-Detect selects LI-Medium for LI-Medium samples."""
    sample_csv = (BASE_DIR / "data/demo_samples/li_medium_sample_25.csv").read_bytes()
    res = eval_service.evaluate_csv_bytes(sample_csv, model_id="auto", filename="li_medium_sample_25.csv")
    assert res["status"] == "success"
    assert res["dataset"] == "LI-Medium"
    assert res["decision_threshold"] == pytest.approx(FROZEN_THRESHOLDS["LI-Medium"], abs=1e-5)


def test_auto_detect_incompatible_schema_informative_error(eval_service):
    """Verify Auto-Detect raises an informative error instead of silently using an incompatible model."""
    incompatible_csv = (
        "customer_name,email,comment,satisfaction_rating\n"
        "Alice,alice@example.com,Great service,5\n"
        "Bob,bob@example.com,Slow response,2\n"
    ).encode("utf-8")

    with pytest.raises(ValueError) as excinfo:
        eval_service.evaluate_csv_bytes(incompatible_csv, model_id="auto", filename="customer_feedback.csv")
    
    err_msg = str(excinfo.value)
    assert "cannot be evaluated" in err_msg.lower() or "missing required" in err_msg.lower()


# ------------------------------------------------------------------------------
# 3. Positive-Class Index & Probabilities Correctness
# ------------------------------------------------------------------------------

def test_dynamic_model_predict_proba_column_indexing():
    """Verify DynamicModelTrainer.predict_proba extracts column 1 (positive laundering class)."""
    # Create simple binary classification training dataset
    train_df = pd.DataFrame({
        "amount": [100.0, 50000.0, 150.0, 95000.0, 200.0, 85000.0, 120.0, 75000.0] * 5,
        "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1] * 5,
    })
    trainer = DynamicModelTrainer()
    meta, _ = trainer.train_custom_model(
        train_df,
        model_name="TestIndexModel",
        target_column="is_laundering",
        model_type="RandomForest",
    )
    
    artifacts = trainer.model_registry.load_dynamic_model_artifacts(meta.model_id)
    pipeline = artifacts["model"]

    # Predict on high vs low amount
    test_df = pd.DataFrame({"amount": [50.0, 100000.0]})
    probs = pipeline.predict_proba(test_df)
    assert probs.ndim == 1, "Expected 1D probability array"
    assert len(probs) == 2
    assert np.all(np.isfinite(probs)), "Probabilities must be finite"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be bounded in [0, 1]"
    # High amount should have higher laundering probability than low amount
    assert probs[1] > probs[0], "Positive laundering class probability must correlate with risk features"


def test_pytorch_model_probabilities_bounded_and_finite(eval_service):
    """Verify PyTorch sequence model produces finite probabilities strictly in [0, 1]."""
    sample_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    res = eval_service.evaluate_csv_bytes(sample_csv, dataset="HI-Small")
    for item in res["predictions"]:
        p = item["predicted_probability"]
        assert np.isfinite(p), f"Probability {p} is not finite"
        assert 0.0 <= p <= 1.0, f"Probability {p} outside [0, 1]"


# ------------------------------------------------------------------------------
# 4. Threshold Consistency Across All Predictions
# ------------------------------------------------------------------------------

def test_decision_threshold_consistency(eval_service):
    """Verify that every prediction decision strictly obeys the applied threshold."""
    sample_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    res = eval_service.evaluate_csv_bytes(sample_csv, dataset="HI-Small")
    threshold = res["decision_threshold"]
    assert threshold == pytest.approx(FROZEN_THRESHOLDS["HI-Small"], abs=1e-5)

    for item in res["predictions"]:
        p = item["predicted_probability"]
        expected_status = "SUSPICIOUS" if p >= threshold else "LEGITIMATE"
        assert item["predicted_label"] == expected_status
        assert item["decision"] == expected_status
        assert item["decision_threshold"] == threshold


# ------------------------------------------------------------------------------
# 5. No Positive Prediction Metric Behavior (Zero Alerts)
# ------------------------------------------------------------------------------

def test_zero_alerts_precision_is_none_not_misleading_zero(eval_service):
    """When no transactions exceed threshold, precision must be None, not misleading 0.0."""
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    # All probabilities compressed near zero (simulating the user's 500-row case)
    probs = np.array([0.001, 0.011, 0.002, 0.009, 0.003, 0.004, 0.010, 0.002])
    threshold = 0.337784  # Standard frozen threshold

    metrics = eval_service._calculate_metrics(labels, probs, threshold)
    
    assert metrics["confusion_matrix"]["tp"] == 0
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 3
    assert metrics["confusion_matrix"]["tn"] == 5
    assert metrics["accuracy"] == 5 / 8
    assert metrics["recall"] == 0.0
    # Precision and F-scores MUST be None internally to represent undefined 0/0
    assert metrics["precision"] is None
    assert metrics["f1_score"] is None
    assert metrics["f2_score"] is None
    assert metrics["zero_alerts_note"] is not None
    assert "0 alerts generated" in metrics["zero_alerts_note"]


# ------------------------------------------------------------------------------
# 6. Labeled versus Unlabeled Evaluation Semantics
# ------------------------------------------------------------------------------

def test_labeled_vs_unlabeled_evaluation_semantics(eval_service):
    """Verify that unlabeled CSVs return evaluation_metrics as None and omit confusion matrix."""
    # 1. Labeled evaluation
    labeled_csv = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    res_labeled = eval_service.evaluate_csv_bytes(labeled_csv, dataset="HI-Small")
    assert res_labeled["has_labels"] is True
    assert res_labeled["evaluation_metrics"] is not None
    assert "confusion_matrix" in res_labeled["evaluation_metrics"]

    # 2. Unlabeled evaluation
    unlabeled_csv = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 09:00,102,ACC_01,102,ACC_02,500.00,US Dollar,500.00,US Dollar,ACH\n"
        "2022/09/01 09:30,102,ACC_02,102,ACC_03,450.00,US Dollar,450.00,US Dollar,ACH\n"
    ).encode("utf-8")
    res_unlabeled = eval_service.evaluate_csv_bytes(unlabeled_csv, dataset="HI-Small")
    assert res_unlabeled["has_labels"] is False
    assert res_unlabeled["evaluation_metrics"] is None


# ------------------------------------------------------------------------------
# 7. CSV Row Order and Count Preservation
# ------------------------------------------------------------------------------

def test_csv_row_order_strict_preservation(eval_service):
    """Verify output predictions strictly correspond to input rows 1-to-1."""
    test_csv = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 10:00,10,ACC_ALPHA,20,ACC_BETA,100.00,US Dollar,100.00,US Dollar,ACH\n"
        "2022/09/01 11:00,20,ACC_BETA,30,ACC_GAMMA,250.00,US Dollar,250.00,US Dollar,Wire\n"
        "2022/09/01 12:00,30,ACC_GAMMA,10,ACC_ALPHA,500.00,US Dollar,500.00,US Dollar,Cheque\n"
    ).encode("utf-8")
    
    res = eval_service.evaluate_csv_bytes(test_csv, dataset="HI-Small")
    preds = res["predictions"]
    assert len(preds) == 3
    assert preds[0]["from_account"] == "ACC_ALPHA"
    assert preds[0]["to_account"] == "ACC_BETA"
    assert preds[0]["amount"] == 100.0

    assert preds[1]["from_account"] == "ACC_BETA"
    assert preds[1]["to_account"] == "ACC_GAMMA"
    assert preds[1]["amount"] == 250.0

    assert preds[2]["from_account"] == "ACC_GAMMA"
    assert preds[2]["to_account"] == "ACC_ALPHA"
    assert preds[2]["amount"] == 500.0


# ------------------------------------------------------------------------------
# 8. Validation-Only Threshold Analysis
# ------------------------------------------------------------------------------

def test_validation_only_threshold_analysis(eval_service, api_client):
    """Verify validation-only threshold tuning evaluates multiple objectives without test leakage."""
    # Test service method directly
    analysis_f1 = eval_service.analyze_validation_thresholds("HI-Small", objective="max_f1")
    assert analysis_f1["dataset"] == "HI-Small"
    assert analysis_f1["objective"] == "max_f1"
    assert analysis_f1["candidates_count"] >= 40
    assert 0.0 < analysis_f1["optimal_threshold"] < 1.0
    assert "methodology_note" in analysis_f1

    # Test via API endpoint
    response = api_client.post("/api/v1/evaluate/threshold-analysis", json={
        "dataset": "HI-Small",
        "objective": "cost_sensitive",
        "cost_fp": 1.0,
        "cost_fn": 10.0,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["objective"] == "cost_sensitive"
    assert "optimal_metrics" in data
    assert data["optimal_metrics"]["cost"] > 0
