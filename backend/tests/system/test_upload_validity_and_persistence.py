"""Unit tests for upload validity banners, download disk persistence, and tabular fallback opt-in."""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from evaluation.metrics import evaluate


def test_zero_alerts_metrics_handling():
    """Verify metrics calculation never crashes on 0 alerts (zero division safety)."""
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int32)
    # Probs all zero -> 0 alerts
    y_probs = np.zeros(len(y_true), dtype=np.float64)

    metrics = evaluate(y_true, y_probs, thr=0.5, n_boot=0)
    assert metrics["f1"] is None or metrics["f1"] == 0.0
    assert metrics["precision"] is None or metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["alerts_per_10k"] == 0.0


def test_validity_banner_clean_demo():
    """Verify that a compliant demo dataset yields VALID status."""
    demo_path = Path("data/demo/demo_ibm_aml_5000.csv")
    if not demo_path.exists():
        pytest.skip("data/demo/demo_ibm_aml_5000.csv does not exist.")

    service = CsvEvaluationService()
    csv_bytes = demo_path.read_bytes()
    result = service.evaluate_csv_bytes(csv_bytes, filename="demo_ibm_aml_5000.csv")

    assert "validity_banner" in result
    banner = result["validity_banner"]
    assert banner["status"] == "VALID"
    assert "LOW_HISTORY" not in banner["flags"]
    assert "PREVALENCE_MISMATCH" not in banner["flags"]


def test_validity_banner_flags_single_tx_history_poor():
    """Verify that a single-transaction per account file triggers LOW_HISTORY flag."""
    # Create 50 rows each with unique sender
    df = pd.DataFrame({
        "timestamp": [f"2022/09/01 {i:02d}:00" for i in range(50)],
        "from_account": [f"sender_{i:04d}" for i in range(50)],
        "to_account": [f"recv_{i:04d}" for i in range(50)],
        "amount": [100.0] * 50,
        "From Bank": [1] * 50,
        "To Bank": [2] * 50,
        "Amount Received": [100.0] * 50,
        "Receiving Currency": ["US Dollar"] * 50,
        "Amount Paid": [100.0] * 50,
        "Payment Currency": ["US Dollar"] * 50,
        "Payment Format": ["ACH"] * 50,
        "Is Laundering": [0] * 50,
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    service = CsvEvaluationService()
    result = service.evaluate_csv_bytes(csv_bytes, filename="single_tx.csv")

    banner = result["validity_banner"]
    assert "LOW_HISTORY" in banner["flags"]
    assert banner["status"] in ("INVALID", "WARNING")


def test_prediction_download_persistence_across_instances():
    """Verify predictions persist to disk and survive across fresh service instances."""
    demo_path = Path("data/demo/demo_ibm_aml_500.csv")
    if not demo_path.exists():
        pytest.skip("data/demo/demo_ibm_aml_500.csv does not exist.")

    service1 = CsvEvaluationService()
    result = service1.evaluate_csv_bytes(demo_path.read_bytes(), filename="demo_500.csv")
    pred_id = result["prediction_id"]

    # Instantiate brand new service instance
    service2 = CsvEvaluationService()
    # Ensure memory cache is empty
    service2._prediction_cache.clear()

    # Load from disk
    csv_content, filename = service2.get_prediction_download(pred_id)
    assert csv_content is not None
    assert filename.startswith("predictions_")
    assert "predicted_probability" in csv_content

    # Expire and verify status EXPIRED
    expired = service2.expire_prediction_download(pred_id)
    assert expired is True

    status, _ = service2.get_prediction_download(pred_id)
    assert status == "EXPIRED"


def test_tabular_fallback_explicit_opt_in():
    """Verify that passing fallback_model='tabular' routes to the LightGBM fallback."""
    demo_path = Path("data/demo/demo_ibm_aml_500.csv")
    if not demo_path.exists():
        pytest.skip("data/demo/demo_ibm_aml_500.csv does not exist.")

    service = CsvEvaluationService()
    result = service.evaluate_csv_bytes(
        demo_path.read_bytes(),
        filename="demo_500.csv",
        fallback_model="tabular",
    )

    assert result["model_id"] == "lightgbm_tabular_fallback"
    assert result["fallback_model_used"] is True
    assert result["is_sequence_model"] is False
    assert len(result["predictions"]) > 0
