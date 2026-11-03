"""Parity tests verifying that CSV serving sequence construction reproduces exact test split predictions."""

import sys
from pathlib import Path

# Add project root and backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
import pytest
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService


def test_test_slice_prediction_parity():
    """Verify that scoring a contiguous slice of test data reproduces saved test predictions to < 1e-5."""
    demo_slice_path = Path("data/demo_slices/test_slice_500.csv")
    if not demo_slice_path.exists():
        pytest.skip("Demo slice not found; run scripts/export_test_slices.py first.")

    saved_preds_path = Path("results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_test_preds.npy")
    if not saved_preds_path.exists():
        pytest.skip("Saved test predictions not found.")

    with open(demo_slice_path, "rb") as f:
        csv_bytes = f.read()

    service = CsvEvaluationService()
    result = service.evaluate_csv_bytes(csv_bytes, dataset="HI-Small", row_limit=500)
    
    saved_preds = np.load(saved_preds_path)[:500]
    service_preds = np.array([p["predicted_probability"] for p in result["predictions"]])
    
    abs_diff = np.abs(service_preds - saved_preds)
    max_diff = float(np.max(abs_diff))
    
    assert max_diff < 1e-5, f"Max difference {max_diff:.8e} exceeds tolerance of 1e-5."
    assert len(service_preds) == 500

