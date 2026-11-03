"""Integration and regression tests for CSV evaluation operating points and score-only mode."""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService


@pytest.fixture(scope="module")
def eval_service():
    return CsvEvaluationService()


def test_zero_alerts_numeric_metrics(eval_service):
    """Verify that a labeled file generating 0 alerts produces numeric F1=0, Recall=0, Specificity=1.0, BalancedAcc=0.5, Precision=None."""
    csv_path = PROJECT_ROOT / "data/demo/demo_ibm_aml_500.csv"
    with open(csv_path, "rb") as f:
        content = f.read()

    res = eval_service.evaluate_csv_bytes(content, filename="demo_500.csv", dataset="HI-Small", row_limit=5000)

    assert res["status"] == "success"
    assert res["has_labels"] is True
    em = res["evaluation_metrics"]
    assert em is not None

    # Zero alerts triggered
    assert em["alert_count"] == 0
    assert em["tp"] == 0
    assert em["fp"] == 0
    assert em["precision"] is None
    # Must be numeric 0.0, NOT None
    assert em["f1_score"] == 0.0
    assert em["f2_score"] == 0.0
    assert em["recall"] == 0.0
    assert em["specificity"] == 1.0
    assert em["balanced_accuracy"] == 0.5
    assert em["accuracy"] > 0.95
    assert em["mcc"] == 0.0
    assert em["cohen_kappa"] == 0.0

    # Operating points must be present
    op = res.get("operating_points")
    assert op is not None
    assert "frozen" in op
    assert "alert_budgets" in op
    assert "10" in op["alert_budgets"]
    assert "50" in op["alert_budgets"]
    assert "100" in op["alert_budgets"]
    assert "best_f1_on_file" in op

    # Small sample notice
    banner = res.get("validity_banner")
    assert banner is not None
    assert "SMALL_SAMPLE" in banner["flags"]
    assert banner["diagnostics"]["expected_positives"] is not None


def test_score_only_mode_for_unlabeled_csv(eval_service):
    """Verify that an unlabeled CSV enters Score-Only Mode without crashing or empty cards."""
    csv_path = PROJECT_ROOT / "data/demo/demo_no_labels_500.csv"
    with open(csv_path, "rb") as f:
        content = f.read()

    res = eval_service.evaluate_csv_bytes(content, filename="demo_no_labels.csv", dataset="HI-Small", row_limit=5000)

    assert res["status"] == "success"
    assert res["has_labels"] is False
    assert res["has_ground_truth"] is False
    assert res["mode"] == "SCORE_ONLY"
    assert res["score_only_message"] is not None
    assert res["evaluation_metrics"] is None
    assert len(res["predictions"]) == 500
    assert len(res["top_risky_transactions"]) > 0
    assert res["risk_distribution"] is not None


def test_alert_budgets_consistency(eval_service):
    """Verify that alert budgets (10, 50, 100 per 10k) produce monotonically non-decreasing alerts and valid metrics."""
    csv_path = PROJECT_ROOT / "data/demo_slices/test_slice_5000.csv"
    with open(csv_path, "rb") as f:
        content = f.read()

    res = eval_service.evaluate_csv_bytes(content, filename="slice_5000.csv", dataset="HI-Small", row_limit=5000)

    em = res["evaluation_metrics"]
    assert em is not None
    ops = res["operating_points"]
    budgets = ops["alert_budgets"]

    alerts_10 = budgets["10"]["alerts"]
    alerts_50 = budgets["50"]["alerts"]
    alerts_100 = budgets["100"]["alerts"]

    assert alerts_10 <= alerts_50 <= alerts_100
    assert budgets["10"]["precision"] is not None
    assert budgets["50"]["precision"] is not None
    assert budgets["100"]["precision"] is not None

    # Best F1 on file
    best_f1_op = ops["best_f1_on_file"]
    assert best_f1_op["f1_score"] >= em["f1_score"]
    assert "what-if" in best_f1_op["label"]
