"""Unit tests for evaluation/metrics module."""

import sys
from pathlib import Path

# Add project root to path so evaluation package is resolved
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from evaluation.metrics import (
    evaluate,
    fast_pr_auc,
    compute_ece,
    compute_recall_at_alerts,
    compute_operating_points,
    compute_metrics_at_threshold,
)


def test_zero_alerts():
    """Verify safe division: F1, F2, Recall, MCC, Kappa are numeric (0.0), Specificity is 1.0, and Precision is None when 0 alerts."""
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.01, 0.02, 0.05, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01])
    
    # Threshold higher than max probability (0.05)
    res = evaluate(y, p, thr=0.50, n_boot=50)
    
    assert res["alerts"] == 0
    assert res["alerts_per_10k"] == 0.0
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 2
    assert res["tn"] == 8
    # Only precision (TP / (TP + FP)) has 0/0 division and is None
    assert res["precision"] is None
    # By formula: 2TP / (2TP + FP + FN) = 0 / 2 = 0.0
    assert res["f1"] == 0.0
    assert res["f1_score"] == 0.0
    assert res["f2"] == 0.0
    assert res["f2_score"] == 0.0
    # Recall = TP / (TP + FN) = 0 / 2 = 0.0
    assert res["recall"] == 0.0
    # Specificity = TN / (TN + FP) = 8 / 8 = 1.0
    assert res["specificity"] == 1.0
    # Balanced Accuracy = 0.5 * (Recall + Specificity) = 0.5 * (0 + 1) = 0.5
    assert res["balanced_accuracy"] == 0.5
    # MCC and Kappa are 0.0 (no correlation, chance agreement)
    assert res["mcc"] == 0.0
    assert res["cohen_kappa"] == 0.0
    assert "zero alerts generated at specified threshold" in res["warnings"]


def test_operating_points_generation():
    """Verify three operating points (frozen, alert budget, best-f1 what-if) are generated."""
    rng = np.random.default_rng(42)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:20] = 1  # 20 positives
    rng.shuffle(y)
    p = np.where(y == 1, rng.beta(2, 5, n), rng.beta(1, 20, n))

    res = evaluate(y, p, thr=0.337784, n_boot=50)
    op = res["operating_points"]

    assert "frozen" in op
    assert "alert_budgets" in op
    assert "best_f1_on_file" in op

    # Frozen
    assert op["frozen"]["threshold"] == pytest.approx(0.337784, rel=1e-3)
    assert "f1_score" in op["frozen"]
    assert "recall" in op["frozen"]

    # Alert Budgets
    assert "10" in op["alert_budgets"]
    assert "50" in op["alert_budgets"]
    assert "100" in op["alert_budgets"]
    # 10 / 10k of 1000 = 1 alert
    assert op["alert_budgets"]["10"]["alerts"] >= 1
    assert op["alert_budgets"]["50"]["alerts"] >= 1
    assert op["alert_budgets"]["100"]["alerts"] >= 1

    # Best F1
    best_f1 = op["best_f1_on_file"]
    assert "what-if" in best_f1["label"]
    assert best_f1["f1_score"] >= 0.0


def test_bootstrap_ci_too_few_positives():
    """Verify bootstrap CI gracefully handles < 2 positives with explicit reason."""
    y = np.zeros(500, dtype=int)
    y[0] = 1  # Only 1 positive
    p = np.linspace(0.001, 0.20, 500)

    res = evaluate(y, p, thr=0.337784, n_boot=50)

    assert res["positives"] == 1
    assert res["pr_auc_ci_lower"] is None
    assert res["pr_auc_ci_upper"] is None
    assert res["pr_auc_ci_reason"] == "too few positives (< 2)"
    assert any("too few positives" in w for w in res["warnings"])


def test_single_class_all_zeros():
    """Verify single-class dataset (all 0s) returns single_class status and defined specificity."""
    y = np.zeros(100, dtype=int)
    p = np.random.uniform(0, 1, 100)
    
    res = evaluate(y, p, thr=0.5, n_boot=20)
    
    assert res["positives"] == 0
    assert res["prevalence"] == 0.0
    assert res["specificity"] is not None
    assert res["accuracy"] is not None
    assert res["recall"] is None
    assert res["f1"] is None
    assert res["pr_auc"] is None
    assert res["roc_auc"] is None
    assert res["pr_auc_ci_lower"] is None
    assert res["status"] == "single_class"


def test_single_class_all_ones():
    """Verify single-class dataset (all 1s) returns single_class status and defined recall."""
    y = np.ones(100, dtype=int)
    p = np.random.uniform(0, 1, 100)
    
    res = evaluate(y, p, thr=0.5, n_boot=20)
    
    assert res["positives"] == 100
    assert res["prevalence"] == 1.0
    assert res["recall"] is not None
    assert res["specificity"] is None
    assert res["pr_auc"] is None
    assert res["roc_auc"] is None
    assert res["status"] == "single_class"


def test_perfect_model():
    """Verify perfect predictions yield perfect metric values."""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    res = evaluate(y, p, thr=0.5, n_boot=50)
    
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["specificity"] == 1.0
    assert res["f1"] == 1.0
    assert res["mcc"] == 1.0
    assert res["pr_auc"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["brier_score"] == 0.0
    assert res["ece"] == 0.0
    assert res["pr_auc_lift"] == pytest.approx(2.0)  # 1.0 / 0.5


def test_recall_at_alerts_calculation():
    """Verify compute_recall_at_alerts logic."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    recalls = compute_recall_at_alerts(y, p, budgets=[1000, 2000, 4000])
    # 10 samples: budget 1000/10000 = 10% = 1 sample (top 1 has 1 pos -> recall = 1/4 = 0.25)
    assert recalls[1000] == 0.25
    # budget 2000/10000 = 20% = 2 samples (top 2 has 2 pos -> recall = 2/4 = 0.50)
    assert recalls[2000] == 0.50
    # budget 4000/10000 = 40% = 4 samples (top 4 has 4 pos -> recall = 4/4 = 1.0)
    assert recalls[4000] == 1.0


def test_ece_perfect_and_imperfect():
    """Verify ECE gives 0 for perfectly calibrated binary output."""
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    ece = compute_ece(y, p, n_bins=10)
    assert ece == pytest.approx(0.0)
    
    # Overconfident wrong predictions
    y_bad = np.array([0, 0])
    p_bad = np.array([0.95, 0.95])
    ece_bad = compute_ece(y_bad, p_bad, n_bins=10)
    assert ece_bad == pytest.approx(0.95)
