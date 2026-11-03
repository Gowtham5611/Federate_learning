"""Unit tests for evaluation/metrics module."""

import sys
from pathlib import Path

# Add project root to path so evaluation package is resolved
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from evaluation.metrics import evaluate, fast_pr_auc, compute_ece, compute_recall_at_alerts


def test_zero_alerts():
    """Verify safe division and None returns when no alerts are triggered."""
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.01, 0.02, 0.05, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01])
    
    # Threshold higher than max probability (0.05)
    res = evaluate(y, p, thr=0.50, n_boot=50)
    
    assert res["alerts"] == 0
    assert res["alerts_per_10k"] == 0.0
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["precision"] is None
    assert res["f1"] is None
    assert res["f2"] is None
    assert res["recall"] == 0.0
    assert res["specificity"] == 1.0
    assert "zero alerts generated at specified threshold" in res["warnings"]
    assert res["status"] == "warning"


def test_single_class_all_zeros():
    """Verify single-class dataset (all 0s) returns None for ranking AUCs and precision/recall."""
    y = np.zeros(100, dtype=int)
    p = np.random.uniform(0, 1, 100)
    
    res = evaluate(y, p, thr=0.5, n_boot=20)
    
    assert res["positives"] == 0
    assert res["prevalence"] == 0.0
    assert res["recall"] is None
    assert res["pr_auc"] is None
    assert res["roc_auc"] is None
    assert res["pr_auc_ci_lower"] is None
    assert res["pr_auc_ci_upper"] is None
    assert "dataset contains only one class" in res["warnings"]
    assert res["status"] == "warning"


def test_single_class_all_ones():
    """Verify single-class dataset (all 1s) returns None for ranking AUCs and specificity."""
    y = np.ones(100, dtype=int)
    p = np.random.uniform(0, 1, 100)
    
    res = evaluate(y, p, thr=0.5, n_boot=20)
    
    assert res["positives"] == 100
    assert res["prevalence"] == 1.0
    assert res["specificity"] is None
    assert res["pr_auc"] is None
    assert res["roc_auc"] is None
    assert "dataset contains only one class" in res["warnings"]
    assert res["status"] == "warning"


def test_tiny_positives_warning():
    """Verify warning is issued when positives are fewer than 30."""
    y = np.zeros(200, dtype=int)
    y[:15] = 1  # 15 positives < 30
    p = np.random.uniform(0, 1, 200)
    
    res = evaluate(y, p, thr=0.5, n_boot=50)
    
    assert res["positives"] == 15
    assert "metrics unstable, sample has <30 positives" in res["warnings"]
    assert res["status"] == "warning"


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


def test_random_model():
    """Verify random model yields PR-AUC close to prevalence baseline."""
    rng = np.random.default_rng(42)
    n = 2000
    # Prevalence ~ 5%
    y = rng.binomial(1, 0.05, n)
    # Random uninformative predictions
    p = rng.uniform(0, 1, n)
    
    res = evaluate(y, p, thr=0.5, n_boot=100, random_state=42)
    
    assert res["positives"] >= 30
    assert res["pr_auc"] is not None
    # Lift for random model is approximately 1.0
    assert 0.5 <= res["pr_auc_lift"] <= 1.8
    assert res["random_baseline_pr_auc"] == pytest.approx(res["prevalence"])


def test_small_sample_500_rows():
    """Verify behavior on realistic 500-row sample (like sample CSV upload)."""
    rng = np.random.default_rng(123)
    n = 500
    y = np.zeros(n, dtype=int)
    y[:150] = 1  # 150 suspicious rows as described in prompt
    rng.shuffle(y)
    
    # Model with moderate signal
    p = np.where(y == 1, rng.beta(2, 5, n), rng.beta(1, 10, n))
    
    res = evaluate(y, p, thr=0.337784, n_boot=200, random_state=123)
    
    assert res["n"] == 500
    assert res["positives"] == 150
    assert res["prevalence"] == pytest.approx(0.30)
    assert res["pr_auc"] > 0.30
    assert res["pr_auc_ci_lower"] is not None
    assert res["pr_auc_ci_upper"] is not None
    assert res["pr_auc_ci_lower"] <= res["pr_auc"] <= res["pr_auc_ci_upper"]
    assert res["brier_score"] < 0.25


def test_large_sample_50000_rows():
    """Verify fast and stable execution on 50,000 rows."""
    rng = np.random.default_rng(999)
    n = 50000
    # Low AML prevalence ~ 0.2% (100 positives)
    y = rng.binomial(1, 0.002, n)
    # Model probabilities
    p = np.where(y == 1, rng.beta(5, 5, n), rng.beta(1, 50, n))
    
    res = evaluate(y, p, thr=0.337784, n_boot=100, random_state=999)
    
    assert res["n"] == 50000
    assert res["positives"] > 30
    assert res["pr_auc"] is not None
    assert res["pr_auc_lift"] > 1.0
    assert res["recall_at_alerts"][10] is not None
    assert res["recall_at_alerts"][50] is not None
    assert res["recall_at_alerts"][100] is not None


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
