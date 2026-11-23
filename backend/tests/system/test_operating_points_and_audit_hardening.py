"""Regression tests for Operating Point behavior, hard top-K alert capacity,
deterministic tie-breaking, synthetic shortcut detection, and baseline hash integrity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    compute_metrics_at_capacity,
    compute_metrics_at_threshold,
    compute_operating_points,
    evaluate as run_robust_evaluate,
)


def test_fixed_threshold_mode_semantics():
    """Mode A: Fixed probability threshold should evaluate p >= threshold without alert capping."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

    thr = 0.5
    op = compute_metrics_at_threshold(y_true, p, thr=thr, name="Test Fixed Thr")

    assert op["operating_mode"] == "fixed_threshold"
    assert op["is_hard_capacity"] is False
    assert op["threshold"] == 0.5
    # Scores >= 0.5 are indices 0, 1, 2, 3 (4 alerts total)
    assert op["alerts"] == 4
    assert op["alerts_per_10k"] == 4000.0
    # TP: y=1 and p>=0.5 -> idx 0, 1 (2 TP)
    assert op["tp"] == 2
    # FP: y=0 and p>=0.5 -> idx 2, 3 (2 FP)
    assert op["fp"] == 2
    assert op["precision"] == 0.5


def test_fixed_alert_capacity_hard_capping_and_tie_breaking():
    """Mode B: Fixed alert capacity MUST enforce strict top-K capacity regardless of score ties.
    
    Verifies the fix for the bug where 50 alerts/10k produced 1,190 alerts on 4,000 test rows.
    """
    n = 4000
    # Synthetic test scenario: 1195 laundering rows all have high probability (e.g. 0.95)
    y_true = np.zeros(n, dtype=int)
    y_true[:1195] = 1
    p = np.zeros(n, dtype=float)
    p[:1195] = 0.95
    p[1195:] = 0.05

    # With capacity 50 per 10k: K = 4000 * (50 / 10000) = 20
    capacity_k = 20
    raw_amounts = np.linspace(100.0, 50000.0, n)  # arbitrary amounts

    op = compute_metrics_at_capacity(
        y_true, p, capacity_k=capacity_k, name="Alert Capacity: 50 / 10k", raw_amounts=raw_amounts
    )

    # CRITICAL: Alerts must be EXACTLY 20, NOT 1195!
    assert op["operating_mode"] == "fixed_capacity"
    assert op["is_hard_capacity"] is True
    assert op["alerts"] == 20
    assert op["alerts_per_10k"] == 50.0
    assert op["target_alerts"] == 20
    assert op["actual_alerts"] == 20
    assert op["tp"] == 20
    assert op["fp"] == 0
    assert op["precision"] == 1.0


def test_deterministic_tie_breaking_lexsort():
    """Verify tie-breaking: Primary (-p), Secondary (-amount), Tertiary (+index)."""
    y_true = np.array([0, 1, 1, 0])
    p = np.array([0.90, 0.90, 0.90, 0.50])
    # Item 1 has larger amount than Item 0; Item 2 has same amount as Item 1
    amounts = np.array([100.0, 500.0, 500.0, 20.0])

    # Request capacity 2:
    # Ties between idx 0, 1, 2 at p=0.90:
    # - idx 1 and idx 2 have amt=500.0 > idx 0 (amt=100.0)
    # - Between idx 1 and idx 2, idx 1 has lower row index (1 < 2)
    # Therefore, top 2 alerts MUST be idx 1 and idx 2!
    op = compute_metrics_at_capacity(y_true, p, capacity_k=2, raw_amounts=amounts)

    assert op["alerts"] == 2
    # Both idx 1 and idx 2 have y=1, so TP should be 2
    assert op["tp"] == 2
    assert op["fp"] == 0
    assert op["precision"] == 1.0


def test_compute_operating_points_structure():
    """Verify compute_operating_points returns distinct operating modes."""
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0] * 100)  # 1000 rows
    p = np.linspace(0.01, 0.99, len(y))

    ops = compute_operating_points(y, p, frozen_thr=0.337784)

    assert "frozen" in ops
    assert ops["frozen"]["operating_mode"] == "fixed_threshold"
    assert ops["frozen"]["is_hard_capacity"] is False

    assert "alert_budgets" in ops
    for b_key in ["10", "50", "100"]:
        assert b_key in ops["alert_budgets"]
        b_op = ops["alert_budgets"][b_key]
        assert b_op["operating_mode"] == "fixed_capacity"
        assert b_op["is_hard_capacity"] is True
        # For 1000 rows, b=50 -> k = round(1000 * 50 / 10000) = 5 alerts
        if b_key == "50":
            assert b_op["alerts"] == 5
            assert b_op["alerts_per_10k"] == 50.0

    assert "best_f1_on_file" in ops
    assert ops["best_f1_on_file"]["operating_mode"] == "what_if_diagnostic"
    assert ops["best_f1_on_file"]["is_diagnostic_only"] is True


def test_synthetic_shortcut_proof():
    """Inspect and prove mathematically that ibm_aml_schema_synthetic_20000.csv
    contains deterministic label shortcuts based on Amount and Payment Format.
    """
    csv_file = Path("data/demo/ibm_aml_schema_synthetic_20000.csv")
    assert csv_file.exists(), "Synthetic 20k file must exist"

    df = pd.read_csv(csv_file)
    amt_col = "Amount Paid"
    format_col = "Payment Format"
    label_col = "Is Laundering"

    laundering = df[df[label_col] == 1]
    normal = df[df[label_col] == 0]

    min_laundering_amt = laundering[amt_col].min()
    max_normal_amt = normal[amt_col].max()

    # Proof of amount separation shortcut:
    # In the generator, laundering amounts are in [50000, 250000], normal in [5, 4500]
    assert min_laundering_amt >= 50000.0, f"Expected min laundering amount >= 50k, got {min_laundering_amt}"
    assert max_normal_amt <= 4500.0, f"Expected max normal amount <= 4.5k, got {max_normal_amt}"
    assert min_laundering_amt > max_normal_amt, "Amount shortcut confirms 100% linear separability"

    # Proof of format separation shortcut:
    laundering_formats = set(laundering[format_col].unique())
    normal_formats = set(normal[format_col].unique())
    assert laundering_formats.isdisjoint(normal_formats), "Payment formats are completely disjoint between classes"


def test_baseline_hashes_preserved():
    """Verify all 273 frozen baseline files remain 100% bit-identical."""
    import subprocess
    res = subprocess.run([".venv/Scripts/python", "scripts/verify_baseline_hashes.py"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "SUCCESS: All 273 baseline models, checkpoints, and phase8_summary.csv are 100% BIT-IDENTICAL!" in res.stdout

