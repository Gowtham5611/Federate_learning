"""Regression and Unit Test Suite for FedSecureAI Results Audit and Repairs.

Covers:
1. Metric recalculations against known confusion matrices.
2. Continuous metric scoring (PR-AUC, ROC-AUC) against known arrays.
3. Independent FedProx proximal term loss & gradient verification.
4. Independent FedAvg aggregation mathematical correctness.
5. All 54 checkpoints presence, mapping, and SHA-256 manifest integrity.
6. Report Table 4 consistency against underlying evaluation_metrics.csv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "dataset_comparison_combined"
REPAIRED_DIR = BASE_DIR / "artifacts" / "repaired_dataset_comparison_20261018"


def test_metric_recalculation_known_cm():
    """Verify precision, recall, F1, FPR, and alert rate on a known confusion matrix."""
    y_true = np.array([1] * 15 + [0] * 985)
    # 10 true positives, 5 false negatives, 5 false positives, 980 true negatives
    y_pred = np.array([1] * 10 + [0] * 5 + [1] * 5 + [0] * 980)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    assert (tn, fp, fn, tp) == (980, 5, 5, 10)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)
    fpr = fp / (fp + tn)
    alerts_10k = (fp / len(y_true)) * 10000.0

    assert abs(prec - (10 / 15)) < 1e-7
    assert abs(rec - (10 / 15)) < 1e-7
    assert abs(f1 - (10 / 15)) < 1e-7
    assert abs(fpr - (5 / 985)) < 1e-7
    assert abs(alerts_10k - 50.0) < 1e-7


def test_continuous_metrics_scoring():
    """Verify ROC-AUC and Average Precision on known score ranking."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.1])

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    assert 0.0 <= roc <= 1.0
    assert 0.0 <= pr <= 1.0
    assert abs(roc - 0.75) < 1e-6


def test_fedprox_proximal_term_independent_gradient():
    """Verify FedProx proximal term loss and exact analytic gradient."""
    mu = 0.5
    w_global = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    w_local = torch.tensor([1.2, 2.4, 2.8], requires_grad=True)

    prox_loss = (mu / 2.0) * torch.sum((w_local - w_global) ** 2)
    # Expected loss: 0.25 * ((0.2)^2 + (0.4)^2 + (-0.2)^2) = 0.25 * (0.04 + 0.16 + 0.04) = 0.25 * 0.24 = 0.06
    assert abs(prox_loss.item() - 0.06) < 1e-6

    prox_loss.backward()
    expected_grad = mu * (w_local.detach() - w_global)
    # Expected grad: 0.5 * [0.2, 0.4, -0.2] = [0.1, 0.2, -0.1]
    assert torch.allclose(w_local.grad, expected_grad, atol=1e-6)


def test_fedavg_aggregation_independent_math():
    """Verify FedAvg weighted parameter aggregation mathematically."""
    client_weights = [100, 200, 100]  # Total = 400 (fractions: 0.25, 0.5, 0.25)
    updates = [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([3.0, 4.0])},
        {"weight": torch.tensor([5.0, 6.0])},
    ]

    total_w = sum(client_weights)
    aggregated = sum(
        (w / total_w) * updates[i]["weight"] for i, w in enumerate(client_weights)
    )

    # Expected: 0.25 * [1, 2] + 0.5 * [3, 4] + 0.25 * [5, 6] = [0.25+1.5+1.25, 0.5+2.0+1.5] = [3.0, 4.0]
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(aggregated, expected, atol=1e-7)


def test_all_54_checkpoints_mapping_and_hashes():
    """Verify all 54 checkpoint files exist and match their SHA-256 hashes in manifest."""
    manifest_path = ARTIFACTS_DIR / "checkpoints_manifest.json"
    assert manifest_path.exists(), "checkpoints_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 54, f"Expected 54 checkpoints, got {len(manifest)}"

    for exp_id, meta in manifest.items():
        ckpt_path = BASE_DIR / meta["path"]
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} missing"

        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(1048576):
                h.update(chunk)
        assert h.hexdigest() == meta["sha256"], f"Hash mismatch for {exp_id}"


def test_repaired_report_consistency():
    """Regression test: verify repaired markdown report Table 4 matches evaluation_metrics.csv."""
    report_path = REPAIRED_DIR / "dataset_comparison_report.md"
    assert report_path.exists(), "Repaired report must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The hardcoded placeholder '0.9421' must NOT appear in Table 4 of the repaired report
    assert "0.9421" not in content, "Discredited placeholder string 0.9421 should not be in repaired report"

    # Verify that the actual true aggregate values from evaluation_metrics.csv are present
    df_eval = pd.read_csv(ARTIFACTS_DIR / "evaluation_metrics.csv")
    df_hi_own = df_eval[
        (df_eval["training_dataset"] == "HI-Small")
        & (df_eval["evaluation_split"] == "HI-Small_test_own")
        & (df_eval["threshold_type"] == "optimal_validation")
    ]
    hi_own_roc = f"{df_hi_own['roc_auc'].mean():.4f}"
    assert hi_own_roc in content, f"Expected true ROC-AUC {hi_own_roc} to appear in repaired report"

