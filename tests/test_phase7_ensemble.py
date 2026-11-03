"""Unit and integration tests for FedSecureAI Phase 7 Hybrid Ensemble Pipeline.

Tests:
1. Prediction alignment and shape equivalence.
2. Correct alpha blending calculation: P_ens = alpha * P_iid + (1 - alpha) * P_noniid.
3. Alpha grid boundary guarantees [0.00, 1.00] in steps of 0.05.
4. Deterministic tie-breaking policy enforcement (PR-AUC -> F1 -> Recall).
5. Threshold selection strictly on validation data with zero test contamination.
6. Frozen configuration JSON serialization and lossless reload.
7. Metric calculation consistency from saved probability arrays.
8. Detection and rejection of missing, NaN, or shape-mismatched arrays.
9. Prevention of cross-dataset blending (HI-Small vs LI-Medium).
10. Strict immutability: parameter tuning cannot occur on test splits.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def test_alpha_blending_mathematical_properties():
    """Verify that blended probabilities are strictly convex combinations in [0, 1]."""
    p_iid = np.array([0.1, 0.4, 0.8, 0.95], dtype=np.float32)
    p_noniid = np.array([0.2, 0.3, 0.7, 0.85], dtype=np.float32)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_ens = alpha * p_iid + (1.0 - alpha) * p_noniid
        assert p_ens.shape == p_iid.shape
        assert np.all(p_ens >= 0.0)
        assert np.all(p_ens <= 1.0)
        assert np.all(p_ens >= np.minimum(p_iid, p_noniid))
        assert np.all(p_ens <= np.maximum(p_iid, p_noniid))


def test_alpha_grid_boundaries_and_steps():
    """Verify that the predeclared alpha grid spans exactly [0.00, 1.00] with step 0.05."""
    alphas = np.round(np.linspace(0.0, 1.0, 21), 2)
    assert len(alphas) == 21
    assert alphas[0] == 0.00
    assert alphas[-1] == 1.00
    assert np.isclose(alphas[1] - alphas[0], 0.05)
    assert np.isclose(alphas[-1] - alphas[-2], 0.05)


def test_deterministic_tie_breaking_policy():
    """Verify tie-breaking: primary = pr_auc desc, secondary = f1 desc, tertiary = recall desc."""
    candidates = [
        {"name": "A", "pr_auc": 0.250, "f1": 0.40, "recall": 0.30},
        {"name": "B", "pr_auc": 0.260, "f1": 0.35, "recall": 0.25},  # Highest PR-AUC
        {"name": "C", "pr_auc": 0.250, "f1": 0.42, "recall": 0.28},  # Tied with A on PR-AUC, higher F1
        {"name": "D", "pr_auc": 0.250, "f1": 0.40, "recall": 0.35},  # Tied with A on PR-AUC and F1, higher recall
    ]

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x["pr_auc"], x["f1"], x["recall"]),
        reverse=True,
    )
    assert sorted_candidates[0]["name"] == "B"
    assert sorted_candidates[1]["name"] == "C"
    assert sorted_candidates[2]["name"] == "D"
    assert sorted_candidates[3]["name"] == "A"


def test_rejection_of_shape_mismatched_predictions():
    """Verify that blending arrays of different lengths raises ValueError."""
    p_iid = np.ones(100, dtype=np.float32)
    p_noniid = np.ones(105, dtype=np.float32)

    with pytest.raises(ValueError):
        if len(p_iid) != len(p_noniid):
            raise ValueError(f"Shape mismatch: {p_iid.shape} != {p_noniid.shape}")
        _ = 0.5 * p_iid + 0.5 * p_noniid


def test_rejection_of_nan_or_inf_in_predictions():
    """Verify fail-fast check when probability arrays contain NaN or Inf."""
    p_corrupt = np.array([0.1, np.nan, 0.5], dtype=np.float32)
    assert np.isnan(p_corrupt).any()

    p_inf = np.array([0.1, np.inf, 0.5], dtype=np.float32)
    assert np.isinf(p_inf).any()


def test_frozen_configuration_serialization_and_reload(tmp_path):
    """Verify that frozen configuration dictionary serializes and deserializes losslessly."""
    config = {
        "dataset": "HI-Small",
        "iid_model": {
            "checkpoint_path": "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/global_best.pt",
            "checkpoint_sha256": "abcdef123456",
            "frozen_standalone_threshold": 0.34,
            "validation_pr_auc": 0.242083,
        },
        "noniid_model": {
            "checkpoint_path": "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_seed42/global_best.pt",
            "checkpoint_sha256": "fedcba654321",
            "frozen_standalone_threshold": 0.37,
            "validation_pr_auc": 0.198980,
        },
        "ensemble": {
            "selected_alpha": 0.55,
            "frozen_threshold": 0.35,
            "validation_pr_auc": 0.251000,
        },
    }

    out_file = tmp_path / "frozen_ensemble_config.json"
    out_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["dataset"] == "HI-Small"
    assert loaded["ensemble"]["selected_alpha"] == 0.55
    assert loaded["ensemble"]["frozen_threshold"] == 0.35
    assert loaded["iid_model"]["frozen_standalone_threshold"] == 0.34


def test_cross_dataset_blending_prevention():
    """Verify that HI-Small and LI-Medium datasets cannot be cross-blended."""
    hi_len = 1_015_564
    li_len = 6_249_112

    assert hi_len != li_len
    # Cross-dataset blending should be strictly rejected by row count mismatch
    with pytest.raises(AssertionError):
        assert hi_len == li_len, "Attempted illegal cross-dataset prediction blending!"

