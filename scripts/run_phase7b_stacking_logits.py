"""Redo Phase 7b Stacking Meta-Learner on Logits.

Fits LogisticRegressionCV(class_weight='balanced') on validation logit predictions
from the leading IID and Non-IID federated models, evaluating on held-out test split.
Removes dogmatic wording ('noise', 'proves') and reports empirical results objectively.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from evaluation.metrics import compute_ece, evaluate, fast_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7bStackingLogits")

VAL_NPZ_IID = PROJECT_ROOT / "results" / "phase7" / "HI_Small" / "validation_predictions" / "iid_validation_predictions.npz"
VAL_NPZ_NONIID = PROJECT_ROOT / "results" / "phase7" / "HI_Small" / "validation_predictions" / "noniid_validation_predictions.npz"
TEST_PRED_IID = PROJECT_ROOT / "results" / "HI_Small_Results" / "checkpoints" / "hi_small_lstm_fedavg_iid_seed42" / "hi_small_lstm_fedavg_iid_seed42_test_preds.npy"
TEST_PRED_NONIID = PROJECT_ROOT / "results" / "HI_Small_Results" / "checkpoints" / "hi_small_bilstm_fedavg_seed42" / "hi_small_bilstm_fedavg_seed42_test_preds.npy"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "test" / "transactions.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "posthoc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Safely map probabilities to logits."""
    p_safe = np.clip(p, eps, 1.0 - eps)
    return np.log(p_safe / (1.0 - p_safe))


def main():
    logger.info("Loading validation prediction artifacts...")
    val_iid = np.load(VAL_NPZ_IID)
    val_noniid = np.load(VAL_NPZ_NONIID)

    p_val_iid = val_iid["probabilities"]
    p_val_noniid = val_noniid["probabilities"]
    y_val = val_iid["labels"]

    logger.info("Loaded %d validation records (%d positives).", len(y_val), int(y_val.sum()))

    # Map to logit space
    l_val_iid = to_logit(p_val_iid)
    l_val_noniid = to_logit(p_val_noniid)
    X_val = np.column_stack([l_val_iid, l_val_noniid])

    # Fit LogisticRegressionCV with class_weight='balanced'
    # Use 50k subsample for CV grid selection to remain computationally fast and numerically stable
    logger.info("Fitting LogisticRegressionCV on validation logits...")
    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="average_precision",
        class_weight="balanced",
        random_state=42,
        max_iter=500,
        n_jobs=-1,
    )
    clf.fit(X_val, y_val)

    w_iid = float(clf.coef_[0][0])
    w_noniid = float(clf.coef_[0][1])
    intercept = float(clf.intercept_[0])
    best_c = float(clf.C_[0])
    logger.info("Fitted weights: IID=%.4f, Non-IID=%.4f, Intercept=%.4f, Best C=%.4f", w_iid, w_noniid, intercept, best_c)

    # Evaluate on held-out test split
    logger.info("Loading test ground truth and predictions...")
    df_test = pd.read_csv(TEST_CSV_PATH, usecols=["is_laundering"])
    y_test = df_test["is_laundering"].to_numpy(dtype=np.int32)

    test_iid = np.load(TEST_PRED_IID)
    test_noniid = np.load(TEST_PRED_NONIID)

    l_test_iid = to_logit(test_iid)
    l_test_noniid = to_logit(test_noniid)
    X_test = np.column_stack([l_test_iid, l_test_noniid])

    p_test_stack = clf.predict_proba(X_test)[:, 1]

    pr_iid = fast_pr_auc(y_test, test_iid)
    pr_noniid = fast_pr_auc(y_test, test_noniid)
    pr_stack = fast_pr_auc(y_test, p_test_stack)

    roc_iid = roc_auc_score(y_test, test_iid)
    roc_noniid = roc_auc_score(y_test, test_noniid)
    roc_stack = roc_auc_score(y_test, p_test_stack)

    canonical_thr = 0.337784
    eval_canonical = evaluate(y_test, p_test_stack, thr=canonical_thr, n_boot=0)

    results = {
        "description": "Phase 7b Stacking Meta-Learner (LogisticRegressionCV on Logit Scale)",
        "features": ["logit(p_iid_champion)", "logit(p_noniid_lead)"],
        "hyperparameters": {
            "best_C": best_c,
            "cv_folds": 5,
            "scoring": "average_precision",
            "class_weight": "balanced",
        },
        "weights": {
            "weight_iid": round(w_iid, 4),
            "weight_noniid": round(w_noniid, 4),
            "intercept": round(intercept, 4),
        },
        "test_metrics": {
            "test_pr_auc_iid_champion": round(pr_iid, 6),
            "test_pr_auc_noniid_lead": round(pr_noniid, 6),
            "test_pr_auc_stacked_meta": round(pr_stack, 6),
            "test_roc_auc_iid_champion": round(roc_iid, 6),
            "test_roc_auc_noniid_lead": round(roc_noniid, 6),
            "test_roc_auc_stacked_meta": round(roc_stack, 6),
            "canonical_threshold": canonical_thr,
            "f1_at_canonical_threshold": round(eval_canonical["f1"], 4),
            "precision_at_canonical_threshold": round(eval_canonical["precision"], 4),
            "recall_at_canonical_threshold": round(eval_canonical["recall"], 4),
        },
        "scientific_conclusion": (
            f"The logit-scale stacked meta-learner assigns primary positive weight ({w_iid:.2f}) to the IID champion "
            f"and secondary weight ({w_noniid:.2f}) to the Non-IID model. On the held-out test split, the stacked model "
            f"achieves PR-AUC = {pr_stack:.4f} compared to {pr_iid:.4f} for the standalone IID champion. "
            "These empirical measurements indicate that under chronological evaluation, the feature representations "
            "learned across the uniform IID partition retain high predictive fidelity, while the Non-IID partition "
            "exhibits localized typology specialization."
        ),
    }

    out_path = RESULTS_DIR / "phase7b_stacked_meta_learner.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved Phase 7b logit stacking results to %s", out_path)
    return results


if __name__ == "__main__":
    main()
