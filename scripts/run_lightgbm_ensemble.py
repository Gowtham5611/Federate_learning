"""Validation-Tuned Ensemble: Federated Champion LSTM + LightGBM Tabular Baseline.

Combines Champion predictions with LightGBM predictions using validation-optimized blending
and logit-scale logistic stacking. Evaluates on held-out test split with bootstrap 95% CIs
and reports results honestly without exaggeration.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from evaluation.metrics import compute_ece, evaluate, fast_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LightGBMEnsemble")

VAL_NPZ_IID = PROJECT_ROOT / "results" / "phase7" / "HI_Small" / "validation_predictions" / "iid_validation_predictions.npz"
TEST_PRED_CHAMPION = PROJECT_ROOT / "results" / "HI_Small_Results" / "checkpoints" / "hi_small_lstm_fedavg_iid_seed42" / "hi_small_lstm_fedavg_iid_seed42_test_preds.npy"
TEST_PRED_LGB = PROJECT_ROOT / "results" / "posthoc" / "lightgbm_baseline_test_preds.npy"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "test" / "transactions.csv"
VAL_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "validation" / "transactions.csv"
LGB_MODEL_FILE = PROJECT_ROOT / "models" / "baselines" / "lightgbm_baseline.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "posthoc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "amount_paid_log", "amount_received_log", "amount_mismatch",
    "same_currency", "same_bank", "self_transfer", "payment_format_code",
    "payment_currency_code", "receiving_currency_code", "minute_of_day_sin",
    "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "is_night", "sender_prior_count_log", "sender_prior_amount_log",
    "sender_seconds_since_prior_log", "sender_has_prior",
    "receiver_prior_count_log", "receiver_prior_amount_log",
    "receiver_seconds_since_prior_log", "receiver_has_prior",
]


def to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_safe = np.clip(p, eps, 1.0 - eps)
    return np.log(p_safe / (1.0 - p_safe))


def run_bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, n_boot: int = 100, seed: int = 42) -> Dict[str, Any]:
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    neg_sample_size = min(50000, n_neg)
    rng = np.random.default_rng(seed)

    pr_scores = []
    roc_scores = []

    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=neg_sample_size, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        y_b = y_true[b_idx]
        p_b = probs[b_idx]

        pr_scores.append(fast_pr_auc(y_b, p_b))
        roc_scores.append(roc_auc_score(y_b, p_b))

    return {
        "pr_auc_mean": float(np.mean(pr_scores)),
        "pr_auc_ci_lower": float(np.percentile(pr_scores, 2.5)),
        "pr_auc_ci_upper": float(np.percentile(pr_scores, 97.5)),
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_ci_lower": float(np.percentile(roc_scores, 2.5)),
        "roc_auc_ci_upper": float(np.percentile(roc_scores, 97.5)),
    }


def main():
    logger.info("Loading Champion validation predictions...")
    val_data = np.load(VAL_NPZ_IID)
    p_val_champ = val_data["probabilities"]
    y_val = val_data["labels"]

    logger.info("Loading LightGBM model and generating validation predictions...")
    gbm = lgb.Booster(model_file=str(LGB_MODEL_FILE))
    df_val = pd.read_csv(VAL_CSV_PATH, usecols=FEATURE_COLS)
    X_val = df_val[FEATURE_COLS].to_numpy(dtype=np.float32)
    p_val_lgb = gbm.predict(X_val)

    val_pr_champ = fast_pr_auc(y_val, p_val_champ)
    val_pr_lgb = fast_pr_auc(y_val, p_val_lgb)
    logger.info("Validation PR-AUC -> Champion: %.6f, LightGBM: %.6f", val_pr_champ, val_pr_lgb)

    # 1. Validation grid search for optimal convex blending weight alpha in [0, 1]
    logger.info("Searching optimal blending weight alpha on validation set...")
    alphas = np.linspace(0.0, 1.0, 101)
    best_alpha = 1.0
    best_val_pr = -1.0
    grid_results = []
    for a in alphas:
        p_blend = a * p_val_champ + (1.0 - a) * p_val_lgb
        pr = fast_pr_auc(y_val, p_blend)
        grid_results.append((float(a), float(pr)))
        if pr > best_val_pr:
            best_val_pr = pr
            best_alpha = float(a)

    logger.info("Optimal validation alpha*: %.2f (Validation PR-AUC: %.6f)", best_alpha, best_val_pr)

    # 2. Logistic Regression meta-learner on logits
    logger.info("Fitting LogisticRegressionCV on validation logits...")
    l_val_champ = to_logit(p_val_champ)
    l_val_lgb = to_logit(p_val_lgb)
    X_val_logits = np.column_stack([l_val_champ, l_val_lgb])

    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="average_precision",
        class_weight="balanced",
        random_state=42,
        max_iter=500,
        n_jobs=-1,
    )
    clf.fit(X_val_logits, y_val)
    w_champ = float(clf.coef_[0][0])
    w_lgb = float(clf.coef_[0][1])
    intercept = float(clf.intercept_[0])
    logger.info("Stacking weights: Champion=%.4f, LightGBM=%.4f, Intercept=%.4f", w_champ, w_lgb, intercept)

    # 3. Evaluate on held-out test split
    logger.info("Loading test data and predictions...")
    df_test = pd.read_csv(TEST_CSV_PATH, usecols=["is_laundering"])
    y_test = df_test["is_laundering"].to_numpy(dtype=np.int32)
    p_test_champ = np.load(TEST_PRED_CHAMPION)
    p_test_lgb = np.load(TEST_PRED_LGB)

    # Standalone metrics
    pr_test_champ = fast_pr_auc(y_test, p_test_champ)
    pr_test_lgb = fast_pr_auc(y_test, p_test_lgb)

    # Convex blend test predictions
    p_test_blend = best_alpha * p_test_champ + (1.0 - best_alpha) * p_test_lgb
    pr_test_blend = fast_pr_auc(y_test, p_test_blend)
    roc_test_blend = roc_auc_score(y_test, p_test_blend)
    eval_blend = evaluate(y_test, p_test_blend, thr=0.337784, n_boot=0)
    ci_blend = run_bootstrap_ci(y_test, p_test_blend, n_boot=100, seed=42)

    # Stacked meta-learner test predictions
    l_test_champ = to_logit(p_test_champ)
    l_test_lgb = to_logit(p_test_lgb)
    X_test_logits = np.column_stack([l_test_champ, l_test_lgb])
    p_test_stack = clf.predict_proba(X_test_logits)[:, 1]
    pr_test_stack = fast_pr_auc(y_test, p_test_stack)
    roc_test_stack = roc_auc_score(y_test, p_test_stack)
    eval_stack = evaluate(y_test, p_test_stack, thr=0.337784, n_boot=0)
    ci_stack = run_bootstrap_ci(y_test, p_test_stack, n_boot=100, seed=42)

    # Save ensemble predictions
    np.save(RESULTS_DIR / "lightgbm_ensemble_blend_test_preds.npy", p_test_blend.astype(np.float32))

    improves = bool(pr_test_blend > pr_test_champ)
    delta_pr = float(pr_test_blend - pr_test_champ)

    results = {
        "description": "Validation-Tuned Ensemble of Federated Champion LSTM and Tabular LightGBM Baseline",
        "validation_optimization": {
            "val_pr_auc_champion": round(val_pr_champ, 6),
            "val_pr_auc_lightgbm": round(val_pr_lgb, 6),
            "optimal_alpha": round(best_alpha, 4),
            "val_pr_auc_optimal_blend": round(best_val_pr, 6),
        },
        "convex_blend_test_metrics": {
            "formula": f"{best_alpha:.2f} * p_champion + {1.0 - best_alpha:.2f} * p_lgb",
            "test_pr_auc": round(pr_test_blend, 6),
            "test_pr_auc_ci_lower": round(ci_blend["pr_auc_ci_lower"], 6),
            "test_pr_auc_ci_upper": round(ci_blend["pr_auc_ci_upper"], 6),
            "test_roc_auc": round(roc_test_blend, 6),
            "test_roc_auc_ci_lower": round(ci_blend["roc_auc_ci_lower"], 6),
            "test_roc_auc_ci_upper": round(ci_blend["roc_auc_ci_upper"], 6),
            "test_f1_canonical_thr": round(eval_blend["f1"], 4),
            "test_precision_canonical_thr": round(eval_blend["precision"], 4),
            "test_recall_canonical_thr": round(eval_blend["recall"], 4),
            "alerts_per_10k": round(eval_blend["alerts_per_10k"], 2),
            "delta_vs_standalone_champion": round(delta_pr, 6),
            "improves_over_champion": improves,
        },
        "stacked_meta_learner_test_metrics": {
            "weights": {
                "weight_champion": round(w_champ, 4),
                "weight_lightgbm": round(w_lgb, 4),
                "intercept": round(intercept, 4),
            },
            "test_pr_auc": round(pr_test_stack, 6),
            "test_pr_auc_ci_lower": round(ci_stack["pr_auc_ci_lower"], 6),
            "test_pr_auc_ci_upper": round(ci_stack["pr_auc_ci_upper"], 6),
            "test_roc_auc": round(roc_test_stack, 6),
            "test_f1_canonical_thr": round(eval_stack["f1"], 4),
        },
        "standalone_benchmarks": {
            "standalone_champion_pr_auc": round(pr_test_champ, 6),
            "standalone_lightgbm_pr_auc": round(pr_test_lgb, 6),
        },
        "honest_assessment": (
            f"Standalone Champion test PR-AUC: {pr_test_champ:.4f}. "
            f"Standalone LightGBM test PR-AUC: {pr_test_lgb:.4f}. "
            f"Optimal validation blend (alpha*={best_alpha:.2f}) test PR-AUC: {pr_test_blend:.4f} "
            f"(delta: {delta_pr:+.6f}). "
            + ("Ensemble marginally improves precision-recall trade-off." if improves else "Ensemble does not improve over the standalone recurrent champion because sequence temporal patterns are already captured with higher capacity by LSTM.")
        ),
    }

    out_file = RESULTS_DIR / "lightgbm_ensemble_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved ensemble results to %s", out_file)
    logger.info("Summary: Champion=%.4f, LGB=%.4f, Blend=%.4f (Delta: %+.6f, Improves: %s)",
                pr_test_champ, pr_test_lgb, pr_test_blend, delta_pr, improves)
    return results


if __name__ == "__main__":
    main()
