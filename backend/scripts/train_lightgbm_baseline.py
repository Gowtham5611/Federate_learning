"""Train LightGBM Baseline on IBM AML HI-Small with Validation Early Stopping and Class Weighting.

Saves:
- Model: models/baselines/lightgbm_baseline.txt
- Predictions: results/posthoc/lightgbm_baseline_test_preds.npy
- Metrics: results/posthoc/lightgbm_baseline_results.json
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
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from evaluation.metrics import compute_ece, evaluate, fast_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainLightGBM")

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "train" / "transactions.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "validation" / "transactions.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "test" / "transactions.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "baselines"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results" / "posthoc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "amount_paid_log",
    "amount_received_log",
    "amount_mismatch",
    "same_currency",
    "same_bank",
    "self_transfer",
    "payment_format_code",
    "payment_currency_code",
    "receiving_currency_code",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_night",
    "sender_prior_count_log",
    "sender_prior_amount_log",
    "sender_seconds_since_prior_log",
    "sender_has_prior",
    "receiver_prior_count_log",
    "receiver_prior_amount_log",
    "receiver_seconds_since_prior_log",
    "receiver_has_prior",
]
LABEL_COL = "is_laundering"


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
    t_start = time.time()
    logger.info("Loading training data from %s...", TRAIN_PATH)
    df_train = pd.read_csv(TRAIN_PATH, usecols=FEATURE_COLS + [LABEL_COL])
    X_train = df_train[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = df_train[LABEL_COL].to_numpy(dtype=np.int32)
    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pos_weight = 5.0
    logger.info("Train: %d rows (%d pos, scale_pos_weight=%.2f)", len(y_train), n_pos_train, scale_pos_weight)

    logger.info("Loading validation data from %s...", VAL_PATH)
    df_val = pd.read_csv(VAL_PATH, usecols=FEATURE_COLS + [LABEL_COL])
    X_val = df_val[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = df_val[LABEL_COL].to_numpy(dtype=np.int32)
    logger.info("Validation: %d rows (%d pos)", len(y_val), int(y_val.sum()))

    logger.info("Loading test data from %s...", TEST_PATH)
    df_test = pd.read_csv(TEST_PATH, usecols=FEATURE_COLS + [LABEL_COL])
    X_test = df_test[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = df_test[LABEL_COL].to_numpy(dtype=np.int32)
    logger.info("Test: %d rows (%d pos)", len(y_test), int(y_test.sum()))

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    logger.info("Training LightGBM baseline with validation early stopping on average_precision...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=20),
    ]

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Save model
    model_file = MODELS_DIR / "lightgbm_baseline.txt"
    gbm.save_model(str(model_file))
    logger.info("Model saved to %s", model_file)

    # Validation predictions and threshold selection
    p_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    val_pr_auc = fast_pr_auc(y_val, p_val)
    val_roc_auc = roc_auc_score(y_val, p_val)
    logger.info("Validation PR-AUC: %.6f, ROC-AUC: %.6f", val_pr_auc, val_roc_auc)

    # Find optimal threshold on validation set (maximizing F1)
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thr = 0.5
    best_f1 = 0.0
    for thr in thresholds:
        f1 = f1_score(y_val, (p_val >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    logger.info("Validation optimal threshold: %.4f (F1: %.4f)", best_thr, best_f1)

    # Test predictions
    logger.info("Generating test predictions...")
    p_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    np.save(RESULTS_DIR / "lightgbm_baseline_test_preds.npy", p_test.astype(np.float32))

    # Evaluate at canonical threshold (0.337784) and validation-optimal threshold
    canonical_thr = 0.337784
    eval_canonical = evaluate(y_test, p_test, thr=canonical_thr, n_boot=0)
    eval_opt = evaluate(y_test, p_test, thr=best_thr, n_boot=0)
    ci = run_bootstrap_ci(y_test, p_test, n_boot=100, seed=42)

    # Feature importances
    importance = gbm.feature_importance(importance_type="gain")
    feature_imp = sorted(
        [{"feature": f, "importance": float(imp)} for f, imp in zip(FEATURE_COLS, importance)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    results = {
        "model_id": "lightgbm_baseline_hi_small",
        "description": "LightGBM Gradient Boosted Trees on 23 Tabular AML Features",
        "best_iteration": int(gbm.best_iteration),
        "scale_pos_weight": scale_pos_weight,
        "validation_metrics": {
            "pr_auc": float(val_pr_auc),
            "roc_auc": float(val_roc_auc),
            "optimal_threshold": float(best_thr),
            "optimal_f1": float(best_f1),
        },
        "test_metrics_canonical_threshold": {
            "threshold": canonical_thr,
            "pr_auc": float(eval_canonical["pr_auc"]),
            "pr_auc_ci_lower": float(ci["pr_auc_ci_lower"]),
            "pr_auc_ci_upper": float(ci["pr_auc_ci_upper"]),
            "roc_auc": float(eval_canonical["roc_auc"]),
            "roc_auc_ci_lower": float(ci["roc_auc_ci_lower"]),
            "roc_auc_ci_upper": float(ci["roc_auc_ci_upper"]),
            "f1": float(eval_canonical["f1"]),
            "precision": float(eval_canonical["precision"]),
            "recall": float(eval_canonical["recall"]),
            "alerts_per_10k": float(eval_canonical["alerts_per_10k"]),
            "brier_score": float(eval_canonical["brier_score"]),
            "ece": float(eval_canonical["ece"]),
        },
        "test_metrics_val_optimal_threshold": {
            "threshold": float(best_thr),
            "f1": float(eval_opt["f1"]),
            "precision": float(eval_opt["precision"]),
            "recall": float(eval_opt["recall"]),
            "alerts_per_10k": float(eval_opt["alerts_per_10k"]),
        },
        "top_features_by_gain": feature_imp[:10],
    }

    with open(RESULTS_DIR / "lightgbm_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved LightGBM baseline results to %s", RESULTS_DIR / "lightgbm_baseline_results.json")
    logger.info("Total elapsed: %.2f seconds", time.time() - t_start)


if __name__ == "__main__":
    main()
