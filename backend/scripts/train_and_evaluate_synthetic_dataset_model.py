"""Script to audit compatibility, train a dedicated cold-start prediction-time model,
and compare side-by-side with frozen baseline and always-negative baselines on untouched test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 1. Verification of File Hash & Provenance
CSV_PATH = Path("data/demo/ibm_aml_schema_synthetic_20000.csv")
with open(CSV_PATH, "rb") as f:
    CSV_SHA256 = hashlib.sha256(f.read()).hexdigest()

BASELINE_MODEL_PATH = Path("models/baselines/lightgbm_baseline.txt")
with open(BASELINE_MODEL_PATH, "rb") as f:
    BASELINE_SHA256 = hashlib.sha256(f.read()).hexdigest()

print(f"Dataset: {CSV_PATH}")
print(f"Dataset SHA-256: {CSV_SHA256}")
print(f"Frozen Baseline Model: {BASELINE_MODEL_PATH}")
print(f"Frozen Baseline SHA-256: {BASELINE_SHA256}")

# 2. Chronological Split (60% Train, 20% Val, 20% Test)
raw_df = pd.read_csv(CSV_PATH)
raw_df["Timestamp_dt"] = pd.to_datetime(raw_df["Timestamp"])
sorted_df = raw_df.sort_values(by="Timestamp_dt").reset_index(drop=True)

N_TOTAL = len(sorted_df)
N_TRAIN = int(N_TOTAL * 0.60)  # 12,000
N_VAL = int(N_TOTAL * 0.20)    # 4,000
N_TEST = N_TOTAL - N_TRAIN - N_VAL  # 4,000

train_df = sorted_df.iloc[:N_TRAIN].copy().reset_index(drop=True)
val_df = sorted_df.iloc[N_TRAIN : N_TRAIN + N_VAL].copy().reset_index(drop=True)
test_df = sorted_df.iloc[N_TRAIN + N_VAL :].copy().reset_index(drop=True)

print(f"\n--- Chronological Split ---")
print(f"Train: {len(train_df)} rows [{train_df['Timestamp_dt'].min()} to {train_df['Timestamp_dt'].max()}], Laundering: {train_df['Is Laundering'].sum()} ({train_df['Is Laundering'].mean():.4f})")
print(f"Val:   {len(val_df)} rows [{val_df['Timestamp_dt'].min()} to {val_df['Timestamp_dt'].max()}], Laundering: {val_df['Is Laundering'].sum()} ({val_df['Is Laundering'].mean():.4f})")
print(f"Test:  {len(test_df)} rows [{test_df['Timestamp_dt'].min()} to {test_df['Timestamp_dt'].max()}], Laundering: {test_df['Is Laundering'].sum()} ({test_df['Is Laundering'].mean():.4f})")

# 3. Fit Prediction-Time Encoders STRICTLY on Training Split
def extract_prediction_time_features(
    df: pd.DataFrame,
    encoders: Dict[str, Dict[Any, int]] | None = None,
    is_training: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
    """Extract features available strictly at prediction time for single transactions.
    
    NO historical velocity features, NO label leakage.
    """
    if encoders is None:
        encoders = {}

    feats = pd.DataFrame(index=df.index)

    # A. Amount features
    amt_paid = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)
    amt_rec = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0.0)
    feats["amount_paid_log"] = np.log1p(np.maximum(amt_paid, 0.0))
    feats["amount_received_log"] = np.log1p(np.maximum(amt_rec, 0.0))
    feats["amount_diff_log"] = np.log1p(np.abs(amt_paid - amt_rec))
    feats["amount_mismatch"] = (np.abs(amt_paid - amt_rec) > 1e-4).astype(np.float32)

    # B. Currencies
    pay_curr = df["Payment Currency"].astype(str)
    rec_curr = df["Receiving Currency"].astype(str)
    feats["same_currency"] = (pay_curr == rec_curr).astype(np.float32)

    # C. Banks
    from_bank = df["From Bank"].astype(str)
    to_bank = df["To Bank"].astype(str)
    feats["same_bank"] = (from_bank == to_bank).astype(np.float32)

    # D. Categorical columns (fit strictly on train)
    cat_cols = ["Payment Format", "Payment Currency", "Receiving Currency", "From Bank", "To Bank"]
    for col in cat_cols:
        series_str = df[col].astype(str)
        if is_training:
            unique_vals = sorted(series_str.unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            encoders[col] = mapping
        
        mapping = encoders.get(col, {})
        # Map unknown categories to -1
        feats[f"{col.lower().replace(' ', '_')}_code"] = series_str.map(mapping).fillna(-1).astype(np.int32)

    # E. Temporal features from timestamp
    ts_dt = pd.to_datetime(df["Timestamp"])
    minute_of_day = ts_dt.dt.hour * 60 + ts_dt.dt.minute
    dow = ts_dt.dt.dayofweek

    feats["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    feats["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    feats["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    feats["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    feats["is_weekend"] = (dow >= 5).astype(np.float32)
    feats["is_night"] = ((ts_dt.dt.hour < 6) | (ts_dt.dt.hour >= 22)).astype(np.float32)

    return feats, encoders

# Extract features
X_train, encoders = extract_prediction_time_features(train_df, is_training=True)
X_val, _ = extract_prediction_time_features(val_df, encoders=encoders, is_training=False)
X_test, _ = extract_prediction_time_features(test_df, encoders=encoders, is_training=False)

y_train = train_df["Is Laundering"].to_numpy(dtype=np.int32)
y_val = val_df["Is Laundering"].to_numpy(dtype=np.int32)
y_test = test_df["Is Laundering"].to_numpy(dtype=np.int32)

feature_names = list(X_train.columns)
print(f"Extracted {len(feature_names)} prediction-time features: {feature_names}")

# 4. Train New Dedicated Cold-Start Model on Training Split
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}

new_model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

# Save the new model checkpoint
OUTPUT_DIR = Path("results/dynamic_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEW_MODEL_PATH = OUTPUT_DIR / "synthetic_cold_start_lightgbm.txt"
new_model.save_model(str(NEW_MODEL_PATH))
print(f"\nSaved new model checkpoint to {NEW_MODEL_PATH}")

# 5. Tune Threshold STRICTLY on Validation Split
val_probs = new_model.predict(X_val)
val_pr_auc = average_precision_score(y_val, val_probs)
val_roc_auc = roc_auc_score(y_val, val_probs)

# Threshold selection on Validation:
# Candidate 1: Default canonical 0.50
# Candidate 2: Max-F1 on validation
cand_thresholds = np.linspace(0.01, 0.99, 100)
best_f1_val = -1.0
best_th_val = 0.5
for th in cand_thresholds:
    preds = (val_probs >= th).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1_val:
        best_f1_val = f1
        best_th_val = th

# Candidate 3: Alert Budget 50 / 10,000 on Validation (top 0.5% alerts)
sorted_val_probs = np.sort(val_probs)[::-1]
target_idx_50 = int(len(val_df) * (50 / 10000))
th_budget_50 = float(sorted_val_probs[target_idx_50]) if target_idx_50 < len(sorted_val_probs) else 0.5

print(f"\n--- Validation Set Tuning ---")
print(f"Validation PR-AUC: {val_pr_auc:.6f}, ROC-AUC: {val_roc_auc:.6f}")
print(f"Validation Selected Max-F1 Threshold: {best_th_val:.4f} (Val F1: {best_f1_val:.4f})")
print(f"Validation Alert Budget 50/10k Threshold: {th_budget_50:.4f}")

# 6. Evaluate Frozen Baseline on the Exact Same Untouched Test Set
# The frozen baseline expects 23 features including history
from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
service = CsvEvaluationService()
test_csv_bytes = test_df.to_csv(index=False).encode("utf-8")
baseline_eval = service.evaluate_csv_bytes(
    csv_bytes=test_csv_bytes,
    filename="untouched_test_4000.csv",
    model_id="lightgbm_tabular_fallback",
    threshold=0.337784,
)

# Extract frozen baseline test predictions
pred_id_baseline = baseline_eval["prediction_id"]
pred_csv_baseline = Path(f"results/uploads/{pred_id_baseline}.csv")
df_pred_baseline = pd.read_csv(pred_csv_baseline)
baseline_test_probs = df_pred_baseline["predicted_probability"].to_numpy(dtype=float)

# 7. Evaluate on UNTOUCHED Test Split (4,000 rows)
test_probs_new = new_model.predict(X_test)
test_amounts = test_df["Amount Paid"].to_numpy(dtype=float)

def calculate_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    alerts = int(tp + fp)
    alert_rate = float(alerts / len(y_true))
    alerts_per_10k = float(alert_rate * 10000.0)

    return {
        "threshold": float(threshold),
        "total_samples": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "alerts": alerts,
        "alert_rate": alert_rate,
        "alerts_per_10k": alerts_per_10k,
    }

# A. Always Negative Baseline on Untouched Test
metrics_always_neg = calculate_full_metrics(y_test, np.zeros_like(y_test, dtype=float), threshold=0.5)

# B. Frozen Baseline Model on Untouched Test
# Mode A: Canonical Fixed Threshold 0.337784
metrics_baseline_canonical = calculate_full_metrics(y_test, baseline_test_probs, threshold=0.337784)
metrics_baseline_canonical["operating_mode"] = "fixed_threshold"
metrics_baseline_canonical["description"] = "Fixed probability threshold calibrated offline on HI-Small validation split"

# Mode B: Strict Top-K Alert Capacity 50/10k (K = 20 alerts on 4,000 rows)
from evaluation.metrics import compute_metrics_at_capacity
k_test_50 = int(round(len(y_test) * (50 / 10000.0)))
metrics_baseline_b50 = compute_metrics_at_capacity(
    y_test, baseline_test_probs, capacity_k=k_test_50, name="Frozen Baseline Alert Capacity (50/10k)", raw_amounts=test_amounts
)

# C. New Cold-Start Model on Untouched Test
# Mode A: Fixed Threshold tuned strictly on Validation (0.2773)
metrics_new_val_tuned = calculate_full_metrics(y_test, test_probs_new, threshold=best_th_val)
metrics_new_val_tuned["operating_mode"] = "fixed_threshold"
metrics_new_val_tuned["description"] = f"Fixed threshold ({best_th_val:.4f}) tuned on validation split for Max-F1"

# Mode B: Strict Top-K Alert Capacity 50/10k (K = 20 alerts on 4,000 rows)
metrics_new_b50 = compute_metrics_at_capacity(
    y_test, test_probs_new, capacity_k=k_test_50, name="New Model Alert Capacity (50/10k)", raw_amounts=test_amounts
)

# Canonical 0.50 Fixed Threshold for New Model
metrics_new_canonical = calculate_full_metrics(y_test, test_probs_new, threshold=0.5)
metrics_new_canonical["operating_mode"] = "fixed_threshold"

results = {
    "provenance": {
        "dataset": str(CSV_PATH),
        "dataset_sha256": CSV_SHA256,
        "frozen_baseline_model": str(BASELINE_MODEL_PATH),
        "frozen_baseline_sha256": BASELINE_SHA256,
        "new_model": str(NEW_MODEL_PATH),
        "split_protocol": "Chronological 60% Train (12,000) / 20% Val (4,000) / 20% Test (4,000)",
        "prediction_id_baseline": pred_id_baseline,
        "feature_names_new": feature_names,
    },
    "test_comparison": {
        "always_negative": metrics_always_neg,
        "frozen_baseline_canonical": metrics_baseline_canonical,
        "frozen_baseline_alert_budget_50": metrics_baseline_b50,
        "new_model_canonical_0_5": metrics_new_canonical,
        "new_model_val_tuned": metrics_new_val_tuned,
        "new_model_alert_budget_50": metrics_new_b50,
    },
}

REPORT_JSON = Path("results/diagnostics/synthetic_20k_model_comparison.json")
with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print(f"{'METRIC':<25} | {'ALWAYS NEGATIVE':<15} | {'FROZEN BASELINE':<18} | {'NEW MODEL (Val-Tuned)':<20}")
print("=" * 90)
m_an = metrics_always_neg
m_fb = metrics_baseline_canonical
m_nm = metrics_new_val_tuned

print(f"{'Threshold':<25} | {m_an['threshold']:<15.4f} | {m_fb['threshold']:<18.4f} | {m_nm['threshold']:<20.4f}")
print(f"{'Positive Rate (Prev)':<25} | {m_an['prevalence']:<15.4f} | {m_fb['prevalence']:<18.4f} | {m_nm['prevalence']:<20.4f}")
print(f"{'Confusion Matrix':<25} | {str(m_an['confusion_matrix']):<15} | {str(m_fb['confusion_matrix']):<18} | {str(m_nm['confusion_matrix']):<20}")
print(f"{'TP / FP / FN / TN':<25} | {m_an['tp']}/{m_an['fp']}/{m_an['fn']}/{m_an['tn']} | {m_fb['tp']}/{m_fb['fp']}/{m_fb['fn']}/{m_fb['tn']} | {m_nm['tp']}/{m_nm['fp']}/{m_nm['fn']}/{m_nm['tn']}")
print(f"{'Precision':<25} | {m_an['precision']:<15.4f} | {m_fb['precision']:<18.4f} | {m_nm['precision']:<20.4f}")
print(f"{'Recall':<25} | {m_an['recall']:<15.4f} | {m_fb['recall']:<18.4f} | {m_nm['recall']:<20.4f}")
print(f"{'F1 Score':<25} | {m_an['f1']:<15.4f} | {m_fb['f1']:<18.4f} | {m_nm['f1']:<20.4f}")
print(f"{'Specificity':<25} | {m_an['specificity']:<15.4f} | {m_fb['specificity']:<18.4f} | {m_nm['specificity']:<20.4f}")
print(f"{'Balanced Accuracy':<25} | {m_an['balanced_accuracy']:<15.4f} | {m_fb['balanced_accuracy']:<18.4f} | {m_nm['balanced_accuracy']:<20.4f}")
print(f"{'Accuracy':<25} | {m_an['accuracy']:<15.4f} | {m_fb['accuracy']:<18.4f} | {m_nm['accuracy']:<20.4f}")
print(f"{'PR-AUC':<25} | {m_an['pr_auc']:<15.4f} | {m_fb['pr_auc']:<18.4f} | {m_nm['pr_auc']:<20.4f}")
print(f"{'ROC-AUC':<25} | {m_an['roc_auc']:<15.4f} | {m_fb['roc_auc']:<18.4f} | {m_nm['roc_auc']:<20.4f}")
print(f"{'Alert Rate':<25} | {m_an['alert_rate']:<15.4f} | {m_fb['alert_rate']:<18.4f} | {m_nm['alert_rate']:<20.4f}")
print(f"{'Alerts / 10k':<25} | {m_an['alerts_per_10k']:<15.1f} | {m_fb['alerts_per_10k']:<18.1f} | {m_nm['alerts_per_10k']:<20.1f}")
print("=" * 90)

print("\n" + "=" * 90)
print(f"{'MODE B: ALERT CAPACITY (50 / 10k, K=20)':<42} | {'FROZEN BASELINE':<20} | {'NEW MODEL (Hard Cap)':<20}")
print("=" * 90)
m_fb_b50 = metrics_baseline_b50
m_nm_b50 = metrics_new_b50
print(f"{'Target Capacity (/10k)':<42} | {50.0:<20.1f} | {50.0:<20.1f}")
print(f"{'Actual Enforced Alerts':<42} | {m_fb_b50['alerts']:<20d} | {m_nm_b50['alerts']:<20d}")
print(f"{'Actual Alerts (/10k)':<42} | {m_fb_b50['alerts_per_10k']:<20.1f} | {m_nm_b50['alerts_per_10k']:<20.1f}")
print(f"{'Effective Cutoff Threshold':<42} | {m_fb_b50['threshold']:<20.6f} | {m_nm_b50['threshold']:<20.6f}")
print(f"{'Confusion Matrix':<42} | {str([[m_fb_b50['tn'], m_fb_b50['fp']], [m_fb_b50['fn'], m_fb_b50['tp']]]):<20} | {str([[m_nm_b50['tn'], m_nm_b50['fp']], [m_nm_b50['fn'], m_nm_b50['tp']]]):<20}")
print(f"{'TP / FP / FN / TN':<42} | {m_fb_b50['tp']}/{m_fb_b50['fp']}/{m_fb_b50['fn']}/{m_fb_b50['tn']} | {m_nm_b50['tp']}/{m_nm_b50['fp']}/{m_nm_b50['fn']}/{m_nm_b50['tn']}")
print(f"{'Precision':<42} | {m_fb_b50['precision']:<20.4f} | {m_nm_b50['precision']:<20.4f}")
print(f"{'Recall':<42} | {m_fb_b50['recall']:<20.4f} | {m_nm_b50['recall']:<20.4f}")
print(f"{'F1 Score':<42} | {m_fb_b50['f1_score']:<20.4f} | {m_nm_b50['f1_score']:<20.4f}")
print(f"{'Specificity':<42} | {m_fb_b50['specificity']:<20.4f} | {m_nm_b50['specificity']:<20.4f}")
print(f"{'Balanced Accuracy':<42} | {m_fb_b50['balanced_accuracy']:<20.4f} | {m_nm_b50['balanced_accuracy']:<20.4f}")
print("=" * 90)
print(f"Saved full artifact report to {REPORT_JSON}")

