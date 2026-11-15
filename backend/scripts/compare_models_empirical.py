"""Empirical Model Comparison Script for FedSecureAI Phase 2.

Evaluates:
1. Centralized LSTM
2. FedAvg LSTM (HI-Small Champion)
3. FedProx LSTM (mu=0.0001)
4. Tabular Fallback Model (LightGBM)

Across identical records on:
- data/demo_slices/test_slice_5000.csv (5,000 real IBM AML test transactions with temporal context)
- data/demo/ibm_aml_schema_synthetic_20000.csv (20,000 synthetic transactions)

Computes and reports:
- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- MCC
- Confusion Matrix (TP, FP, TN, FN)
- Calibration Metrics (Brier Score, ECE)
- Prediction Probability Distributions (min, 25%, median, mean, 75%, max, std)
- Decision threshold used
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, brier_score_loss, accuracy_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.infrastructure.models.tabular_fallback import TabularFallbackModel, FEATURE_COLS
from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from evaluation.metrics import fast_pr_auc, compute_ece


def evaluate_model_on_dataset(service: CsvEvaluationService, csv_path: Path, model_id: str, threshold: float = None):
    csv_bytes = csv_path.read_bytes()
    filename = csv_path.name
    
    if model_id == "lightgbm_tabular_fallback":
        res = service.evaluate_csv_bytes(
            csv_bytes=csv_bytes,
            filename=filename,
            fallback_model="tabular",
            threshold=threshold
        )
    else:
        res = service.evaluate_csv_bytes(
            csv_bytes=csv_bytes,
            filename=filename,
            model_id=model_id,
            threshold=threshold
        )
    
    predictions = res.get("predictions", [])
    probs = np.array([p["predicted_probability"] for p in predictions], dtype=np.float64)
    labels = np.array([p["ground_truth_label"] for p in predictions], dtype=np.int64)
    eff_thresh = res.get("decision_threshold", 0.337784)
    
    preds = (probs >= eff_thresh).astype(int)
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(preds)) > 1 and len(np.unique(labels)) > 1 else 0.0
    
    pr_auc = fast_pr_auc(labels, probs)
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    brier = brier_score_loss(labels, probs)
    ece = compute_ece(labels, probs, n_bins=10)
    
    dist = {
        "min": float(np.min(probs)),
        "p25": float(np.percentile(probs, 25)),
        "median": float(np.median(probs)),
        "mean": float(np.mean(probs)),
        "p75": float(np.percentile(probs, 75)),
        "max": float(np.max(probs)),
        "std": float(np.std(probs)),
    }
    
    return {
        "model_id": model_id,
        "model_name": res.get("model_name"),
        "total_records": len(labels),
        "positive_records": int(np.sum(labels)),
        "negative_records": int(len(labels) - np.sum(labels)),
        "decision_threshold": float(eff_thresh),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "accuracy": float(acc),
        "mcc": float(mcc),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "calibration": {
            "brier_score": float(brier),
            "ece": float(ece)
        },
        "probability_distribution": dist,
        "flagged_alerts": int(np.sum(preds))
    }


def main():
    service = CsvEvaluationService()
    
    models = [
        ("centralized_lstm", 0.703287),
        ("hi_small_lstm_fedavg_iid_seed42", 0.337784),
        ("hi_small_lstm_fedprox_mu0.0001_seed42", 0.360000),
        ("lightgbm_tabular_fallback", 0.337784),
    ]
    
    datasets = [
        PROJECT_ROOT / "data" / "demo_slices" / "test_slice_5000.csv",
        PROJECT_ROOT / "data" / "demo" / "ibm_aml_schema_synthetic_20000.csv"
    ]
    
    all_results = {}
    
    for d_path in datasets:
        print(f"\n========================================================")
        print(f"Evaluating Dataset: {d_path.name}")
        print(f"========================================================")
        all_results[d_path.name] = []
        
        for m_id, th in models:
            print(f"  Evaluating {m_id} (threshold={th})...")
            res = evaluate_model_on_dataset(service, d_path, m_id, threshold=th)
            all_results[d_path.name].append(res)
            print(f"    PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | F1: {res['f1_score']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | Alerts: {res['flagged_alerts']}")
    
    out_file = PROJECT_ROOT / "results" / "phase2_model_comparison_empirical.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_file}")


if __name__ == "__main__":
    main()

