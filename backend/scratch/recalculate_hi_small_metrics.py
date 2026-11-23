import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

def main():
    print("=================================================================")
    print("HI-SMALL METRIC RECALCULATION & VALIDATION (CONFIGS 1-12)")
    print("=================================================================")

    test_csv = Path("data/processed/ibm_aml/HI-Small/test/transactions.csv")
    print(f"Reading test labels from {test_csv}...")
    
    # Read only target column
    # Check column name first
    header_df = pd.read_csv(test_csv, nrows=5)
    print("Available columns in test CSV:", list(header_df.columns))
    target_col = [c for c in header_df.columns if "laundering" in c.lower()][0]
    print(f"Target column identified: {target_col}")

    # Read entire column in chunks or single series
    df_y = pd.read_csv(test_csv, usecols=[target_col])
    y_test = df_y[target_col].to_numpy().astype(int)
    print(f"Loaded {len(y_test)} labels: {np.sum(y_test == 1)} positive, {np.sum(y_test == 0)} negative")

    # Locate HI-Small predictions
    pred_dir = Path("colab work and results/artifacts/fedavg_fedprox_full_comparison/predictions")
    manifest_p = Path("colab work and results/artifacts/fedavg_fedprox_full_comparison/resume_manifest.json")
    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    
    hi_manifest_cfgs = {c["exp_id"]: c for c in manifest_data["configurations"] if c["dataset"] == "HI-Small"}

    recalculated_list = []
    for exp_id, m_cfg in hi_manifest_cfgs.items():
        pred_file = pred_dir / f"{exp_id}_test_preds.npy"
        if not pred_file.exists():
            print(f"Missing predictions file: {pred_file}")
            continue

        probs = np.load(pred_file)
        if len(probs) != len(y_test):
            print(f"Length mismatch for {exp_id}: preds={len(probs)}, labels={len(y_test)}")
            continue

        threshold = float(m_cfg.get("frozen_classification_threshold") or 0.5)
        preds = (probs >= threshold).astype(int)

        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        acc = float(accuracy_score(y_test, preds))

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        alerts_per_10k = float(((tp + fp) / len(y_test)) * 10000)

        rep_roc = m_cfg.get("test_roc_auc")
        rep_pr = m_cfg.get("test_pr_auc")

        recalculated_list.append({
            "exp_id": exp_id,
            "architecture": m_cfg.get("architecture"),
            "partition_strategy": m_cfg.get("partition_strategy"),
            "algorithm": m_cfg.get("algorithm"),
            "frozen_threshold": threshold,
            "recalculated_roc_auc": roc_auc,
            "reported_roc_auc": rep_roc,
            "recalculated_pr_auc": pr_auc,
            "reported_pr_auc": rep_pr,
            "recalculated_f1": f1,
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts_per_10k": alerts_per_10k
        })

        print(f"\n--- {exp_id} ---")
        print(f"  ROC-AUC: Recalc={roc_auc:.6f} | Reported={rep_roc}")
        print(f"  PR-AUC (AP): Recalc={pr_auc:.6f} | Reported={rep_pr}")
        print(f"  F1-Score: {f1:.6f} | Precision: {prec:.6f} | Recall: {rec:.6f}")
        print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn} | Alerts/10k={alerts_per_10k:.2f}")

    Path("scratch/hi_small_recalculated_metrics.json").write_text(
        json.dumps(recalculated_list, indent=2), encoding="utf-8"
    )
    print("\nSaved full recalculated metrics to scratch/hi_small_recalculated_metrics.json")

if __name__ == "__main__":
    main()

