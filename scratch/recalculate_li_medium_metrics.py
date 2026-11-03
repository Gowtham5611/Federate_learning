import json
import numpy as np
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

def evaluate_config(pred_npz_path, json_metrics_path=None):
    data = np.load(pred_npz_path)
    labels = data["labels"].astype(int)
    probs = data["probabilities"].astype(float)
    threshold = float(data["threshold"]) if "threshold" in data else 0.5

    # Predictions based on frozen threshold
    preds = (probs >= threshold).astype(int)

    # Calculate metrics
    roc_auc = float(roc_auc_score(labels, probs))
    ap_score = float(average_precision_score(labels, probs))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    acc = float(accuracy_score(labels, preds))

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total = len(labels)
    total_alerts = tp + fp
    alerts_per_10k = float((total_alerts / total) * 10000)

    recalculated = {
        "total_test_samples": total,
        "positive_prevalence": float(np.mean(labels)),
        "frozen_threshold": threshold,
        "test_roc_auc": roc_auc,
        "test_pr_auc_average_precision": ap_score,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_accuracy": acc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_alerts": total_alerts,
        "alerts_per_10k": alerts_per_10k,
    }

    reported = {}
    if json_metrics_path and Path(json_metrics_path).exists():
        reported = json.loads(Path(json_metrics_path).read_text(encoding="utf-8"))

    return recalculated, reported

def main():
    print("=================================================================")
    print("LI-MEDIUM METRIC RECALCULATION & VALIDATION (CONFIGS 1-12)")
    print("=================================================================")

    # Configs 1-6 in Colab Results/artifacts/li_medium_experiment
    # Configs 7-12 in Colab Results/FedSecureAI_LI_Medium_configs_7_12

    configs = {}
    for i in range(1, 7):
        configs[f"config_{i}"] = (
            Path(f"Colab Results/artifacts/li_medium_experiment/predictions/config_{i}/test_predictions.npz"),
            Path(f"Colab Results/artifacts/li_medium_experiment/results/config_{i}_metrics.json")
        )

    for i in range(7, 13):
        configs[f"config_{i}"] = (
            Path(f"Colab Results/FedSecureAI_LI_Medium_configs_7_12/predictions/config_{i}/test_predictions.npz"),
            Path(f"Colab Results/FedSecureAI_LI_Medium_configs_7_12/results/config_{i}_metrics.json")
        )

    results_table = []
    for cid, (pred_path, json_path) in configs.items():
        if not pred_path.exists():
            print(f"Skipping {cid}: {pred_path} does not exist")
            continue
        recalc, rep = evaluate_config(pred_path, json_path)
        
        rep_f1 = rep.get("test_f1", rep.get("f1"))
        rep_roc = rep.get("test_roc_auc", rep.get("roc_auc"))
        rep_pr = rep.get("test_pr_auc", rep.get("pr_auc"))

        results_table.append({
            "config_id": cid,
            "recalculated": recalc,
            "reported": rep
        })

        print(f"\n--- {cid} ---")
        print(f"  Frozen Threshold: {recalc['frozen_threshold']:.6f}")
        print(f"  ROC-AUC: Recalc={recalc['test_roc_auc']:.6f} | Reported={rep_roc}")
        print(f"  PR-AUC (AP): Recalc={recalc['test_pr_auc_average_precision']:.6f} | Reported={rep_pr}")
        print(f"  F1-Score: Recalc={recalc['test_f1']:.6f} | Reported={rep_f1}")
        print(f"  Precision: {recalc['test_precision']:.6f} | Recall: {recalc['test_recall']:.6f}")
        print(f"  Confusion: TP={recalc['tp']}, FP={recalc['fp']}, FN={recalc['fn']}, TN={recalc['tn']}")
        print(f"  Alerts/10k: {recalc['alerts_per_10k']:.2f}")

    Path("scratch/li_medium_recalculated_metrics.json").write_text(
        json.dumps(results_table, indent=2), encoding="utf-8"
    )
    print("\nSaved full recalculated metrics to scratch/li_medium_recalculated_metrics.json")

if __name__ == "__main__":
    main()

