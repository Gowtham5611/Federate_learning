import sys
from pathlib import Path
sys.path.extend([".", "backend"])

import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from scratch.test_recurring_6k_14k import generate_recurring_6k_14k

def main():
    print("=== GENERATING REVISED 70/30 THRESHOLD TRIGGER DEMONSTRATION CSV ===")
    df, patterns = generate_recurring_6k_14k(seed=42)
    
    # Target path
    csv_path = Path("data/demo/threshold_trigger_demo_only.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    csv_bytes = csv_path.read_bytes()
    file_size = len(csv_bytes)
    sha256_hash = hashlib.sha256(csv_bytes).hexdigest().lower()
    
    print(f"Saved CSV to: {csv_path.as_posix()}")
    print(f"File Size: {file_size:,} bytes")
    print(f"SHA-256 Digest: {sha256_hash}")
    assert len(sha256_hash) == 64 and all(c in "0123456789abcdef" for c in sha256_hash)
    
    # Pattern distribution
    pattern_counts = patterns.value_counts().to_dict()
    label_counts = df["Is Laundering"].value_counts().to_dict()
    print("\nPattern Breakdown:")
    for pat, cnt in pattern_counts.items():
        print(f"  - {pat}: {cnt:,} rows")
        
    print("\nTarget Label Breakdown:")
    print(f"  - Actual Positives (Is Laundering = 1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df):.1%})")
    print(f"  - Actual Negatives (Is Laundering = 0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df):.1%})")
    print(f"  - Total Rows: {len(df):,}")
    assert label_counts[1] == 6000, f"Expected exactly 6,000 positives, got {label_counts[1]}"
    assert label_counts[0] == 14000, f"Expected exactly 14,000 negatives, got {label_counts[0]}"
    assert len(df) == 20000, f"Expected exactly 20,000 rows, got {len(df)}"
    
    # Run through full CsvEvaluationService
    print("\n=== RUNNING THROUGH FULL PRODUCTION INFERENCE PIPELINE ===")
    service = CsvEvaluationService()
    result = service.evaluate_csv_bytes(
        csv_bytes=csv_bytes,
        filename="threshold_trigger_demo_only.csv",
        model_id="HI-Small", # Target sequence LSTM
        threshold=0.337784,
    )
    
    run_id = result["run_id"]
    model_name = result["model_name"]
    model_id = result["model_id"]
    thr = result["decision_threshold"]
    
    print(f"Evaluation Run ID: {run_id}")
    print(f"Evaluated Model Name: {model_name}")
    print(f"Evaluated Model ID: {model_id}")
    print(f"Decision Threshold: {thr}")
    
    em = result.get("evaluation_metrics", {})
    cm = em.get("confusion_matrix", {})
    tp = cm.get("tp", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)
    tn = cm.get("tn", 0)
    total_alerts = tp + fp
    
    prec = em.get("precision", 0.0)
    rec = em.get("recall", 0.0)
    f1 = em.get("f1_score", 0.0)
    pr_auc = em.get("pr_auc", 0.0)
    roc_auc = em.get("roc_auc", 0.0)
    
    print("\nConfusion Matrix Results:")
    print(f"  TP: {tp:,}")
    print(f"  FP: {fp:,}")
    print(f"  FN: {fn:,}")
    print(f"  TN: {tn:,}")
    print(f"  Total Flagged Alerts (TP+FP): {total_alerts:,}")
    print(f"  Precision: {prec:.4%}")
    print(f"  Recall: {rec:.4%}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    assert tp + fn == label_counts[1], f"TP+FN ({tp+fn}) != pos ({label_counts[1]})"
    assert fp + tn == label_counts[0], f"FP+TN ({fp+tn}) != neg ({label_counts[0]})"
    assert tp + fp + fn + tn == 20000, f"Sum ({tp+fp+fn+tn}) != 20000"
    assert total_alerts >= 501, f"Expected at least 501 alerts, got {total_alerts}"
    
    # Generate PDF report
    pdf_res = service.get_prediction_pdf_report(run_id)
    pdf_bytes = pdf_res[0] if isinstance(pdf_res, tuple) else pdf_res
    pdf_path = Path(f"results/uploads/final_demo_report_{run_id}.pdf")
    pdf_path.write_bytes(pdf_bytes)
    print(f"\nGenerated PDF Report: {pdf_path.as_posix()} ({len(pdf_bytes):,} bytes)")
    
    # Save manifest record
    manifest = {
        "dataset_name": "threshold_trigger_demo_only.csv",
        "purpose": "Purpose-built synthetic demonstration: threshold-trigger demo only — not an independent performance benchmark and not evidence of real-world AML detection.",
        "file_size_bytes": file_size,
        "sha256_hash": sha256_hash,
        "total_rows": len(df),
        "pattern_breakdown": pattern_counts,
        "label_breakdown": label_counts,
        "evaluation_run_id": run_id,
        "evaluated_model_id": model_id,
        "evaluated_model_name": model_name,
        "decision_threshold": thr,
        "total_flagged_scores_ge_threshold": total_alerts,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1_score": f1,
            "pr_auc": pr_auc, "roc_auc": roc_auc,
        },
        "pdf_report_path": pdf_path.as_posix(),
    }
    manifest_path = Path("results/uploads/threshold_trigger_demo_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Demonstration Manifest to: {manifest_path.as_posix()}")

if __name__ == "__main__":
    main()
