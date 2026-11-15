"""Phase H1: Authoritative artifact, checkpoint, and hash verification script."""

import csv
import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def audit_artifacts():
    print("=" * 80)
    print("PHASE H1: ARTIFACT & CHECKPOINT INTEGRITY AUDIT")
    print("=" * 80)

    hi_csv = BASE_DIR / "results/authoritative_results/HI_Small_results.csv"
    li_csv = BASE_DIR / "results/authoritative_results/LI_Medium_results.csv"
    mu_csv = BASE_DIR / "results/authoritative_results/fedprox_mu_verification.csv"
    registry_csv = BASE_DIR / "results/authoritative_results/experiment_registry.csv"

    for f in [hi_csv, li_csv, mu_csv, registry_csv]:
        assert f.exists(), f"Missing file: {f}"
        print(f"File verified: {f.name} ({f.stat().st_size} bytes)")

    hi_rows = list(csv.DictReader(open(hi_csv, encoding="utf-8")))
    li_rows = list(csv.DictReader(open(li_csv, encoding="utf-8")))

    print(f"\nAuditing {len(hi_rows)} HI-Small checkpoints & predictions...")
    hi_mismatches = 0
    for r in hi_rows:
        ckpt_path = BASE_DIR / r["checkpoint_path"]
        preds_path = BASE_DIR / r["predictions_path"]
        expected_sha = r["checkpoint_sha256"]
        if not ckpt_path.exists():
            print(f"FAIL: Missing ckpt {ckpt_path}")
            hi_mismatches += 1
            continue
        if not preds_path.exists():
            print(f"FAIL: Missing preds {preds_path}")
            hi_mismatches += 1
            continue
        actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            print(f"FAIL: SHA256 mismatch for {r['config_id']}: expected {expected_sha}, got {actual_sha}")
            hi_mismatches += 1
        else:
            print(f"  [OK] {r['config_id']} ({r['model_name']}) - SHA256 verified")

    print(f"\nAuditing {len(li_rows)} LI-Medium checkpoints & predictions...")
    li_mismatches = 0
    for r in li_rows:
        ckpt_path = BASE_DIR / r["checkpoint_path"]
        preds_path = BASE_DIR / r["predictions_path"]
        expected_sha = r["checkpoint_sha256"]
        if not ckpt_path.exists():
            print(f"FAIL: Missing ckpt {ckpt_path}")
            li_mismatches += 1
            continue
        if not preds_path.exists():
            print(f"FAIL: Missing preds {preds_path}")
            li_mismatches += 1
            continue
        actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            print(f"FAIL: SHA256 mismatch for {r['config_id']}: expected {expected_sha}, got {actual_sha}")
            li_mismatches += 1
        else:
            print(f"  [OK] {r['config_id']} ({r['model_name']}) - SHA256 verified")

    print(f"\nSummary: HI mismatches: {hi_mismatches}, LI mismatches: {li_mismatches}")


if __name__ == "__main__":
    audit_artifacts()

