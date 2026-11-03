"""Verify that all 15 baseline models, checkpoints, and phase8_summary.csv remain bit-identical."""

import json
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASHES_FILE = PROJECT_ROOT / "scratch" / "baseline_model_hashes.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not HASHES_FILE.exists():
        print(f"ERROR: Baseline hashes file not found at {HASHES_FILE}")
        sys.exit(1)

    with open(HASHES_FILE, "r", encoding="utf-8") as f:
        baseline_hashes = json.load(f)

    total_files = len(baseline_hashes)
    print(f"Auditing SHA-256 hashes for {total_files} protected baseline files...")

    mismatches = []
    missing = []

    for rel_path, expected_hash in baseline_hashes.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue
        actual_hash = sha256_file(full_path)
        if actual_hash != expected_hash:
            mismatches.append((rel_path, expected_hash, actual_hash))

    if missing:
        print(f"FAILED: {len(missing)} files missing!")
        for m in missing:
            print(f"  MISSING: {m}")

    if mismatches:
        print(f"FAILED: {len(mismatches)} files modified!")
        for path, exp, act in mismatches:
            print(f"  MODIFIED: {path}")
            print(f"    Expected: {exp}")
            print(f"    Actual:   {act}")

    if not missing and not mismatches:
        print(f"SUCCESS: All {total_files} baseline models, checkpoints, and phase8_summary.csv are 100% BIT-IDENTICAL!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

