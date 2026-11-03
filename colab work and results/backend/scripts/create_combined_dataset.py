"""Phase 3: Create Combined IBM AML Dataset (HI-Small + LI-Medium).

Combines compatible preprocessed records from HI-Small and LI-Medium:
1. Adds 'dataset_source' identifier ('hi_small' vs 'li_medium') for provenance and subgroup auditing.
2. Ensures strict chronological sorting across combined splits.
3. Maintains identical feature schema (23 features, 3 identifiers, 1 target).
4. Generates complete metadata (schema.json, splits.json, transformers.json, preprocessing_report.json, feature_metadata.json).
5. Persists splits under data/processed/ibm_aml/Combined/.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
backend_path = BASE_DIR / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("create_combined")

PROCESSED_BASE = BASE_DIR / "data" / "processed" / "ibm_aml"
HI_DIR = PROCESSED_BASE / "HI-Small"
LI_DIR = PROCESSED_BASE / "LI-Medium"
COMBINED_DIR = PROCESSED_BASE / "Combined"


def build_combined_split(
    split_name: str,
    hi_csv: Path,
    li_csv: Path,
    out_csv: Path,
    hi_limit: int | None = None,
    li_limit: int | None = None,
) -> dict[str, Any]:
    """Combine, tag, and sort records for a split."""
    logger.info("Building combined %s split...", split_name)
    started = time.perf_counter()

    # Read HI-Small split
    logger.info("Reading HI-Small %s from %s (limit=%s)...", split_name, hi_csv.name, hi_limit)
    df_hi = pd.read_csv(hi_csv, nrows=hi_limit)
    df_hi["dataset_source"] = "hi_small"

    # Read LI-Medium split
    logger.info("Reading LI-Medium %s from %s (limit=%s)...", split_name, li_csv.name, li_limit)
    df_li = pd.read_csv(li_csv, nrows=li_limit)
    df_li["dataset_source"] = "li_medium"

    # Concatenate
    df_combined = pd.concat([df_hi, df_li], ignore_index=True)
    del df_hi, df_li

    # Sort strictly chronologically by timestamp
    logger.info("Sorting %s split chronologically (%d total rows)...", split_name, len(df_combined))
    df_combined.sort_values(by="timestamp", ascending=True, kind="mergesort", inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(out_csv, index=False)
    elapsed = time.perf_counter() - started

    positives = int((df_combined["is_laundering"] == 1).sum())
    negatives = int((df_combined["is_laundering"] == 0).sum())
    total = len(df_combined)
    prevalence = float(positives / total) if total > 0 else 0.0

    logger.info(
        "Saved %s: %d rows (%d pos, %.4f%% prev) in %.2fs",
        out_csv, total, positives, prevalence * 100, elapsed
    )

    return {
        "name": split_name,
        "rows": total,
        "positive": positives,
        "negative": negatives,
        "positive_rate": prevalence,
        "first_timestamp": str(df_combined["timestamp"].iloc[0]),
        "last_timestamp": str(df_combined["timestamp"].iloc[-1]),
    }


def main() -> None:
    started_total = time.perf_counter()
    logger.info("=================================================================")
    logger.info("Creating Combined IBM AML Dataset (HI-Small + LI-Medium)")
    logger.info("=================================================================")

    # Ensure output directories exist
    for sub in ["train", "validation", "test", "metadata"]:
        (COMBINED_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Copy metadata template from HI-Small (features and categorical vocabs are identical)
    for meta_file in ["schema.json", "transformers.json", "feature_metadata.json"]:
        src = HI_DIR / "metadata" / meta_file
        dst = COMBINED_DIR / "metadata" / meta_file
        shutil.copy2(src, dst)
        logger.info("Copied metadata template %s -> %s", src.name, dst)

    # In our controlled experimental design:
    # Train: 300,000 rows (150,000 HI-Small + 150,000 LI-Medium)
    # Validation: 100,000 rows (50,000 HI-Small + 50,000 LI-Medium)
    # Test: 100,000 rows (50,000 HI-Small + 50,000 LI-Medium)
    # Plus full HI-Small test (1,015,564) and LI-Medium test (6,249,112) for holdout benchmarks
    split_configs = [
        ("train", HI_DIR / "train" / "transactions.csv", LI_DIR / "train" / "transactions.csv", COMBINED_DIR / "train" / "transactions.csv", 150000, 150000),
        ("validation", HI_DIR / "validation" / "transactions.csv", LI_DIR / "validation" / "transactions.csv", COMBINED_DIR / "validation" / "transactions.csv", 50000, 50000),
        ("test", HI_DIR / "test" / "transactions.csv", LI_DIR / "test" / "transactions.csv", COMBINED_DIR / "test" / "transactions.csv", 50000, 50000),
    ]

    split_reports = []
    for s_name, hi_p, li_p, out_p, hi_lim, li_lim in split_configs:
        rep = build_combined_split(s_name, hi_p, li_p, out_p, hi_lim, li_lim)
        split_reports.append(rep)

    # Build Combined splits.json
    splits_meta = {
        "dataset_name": "IBM AML Combined (HI-Small + LI-Medium)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "Balanced Chronological Composite for Controlled Empirical Comparison",
        "per_split": split_reports,
        "target": {
            "source_column": "is_laundering",
            "mapping": {"0": 0, "1": 1},
            "class_counts": {
                "0": sum(s["negative"] for s in split_reports),
                "1": sum(s["positive"] for s in split_reports),
            },
        },
    }
    with open(COMBINED_DIR / "metadata" / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_meta, f, indent=2)
    logger.info("Saved %s", COMBINED_DIR / "metadata" / "splits.json")

    # Build preprocessing report
    prep_report = {
        "dataset_name": "IBM AML Combined",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.perf_counter() - started_total,
        "splits": split_reports,
        "notes": (
            "Combined dataset harmonizes HI-Small and LI-Medium under strict chronological alignment, "
            "preserving all 23 engineered features and tracking dataset_source ('hi_small' vs 'li_medium')."
        ),
    }
    with open(COMBINED_DIR / "metadata" / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(prep_report, f, indent=2)
    logger.info("Saved %s", COMBINED_DIR / "metadata" / "preprocessing_report.json")

    logger.info("Combined Dataset Generation Completed in %.2fs", time.perf_counter() - started_total)


if __name__ == "__main__":
    main()

