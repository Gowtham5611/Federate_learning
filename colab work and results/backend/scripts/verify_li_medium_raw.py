"""Verification and summary generator for raw LI-Medium_Trans.csv dataset."""

import csv
import sys
import time
from collections import Counter
from pathlib import Path
import pandas as pd

def analyze_raw_li_medium(
    raw_path: Path = Path("data/raw/ibm_aml/LI-Medium_Trans.csv"),
    chunksize: int = 500000,
) -> dict:
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} does not exist.")

    file_size = raw_path.stat().st_size
    print(f"Inspecting {raw_path} ({file_size:,} bytes)...")

    # Read header and first row
    first_chunk = pd.read_csv(raw_path, nrows=5)
    columns = list(first_chunk.columns)
    print(f"Columns ({len(columns)}): {columns}")

    target_col = None
    for c in ["Is Laundering", "is_laundering", "Is_Laundering"]:
        if c in columns:
            target_col = c
            break

    if target_col is None:
        raise ValueError(f"Target column not found in columns: {columns}")

    print(f"Target column detected: '{target_col}'")

    total_rows = 0
    label_counts = Counter()
    distinct_timestamps = set()
    first_ts = None
    last_ts = None

    t0 = time.perf_counter()
    chunk_idx = 0

    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
        chunk_idx += 1
        n = len(chunk)
        total_rows += n

        ts_series = chunk["Timestamp"]
        c_min, c_max = ts_series.min(), ts_series.max()
        first_ts = c_min if first_ts is None else min(first_ts, c_min)
        last_ts = c_max if last_ts is None else max(last_ts, c_max)

        counts = chunk[target_col].value_counts()
        for k, v in counts.items():
            label_counts[str(k)] += int(v)

        if chunk_idx % 5 == 0:
            print(f"Processed {total_rows:,} rows... Elapsed: {time.perf_counter() - t0:.1f}s")

    elapsed = time.perf_counter() - t0
    pos_count = label_counts.get("1", 0)
    neg_count = label_counts.get("0", 0)
    pos_rate = pos_count / total_rows if total_rows > 0 else 0.0

    print("\n" + "=" * 50)
    print("LI-MEDIUM RAW DATASET AUDIT COMPLETE")
    print("=" * 50)
    print(f"File Size:          {file_size:,} bytes")
    print(f"Total Rows:         {total_rows:,}")
    print(f"Legitimate (0):     {neg_count:,} ({neg_count / total_rows * 100:.4f}%)")
    print(f"Laundering (1):     {pos_count:,} ({pos_rate * 100:.4f}%)")
    print(f"Timestamp Span:     {first_ts} to {last_ts}")
    print(f"Time Taken:         {elapsed:.1f}s")
    print("=" * 50)

    # Update dataset_verification.md
    res = {
        "file_size": file_size,
        "total_rows": total_rows,
        "neg_count": neg_count,
        "pos_count": pos_count,
        "pos_rate": pos_rate,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "columns": columns,
    }
    return res

if __name__ == "__main__":
    analyze_raw_li_medium()

