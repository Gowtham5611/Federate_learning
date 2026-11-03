"""Export contiguous slices of the held-out test split for offline testing and validation.

Generates:
- data/demo_slices/test_slice_500.csv (500 rows)
- data/demo_slices/test_slice_5000.csv (5,000 rows)
- data/demo_slices/test_slice_50000.csv (50,000 rows)

Includes ground-truth labels and real temporal sequence context.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ibm_aml" / "HI-Small" / "test" / "transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "demo_slices"


def export_slices():
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Test split not found at {TEST_CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading test split from {TEST_CSV_PATH}...")

    # Read maximum needed rows
    max_needed = 50000
    df = pd.read_csv(TEST_CSV_PATH, nrows=max_needed)
    print(f"Successfully loaded {len(df)} test rows. Slicing...")

    # Export 500 rows
    slice_500 = df.iloc[:500]
    out_500 = OUTPUT_DIR / "test_slice_500.csv"
    slice_500.to_csv(out_500, index=False)
    print(f"Exported {out_500} ({len(slice_500)} rows, {slice_500['is_laundering'].sum()} positives)")

    # Export 5,000 rows
    slice_5000 = df.iloc[:5000]
    out_5000 = OUTPUT_DIR / "test_slice_5000.csv"
    slice_5000.to_csv(out_5000, index=False)
    print(f"Exported {out_5000} ({len(slice_5000)} rows, {slice_5000['is_laundering'].sum()} positives)")

    # Export 50,000 rows
    slice_50000 = df.iloc[:50000]
    out_50000 = OUTPUT_DIR / "test_slice_50000.csv"
    slice_50000.to_csv(out_50000, index=False)
    print(f"Exported {out_50000} ({len(slice_50000)} rows, {slice_50000['is_laundering'].sum()} positives)")


if __name__ == "__main__":
    export_slices()

