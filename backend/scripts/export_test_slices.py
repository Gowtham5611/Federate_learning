"""Export contiguous slices of the held-out test split for offline testing and validation.

Generates:
- data/demo_slices/test_slice_500.csv (500 rows)
- data/demo_slices/test_slice_500_with_context.csv (5,000 warm-up + 500 eval rows)
- data/demo_slices/test_slice_5000.csv (5,000 rows)
- data/demo_slices/test_slice_5000_with_context.csv (25,000 warm-up + 5,000 eval rows)
- data/demo_slices/test_slice_50000.csv (50,000 rows)
- data/demo_slices/test_slice_50000_with_context.csv (100,000 warm-up + 50,000 eval rows)

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

    # Load up to 150,000 rows to support 100k warm-up context for the 50k slice
    max_needed = 150000
    df = pd.read_csv(TEST_CSV_PATH, nrows=max_needed)
    print(f"Successfully loaded {len(df)} test rows. Slicing...")

    # 1. 500 rows slice (slice-only)
    slice_500 = df.iloc[:500].copy()
    out_500 = OUTPUT_DIR / "test_slice_500.csv"
    slice_500.to_csv(out_500, index=False)
    print(f"Exported {out_500.name} ({len(slice_500)} rows, {slice_500['is_laundering'].sum()} positives)")

    # 1b. 500 rows slice with 5,000 preceding rows as warm-up context
    ctx_500 = df.iloc[:5500].copy()
    ctx_500['Is_Context'] = 1
    ctx_500.loc[5000:, 'Is_Context'] = 0
    out_500_ctx = OUTPUT_DIR / "test_slice_500_with_context.csv"
    ctx_500.to_csv(out_500_ctx, index=False)
    print(f"Exported {out_500_ctx.name} ({len(ctx_500)} rows: 5,000 context, 500 eval)")

    # 2. 5,000 rows slice (slice-only)
    slice_5000 = df.iloc[:5000].copy()
    out_5000 = OUTPUT_DIR / "test_slice_5000.csv"
    slice_5000.to_csv(out_5000, index=False)
    print(f"Exported {out_5000.name} ({len(slice_5000)} rows, {slice_5000['is_laundering'].sum()} positives)")

    # 2b. 5,000 rows slice with 25,000 preceding rows as warm-up context
    ctx_5000 = df.iloc[:30000].copy()
    ctx_5000['Is_Context'] = 1
    ctx_5000.loc[25000:, 'Is_Context'] = 0
    out_5000_ctx = OUTPUT_DIR / "test_slice_5000_with_context.csv"
    ctx_5000.to_csv(out_5000_ctx, index=False)
    print(f"Exported {out_5000_ctx.name} ({len(ctx_5000)} rows: 25,000 context, 5,000 eval)")

    # 3. 50,000 rows slice (slice-only)
    slice_50000 = df.iloc[:50000].copy()
    out_50000 = OUTPUT_DIR / "test_slice_50000.csv"
    slice_50000.to_csv(out_50000, index=False)
    print(f"Exported {out_50000.name} ({len(slice_50000)} rows, {slice_50000['is_laundering'].sum()} positives)")

    # 3b. 50,000 rows slice with 100,000 preceding rows as warm-up context
    ctx_50000 = df.iloc[:150000].copy()
    ctx_50000['Is_Context'] = 1
    ctx_50000.loc[100000:, 'Is_Context'] = 0
    out_50000_ctx = OUTPUT_DIR / "test_slice_50000_with_context.csv"
    ctx_50000.to_csv(out_50000_ctx, index=False)
    print(f"Exported {out_50000_ctx.name} ({len(ctx_50000)} rows: 100,000 context, 50,000 eval)")


if __name__ == "__main__":
    export_slices()
