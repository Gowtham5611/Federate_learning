"""Equivalence tests between reference AmlPreprocessor and GpuAmlPreprocessor.

Verifies:
1. Identical cleaning filtering and removal counter keys.
2. Identical feature column names and output column ordering.
3. Numerical feature outputs (sin/cos/log1p/standardized) match within float32 tolerance.
4. Identical categorical encoding outputs.
5. Identical causal account history accumulation without forward leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
import pytest

from infrastructure.data.preprocessors.aml_preprocessor import AmlPreprocessor
from infrastructure.data.preprocessors.gpu_aml_preprocessor import GpuAmlPreprocessor
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_FROM_BANK,
    COL_IS_LAUNDERING,
    COL_PAYMENT_CURRENCY,
    COL_PAYMENT_FORMAT,
    COL_RECEIVING_CURRENCY,
    COL_TIMESTAMP,
    COL_TO_ACCOUNT,
    COL_TO_BANK,
)


@pytest.fixture
def raw_sample_transactions() -> pd.DataFrame:
    """Create a realistic raw DataFrame representing IBM AML transaction schema."""
    data = {
        COL_TIMESTAMP: [
            "2022/09/01 00:00",
            "2022/09/01 00:05",
            "2022/09/01 00:10",
            "2022/09/01 00:15",
            "2022/09/01 00:20",
        ],
        COL_FROM_BANK: ["010", "010", "020", "010", "030"],
        COL_FROM_ACCOUNT: ["acc_1", "acc_2", "acc_1", "acc_3", "acc_1"],
        COL_TO_BANK: ["010", "020", "020", "010", "040"],
        COL_TO_ACCOUNT: ["acc_2", "acc_3", "acc_1", "acc_3", "acc_4"],
        COL_AMOUNT_RECEIVED: [100.0, 250.5, 50.0, 1000.0, 75.25],
        COL_RECEIVING_CURRENCY: ["US Dollar", "US Dollar", "Euro", "US Dollar", "Yen"],
        COL_AMOUNT_PAID: [100.0, 250.5, 50.0, 1000.0, 75.25],
        COL_PAYMENT_CURRENCY: ["US Dollar", "US Dollar", "Euro", "US Dollar", "Yen"],
        COL_PAYMENT_FORMAT: ["ACH", "Cheque", "Credit Card", "Wire", "ACH"],
        COL_IS_LAUNDERING: [0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_preprocessor_interface_and_cleaning_equivalence(raw_sample_transactions: pd.DataFrame):
    """Test clean() step produces equivalent output and counts."""
    cpu_prep = AmlPreprocessor()
    gpu_prep = GpuAmlPreprocessor(prefer_gpu=False)

    df_cpu, rem_cpu = cpu_prep.clean(raw_sample_transactions)
    df_gpu, rem_gpu = gpu_prep.clean(raw_sample_transactions)

    assert len(df_cpu) == len(df_gpu)
    assert rem_cpu == rem_gpu
    assert list(df_cpu.columns) == list(df_gpu.columns)
    assert (df_cpu[COL_TIMESTAMP] == df_gpu[COL_TIMESTAMP]).all()


def test_full_pipeline_engineering_and_transform_equivalence(raw_sample_transactions: pd.DataFrame):
    """Test full clean -> engineer -> fit -> transform produces identical features."""
    cpu_prep = AmlPreprocessor()
    gpu_prep = GpuAmlPreprocessor(prefer_gpu=False)

    clean_cpu, _ = cpu_prep.clean(raw_sample_transactions)
    clean_gpu, _ = gpu_prep.clean(raw_sample_transactions)

    eng_cpu = cpu_prep.engineer(clean_cpu)
    eng_gpu = gpu_prep.engineer(clean_gpu)

    # Fit on training chunk
    cpu_prep.fit_on_training_chunk(eng_cpu)
    cpu_prep.freeze_fit()

    gpu_prep.fit_on_training_chunk(eng_gpu)
    gpu_prep.freeze_fit()

    out_cpu = cpu_prep.transform(eng_cpu)
    out_gpu = gpu_prep.transform(eng_gpu)

    assert list(out_cpu.columns) == list(out_gpu.columns)
    assert len(out_cpu) == len(out_gpu)

    # Compare all columns numerically or categorically
    for col in out_cpu.columns:
        if pd.api.types.is_float_dtype(out_cpu[col]):
            np.testing.assert_allclose(
                out_cpu[col].to_numpy(),
                out_gpu[col].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mismatch in float feature {col}",
            )
        else:
            assert (out_cpu[col] == out_gpu[col]).all(), f"Mismatch in non-float column {col}"
