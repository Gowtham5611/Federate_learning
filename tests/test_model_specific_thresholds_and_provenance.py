"""Regression test suite for Model-Specific Threshold Selection,
Cryptographic Hash Provenance, and Zero-Leakage History Feature Engineering.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.config.threshold_config import get_canonical_threshold
from backend.infrastructure.data.readers.universal_csv_reader import UniversalCsvReader
from backend.infrastructure.evaluation.pdf_report_generator import PdfReportGenerator
from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from evaluation.metrics import (
    compute_metrics_at_capacity,
    compute_metrics_at_threshold,
)


def test_model_specific_threshold_lookup():
    """Verify each model/checkpoint has its own canonical threshold calibrated on validation data."""
    # Sequence Models (HI-Small LSTM & LI-Medium BiLSTM)
    hi_small_thr = get_canonical_threshold("HI-Small")
    assert abs(hi_small_thr - 0.337784) < 1e-6, "HI-Small sequence LSTM must use 0.337784"

    li_medium_thr = get_canonical_threshold("LI-Medium")
    assert abs(li_medium_thr - 0.238483) < 1e-6, "LI-Medium sequence BiLSTM must use 0.238483"

    # Tabular LightGBM Fallback Model
    lightgbm_thr = get_canonical_threshold("lightgbm_tabular_fallback")
    assert abs(lightgbm_thr - 0.801919) < 1e-6, "LightGBM tabular fallback must use 0.801919 (val-optimal F1)"

    tabular_thr = get_canonical_threshold("tabular_fallback")
    assert abs(tabular_thr - 0.801919) < 1e-6, "tabular_fallback alias must resolve to 0.801919"


def test_tabular_fallback_evaluates_with_model_specific_default():
    """Verify tabular fallback automatically uses 0.801919 when no threshold is passed."""
    service = CsvEvaluationService()
    raw_csv = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/01/01 10:00,1,A1,2,B1,100,US Dollar,100,US Dollar,ACH,0\n"
        "2022/01/01 11:00,1,A2,2,B2,200,US Dollar,200,US Dollar,Wire,1\n"
    )
    b = raw_csv.encode("utf-8")
    res = service.evaluate_csv_bytes(b, filename="test.csv", fallback_model="tabular")

    assert res["model_id"] == "lightgbm_tabular_fallback"
    # When threshold=None, decision_threshold must be LightGBM's 0.801919, NOT LSTM's 0.337784!
    assert abs(res["decision_threshold"] - 0.801919) < 1e-4, f"Expected 0.801919, got {res['decision_threshold']}"


def test_hash_provenance_validation_and_rejection():
    """Verify SHA-256 validation strictly accepts 64-hex and rejects malformed/truncated hashes."""
    # 1. Valid SHA-256
    valid_bytes = b"sample_transaction_data_for_testing"
    valid_hash = hashlib.sha256(valid_bytes).hexdigest().lower()
    assert len(valid_hash) == 64
    assert bool(re.match(r"^[0-9a-f]{64}$", valid_hash))

    # 2. Rejection of truncated hash (e.g. 40 hex chars)
    truncated_hash = valid_hash[:40]
    with pytest.raises(ValueError, match="Invalid.*SHA-256 digest"):
        PdfReportGenerator.generate_pdf_bytes(
            meta={"file_hash": truncated_hash, "run_id": "test_run"},
            predictions_df=pd.DataFrame({"x": [1]}),
        )

    # 3. Rejection of corrupted string with non-hex characters (e.g. base64 token spliced onto prefix)
    corrupted_hash = valid_hash[:40] + "BSN7gxmPWYCAKJgA9aH4yNuERpTvs4uiX"
    with pytest.raises(ValueError, match="Invalid.*SHA-256 digest"):
        PdfReportGenerator.generate_pdf_bytes(
            meta={"file_hash": corrupted_hash, "run_id": "test_run"},
            predictions_df=pd.DataFrame({"x": [1]}),
        )

    # 4. Rejection of malformed checkpoint hash
    with pytest.raises(ValueError, match="Invalid checkpoint SHA-256 digest"):
        PdfReportGenerator.generate_pdf_bytes(
            meta={"file_hash": valid_hash, "model_checkpoint_sha256": "invalid_checkpoint_hash", "run_id": "test_run"},
            predictions_df=pd.DataFrame({"x": [1]}),
        )


def test_account_history_construction_zero_future_leakage():
    """Verify account historical velocity features evaluate strictly chronologically without future leakage."""
    from backend.infrastructure.data.features.account_history import AccountHistoryBuilder

    # Create sequence of 3 transactions for same sender account A1
    timestamps = [
        pd.Timestamp("2022-01-01 10:00:00"),
        pd.Timestamp("2022-01-01 11:00:00"),
        pd.Timestamp("2022-01-01 12:00:00"),
    ]
    amounts = [100.0, 500.0, 2000.0]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "From Bank": [1, 1, 1],
        "Account": ["ACC_A1", "ACC_A1", "ACC_A1"],
        "To Bank": [2, 2, 2],
        "Account.1": ["ACC_B1", "ACC_B2", "ACC_B3"],
        "Amount Paid": amounts,
        "Amount Received": amounts,
        "Payment Format": ["ACH", "ACH", "ACH"],
        "Payment Currency": ["US Dollar", "US Dollar", "US Dollar"],
        "Receiving Currency": ["US Dollar", "US Dollar", "US Dollar"],
    })

    df["_epoch_seconds"] = (df["Timestamp"].astype("int64") // 10**9).astype(float)

    builder = AccountHistoryBuilder()
    features = builder.apply(df)

    # Tx 0: Sender has NO prior transactions (cold-start for Tx 0)
    assert features["sender_has_prior"].iloc[0] == 0
    assert features["sender_prior_count_log"].iloc[0] == 0.0
    assert features["sender_prior_amount_log"].iloc[0] == 0.0

    # Tx 1: Sender has 1 prior transaction (Amount = 100.0)
    assert features["sender_has_prior"].iloc[1] == 1
    assert features["sender_prior_count_log"].iloc[1] > 0.0
    # Tx 1 must NOT know about Tx 2 ($2000.0)
    assert abs(features["sender_prior_amount_log"].iloc[1] - np.log1p(100.0)) < 1e-4

    # Tx 2: Sender has 2 prior transactions (Cumulative Amount = 100 + 500 = 600.0)
    assert features["sender_has_prior"].iloc[2] == 1
    assert abs(features["sender_prior_amount_log"].iloc[2] - np.log1p(600.0)) < 1e-4


def test_cross_surface_manifest_consistency():
    """Verify API metadata, saved _meta.json, downloaded CSV, and PDF report all match."""
    service = CsvEvaluationService()
    raw_csv = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/01/01 10:00,1,ACC_S1,2,ACC_R1,100,US Dollar,100,US Dollar,ACH,0\n"
        "2022/01/01 11:00,1,ACC_S2,2,ACC_R2,200,US Dollar,200,US Dollar,Wire,1\n"
    )
    b = raw_csv.encode("utf-8")
    res = service.evaluate_csv_bytes(b, filename="provenance_test.csv", fallback_model="tabular")

    run_id = res["run_id"]
    file_hash = res["file_hash"]
    ckpt_sha = res["model_checkpoint_sha256"]

    # 1. API Response has full 64-hex hash
    assert len(file_hash) == 64
    assert bool(re.match(r"^[0-9a-f]{64}$", file_hash))

    # 2. _meta.json matches
    meta_path = Path(f"results/uploads/{run_id}_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["file_hash"] == file_hash
    assert meta["run_id"] == run_id
    assert meta["model_checkpoint_sha256"] == ckpt_sha

    # 3. Downloaded CSV matches
    csv_path = Path(f"results/uploads/{run_id}.csv")
    assert csv_path.exists()
    pred_df = pd.read_csv(csv_path)
    assert pred_df["eval_file_hash"].iloc[0] == file_hash
    assert pred_df["eval_run_id"].iloc[0] == run_id
    assert pred_df["eval_model_checkpoint_sha256"].iloc[0] == ckpt_sha

    # 4. PDF report matches
    pdf_result = service.get_prediction_pdf_report(run_id)
    pdf_bytes = pdf_result[0] if isinstance(pdf_result, tuple) else pdf_result
    assert len(pdf_bytes) > 0
    # In uncompressed PDF, Courier split hash halves are present
    assert file_hash[:32].encode("latin1") in pdf_bytes
    assert file_hash[32:].encode("latin1") in pdf_bytes
