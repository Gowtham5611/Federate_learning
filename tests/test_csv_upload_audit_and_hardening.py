"""Focused Regression Suite for CSV Upload Audit and Hardening.

Covers:
1. Duplicate 'Account' headers handling before dataframe parsing.
2. Label and probability orientation (positive class = 1, [0, 1] bounded).
3. Row alignment between input CSV rows and output prediction items.
4. Cold-start behavior and velocity feature defaults.
5. Threshold metrics and operating points (ranking invariance).
6. Run and cache identity (file_hash, run_id, no stale leaks).
"""

import hashlib
import io
import json
import pytest
import numpy as np
import pandas as pd

from backend.infrastructure.data.readers.universal_csv_reader import UniversalCsvReader
from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService


@pytest.fixture
def csv_reader():
    return UniversalCsvReader()


@pytest.fixture
def eval_service():
    return CsvEvaluationService()


def test_duplicate_account_headers_sanitization(csv_reader):
    """Verify duplicate 'Account' headers are disambiguated stably before dataframe parsing."""
    raw_csv = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Paid,Is Laundering\n"
        "2022/01/01 12:00,1,ACC_SENDER,2,ACC_RECEIVER,1500.0,0\n"
        "2022/01/01 12:05,1,ACC_SENDER2,2,ACC_RECEIVER2,250000.0,1\n"
    )

    sanitized_text, had_dups, cols = csv_reader.sanitize_duplicate_headers(raw_csv)
    assert had_dups is True
    assert cols == ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Is Laundering"]

    df, meta = csv_reader.read_dataframe(raw_csv.encode("utf-8"))
    assert list(df.columns) == ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Is Laundering"]
    assert df["Account"].iloc[0] == "ACC_SENDER"
    assert df["Account.1"].iloc[0] == "ACC_RECEIVER"
    assert any("duplicate" in w.lower() for w in meta.warnings)


def test_label_and_probability_orientation(eval_service):
    """Verify label 1 is positive laundering, and probabilities are properly oriented in [0, 1]."""
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    probs = np.array([0.01, 0.05, 0.20, 0.70, 0.95], dtype=float)

    metrics = eval_service._calculate_metrics(labels, probs, threshold=0.5)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_row_alignment_preservation(eval_service):
    """Verify prediction items match input CSV row count, transaction IDs, and chronological ordering."""
    rows = []
    for i in range(10):
        rows.append(
            f"2022/01/01 {i:02d}:00,1,SENDER_{i:03d},2,RECEIVER_{i:03d},{100.0 + i * 50},US Dollar,{100.0 + i * 50},US Dollar,ACH,0"
        )
    csv_text = "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n" + "\n".join(rows)

    res = eval_service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="alignment_test.csv")
    preds = res["predictions"]
    assert len(preds) == 10
    for idx, item in enumerate(preds):
        assert item["row_id"] == idx + 1
        assert item["from_account"] == f"SENDER_{idx:03d}"
        assert item["to_account"] == f"RECEIVER_{idx:03d}"
        assert abs(item["amount"] - (100.0 + idx * 50)) < 1e-4


def test_cold_start_velocity_defaults_and_domain_shift(eval_service):
    """Verify zero-history accounts trigger cold-start warnings and domain shift diagnosis."""
    # 5 single-transaction rows (unique_sender_ratio = 1.0)
    rows = [
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering",
        "2022/01/01 10:00,1,ACC_A1,2,ACC_B1,500.0,US Dollar,500.0,US Dollar,ACH,0",
        "2022/01/01 11:00,1,ACC_A2,2,ACC_B2,1200.0,US Dollar,1200.0,US Dollar,Wire,1",
        "2022/01/01 12:00,1,ACC_A3,2,ACC_B3,350.0,US Dollar,350.0,US Dollar,ACH,0",
        "2022/01/01 13:00,1,ACC_A4,2,ACC_B4,250000.0,US Dollar,250000.0,US Dollar,Cash,1",
        "2022/01/01 14:00,1,ACC_A5,2,ACC_B5,800.0,US Dollar,800.0,US Dollar,Credit Card,0",
    ]
    csv_bytes = "\n".join(rows).encode("utf-8")

    res = eval_service.evaluate_csv_bytes(csv_bytes, filename="cold_start_test.csv", fallback_model="tabular")
    assert res["model_id"] == "lightgbm_tabular_fallback"
    assert res["is_fallback"] is True
    assert res["validity_banner"]["status"] == "INVALID"
    assert "NO_ACCOUNT_HISTORY" in res["validity_banner"]["flags"]
    assert "PREVALENCE_MISMATCH" in res["validity_banner"]["flags"]
    assert res["domain_shift_report"]["cold_start_active"] is True
    assert res["domain_shift_report"]["velocity_features_available"] is False


def test_threshold_metrics_and_ranking_invariance(eval_service):
    """Verify threshold tuning slides along classification boundary without altering ranking (PR-AUC / ROC-AUC)."""
    labels = np.array([0] * 70 + [1] * 30, dtype=int)
    np.random.seed(42)
    # Generate scores with moderate discrimination
    probs = np.clip(np.where(labels == 1, np.random.beta(2, 5, 100), np.random.beta(1, 10, 100)), 0.0, 1.0)

    m_low = eval_service._calculate_metrics(labels, probs, threshold=0.1)
    m_high = eval_service._calculate_metrics(labels, probs, threshold=0.5)

    # Ranking metrics are mathematically invariant to threshold
    assert abs(m_low["pr_auc"] - m_high["pr_auc"]) < 1e-6
    assert abs(m_low["roc_auc"] - m_high["roc_auc"]) < 1e-6

    # Classification trade-off: lower threshold increases recall, reduces precision
    assert m_low["recall"] >= m_high["recall"]
    if m_high["tp"] + m_high["fp"] > 0 and m_low["tp"] + m_low["fp"] > 0:
        assert m_low["precision"] <= m_high["precision"]


def test_provenance_and_run_cache_identity(eval_service):
    """Verify each evaluation returns cryptographic file_hash and unique run_id with no stale leaks."""
    csv1 = "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n2022/01/01 10:00,1,A1,2,B1,100,US Dollar,100,US Dollar,ACH,0\n"
    csv2 = "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n2022/01/01 10:00,1,A2,2,B2,200,US Dollar,200,US Dollar,ACH,0\n"

    b1 = csv1.encode("utf-8")
    b2 = csv2.encode("utf-8")

    res1 = eval_service.evaluate_csv_bytes(b1, filename="file1.csv", fallback_model="tabular")
    res2 = eval_service.evaluate_csv_bytes(b2, filename="file2.csv", fallback_model="tabular")

    assert res1["file_hash"] == hashlib.sha256(b1).hexdigest()
    assert res2["file_hash"] == hashlib.sha256(b2).hexdigest()
    assert res1["file_hash"] != res2["file_hash"]
    assert res1["run_id"] != res2["run_id"]
    assert res1["predictions"][0]["from_account"] == "A1"
    assert res2["predictions"][0]["from_account"] == "A2"

