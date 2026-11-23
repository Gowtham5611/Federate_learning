"""Phase E: Real-Data Validation, Sequence Context, and Model Performance Audit Tests.

Tests cover:
1. Account grouping (same account index chaining)
2. Chronological temporal ordering
3. No future transaction leakage (backward walks only)
4. Correct padding behavior (right-padded, masked)
5. Sequence-history diagnostics per row and dataset-level
6. Threshold freezing immutability
7. Model/data compatibility checking
8. Supervised evaluation metrics calculation
9. Insufficient-history warning behavior for unlinked/standalone transaction files
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService
from interfaces.api.schemas.common import FROZEN_THRESHOLDS, MAX_SEQUENCE_LENGTH
from infrastructure.data.sequences.account_sequence_dataset import AccountSequenceDataset, NO_PREVIOUS_ROW
from infrastructure.data.partitioners.client_sequence_provider import build_previous_row_index
from domain.enums.split_name import SplitName

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def eval_service():
    return CsvEvaluationService()


@pytest.fixture(scope="module")
def model_service():
    ms = ModelService()
    ms.load_all_models()
    return ms


# ------------------------------------------------------------------------------
# 1. Correct Account Grouping
# ------------------------------------------------------------------------------

def test_account_grouping_builds_correct_history_index():
    """Verify that build_previous_row_index links transactions for the same account."""
    accounts = np.array([101, 102, 101, 103, 102, 101], dtype=np.int64)
    prev_idx = build_previous_row_index(accounts)
    
    assert prev_idx[0] == NO_PREVIOUS_ROW  # 101 first appearance
    assert prev_idx[1] == NO_PREVIOUS_ROW  # 102 first appearance
    assert prev_idx[2] == 0                # 101 previous is row 0
    assert prev_idx[3] == NO_PREVIOUS_ROW  # 103 first appearance
    assert prev_idx[4] == 1                # 102 previous is row 1
    assert prev_idx[5] == 2                # 101 previous is row 2


# ------------------------------------------------------------------------------
# 2. Correct Temporal Ordering & 3. No Future Transaction Leakage
# ------------------------------------------------------------------------------

def test_no_future_transaction_leakage():
    """Verify that AccountSequenceDataset walk strictly moves backward in time."""
    n_rows = 10
    features = np.zeros((n_rows, 5), dtype=np.float32)
    # Put row index into feature column 0 to trace origin
    for i in range(n_rows):
        features[i, 0] = float(i)
        
    targets = np.zeros(n_rows, dtype=np.float32)
    # Single account spanning all 10 rows chronologically
    accounts = np.zeros(n_rows, dtype=np.int64)
    prev_idx = build_previous_row_index(accounts)
    
    ds = AccountSequenceDataset(
        split=SplitName.TEST,
        features=features,
        targets=targets,
        previous_row=prev_idx,
        sequence_length=5,
        stride=1,
    )
    
    batch = ds._assemble(ds.target_rows)
    for row_idx in range(n_rows):
        length = batch.lengths[row_idx]
        seq = batch.features[row_idx]
        # Real steps are from position 0 up to length-1
        real_steps = seq[:length, 0]
        # Every step in real_steps must be <= row_idx (strictly no future rows)
        assert np.all(real_steps <= row_idx), "Future transaction information detected in sequence!"
        # Target transaction is at position length - 1
        assert real_steps[-1] == float(row_idx), "Target transaction must be the final real step"
        # History is ordered chronologically ascending
        if len(real_steps) > 1:
            assert np.all(np.diff(real_steps) > 0), "History steps must be chronologically ascending"


# ------------------------------------------------------------------------------
# 4. Correct Padding Behavior
# ------------------------------------------------------------------------------

def test_correct_padding_behavior():
    """Verify sequences with length < MAX_SEQUENCE_LENGTH are right-padded with neutral zeros."""
    features = np.ones((3, 4), dtype=np.float32)
    targets = np.zeros(3, dtype=np.float32)
    prev_idx = np.array([-1, -1, -1], dtype=np.int64)  # 3 isolated accounts
    
    ds = AccountSequenceDataset(
        split=SplitName.TEST,
        features=features,
        targets=targets,
        previous_row=prev_idx,
        sequence_length=10,
        stride=1,
    )
    
    batch = ds._assemble(ds.target_rows)
    assert np.all(batch.lengths == 1)
    
    for i in range(3):
        # Step 0 is real
        assert np.all(batch.features[i, 0, :] == 1.0)
        # Steps 1..9 are zeros
        assert np.all(batch.features[i, 1:, :] == 0.0)


# ------------------------------------------------------------------------------
# 5. Sequence-History Diagnostics & 9. Insufficient-History Warning
# ------------------------------------------------------------------------------

def test_sequence_history_diagnostics_and_warning_on_unlinked_csv(eval_service):
    """Verify that a standalone CSV with unlinked accounts reports diagnostics and triggers warning."""
    # 5 isolated transactions with 5 unique accounts
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 10:00,001,ACC_1,002,ACC_91,500.0,USD,500.0,USD,Wire\n"
        "2022/09/01 10:01,001,ACC_2,002,ACC_92,750.0,USD,750.0,USD,Wire\n"
        "2022/09/01 10:02,001,ACC_3,002,ACC_93,1200.0,USD,1200.0,USD,Wire\n"
        "2022/09/01 10:03,001,ACC_4,002,ACC_94,300.0,USD,300.0,USD,Wire\n"
        "2022/09/01 10:04,001,ACC_5,002,ACC_95,950.0,USD,950.0,USD,Wire\n"
    )
    res = eval_service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="isolated.csv")
    
    assert res["status"] == "success"
    assert "sequence_context_diagnostics" in res
    diag = res["sequence_context_diagnostics"]
    
    assert diag["average_sequence_length"] == 1.0
    assert diag["single_transaction_ratio"] == 1.0
    assert diag["mean_padding_ratio"] == 0.90
    assert diag["insufficient_history_warning"] is True
    assert diag["operational_context_status"] == "INSUFFICIENT_HISTORY_TRUNCATED"
    assert "Sequence history is truncated" in diag["warning_message"]
    
    # Check itemized row diagnostics
    first_pred = res["predictions"][0]
    assert first_pred["sequence_length"] == 1
    assert first_pred["valid_history_count"] == 0
    assert first_pred["padding_ratio"] == 0.90
    assert first_pred["is_history_sufficient"] is False


# ------------------------------------------------------------------------------
# 6. Threshold Freezing Immutability
# ------------------------------------------------------------------------------

def test_frozen_threshold_immutability(eval_service):
    """Verify that frozen thresholds for HI-Small and LI-Medium remain strictly constant."""
    assert FROZEN_THRESHOLDS["HI-Small"] == pytest.approx(0.33778378, abs=1e-5)
    assert FROZEN_THRESHOLDS["LI-Medium"] == pytest.approx(0.23848299, abs=1e-5)
    
    sample_hi = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    res_hi = eval_service.evaluate_csv_bytes(sample_hi, dataset="HI-Small")
    assert res_hi["decision_threshold"] == pytest.approx(0.33778378, abs=1e-5)
    
    sample_li = (BASE_DIR / "data/demo_samples/li_medium_sample_25.csv").read_bytes()
    res_li = eval_service.evaluate_csv_bytes(sample_li, dataset="LI-Medium")
    assert res_li["decision_threshold"] == pytest.approx(0.23848299, abs=1e-5)


# ------------------------------------------------------------------------------
# 7. Model/Data Compatibility
# ------------------------------------------------------------------------------

def test_model_data_compatibility_detection(eval_service):
    """Verify schema detection and compatibility reporting."""
    sample_hi = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_bytes()
    compat = eval_service.check_compatibility(sample_hi, filename="hi.csv")
    assert compat["status"] == "success"
    assert "results" in compat
    assert len(compat["results"]) >= 2
    assert any(r["status"] == "COMPATIBLE" for r in compat["results"])


# ------------------------------------------------------------------------------
# 8. Correct Supervised Evaluation Metrics
# ------------------------------------------------------------------------------

def test_supervised_evaluation_metrics_formulae(eval_service):
    """Verify metric calculations match AML standard definitions."""
    labels = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.01, 0.02])
    threshold = 0.5
    
    metrics = eval_service._calculate_metrics(labels, probs, threshold)
    # Positives predicted: idx 0, 1, 2 (probs 0.9, 0.8, 0.7 >= 0.5)
    # TP: idx 0 (Y=1), idx 1 (Y=1) -> TP = 2
    # FP: idx 2 (Y=0) -> FP = 1
    # FN: idx 4 (Y=1, prob 0.1 < 0.5) -> FN = 1
    # TN: idx 3, 5, 6, 7 (Y=0, probs < 0.5) -> TN = 4
    
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tn"] == 4
    
    assert metrics["accuracy"] == pytest.approx(6.0 / 8.0)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["f1_score"] == pytest.approx(2.0 / 3.0)
    assert metrics["f2_score"] == pytest.approx(2.0 / 3.0)
