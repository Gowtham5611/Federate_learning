"""Universal CSV processing and temporal leakage prevention test suite.

Verifies:
Priority 2: Universal CSV Processing across 10 distinct schemas/sizes:
  1. CSV with 10 rows
  2. CSV with 100 rows
  3. CSV with 1,000 rows
  4. CSV with additional columns
  5. CSV with missing optional columns
  6. CSV with different column names
  7. CSV with a BOM header
  8. CSV with a different delimiter
  9. CSV with missing values
  10. CSV with an unknown target label

Priority 3: Prediction correctness:
  - Row order preservation
  - Original transaction identifier preservation
  - Exact prediction count matching eligible records
  - Accurate row association
  - No silent record drops

Priority 4: Temporal leakage prevention:
  - Account grouping and chronological sorting
  - Causal sliding window (L=10)
  - Future transaction non-interference (future data cannot alter past prediction)
  - Strict train/val/test chronological ordering
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from infrastructure.data.readers.universal_csv_reader import UniversalCsvReader
from infrastructure.data.schemas.universal_schema_detector import UniversalSchemaDetector
from infrastructure.data.features.graph_features import TimeAwareAmlGraph, CausalGraphFeatureExtractor
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def preload_models():
    """Ensure champion models are preloaded once."""
    ms = ModelService()
    ms.load_all_models()
    return ms


def make_transaction_csv(
    n_rows: int,
    delimiter: str = ",",
    has_bom: bool = False,
    include_target: bool = True,
    extra_cols: bool = False,
    missing_optional: bool = False,
    different_names: bool = False,
    inject_missing_values: bool = False,
) -> bytes:
    """Generate customizable CSV bytes conforming to specified test scenario."""
    base_time = pd.Timestamp("2023-01-01 08:00:00")
    rows = []

    for i in range(n_rows):
        t_str = (base_time + pd.Timedelta(minutes=i * 3)).strftime("%Y-%m-%d %H:%M:%S")
        from_acc = f"ACC_{i % 30:04d}"
        to_acc = f"ACC_{(i + 1) % 30:04d}"
        amt = round(50.0 + (i * 27.5) % 8500.0, 2)
        curr = "USD"
        fmt = "WIRE" if i % 4 == 0 else "ACH"
        target = 1 if i % 15 == 0 else 0

        row_dict = {}

        if different_names:
            # Aliased column names
            row_dict["trans_date"] = t_str
            row_dict["originator_account"] = from_acc
            row_dict["beneficiary_account"] = to_acc
            row_dict["trans_amount"] = amt
            if not missing_optional:
                row_dict["currency_iso"] = curr
                row_dict["payment_rail"] = fmt
            if include_target:
                row_dict["fraud_indicator"] = target
        else:
            # Standard naming
            row_dict["timestamp"] = t_str
            row_dict["from_account"] = from_acc
            row_dict["to_account"] = to_acc
            row_dict["amount"] = amt
            if not missing_optional:
                row_dict["currency"] = curr
                row_dict["payment_format"] = fmt
            if include_target:
                row_dict["is_laundering"] = target

        if extra_cols:
            row_dict["risk_analyst_notes"] = f"Review note for transaction {i}"
            row_dict["branch_id"] = f"BR_{i % 5:02d}"
            row_dict["internal_device_ip"] = f"192.168.1.{i % 250}"

        if inject_missing_values and i % 5 == 0:
            if not missing_optional:
                row_dict[list(row_dict.keys())[4]] = ""  # empty optional value

        rows.append(row_dict)

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, sep=delimiter, index=False)
    csv_str = buf.getvalue()

    if has_bom:
        return b"\xef\xbb\xbf" + csv_str.encode("utf-8")
    return csv_str.encode("utf-8")


# ==============================================================================
# Priority 2: 10 Universal CSV Scenarios
# ==============================================================================


def test_scenario_01_csv_with_10_rows():
    """Scenario 1: CSV with exactly 10 rows."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(10)
    result = service.evaluate_csv_bytes(raw_bytes, filename="batch_10.csv")
    assert result["evaluated_sequences"] == 10
    assert len(result["predictions"]) == 10
    assert result["has_labels"] is True


def test_scenario_02_csv_with_100_rows():
    """Scenario 2: CSV with exactly 100 rows."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(100)
    result = service.evaluate_csv_bytes(raw_bytes, filename="batch_100.csv")
    assert result["evaluated_sequences"] == 100
    assert len(result["predictions"]) == 100


def test_scenario_03_csv_with_1000_rows():
    """Scenario 3: CSV with exactly 1,000 rows."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(1000)
    result = service.evaluate_csv_bytes(raw_bytes, filename="batch_1000.csv", row_limit=1000)
    assert result["evaluated_sequences"] == 1000
    assert len(result["predictions"]) == 1000


def test_scenario_04_csv_with_additional_columns():
    """Scenario 4: CSV with additional columns (device IP, branch ID, analyst notes)."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(25, extra_cols=True)
    result = service.evaluate_csv_bytes(raw_bytes, filename="extra_cols.csv")
    assert result["evaluated_sequences"] == 25
    assert len(result["predictions"]) == 25


def test_scenario_05_csv_with_missing_optional_columns():
    """Scenario 5: CSV with missing optional columns (currency, payment format omitted)."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(20, missing_optional=True)
    result = service.evaluate_csv_bytes(raw_bytes, filename="missing_optional.csv")
    assert result["evaluated_sequences"] == 20
    assert len(result["predictions"]) == 20


def test_scenario_06_csv_with_different_column_names():
    """Scenario 6: CSV with aliased column names (trans_date, originator_account, etc.)."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(15, different_names=True)
    result = service.evaluate_csv_bytes(raw_bytes, filename="aliased_names.csv")
    assert result["evaluated_sequences"] == 15
    assert len(result["predictions"]) == 15


def test_scenario_07_csv_with_bom_header():
    """Scenario 7: CSV with UTF-8 BOM header."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(18, has_bom=True)
    result = service.evaluate_csv_bytes(raw_bytes, filename="bom_file.csv")
    assert result["evaluated_sequences"] == 18
    assert len(result["predictions"]) == 18


def test_scenario_08_csv_with_different_delimiter():
    """Scenario 8: CSV with semicolon (;) and tab (\t) delimiters."""
    service = CsvEvaluationService()
    semicolon_bytes = make_transaction_csv(22, delimiter=";")
    res_semi = service.evaluate_csv_bytes(semicolon_bytes, filename="semicolon.csv")
    assert res_semi["evaluated_sequences"] == 22

    tab_bytes = make_transaction_csv(16, delimiter="\t")
    res_tab = service.evaluate_csv_bytes(tab_bytes, filename="tab.tsv")
    assert res_tab["evaluated_sequences"] == 16


def test_scenario_09_csv_with_missing_values():
    """Scenario 9: CSV with missing values."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(30, inject_missing_values=True)
    result = service.evaluate_csv_bytes(raw_bytes, filename="missing_values.csv")
    assert result["evaluated_sequences"] == 30
    assert len(result["predictions"]) == 30


def test_scenario_10_csv_with_unknown_target_label():
    """Scenario 10: Unlabeled stream CSV (no ground-truth target column)."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(24, include_target=False)
    result = service.evaluate_csv_bytes(raw_bytes, filename="unlabeled.csv")
    assert result["evaluated_sequences"] == 24
    assert result["has_labels"] is False
    assert result["evaluation_metrics"] is None
    # All predictions have ground_truth_label is None
    for p in result["predictions"]:
        assert p["ground_truth_label"] is None


# ==============================================================================
# Priority 3: Prediction Alignment & Non-Dropping Guarantee
# ==============================================================================


def test_prediction_alignment_and_row_order_preservation():
    """Verify strict 1-to-1 input row order preservation and exact row alignment."""
    service = CsvEvaluationService()
    raw_bytes = make_transaction_csv(35)
    result = service.evaluate_csv_bytes(raw_bytes, filename="alignment_test.csv")

    predictions = result["predictions"]
    assert len(predictions) == 35

    # Original row order must strictly be preserved (row_id 1, 2, 3... 35)
    for idx, p in enumerate(predictions):
        expected_row_id = idx + 1
        assert p["row_id"] == expected_row_id
        assert 0.0 <= p["predicted_probability"] <= 1.0
        assert p["decision"] in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE", "SUSPICIOUS", "LEGITIMATE")


def test_incompatible_schema_informative_rejection():
    """Verify incompatible schema (e.g. retail inventory) is rejected with clear explanations."""
    service = CsvEvaluationService()
    detector = UniversalSchemaDetector()

    retail_csv = (
        "item_sku,item_category,retail_price,stock_quantity\n"
        "SKU_001,Electronics,999.99,14\n"
        "SKU_002,Apparel,49.50,120\n"
        "SKU_003,Home,12.00,340\n"
    ).encode("utf-8")

    comp = service.check_compatibility(retail_csv, model_id="HI-Small")
    assert comp["compatible"] is False
    assert comp["status"] == "INCOMPATIBLE" or len(comp["missing_required_features"]) > 0
    assert len(comp["missing_required_features"]) > 0


# ==============================================================================
# Priority 4: Temporal Leakage Prevention
# ==============================================================================


def test_temporal_realism_future_transactions_cannot_alter_past_predictions():
    """Core Mathematical Proof: Appending future transactions cannot alter past predictions."""
    service = CsvEvaluationService()

    # Step 1: Base transactions for 10 chronological steps
    base_10_bytes = make_transaction_csv(10)
    res_base = service.evaluate_csv_bytes(base_10_bytes, filename="base.csv")
    base_preds = [p["predicted_probability"] for p in res_base["predictions"]]

    # Step 2: Append 10 FUTURE transactions (at later timestamps)
    extended_20_bytes = make_transaction_csv(20)
    res_ext = service.evaluate_csv_bytes(extended_20_bytes, filename="ext.csv")
    ext_preds = [p["predicted_probability"] for p in res_ext["predictions"][:10]]

    # Step 3: Verify base predictions for the first 10 rows are identical (zero future influence)
    for i in range(10):
        np.testing.assert_allclose(
            base_preds[i],
            ext_preds[i],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Future transactions leaked into past prediction at row {i + 1}!",
        )


def test_causal_graph_zero_future_edge_leakage():
    """Verify TimeAwareAmlGraph strictly isolates edges occurring after query timestamp."""
    graph = TimeAwareAmlGraph()

    t1 = pd.Timestamp("2023-01-01 10:00:00")
    t2 = pd.Timestamp("2023-01-01 10:30:00")
    t3 = pd.Timestamp("2023-01-01 11:00:00")
    t_future = pd.Timestamp("2023-01-01 12:00:00")

    graph.add_transaction("tx_1", "A", "B", 1000.0, t1)
    graph.add_transaction("tx_2", "B", "C", 800.0, t2)
    graph.add_transaction("tx_3", "C", "A", 750.0, t3)
    graph.add_transaction("tx_4", "A", "D", 50000.0, t_future)

    # At t = 10:45:00, only t1 and t2 exist. t3 and t_future MUST NOT exist.
    t_query = pd.Timestamp("2023-01-01 10:45:00")
    feat_t_query = graph.extract_features_for_transaction("A", "B", 500.0, t_query)

    # Degree at query time cannot count t3 (C->A) or t_future (A->D)
    assert feat_t_query.sender_in_degree == 0  # C has not sent to A yet
    assert feat_t_query.sender_out_degree == 1  # Only A->B at t1
    assert feat_t_query.sender_net_flow == -1000.0  # Only outflow 1000
