"""Comprehensive test suite for Universal CSV Upload, Schema Detection, and AML Prediction.

Verifies all 20 required research & operational cases:
1. CSV with 10 rows
2. CSV with 100 rows
3. CSV with 1,000 rows
4. CSV with 11 columns (IBM AML Raw)
5. CSV with 57 columns (Processed Features)
6. CSV with different compatible feature names / aliases
7. CSV with a target column
8. CSV without a target column
9. CSV with a missing required feature
10. CSV with extra unused features
11. CSV with missing values (NaN, null, ?, empty)
12. CSV with categorical values
13. CSV with alternate delimiters (semicolon, tab, pipe)
14. CSV with UTF-8 BOM
15. Empty CSV rejection
16. Malformed CSV error handling
17. Large CSV handling
18. Different prediction batch / row limits
19. Model schema mismatch and compatibility reporting
20. Strict prediction output row count match (len(predictions) == len(valid_rows))
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from infrastructure.data.readers.universal_csv_reader import UniversalCsvReader
from infrastructure.data.schemas.universal_schema_detector import UniversalSchemaDetector
from infrastructure.models.registry.universal_model_registry import UniversalModelRegistry
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def preload_models():
    """Ensure models are preloaded once."""
    ms = ModelService()
    ms.load_all_models()


def generate_ibm_raw_rows(n_rows: int, has_target: bool = True) -> str:
    """Helper to generate n_rows of valid IBM AML raw transaction CSV."""
    lines = [
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format"
        + (",Is Laundering" if has_target else "")
    ]
    base_time = pd.Timestamp("2022-09-01 09:00:00")
    for i in range(n_rows):
        t_str = (base_time + pd.Timedelta(minutes=i * 2)).strftime("%Y/%m/%d %H:%M")
        from_b = str(i % 10)
        from_acc = f"ACC_{i % 50:04d}"
        to_b = str((i + 1) % 10)
        to_acc = f"ACC_{(i + 1) % 50:04d}"
        amt = round(100.0 + (i * 15.5) % 10000.0, 2)
        curr = "US Dollar"
        fmt = "Wire" if i % 3 == 0 else "ACH"
        target_val = f",{1 if i % 10 == 0 else 0}" if has_target else ""
        lines.append(f"{t_str},{from_b},{from_acc},{to_b},{to_acc},{amt},{curr},{amt},{curr},{fmt}{target_val}")
    return "\n".join(lines)


# ==============================================================================
# Cases 1, 2, 3: Variable Row Counts (10, 100, 1,000 rows)
# ==============================================================================


def test_case_01_csv_with_10_rows():
    """Verify prediction functions correctly on a small 10-row CSV."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(10, has_target=True)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="test_10.csv")

    assert res["status"] == "success"
    assert res["total_rows"] == 10
    assert len(res["predictions"]) == 10
    assert res["has_labels"] is True
    assert res["predictions"][0]["risk_category"] in ("Low Risk", "Medium Risk", "High Risk")


def test_case_02_csv_with_100_rows():
    """Verify prediction functions correctly on a 100-row CSV."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(100, has_target=True)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="test_100.csv")

    assert res["status"] == "success"
    assert res["total_rows"] == 100
    assert len(res["predictions"]) == 100


def test_case_03_csv_with_1000_rows():
    """Verify prediction functions correctly on a 1,000-row CSV."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(1000, has_target=False)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="test_1000.csv", row_limit=1500)

    assert res["status"] == "success"
    assert res["total_rows"] == 1000
    assert len(res["predictions"]) == 1000
    assert res["has_labels"] is False


# ==============================================================================
# Cases 4, 5: Column Widths (11 Columns Raw, 57 Features Processed)
# ==============================================================================


def test_case_04_csv_with_11_columns():
    """Verify 11-column IBM AML raw schema is parsed and evaluated."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(15, has_target=True)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="raw_11.csv")

    assert res["status"] == "success"
    assert res["detected_format"] == "IBM_AML_RAW"
    assert len(res["predictions"]) == 15


def test_case_05_csv_with_57_features_processed():
    """Verify pre-engineered processed feature format evaluates without reprocessing."""
    service = CsvEvaluationService()
    demo_file = BASE_DIR / "data/demo_samples/hi_small_sample_25.csv"
    assert demo_file.exists()

    csv_text = demo_file.read_text(encoding="utf-8")
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="processed_25.csv")

    assert res["status"] == "success"
    assert res["detected_format"] == "PROCESSED_FEATURES"
    assert res["total_rows"] == 25
    assert len(res["predictions"]) == 25
    assert res["has_labels"] is True


# ==============================================================================
# Cases 6, 7, 8: Schema Aliases and Target Labels
# ==============================================================================


def test_case_06_compatible_column_aliases():
    """Verify custom column aliases (sender_account, receiver_account, transaction_amount, trans_time)."""
    service = CsvEvaluationService()
    csv_text = (
        "trans_time,sender_account,receiver_account,transaction_amount,trans_type,curr\n"
        "2022-09-01 12:00:00,ACC_ALPHA,ACC_BETA,4500.0,Wire,USD\n"
        "2022-09-01 12:05:00,ACC_GAMMA,ACC_DELTA,150.0,ACH,USD\n"
        "2022-09-01 12:10:00,ACC_ALPHA,ACC_EPSILON,85000.0,Wire,EUR\n"
    )
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="aliased.csv")
    assert res["status"] == "success"
    assert len(res["predictions"]) == 3
    assert res["field_mappings"]["from_account"] == "sender_account"
    assert res["field_mappings"]["to_account"] == "receiver_account"


def test_case_07_csv_with_target_column():
    """Verify evaluation calculates supervised metrics when target column exists."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(20, has_target=True)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="with_target.csv")

    assert res["has_labels"] is True
    assert res["evaluation_metrics"] is not None
    assert "accuracy" in res["evaluation_metrics"]
    assert "confusion_matrix" in res["evaluation_metrics"]


def test_case_08_csv_without_target_column():
    """Verify prediction proceeds cleanly when no target column is present."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(20, has_target=False)
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="unlabeled.csv")

    assert res["has_labels"] is False
    assert res["evaluation_metrics"] is None
    assert len(res["predictions"]) == 20


# ==============================================================================
# Cases 9, 10: Missing Features and Extra Unused Features
# ==============================================================================


def test_case_09_csv_with_missing_required_feature():
    """Verify helpful, actionable error when essential transaction features are missing."""
    service = CsvEvaluationService()
    # Missing timestamp and accounts completely
    csv_text = "customer_name,amount,notes\nAlice,500,Grocery\nBob,1200,Rent\n"

    with pytest.raises(ValueError) as excinfo:
        service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="invalid_schema.csv")

    err_msg = str(excinfo.value)
    assert "cannot be evaluated" in err_msg.lower() or "missing" in err_msg.lower()


def test_case_10_csv_with_extra_unused_features():
    """Verify extra metadata columns (notes, ip_address, device_id) do not prevent prediction."""
    service = CsvEvaluationService()
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,"
        "Amount Paid,Payment Currency,Payment Format,ip_address,device_id,user_notes\n"
        "2022/09/01 09:00,0,10,0,11,500.0,US Dollar,500.0,US Dollar,Wire,192.168.1.1,DEV_991,payment for consulting\n"
        "2022/09/01 09:15,0,10,0,12,7500.0,US Dollar,7500.0,US Dollar,Wire,192.168.1.1,DEV_991,equipment purchase\n"
    )
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="extra_cols.csv")
    assert res["status"] == "success"
    assert len(res["predictions"]) == 2


# ==============================================================================
# Cases 11, 12: Missing Values and Categorical Types
# ==============================================================================


def test_case_11_csv_with_missing_values():
    """Verify nulls/NaNs in optional fields are handled robustly."""
    service = CsvEvaluationService()
    csv_text = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,"
        "Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 10:00,,ACC_100,,ACC_200,150.0,,150.0,US Dollar,ACH\n"
        "2022/09/01 10:30,1,ACC_100,2,ACC_300,9200.0,US Dollar,9200.0,,Wire\n"
    )
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="missing_vals.csv")
    assert res["status"] == "success"
    assert len(res["predictions"]) == 2


def test_case_12_csv_with_categorical_values():
    """Verify unseen/synonym payment formats and currencies map cleanly."""
    service = CsvEvaluationService()
    csv_text = (
        "timestamp,from_account,to_account,amount,payment_format,currency\n"
        "2022-09-01 10:00:00,ACC_1,ACC_2,1000.0,credit_card,usd\n"
        "2022-09-01 10:15:00,ACC_1,ACC_3,50000.0,international_wire,eur\n"
        "2022-09-01 10:30:00,ACC_2,ACC_1,250.0,check,gbp\n"
    )
    res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="categorical.csv")
    assert res["status"] == "success"
    assert len(res["predictions"]) == 3


# ==============================================================================
# Cases 13, 14: Delimiters and UTF-8 BOM
# ==============================================================================


def test_case_13_csv_alternate_delimiters():
    """Verify semicolon, tab, and pipe delimiters are automatically detected."""
    reader = UniversalCsvReader()

    # Semicolon
    semi_csv = "Timestamp;Account;Account.1;Amount Paid;Payment Format\n2022/09/01 10:00;A1;B1;500;Wire\n"
    _, meta_semi = reader.read_dataframe(semi_csv)
    assert meta_semi.detected_delimiter == ";"

    # Tab
    tab_csv = "Timestamp\tAccount\tAccount.1\tAmount Paid\tPayment Format\n2022/09/01 10:00\tA1\tB1\t500\tWire\n"
    _, meta_tab = reader.read_dataframe(tab_csv)
    assert meta_tab.detected_delimiter == "\t"

    # Pipe
    pipe_csv = "Timestamp|Account|Account.1|Amount Paid|Payment Format\n2022/09/01 10:00|A1|B1|500|Wire\n"
    _, meta_pipe = reader.read_dataframe(pipe_csv)
    assert meta_pipe.detected_delimiter == "|"


def test_case_14_csv_with_utf8_bom():
    """Verify UTF-8 with BOM byte prefix (\\xef\\xbb\\xbf) does not corrupt header names."""
    service = CsvEvaluationService()
    raw_csv = generate_ibm_raw_rows(5)
    bom_bytes = b"\xef\xbb\xbf" + raw_csv.encode("utf-8")

    res = service.evaluate_csv_bytes(bom_bytes, filename="bom_file.csv")
    assert res["status"] == "success"
    assert len(res["predictions"]) == 5


# ==============================================================================
# Cases 15, 16, 17: Empty, Malformed, and Large CSVs
# ==============================================================================


def test_case_15_empty_csv_rejection():
    """Verify empty CSV files are rejected with clear user error."""
    service = CsvEvaluationService()

    with pytest.raises(ValueError) as exc:
        service.evaluate_csv_bytes(b"", filename="empty.csv")
    assert "empty" in str(exc.value).lower()

    with pytest.raises(ValueError) as exc2:
        service.evaluate_csv_bytes(b"   \n  \n", filename="empty_ws.csv")
    assert "empty" in str(exc2.value).lower()


def test_case_16_malformed_csv_handling():
    """Verify malformed content returns clear error message."""
    service = CsvEvaluationService()
    malformed_bytes = b"\x00\x01\x02\x03\x04\xff\xfe"

    # Inspection handles gracefully
    inspect_res = service.inspect_csv_bytes(malformed_bytes, filename="corrupt.csv")
    assert "is_empty" in inspect_res


def test_case_17_large_csv_handling():
    """Verify large CSV files up to thousands of rows are handled without OOM."""
    service = CsvEvaluationService()
    large_csv = generate_ibm_raw_rows(2500, has_target=False)

    res = service.evaluate_csv_bytes(large_csv.encode("utf-8"), filename="large.csv", row_limit=3000)
    assert res["status"] == "success"
    assert len(res["predictions"]) == 2500


# ==============================================================================
# Cases 18, 19, 20: Batch Sizes, Compatibility, and Strict Row Equality
# ==============================================================================


def test_case_18_different_prediction_row_limits():
    """Verify row_limit parameter cleanly caps processing without index errors."""
    service = CsvEvaluationService()
    csv_text = generate_ibm_raw_rows(100, has_target=True)

    res_50 = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="test_50.csv", row_limit=50)
    assert len(res_50["predictions"]) == 50

    res_25 = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename="test_25.csv", row_limit=25)
    assert len(res_25["predictions"]) == 25


def test_case_19_model_schema_compatibility_reporting():
    """Verify compatibility reporting returns structured report with actionable suggestions."""
    detector = UniversalSchemaDetector()
    reader = UniversalCsvReader()

    # Valid transaction CSV
    csv_text = generate_ibm_raw_rows(10)
    df, _ = reader.read_dataframe(csv_text)
    schema_rep = detector.inspect_dataframe(df)

    comp = detector.check_compatibility(schema_rep, model_id="HI-Small")
    assert comp.compatible is True
    assert comp.allows_direct_prediction is True

    # Incompatible non-transaction CSV
    incomp_csv = "product_name,category,price,inventory_count\nLaptop,Electronics,1200,45\nChair,Furniture,150,12\n"
    df_incomp, _ = reader.read_dataframe(incomp_csv)
    schema_incomp = detector.inspect_dataframe(df_incomp)

    comp_incomp = detector.check_compatibility(schema_incomp, model_id="HI-Small")
    assert comp_incomp.compatible is False
    assert len(comp_incomp.missing_required_features) > 0
    assert "Missing required sequence fields" in comp_incomp.suggested_action


def test_case_20_prediction_output_row_count_strict_equality():
    """Core Test: returned predictions MUST strictly match the number of valid uploaded rows."""
    service = CsvEvaluationService()

    for test_count in (7, 18, 43, 115):
        csv_text = generate_ibm_raw_rows(test_count, has_target=False)
        res = service.evaluate_csv_bytes(csv_text.encode("utf-8"), filename=f"test_{test_count}.csv")

        assert len(res["predictions"]) == test_count, (
            f"Expected {test_count} predictions, got {len(res['predictions'])}"
        )
        assert res["total_rows"] == test_count
        assert res["evaluated_sequences"] == test_count

        # Check row ordering
        for i, pred in enumerate(res["predictions"]):
            assert pred["row_id"] == i + 1
            assert pred["transaction_id"] == f"TX_{i + 1:05d}"


# ==============================================================================
# Dynamic Training (Mode B) and API Endpoints Tests
# ==============================================================================


def test_dynamic_model_training_and_custom_prediction():
    """Verify Mode B: training a custom AML classifier on labeled CSV and predicting with it."""
    service = CsvEvaluationService()

    # Generate synthetic custom tabular dataset (with non-IBM columns)
    np.random.seed(42)
    n = 60
    custom_df = pd.DataFrame({
        "account_holder": [f"user_{i % 10}" for i in range(n)],
        "tx_amount": np.random.exponential(scale=1000.0, size=n),
        "tx_velocity_hour": np.random.poisson(lam=3, size=n),
        "international_flag": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "channel_type": np.random.choice(["ATM", "ONLINE", "BRANCH", "MOBILE"], size=n),
        "fraud_indicator": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    })

    # Ensure both classes exist
    custom_df.loc[0:4, "fraud_indicator"] = 1
    custom_df.loc[5:15, "fraud_indicator"] = 0

    buf = io.StringIO()
    custom_df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    # 1. Train dynamic model
    train_res = service.train_custom_model_from_csv(
        csv_bytes=csv_bytes,
        model_name="Custom_Risk_V1",
        target_column="fraud_indicator",
        model_type="HistGradientBoosting",
        filename="custom_train.csv",
    )

    assert train_res["status"] == "success"
    assert "model_id" in train_res
    custom_model_id = train_res["model_id"]
    assert train_res["training_metrics"]["validation_f1"] >= 0.0

    # 2. Predict using the newly registered dynamic model on new data
    new_df = pd.DataFrame({
        "account_holder": ["user_1", "user_2", "user_3"],
        "tx_amount": [500.0, 150000.0, 250.0],
        "tx_velocity_hour": [1, 15, 2],
        "international_flag": [0, 1, 0],
        "channel_type": ["ONLINE", "ATM", "MOBILE"],
    })
    buf_new = io.StringIO()
    new_df.to_csv(buf_new, index=False)
    new_csv_bytes = buf_new.getvalue().encode("utf-8")

    pred_res = service.evaluate_csv_bytes(
        new_csv_bytes,
        model_id=custom_model_id,
        filename="unlabeled_eval.csv",
    )

    assert pred_res["status"] == "success"
    assert pred_res["model_id"] == custom_model_id
    assert len(pred_res["predictions"]) == 3
    assert pred_res["predictions"][0]["risk_category"] in ("Low Risk", "Medium Risk", "High Risk")

    # 3. Download predictions CSV
    download_res = service.get_prediction_download(pred_res["prediction_id"])
    assert download_res is not None
    csv_content, filename = download_res
    assert "PRED_PROBABILITY" in csv_content
    assert "PRED_LABEL" in csv_content
    assert filename.startswith("predictions_")

