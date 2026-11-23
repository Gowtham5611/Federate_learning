"""Tests for FedSecureAI Universal Adaptive CSV Evaluation System.

Validates schema detection, canonical alias resolution, target leakage prevention,
pipeline selection, frozen threshold preservation, labeled/unlabeled metric handling,
and operational AML labeling.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi import HTTPException
import pandas as pd
import numpy as np

from infrastructure.data.schemas.adaptive_schema_detector import (
    AdaptiveSchemaDetector,
    DetectionStatus,
    FormatCategory,
)
from interfaces.api.routes.evaluation import evaluate_csv, validate_schema, CsvEvaluateRequest
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService

BASE_DIR = Path(__file__).resolve().parents[1]

# Ensure models are loaded once
@pytest.fixture(scope="module", autouse=True)
def preload_models():
    ms = ModelService()
    ms.load_all_models()


RAW_IBM_SAMPLE = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,0,10,0,10,6.59,US Dollar,6.59,US Dollar,ACH,0
2022/09/01 01:05,0,11,0,12,12000.00,US Dollar,12000.00,US Dollar,Wire,1
2022/09/01 02:30,0,13,0,14,350.50,US Dollar,350.50,US Dollar,Credit Card,0
"""

SYNTHETIC_BANKING_SAMPLE = """timestamp,sender_account,receiver_account,transaction_amount,transaction_type,currency,is_fraud,structuring_flag
2022-09-01 09:15:00,ACC_749102,ACC_839104,12450.00,TRANSFER,USD,0,0
2022-09-01 09:20:00,ACC_109284,ACC_491823,890.50,PAYMENT,USD,0,0
2022-09-01 09:45:00,ACC_749102,ACC_901842,48500.00,WIRE,EUR,1,1
2022-09-01 10:12:00,ACC_381920,ACC_839104,320.00,DEBIT,USD,0,0
2022-09-01 10:30:00,ACC_491823,ACC_109284,95000.00,CASH_OUT,USD,1,1
"""

UNLABELED_SAMPLE = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format
2022/09/01 09:00,102,ACC_91028,102,ACC_81920,5420.00,US Dollar,5420.00,US Dollar,ACH
2022/09/01 09:14,204,ACC_49182,102,ACC_10294,1280.50,US Dollar,1280.50,US Dollar,Credit Card
2022/09/01 09:30,305,ACC_77491,305,ACC_88392,67400.00,Euro,67400.00,Euro,Wire
"""


class TestAdaptiveCSVEvaluation:
    """Comprehensive test suite for universal adaptive CSV evaluation."""

    def test_01_ibm_raw_csv_schema(self):
        """Verify standard 11-column IBM AML raw schema is recognized as compatible."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(RAW_IBM_SAMPLE.encode("utf-8"), filename="raw_stream.csv")
        assert result["status"] == "success"
        assert result["schema_status"] == "COMPATIBLE"
        assert result["schema_message"] == "CSV format recognized. Required fields mapped successfully."
        assert result["dataset"] == "HI-Small"
        assert result["decision_threshold"] == pytest.approx(0.33778378, abs=1e-5)
        assert result["has_labels"] is True
        assert result["evaluated_sequences"] == 3

    def test_02_synthetic_banking_csv_schema(self):
        """Verify synthetic mobile banking format with aliased columns is adapted seamlessly."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(SYNTHETIC_BANKING_SAMPLE.encode("utf-8"), filename="mobile_banking.csv")
        assert result["status"] == "success"
        assert result["schema_status"] == "COMPATIBLE"
        assert result["schema_message"] == "CSV format recognized. Required fields mapped successfully."
        assert result["dataset"] == "HI-Small"
        assert result["evaluated_sequences"] == 5
        assert result["has_labels"] is True
        assert result["evaluation_metrics"] is not None
        assert "accuracy" in result["evaluation_metrics"]
        assert "confusion_matrix" in result["evaluation_metrics"]

    def test_03_different_column_aliases(self):
        """Verify varied aliases for transaction fields (date, trans_time, originator, beneficiary, value, type)."""
        aliased_csv = """trans_date_trans_time,originator,beneficiary,value,type,ccy,laundering
2022-09-01 10:00:00,USER_A,USER_B,500.00,wire_transfer,USD,0
2022-09-01 10:15:00,USER_C,USER_D,150000.00,direct_debit,EUR,1
"""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(aliased_csv.encode("utf-8"), filename="custom_stream.csv")
        assert result["status"] == "success"
        assert result["schema_status"] == "COMPATIBLE"
        assert result["evaluated_sequences"] == 2
        assert result["has_labels"] is True
        # Check field mappings recorded
        mappings = result["field_mappings"]
        assert mappings["timestamp"] == "trans_date_trans_time"
        assert mappings["from_account"] == "originator"
        assert mappings["to_account"] == "beneficiary"
        assert mappings["amount_paid"] == "value"

    def test_04_missing_required_fields_partially_compatible(self):
        """Verify CSV missing required account fields is categorized as PARTIALLY_COMPATIBLE."""
        partial_csv = """timestamp,amount,payment_format
2022-09-01 10:00:00,500.00,Wire
2022-09-01 11:00:00,1200.00,ACH
"""
        detector = AdaptiveSchemaDetector()
        df = pd.read_csv(io.StringIO(partial_csv))
        res = detector.inspect_and_adapt(df, filename="partial.csv")
        assert res.status == DetectionStatus.PARTIALLY_COMPATIBLE
        assert "Some fields could not be mapped" in res.user_message
        assert "from_account" in res.missing_required_fields
        assert "to_account" in res.missing_required_fields

        # When evaluated via service, must raise 400 with user-friendly message
        service = CsvEvaluationService()
        with pytest.raises(ValueError) as exc:
            service.evaluate_csv_bytes(partial_csv.encode("utf-8"), filename="partial.csv")
        assert "Some fields could not be mapped" in str(exc.value)

    def test_05_ambiguous_field_mappings(self):
        """Verify conflicting mappings for same canonical field return AMBIGUOUS status."""
        ambiguous_csv = """timestamp,from_account,sender_account,to_account,amount
2022-09-01 10:00:00,ACC_1,ACC_99,ACC_2,500.00
2022-09-01 11:00:00,ACC_3,ACC_88,ACC_4,1200.00
"""
        detector = AdaptiveSchemaDetector()
        df = pd.read_csv(io.StringIO(ambiguous_csv))
        res = detector.inspect_and_adapt(df, filename="ambiguous.csv")
        assert res.status == DetectionStatus.AMBIGUOUS
        assert "Multiple possible field mappings were detected" in res.user_message

        service = CsvEvaluationService()
        with pytest.raises(ValueError) as exc:
            service.evaluate_csv_bytes(ambiguous_csv.encode("utf-8"), filename="ambiguous.csv")
        assert "Multiple possible field mappings were detected" in str(exc.value)

    def test_06_unsupported_schema(self):
        """Verify non-transaction data (e.g. Iris flowers or customer demographics) is flagged as UNSUPPORTED."""
        iris_csv = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
"""
        detector = AdaptiveSchemaDetector()
        df = pd.read_csv(io.StringIO(iris_csv))
        res = detector.inspect_and_adapt(df, filename="iris.csv")
        assert res.status == DetectionStatus.UNSUPPORTED
        assert "This CSV cannot be evaluated using the currently available trained models." in res.user_message

        service = CsvEvaluationService()
        with pytest.raises(ValueError) as exc:
            service.evaluate_csv_bytes(iris_csv.encode("utf-8"), filename="iris.csv")
        assert "This CSV cannot be evaluated" in str(exc.value)

    def test_07_target_leakage_prevention(self):
        """Verify target-derived fields (structuring_flag, sar_filed) are excluded from model inputs."""
        leakage_csv = """timestamp,from_account,to_account,amount,is_laundering,structuring_flag,sar_filed
2022-09-01 10:00:00,ACC_01,ACC_02,9990.00,1,1,1
2022-09-01 11:00:00,ACC_03,ACC_04,150.00,0,0,0
"""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(leakage_csv.encode("utf-8"), filename="leakage_test.csv")
        assert result["status"] == "success"
        assert "structuring_flag" in result["excluded_leakage_fields"]
        assert "sar_filed" in result["excluded_leakage_fields"]
        assert any("Target-derived field" in note for note in result["audit_notes"])

    def test_08_pipeline_selection_hi_small(self):
        """Verify standard raw files route to HI-Small with frozen threshold 0.337784."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(RAW_IBM_SAMPLE.encode("utf-8"), filename="transactions.csv")
        assert result["dataset"] == "HI-Small"
        assert result["decision_threshold"] == pytest.approx(0.33778378, abs=1e-5)
        assert result["model_name"] == "hi_small_lstm_fedavg_iid_seed42"

    def test_09_pipeline_selection_li_medium(self):
        """Verify files with li_medium in filename route to LI-Medium with frozen threshold 0.238483."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(RAW_IBM_SAMPLE.encode("utf-8"), filename="li_medium_batch.csv")
        assert result["dataset"] == "LI-Medium"
        assert result["decision_threshold"] == pytest.approx(0.23848299, abs=1e-5)
        assert result["model_name"] == "config_5_bilstm_fedavg_iid"

    def test_10_unlabeled_metric_handling(self):
        """Verify unlabeled transactions compute predictions and risk histogram but do NOT fabricate metrics."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(UNLABELED_SAMPLE.encode("utf-8"), filename="unlabeled.csv")
        assert result["status"] == "success"
        assert result["has_labels"] is False
        assert result["evaluation_metrics"] is None
        assert result["evaluated_sequences"] == 3
        assert len(result["risk_distribution"]) == 10
        assert all(p["actual_label"] is None for p in result["predictions"])

    def test_11_operational_status_labels(self):
        """Verify transactions are strictly classified into SUSPICIOUS and LEGITIMATE."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(SYNTHETIC_BANKING_SAMPLE.encode("utf-8"))
        for p in result["predictions"]:
            assert p["predicted_label"] in ("SUSPICIOUS", "LEGITIMATE")
            assert p["decision"] in ("SUSPICIOUS", "LEGITIMATE")
            assert p["status"] in ("SUSPICIOUS", "LEGITIMATE")
            assert isinstance(p["predicted_probability"], float)
            assert 0.0 <= p["predicted_probability"] <= 1.0

    def test_12_validate_schema_endpoint(self):
        """Verify the /evaluate/validate-schema API endpoint returns pre-flight schema analysis."""
        req = CsvEvaluateRequest(
            csv_content=SYNTHETIC_BANKING_SAMPLE,
            filename="synthetic_test.csv",
        )
        res = validate_schema(req)
        assert res["status"] == "COMPATIBLE"
        assert res["user_message"] == "CSV format recognized. Required fields mapped successfully."
        assert "structuring_flag" in res["excluded_leakage_fields"]
        assert res["target_pipeline"] == "HI-Small"

    def test_13_evaluate_csv_endpoint_error_handling(self):
        """Verify FastAPI route handles invalid/unsupported CSV with HTTP 400 and clear message."""
        bad_req = CsvEvaluateRequest(
            csv_content="customer_id,age,churn\n1,34,0\n2,55,1",
            filename="customer_churn.csv",
        )
        with pytest.raises(HTTPException) as exc:
            evaluate_csv(bad_req)
        assert exc.value.status_code == 400
        assert "This CSV cannot be evaluated" in exc.value.detail

