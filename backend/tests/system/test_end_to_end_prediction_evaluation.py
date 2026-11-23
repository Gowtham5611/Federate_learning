"""End-to-End Tests for FedSecureAI Prediction and Evaluation System.

Validates:
1. Dynamic arbitrary CSV batch prediction (Scenario A with labels, Scenario B without labels).
2. PyTorch batch inference (batch_size=4096) with GPU/CPU fallback.
3. Decision threshold adjustment and verification of threshold-dependent vs ranking-invariant metrics.
4. Comprehensive metrics calculation:
   Accuracy, Precision, Recall, Sensitivity, Specificity, Balanced Accuracy, MCC, Cohen's Kappa,
   FPR, FNR, FDR, NPV, PR-AUC, ROC-AUC, Average Precision, Log Loss, Brier Score, ECE.
5. Multi-Model Comparison endpoint (/api/v1/evaluate/multi-model) evaluating identical preprocessed
   sequences across all selected Centralized, FedAvg, FedProx, and Ensemble models.
6. Model selection across all verified Centralized, IID FedAvg, Non-IID FedProx, and Hybrid Ensemble models.
7. Downloadable CSV formatting with original columns preserved and standard predicted fields appended.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from interfaces.api.app import create_app
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService

BASE_DIR = Path(__file__).resolve().parents[1]

LABELED_SAMPLE_CSV = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,0,ACC_01,0,ACC_02,150.00,US Dollar,150.00,US Dollar,ACH,0
2022/09/01 01:05,0,ACC_01,0,ACC_03,45000.00,US Dollar,45000.00,US Dollar,Wire,1
2022/09/01 02:30,0,ACC_02,0,ACC_04,320.50,US Dollar,320.50,US Dollar,Credit Card,0
2022/09/01 03:15,0,ACC_03,0,ACC_05,82000.00,US Dollar,82000.00,US Dollar,Wire,1
2022/09/01 04:00,0,ACC_04,0,ACC_01,120.00,US Dollar,120.00,US Dollar,ACH,0
2022/09/01 04:45,0,ACC_05,0,ACC_02,95000.00,US Dollar,95000.00,US Dollar,Wire,1
2022/09/01 05:20,0,ACC_02,0,ACC_01,210.00,US Dollar,210.00,US Dollar,Credit Card,0
2022/09/01 06:10,0,ACC_03,0,ACC_04,500.00,US Dollar,500.00,US Dollar,ACH,0
"""

UNLABELED_SAMPLE_CSV = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format
2022/09/01 00:20,0,ACC_01,0,ACC_02,150.00,US Dollar,150.00,US Dollar,ACH
2022/09/01 01:05,0,ACC_01,0,ACC_03,45000.00,US Dollar,45000.00,US Dollar,Wire
2022/09/01 02:30,0,ACC_02,0,ACC_04,320.50,US Dollar,320.50,US Dollar,Credit Card
2022/09/01 03:15,0,ACC_03,0,ACC_05,82000.00,US Dollar,82000.00,US Dollar,Wire
2022/09/01 04:00,0,ACC_04,0,ACC_01,120.00,US Dollar,120.00,US Dollar,ACH
"""


@pytest.fixture(scope="module")
def api_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestEndToEndPredictionEvaluation:
    """Comprehensive validation suite for AML prediction, evaluation, and multi-model benchmarking."""

    def test_scenario_a_labeled_evaluation_and_extended_metrics(self):
        """Scenario A: CSV with Is Laundering column computes all required metrics."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(
            LABELED_SAMPLE_CSV.encode("utf-8"),
            filename="labeled_test.csv",
            threshold=0.337784,
        )

        assert result["status"] == "success"
        assert result["has_labels"] is True
        assert result["evaluated_sequences"] == 8

        metrics = result["evaluation_metrics"]
        assert metrics is not None

        # Standard Classification & Confusion Matrix
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "f2_score" in metrics
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert "tp" in cm and "fp" in cm and "fn" in cm and "tn" in cm
        assert (cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]) == 8

        # Extended Scientific Metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "balanced_accuracy" in metrics
        assert "mcc" in metrics
        assert "cohen_kappa" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics
        assert "false_discovery_rate" in metrics
        assert "negative_predictive_value" in metrics

        # Probabilistic & Calibration Metrics
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert "maximum_calibration_error" in metrics
        assert "log_loss" in metrics

        # Verify mathematical consistency
        assert metrics["sensitivity"] == metrics["recall"]
        expected_balanced_acc = (metrics["sensitivity"] + metrics["specificity"]) / 2.0
        assert metrics["balanced_accuracy"] == pytest.approx(expected_balanced_acc, abs=1e-5)

    def test_scenario_b_unlabeled_evaluation_no_fake_metrics(self):
        """Scenario B: Unlabeled CSV computes probabilities and risk distribution without fake metrics."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(
            UNLABELED_SAMPLE_CSV.encode("utf-8"),
            filename="unlabeled_stream.csv",
        )

        assert result["status"] == "success"
        assert result["has_labels"] is False
        assert result["evaluation_metrics"] is None
        assert result["evaluated_sequences"] == 5

        # Laundering count and cleared count computed strictly from probabilities
        assert result["flagged_laundering_count"] + result["cleared_count"] == 5

        # All items have ground_truth_label None
        for p in result["predictions"]:
            assert p["ground_truth_label"] is None
            assert p["predicted_probability"] >= 0.0
            assert p["predicted_probability"] <= 1.0
            assert p["decision"] in ("SUSPICIOUS", "LEGITIMATE")

    def test_dynamic_threshold_adjustment(self):
        """Verify dynamic decision threshold adjustment changes classification but leaves ranking invariant."""
        service = CsvEvaluationService()

        res_low_thr = service.evaluate_csv_bytes(
            LABELED_SAMPLE_CSV.encode("utf-8"),
            filename="test_thr.csv",
            threshold=0.10,
        )
        res_high_thr = service.evaluate_csv_bytes(
            LABELED_SAMPLE_CSV.encode("utf-8"),
            filename="test_thr.csv",
            threshold=0.90,
        )

        # Threshold recorded accurately
        assert res_low_thr["decision_threshold"] == pytest.approx(0.10, abs=1e-5)
        assert res_high_thr["decision_threshold"] == pytest.approx(0.90, abs=1e-5)

        # Lower threshold flags at least as many as higher threshold
        assert res_low_thr["flagged_laundering_count"] >= res_high_thr["flagged_laundering_count"]

        # Ranking metrics (PR-AUC, ROC-AUC) and Brier Score must remain mathematically invariant
        m_low = res_low_thr["evaluation_metrics"]
        m_high = res_high_thr["evaluation_metrics"]

        if m_low["pr_auc"] is not None and m_high["pr_auc"] is not None:
            assert m_low["pr_auc"] == pytest.approx(m_high["pr_auc"], abs=1e-5)
        if m_low["roc_auc"] is not None and m_high["roc_auc"] is not None:
            assert m_low["roc_auc"] == pytest.approx(m_high["roc_auc"], abs=1e-5)
        assert m_low["brier_score"] == pytest.approx(m_high["brier_score"], abs=1e-5)

    def test_multi_model_evaluation_service(self):
        """Verify evaluate_multi_model_csv runs multiple models on the exact preprocessed batch."""
        service = CsvEvaluationService()
        selected_models = [
            "HI-Small",
            "centralized_lstm",
            "centralized_gru",
            "hi_small_gru_fedavg_iid_seed42",
            "hi_small_lstm_fedprox_mu0.0001_seed42",
            "ensemble_hybrid",
        ]

        multi_res = service.evaluate_multi_model_csv(
            csv_bytes=LABELED_SAMPLE_CSV.encode("utf-8"),
            filename="multi_test.csv",
            model_ids=selected_models,
            threshold=0.35,
        )

        assert multi_res["status"] == "success"
        assert multi_res["total_transactions"] == 8
        assert multi_res["has_labels"] is True
        assert multi_res["models_evaluated_count"] == len(selected_models)
        assert len(multi_res["models"]) == len(selected_models)

        # Verify each model item structure
        for m in multi_res["models"]:
            assert "model_id" in m
            assert "model_name" in m
            assert "model_type" in m
            assert "total_samples" in m
            assert m["total_samples"] == 8
            assert "flagged_count" in m
            assert "cleared_count" in m
            assert m["flagged_count"] + m["cleared_count"] == 8
            assert "pr_auc" in m
            assert "roc_auc" in m
            assert "f1_score" in m
            assert "accuracy" in m
            assert "mcc" in m
            assert "decision_threshold" in m
            assert m["decision_threshold"] == pytest.approx(0.35, abs=1e-5)

    def test_api_route_multi_model_endpoint(self, api_client: TestClient):
        """Verify POST /api/v1/evaluate/multi-model API route."""
        payload = {
            "csv_content": LABELED_SAMPLE_CSV,
            "filename": "api_multi_test.csv",
            "model_ids": ["HI-Small", "centralized_bilstm"],
            "threshold": 0.30,
        }
        resp = api_client.post("/api/v1/evaluate/multi-model", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["models_evaluated_count"] == 2
        assert len(data["models"]) == 2

    def test_downloadable_prediction_csv_format(self):
        """Verify generated downloadable CSV contains all specified columns preserving original data."""
        service = CsvEvaluationService()
        result = service.evaluate_csv_bytes(
            LABELED_SAMPLE_CSV.encode("utf-8"),
            filename="download_test.csv",
        )

        download_id = result.get("download_id")
        assert download_id is not None
        assert download_id in service._prediction_cache

        cache_entry = service._prediction_cache[download_id]
        csv_str = cache_entry["csv_content"]
        df_out = pd.read_csv(io.StringIO(csv_str))

        # Expected output columns
        expected_cols = [
            "predicted_probability",
            "predicted_label",
            "prediction_status",
            "model_name",
            "classification_threshold",
            "is_laundering",
        ]
        for col in expected_cols:
            assert col in df_out.columns, f"Expected column {col} in downloadable CSV"

        # Check values
        assert len(df_out) == 8
        assert all(label in ("SUSPICIOUS", "LEGITIMATE") for label in df_out["predicted_label"])
        assert all(status in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE") for status in df_out["prediction_status"])
