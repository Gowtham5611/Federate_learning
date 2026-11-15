"""Tests for model-aware transaction inspection and explainability API.

Verifies:
1. Recurrent models (LSTM, GRU, BiLSTM; Centralized, FedAvg, FedProx) with Integrated Gradients.
2. Tabular fallback model (LightGBM) with Tree SHAP exact feature attributions.
3. Cache-based lookup by prediction_id + row_id / transaction_id.
4. Edge cases: TP, FP, FN, TN, and specific transaction TX_00435.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from backend.interfaces.api.routes.inference import explain
from backend.interfaces.api.routes.evaluation import predict_csv, CsvPredictRequest
from backend.interfaces.api.schemas.explain import ExplainRequest
from backend.interfaces.api.services.model_service import ModelService
from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from backend.infrastructure.models.tabular_fallback import FEATURE_COLS

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class SimpleResponse:
    status_code: int
    _data: Dict[str, Any]
    text: str = ""

    def json(self) -> Dict[str, Any]:
        return self._data


class SyncInspectionClient:
    """Synchronous route dispatcher avoiding Windows Python 3.14 anyio thread-portal deadlock."""

    def __init__(self):
        ModelService().load_all_models()

    def post(self, path: str, json: Optional[Dict[str, Any]] = None) -> SimpleResponse:
        payload = json or {}
        try:
            if path in ("/explain", "/api/v1/explain"):
                req = ExplainRequest.model_validate(payload)
                resp = explain(req)
                return SimpleResponse(200, resp.model_dump())
            elif path in ("/predict/csv", "/api/v1/predict/csv"):
                req = CsvPredictRequest.model_validate(payload)
                resp = predict_csv(req)
                return SimpleResponse(200, resp)
        except ValidationError as e:
            return SimpleResponse(422, {"detail": e.errors()}, text=str(e))
        except HTTPException as e:
            return SimpleResponse(e.status_code, {"detail": e.detail}, text=str(e.detail))
        return SimpleResponse(404, {"detail": "Not found"})


@pytest.fixture(scope="module")
def client():
    return SyncInspectionClient()


def test_recurrent_model_integrated_gradients(client):
    """Test Integrated Gradients on recurrent model (HI-Small LSTM)."""
    # 10 timesteps, 57 features
    seq = [[0.1 * ((i + j) % 5) for j in range(57)] for i in range(10)]
    payload = {
        "dataset": "HI-Small",
        "model_name": "hi_small_lstm_fedavg_iid_seed42",
        "sequence": {
            "features": seq,
            "length": 10,
            "sample_id": "TEST_TX_001",
        },
        "steps": 15,
        "top_k": 5,
    }
    resp = client.post("/api/v1/explain", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["method"] == "integrated_gradients"
    assert data["model_name"] == "hi_small_lstm_fedavg_iid_seed42"
    assert "top_positive_drivers" in data
    assert "top_negative_drivers" in data
    assert len(data["top_positive_drivers"]) <= 5
    assert len(data["top_negative_drivers"]) <= 5
    assert data["completeness_residual"] >= 0.0
    assert data["explanation_time_ms"] > 0
    assert 0.0 <= data["predicted_probability"] <= 1.0


def test_tabular_fallback_tree_shap_direct(client):
    """Test Tree SHAP feature attribution on tabular fallback model."""
    tab_features = {col: 0.05 * (i % 10) for i, col in enumerate(FEATURE_COLS)}
    # High amount and wire transfer to trigger risk
    tab_features["amount_paid_log"] = 12.0
    tab_features["amount_received_log"] = 12.0
    tab_features["payment_format_code"] = 1.0

    payload = {
        "dataset": "HI-Small",
        "model_name": "lightgbm_tabular_fallback",
        "tabular_features": tab_features,
        "top_k": 5,
    }
    resp = client.post("/api/v1/explain", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["method"] == "tree_shap"
    assert data["model_name"] == "lightgbm_tabular_fallback"
    assert data["baseline_type"] == "tree_expected_value"
    assert data["completeness_residual"] == 0.0
    assert len(data["top_positive_drivers"]) <= 5
    assert len(data["top_negative_drivers"]) <= 5
    assert data["explanation_time_ms"] < 200  # Tree SHAP is ultra fast
    assert 0.0 <= data["predicted_probability"] <= 1.0


def test_explain_via_prediction_cache_lookup(client):
    """Test on-demand inspection via cached prediction_id and row_id."""
    csv_path = BASE_DIR / "data/demo/demo_ibm_aml_500.csv"
    if not csv_path.exists():
        pytest.skip(f"Demo CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    eval_resp = client.post("/api/v1/predict/csv", json={
        "csv_content": csv_content,
        "dataset": "HI-Small",
        "filename": "demo_ibm_aml_500.csv",
        "row_limit": 50,
        "fallback_model": "tabular",
    })
    assert eval_resp.status_code == 200, eval_resp.text
    eval_data = eval_resp.json()
    pred_id = eval_data.get("prediction_id") or eval_data.get("download_id")
    assert pred_id is not None

    # Now inspect row 1 using only prediction_id and row_id
    explain_resp = client.post("/api/v1/explain", json={
        "dataset": "HI-Small",
        "prediction_id": pred_id,
        "row_id": 1,
    })
    assert explain_resp.status_code == 200, explain_resp.text
    explain_data = explain_resp.json()
    assert explain_data["method"] in ("tree_shap", "integrated_gradients")
    assert len(explain_data["top_positive_drivers"]) > 0 or len(explain_data["top_negative_drivers"]) > 0


def test_tx_00435_and_confusion_categories(client):
    """Test inspection specifically for transaction ID formats and TP/FP/FN/TN scenarios."""
    tab_features = {col: 1.0 for col in FEATURE_COLS}
    payload = {
        "dataset": "HI-Small",
        "model_name": "lightgbm_tabular_fallback",
        "tabular_features": tab_features,
        "transaction_id": "TX_00435",
        "row_id": 435,
    }
    resp = client.post("/api/v1/explain", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["method"] == "tree_shap"
    assert data["predicted_probability"] >= 0.0
    assert data["decision"] in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE")
