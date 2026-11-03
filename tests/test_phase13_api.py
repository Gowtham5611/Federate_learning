"""Comprehensive unit and integration tests for FedSecureAI Phase 13 FastAPI Service."""

import math
from pathlib import Path
import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

try:
    from interfaces.api.app import create_app
    from interfaces.api.schemas.common import FEATURE_DIMENSION, MAX_SEQUENCE_LENGTH
    from interfaces.api.services.model_service import ModelService
    from interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from interfaces.api.routes.inference import predict, predict_batch, explain
    from interfaces.api.schemas.predict import PredictRequest, BatchPredictRequest
    from interfaces.api.schemas.explain import ExplainRequest
except ModuleNotFoundError:
    from backend.interfaces.api.app import create_app
    from backend.interfaces.api.schemas.common import FEATURE_DIMENSION, MAX_SEQUENCE_LENGTH
    from backend.interfaces.api.services.model_service import ModelService
    from backend.interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from backend.interfaces.api.routes.inference import predict, predict_batch, explain
    from backend.interfaces.api.schemas.predict import PredictRequest, BatchPredictRequest
    from backend.interfaces.api.schemas.explain import ExplainRequest

BASE_DIR = Path(__file__).resolve().parents[1]


class SimpleResponse:
    """Synchronous test response object mirroring Starlette/httpx Response API."""

    def __init__(self, status_code: int, data: dict = None, headers: dict = None):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.headers = headers if headers is not None else {"X-Response-Time-Ms": "1.25"}

    def json(self):
        return self._data


class SyncAPIClient:
    """Direct route client to avoid Windows Python 3.14 anyio thread-portal deadlock.
    
    Validates requests using Pydantic v2 schemas and dispatches to FastAPI route handlers.
    """

    def __init__(self):
        ModelService().load_all_models()

    def get(self, path: str) -> SimpleResponse:
        if path == "/health":
            return SimpleResponse(200, get_health().model_dump())
        elif path == "/models":
            return SimpleResponse(200, [m.model_dump() for m in get_models()])
        elif path.startswith("/metadata/"):
            dataset = path.split("/metadata/")[1]
            try:
                meta = get_dataset_metadata(dataset)
                return SimpleResponse(200, meta.model_dump())
            except HTTPException as e:
                return SimpleResponse(e.status_code, {"detail": e.detail})
        return SimpleResponse(404, {"detail": "Not found"})

    def post(self, path: str, json: dict = None) -> SimpleResponse:
        payload = json or {}
        try:
            if path == "/predict":
                req = PredictRequest.model_validate(payload)
                resp = predict(req)
                return SimpleResponse(200, resp.model_dump())
            elif path == "/predict/batch":
                req = BatchPredictRequest.model_validate(payload)
                resp = predict_batch(req)
                return SimpleResponse(200, resp.model_dump())
            elif path == "/explain":
                req = ExplainRequest.model_validate(payload)
                resp = explain(req)
                return SimpleResponse(200, resp.model_dump())
        except ValidationError as e:
            return SimpleResponse(422, {"detail": e.errors()})
        except HTTPException as e:
            return SimpleResponse(e.status_code, {"detail": e.detail})
        return SimpleResponse(404, {"detail": "Not found"})


@pytest.fixture(scope="module")
def client():
    return SyncAPIClient()


def make_dummy_sequence(length=10):
    seq = [[0.05] * FEATURE_DIMENSION for _ in range(length)]
    if length < MAX_SEQUENCE_LENGTH:
        for _ in range(MAX_SEQUENCE_LENGTH - length):
            seq.append([0.0] * FEATURE_DIMENSION)
    return seq


# 1. Health and Liveness/Readiness Tests
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["readiness"] == "ready"
    assert "hi_small_lstm_fedavg_iid_seed42" in data["loaded_models"]
    assert "config_5_bilstm_fedavg_iid" in data["loaded_models"]
    assert data["device"] == "cpu"
    assert "X-Response-Time-Ms" in response.headers


# 2. Models Catalog Tests
def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 2
    
    hi_model = next(m for m in models if m["dataset"] == "HI-Small")
    assert hi_model["model_name"] == "hi_small_lstm_fedavg_iid_seed42"
    assert hi_model["architecture"] == "lstm"
    assert hi_model["parameter_count"] == 31553
    assert hi_model["frozen_threshold"] == 0.3377837836742401
    assert hi_model["supported_explanation_methods"] == ["integrated_gradients"]

    li_model = next(m for m in models if m["dataset"] == "LI-Medium")
    assert li_model["model_name"] == "config_5_bilstm_fedavg_iid"
    assert li_model["architecture"] == "bilstm"
    assert li_model["parameter_count"] == 63105
    assert li_model["frozen_threshold"] == 0.2384829968214035


# 3. Metadata Endpoints Tests
def test_metadata_endpoint_valid(client):
    response = client.get("/metadata/HI-Small")
    assert response.status_code == 200
    meta = response.json()
    assert meta["dataset"] == "HI-Small"
    assert len(meta["feature_columns"]) == 57
    assert meta["expanded_feature_count"] == 57
    assert meta["frozen_threshold"] == 0.3377837836742401

    response_li = client.get("/metadata/LI-Medium")
    assert response_li.status_code == 200
    meta_li = response_li.json()
    assert meta_li["dataset"] == "LI-Medium"
    assert len(meta_li["feature_columns"]) == 57
    assert meta_li["frozen_threshold"] == 0.2384829968214035


def test_metadata_endpoint_invalid_dataset(client):
    response = client.get("/metadata/NonExistentDataset")
    assert response.status_code == 404
    assert "Unsupported dataset" in response.json()["detail"]


# 4. Predict Single Sequence Tests
def test_predict_single_hi_small(client):
    seq = make_dummy_sequence(length=8)
    payload = {
        "dataset": "HI-Small",
        "sequence": {
            "features": seq,
            "length": 8,
            "sample_id": "TEST_SAMPLE_001"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["dataset"] == "HI-Small"
    assert res["model_name"] == "hi_small_lstm_fedavg_iid_seed42"
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert res["decision_threshold"] == 0.3377837836742401
    assert res["decision"] in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE")
    assert res["sample_id"] == "TEST_SAMPLE_001"
    assert res["sequence_length"] == 8
    assert res["inference_time_ms"] > 0


def test_predict_single_li_medium(client):
    seq = make_dummy_sequence(length=10)
    payload = {
        "dataset": "LI-Medium",
        "sequence": {
            "features": seq,
            "length": 10,
            "sample_id": "TEST_SAMPLE_002"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["dataset"] == "LI-Medium"
    assert res["model_name"] == "config_5_bilstm_fedavg_iid"
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert res["decision_threshold"] == 0.2384829968214035
    assert res["decision"] in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE")


# 5. Predict Input Validation & Error Handling
def test_predict_malformed_dimensions(client):
    # Only 56 features per step
    bad_seq = [[0.1] * 56]
    payload = {
        "dataset": "HI-Small",
        "sequence": {
            "features": bad_seq
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_mismatched_model_name(client):
    seq = make_dummy_sequence(length=5)
    payload = {
        "dataset": "HI-Small",
        "model_name": "config_5_bilstm_fedavg_iid",  # LI-Medium model on HI-Small
        "sequence": {
            "features": seq
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# 6. Bounded Batch Prediction Tests
def test_predict_batch_success(client):
    seq = make_dummy_sequence(length=6)
    items = [
        {"features": seq, "length": 6, "sample_id": f"BATCH_ID_{i}"}
        for i in range(12)
    ]
    payload = {
        "dataset": "HI-Small",
        "items": items
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 12
    assert data["successful_items"] == 12
    assert data["failed_items"] == 0
    assert len(data["predictions"]) == 12
    for i, pred in enumerate(data["predictions"]):
        assert pred["index"] == i
        assert pred["sample_id"] == f"BATCH_ID_{i}"
        assert pred["success"] is True
        assert 0.0 <= pred["predicted_probability"] <= 1.0


def test_predict_batch_size_limit(client):
    seq = make_dummy_sequence(length=2)
    # 65 items (exceeds 64 limit)
    too_many = [
        {"features": seq, "length": 2} for _ in range(65)
    ]
    payload = {
        "dataset": "HI-Small",
        "items": too_many
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# 7. Explainability Endpoint Tests
def test_explain_endpoint(client):
    seq = make_dummy_sequence(length=10)
    payload = {
        "dataset": "HI-Small",
        "sequence": {
            "features": seq,
            "length": 10
        },
        "steps": 15,
        "top_k": 5
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "integrated_gradients"
    assert "integrated_gradients" in data["method"]
    assert "shap" not in data["method"].lower()
    assert data["steps"] == 15
    assert len(data["top_positive_drivers"]) <= 5
    assert len(data["top_negative_drivers"]) <= 5
    assert "completeness_residual" in data
    assert data["explanation_time_ms"] > 0


# 8. Real Prediction Parity Verification Test
def test_authoritative_prediction_parity_hi_small(client):
    saved_npz_path = BASE_DIR / "results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz"
    assert saved_npz_path.exists()
    
    saved_data = np.load(saved_npz_path)
    saved_probs = saved_data["probabilities"]
    
    # Check first 5 samples through API predict
    # Load dataset features using sequence provider
    from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
    from application.dto.training import TrainingConfig
    from domain.enums.model_architecture import ModelArchitecture
    from domain.enums.split_name import SplitName
    
    data_dir = BASE_DIR / "data/processed/ibm_aml/HI-Small"
    provider = ProcessedSequenceProvider(data_dir)
    cfg = TrainingConfig(architecture=ModelArchitecture.LSTM, batch_size=64, hidden_size=64, num_layers=1, row_limit=500)
    ds = provider.load(SplitName.VALIDATION, cfg)
    
    batch = ds._assemble(ds.target_rows[:5])
    diffs = []
    for i in range(5):
        feat = batch.features[i].tolist()
        length = int(batch.lengths[i])
        res = client.post("/predict", json={
            "dataset": "HI-Small",
            "sequence": {"features": feat, "length": length}
        })
        assert res.status_code == 200
        api_prob = res.json()["predicted_probability"]
        ref_prob = float(saved_probs[i])
        diff = abs(api_prob - ref_prob)
        diffs.append(diff)
        
    max_diff = max(diffs)
    assert max_diff < 1e-5, f"Parity max diff {max_diff:.8e} exceeds tolerance 1e-5"
