"""Phase 15: End-to-End System Validation and Deployment Readiness Test Suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

try:
    from interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
    )
    from interfaces.api.services.model_service import ModelService
    from interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from interfaces.api.routes.inference import predict, predict_batch, explain
    from interfaces.api.schemas.predict import PredictRequest, BatchPredictRequest
    from interfaces.api.schemas.explain import ExplainRequest
except ModuleNotFoundError:
    from backend.interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
    )
    from backend.interfaces.api.services.model_service import ModelService
    from backend.interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from backend.interfaces.api.routes.inference import predict, predict_batch, explain
    from backend.interfaces.api.schemas.predict import PredictRequest, BatchPredictRequest
    from backend.interfaces.api.schemas.explain import ExplainRequest

BASE_DIR = Path(__file__).resolve().parents[1]


class SimpleResponse:
    """Synchronous test response mirroring the requests/httpx Response interface."""

    def __init__(self, status_code: int, data=None, text: str = ""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text or (json.dumps(self._data) if isinstance(self._data, dict) else str(self._data))

    def json(self):
        return self._data


class SyncAPIClient:
    """Synchronous route dispatcher avoiding Windows Python 3.14 anyio thread-portal deadlock."""

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
                return SimpleResponse(e.status_code, {"detail": e.detail}, text=str(e.detail))
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
            return SimpleResponse(422, {"detail": e.errors()}, text=str(e))
        except HTTPException as e:
            return SimpleResponse(e.status_code, {"detail": e.detail}, text=str(e.detail))
        return SimpleResponse(404, {"detail": "Not found"})


@pytest.fixture(scope="module")
def api_client():
    """Initializes a synchronous test client with preloaded models."""
    return SyncAPIClient()


def load_demo_samples() -> list:
    ts_file = BASE_DIR / "frontend/src/demo/demoSamples.ts"
    assert ts_file.exists(), "frontend/src/demo/demoSamples.ts must exist"
    content = ts_file.read_text(encoding="utf-8")
    match = re.search(r"export const DEMO_SAMPLES: DemoSample\[\] = (\[.*\]);", content, re.DOTALL)
    assert match, "Could not extract DEMO_SAMPLES from demoSamples.ts"
    return json.loads(match.group(1))


# -------------------------------------------------------------------------
# A. End-to-End API Integration: Service & Metadata Tests
# -------------------------------------------------------------------------

def test_e2e_health_check(api_client: SyncAPIClient):
    """Verifies GET /health returns online status and both loaded models."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["readiness"] == "ready"
    assert "hi_small_lstm_fedavg_iid_seed42" in data["loaded_models"]
    assert "config_5_bilstm_fedavg_iid" in data["loaded_models"]


def test_e2e_models_list(api_client: SyncAPIClient):
    """Verifies GET /models returns verified specs for HI-Small and LI-Medium."""
    response = api_client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) == 2
    by_ds = {m["dataset"]: m for m in models}

    hi = by_ds["HI-Small"]
    assert hi["architecture"].upper() == "LSTM"
    assert hi["feature_dimension"] == FEATURE_DIMENSION
    assert hi["sequence_length"] == MAX_SEQUENCE_LENGTH
    assert hi["frozen_threshold"] == FROZEN_THRESHOLDS["HI-Small"]
    assert hi["checkpoint_verified"] is True
    assert "integrated_gradients" in hi["supported_explanation_methods"]

    li = by_ds["LI-Medium"]
    assert li["architecture"].upper() == "BILSTM"
    assert li["feature_dimension"] == FEATURE_DIMENSION
    assert li["sequence_length"] == MAX_SEQUENCE_LENGTH
    assert li["frozen_threshold"] == FROZEN_THRESHOLDS["LI-Medium"]
    assert li["checkpoint_verified"] is True
    assert "integrated_gradients" in li["supported_explanation_methods"]


def test_e2e_metadata_endpoints(api_client: SyncAPIClient):
    """Verifies GET /metadata/{dataset} for valid and invalid datasets."""
    for ds_name in ("HI-Small", "LI-Medium"):
        resp = api_client.get(f"/metadata/{ds_name}")
        assert resp.status_code == 200
        meta = resp.json()
        assert meta["dataset"] == ds_name
        assert meta["frozen_threshold"] == FROZEN_THRESHOLDS[ds_name]
        assert meta["expanded_feature_count"] == 57
        assert meta["sequence_length"] == 10
        assert len(meta["feature_columns"]) == 57

    # Unknown dataset must return 404
    bad_resp = api_client.get("/metadata/Unknown-Dataset")
    assert bad_resp.status_code == 404
    assert "Unsupported dataset" in bad_resp.text or "not found" in bad_resp.text.lower()


# -------------------------------------------------------------------------
# B. End-to-End Single Prediction Tests (4 Demo Scenarios)
# -------------------------------------------------------------------------

def test_e2e_predict_single_demo_scenarios(api_client: SyncAPIClient):
    """Executes POST /predict on all 4 demonstration scenarios from Phase 14B."""
    samples = load_demo_samples()
    assert len(samples) >= 4

    for sample in samples:
        payload = {
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
        }
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 200, f"Failed prediction for {sample['id']}: {resp.text}"
        res = resp.json()

        assert res["dataset"] == sample["dataset"]
        assert res["sample_id"] == sample["sequence"]["sample_id"]
        assert 0.0 <= res["predicted_probability"] <= 1.0
        assert res["decision_threshold"] == FROZEN_THRESHOLDS[sample["dataset"]]
        assert res["inference_time_ms"] > 0

        if "legit" in sample["id"]:
            assert res["predicted_probability"] < res["decision_threshold"], (
                f"Legit sample {sample['id']} scored {res['predicted_probability']} >= {res['decision_threshold']}"
            )
            assert res["decision"] == "CLEARED_LEGITIMATE"
        elif "alert" in sample["id"]:
            assert res["predicted_probability"] >= res["decision_threshold"], (
                f"Alert sample {sample['id']} scored {res['predicted_probability']} < {res['decision_threshold']}"
            )
            assert res["decision"] == "FLAGGED_LAUNDERING"


# -------------------------------------------------------------------------
# C. End-to-End Batch Prediction Tests
# -------------------------------------------------------------------------

def test_e2e_predict_batch_demo_scenarios(api_client: SyncAPIClient):
    """Executes POST /predict/batch containing all demo sequences in a single call."""
    samples = load_demo_samples()
    # Group by dataset
    hi_samples = [s for s in samples if s["dataset"] == "HI-Small"]
    li_samples = [s for s in samples if s["dataset"] == "LI-Medium"]

    for ds_name, group in [("HI-Small", hi_samples), ("LI-Medium", li_samples)]:
        batch_payload = {
            "dataset": ds_name,
            "items": [s["sequence"] for s in group],
        }
        resp = api_client.post("/predict/batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["dataset"] == ds_name
        assert data["total_items"] == len(group)
        assert data["successful_items"] == len(group)
        assert len(data["predictions"]) == len(group)
        assert data["batch_inference_time_ms"] > 0

        # Verify element-by-element fidelity
        for i, pred in enumerate(data["predictions"]):
            expected_sample = group[i]
            assert pred["sample_id"] == expected_sample["sequence"]["sample_id"]
            assert pred["decision_threshold"] == FROZEN_THRESHOLDS[ds_name]
            if "legit" in expected_sample["id"]:
                assert pred["decision"] == "CLEARED_LEGITIMATE"
            else:
                assert pred["decision"] == "FLAGGED_LAUNDERING"


# -------------------------------------------------------------------------
# D. End-to-End Explainability Tests (Integrated Gradients)
# -------------------------------------------------------------------------

def test_e2e_explain_demo_alert_scenarios(api_client: TestClient):
    """Executes POST /explain on high-risk alert samples for both datasets."""
    samples = load_demo_samples()
    alert_samples = [s for s in samples if "alert" in s["id"]]

    for sample in alert_samples:
        payload = {
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "steps": 25,
            "top_k": 5,
        }
        resp = api_client.post("/explain", json=payload)
        assert resp.status_code == 200, f"Explain failed: {resp.text}"
        data = resp.json()

        assert data["dataset"] == sample["dataset"]
        assert data["method"] == "integrated_gradients"
        assert data["steps"] == 25
        assert data["predicted_probability"] >= FROZEN_THRESHOLDS[sample["dataset"]]
        assert len(data["top_positive_drivers"]) <= 5
        assert len(data["top_negative_drivers"]) <= 5
        assert data["completeness_residual"] < 0.05
        assert data["explanation_time_ms"] > 0

        valid_groups = {
            "Transaction Attributes",
            "Payment Format",
            "Currency Rails",
            "Temporal Dynamics",
            "Sender Causal History",
            "Receiver Causal History",
            "Other",
        }
        for driver in data["top_positive_drivers"]:
            assert driver["feature_group"] in valid_groups
            assert driver["attribution"] > 0
        for driver in data["top_negative_drivers"]:
            assert driver["feature_group"] in valid_groups
            assert driver["attribution"] < 0


# -------------------------------------------------------------------------
# E. Robustness & Error Handling: Invalid Requests
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_payload,expected_status,detail_snippet",
    [
        # 1. Invalid feature dimension: 56
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[0.1] * 56] + [[0.0] * 56] * 9,
                    "length": 1,
                    "sample_id": "test_dim_56",
                },
            },
            422,
            "dimension",
        ),
        # 2. Invalid feature dimension: 58
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[0.1] * 58] + [[0.0] * 58] * 9,
                    "length": 1,
                    "sample_id": "test_dim_58",
                },
            },
            422,
            "dimension",
        ),
        # 3. Invalid sequence length: 0
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[0.0] * 57] * 10,
                    "length": 0,
                    "sample_id": "test_len_0",
                },
            },
            422,
            "length",
        ),
        # 4. Invalid sequence length: 11
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[0.1] * 57] * 10,
                    "length": 11,
                    "sample_id": "test_len_11",
                },
            },
            422,
            "length",
        ),
        # 5. Invalid dataset name
        (
            {
                "dataset": "NonExistentDataset",
                "sequence": {
                    "features": [[0.1] * 57] + [[0.0] * 57] * 9,
                    "length": 1,
                    "sample_id": "test_bad_ds",
                },
            },
            422,
            "dataset",
        ),
        # 6. Non-zero padding violation
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[0.1] * 57] * 2 + [[0.5] * 57] + [[0.0] * 57] * 7,
                    "length": 2,  # index 2 must be 0, but is 0.5
                    "sample_id": "test_nonzero_pad",
                },
            },
            422,
            "padding",
        ),
        # 7. NaN in features
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[float("nan")] * 57] + [[0.0] * 57] * 9,
                    "length": 1,
                    "sample_id": "test_nan",
                },
            },
            422,
            "NaN",
        ),
        # 8. Positive Infinity in features
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[float("inf")] * 57] + [[0.0] * 57] * 9,
                    "length": 1,
                    "sample_id": "test_inf",
                },
            },
            422,
            "Infinity",
        ),
        # 9. Negative Infinity in features
        (
            {
                "dataset": "HI-Small",
                "sequence": {
                    "features": [[float("-inf")] * 57] + [[0.0] * 57] * 9,
                    "length": 1,
                    "sample_id": "test_neginf",
                },
            },
            422,
            "Infinity",
        ),
    ],
)
def test_e2e_invalid_prediction_inputs_rejected(
    api_client: TestClient, invalid_payload: dict, expected_status: int, detail_snippet: str
):
    """Ensures input validation strictly rejects malformed sequences and exposes no stack traces."""
    resp = api_client.post("/predict", json=invalid_payload)
    assert resp.status_code == expected_status
    body = resp.text
    assert "Traceback (most recent call last)" not in body, "Stack trace leaked in error response"
    assert "File \"" not in body, "Internal file paths leaked in error response"


def test_e2e_batch_limit_enforcement(api_client: TestClient):
    """Ensures batch predictions above 64 sequences are rejected."""
    seq = {
        "features": [[0.1] * 57] + [[0.0] * 57] * 9,
        "length": 1,
        "sample_id": "item",
    }
    payload = {
        "dataset": "HI-Small",
        "sequences": [seq] * 65,  # Exceeds max 64
    }
    resp = api_client.post("/predict/batch", json=payload)
    assert resp.status_code == 422
    assert "batch" in resp.text.lower() or "64" in resp.text


def test_e2e_mismatched_model_name_rejected(api_client: TestClient):
    """Ensures arbitrary or mismatched model_name parameters are rejected."""
    seq = {
        "features": [[0.1] * 57] + [[0.0] * 57] * 9,
        "length": 1,
        "sample_id": "test_mismatch",
    }
    payload = {
        "dataset": "HI-Small",
        "sequence": seq,
        "model_name": "config_5_bilstm_fedavg_iid",  # Belongs to LI-Medium, not HI-Small
    }
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code in (400, 422)
    assert "is not valid for dataset" in resp.text or "mismatched model" in resp.text.lower()


# -------------------------------------------------------------------------
# F. Production Frontend & Security Boundary Verification
# -------------------------------------------------------------------------

def test_e2e_frontend_production_bundle_security_audit():
    """Audits the compiled frontend static bundle for privacy and security hygiene."""
    dist_dir = BASE_DIR / "frontend/dist"
    assert dist_dir.exists(), "frontend/dist must exist"

    # 1. HTML5 entrypoint check
    index_html = (dist_dir / "index.html").read_text(encoding="utf-8")
    assert "FedSecureAI" in index_html
    # Ensure no external scripts
    external_script_tags = re.findall(r'<script[^>]+src=["\'](http[s]?://[^"\']+)["\']', index_html)
    assert len(external_script_tags) == 0, f"Found external scripts in index.html: {external_script_tags}"

    # 2. JS bundle scan
    js_files = list((dist_dir / "assets").glob("*.js"))
    assert len(js_files) >= 1, "At least one JS bundle must exist"
    bundle_code = js_files[0].read_text(encoding="utf-8")

    # Verify no persistent browser storage of transaction data
    assert "localStorage.setItem" not in bundle_code
    assert "sessionStorage.setItem" not in bundle_code
    assert "indexedDB.open" not in bundle_code

    # Verify zero external CDN/telemetry endpoints
    forbidden_domains = [
        "google-analytics.com",
        "googletagmanager.com",
        "cdnjs.cloudflare.com",
        "cdn.jsdelivr.net",
        "fonts.googleapis.com",
    ]
    for domain in forbidden_domains:
        assert domain not in bundle_code, f"Bundle contains forbidden external domain: {domain}"

    # Verify zero gateway credentials or authorization headers in client bundle
    assert "Bearer eyJ" not in bundle_code
    assert "AWS_SECRET_ACCESS_KEY" not in bundle_code
    assert "CLIENT_SECRET" not in bundle_code


def test_e2e_deployment_packaging_security_audit():
    """Audits Phase 14A Dockerfile and docker-compose configurations."""
    df_path = BASE_DIR / "docker/Dockerfile.inference"
    dc_path = BASE_DIR / "docker-compose.yml"
    assert df_path.exists(), f"Dockerfile not found at {df_path}"
    assert dc_path.exists(), f"docker-compose.yml not found at {dc_path}"

    df_content = df_path.read_text(encoding="utf-8")
    assert "USER 10001:10001" in df_content or "USER fedsecure" in df_content or "USER appuser" in df_content
    assert "HEALTHCHECK" in df_content
    assert "/health" in df_content

    dc_content = dc_path.read_text(encoding="utf-8")
    assert "read_only: true" in dc_content
    assert "cap_drop:" in dc_content
    assert "- ALL" in dc_content
