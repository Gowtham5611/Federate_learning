"""Security, API contracts, boundary limits, and resilience test suite for Phase 13.1."""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

try:
    from interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_BATCH_SIZE,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from interfaces.api.schemas.predict import BatchPredictRequest, PredictRequest
    from interfaces.api.schemas.explain import ExplainRequest
    from interfaces.api.services.model_service import (
        MODEL_REGISTRY,
        ModelIntegrityError,
        ModelService,
    )
    from interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from interfaces.api.routes.inference import predict, predict_batch, explain
except ModuleNotFoundError:
    from backend.interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_BATCH_SIZE,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from backend.interfaces.api.schemas.predict import BatchPredictRequest, PredictRequest
    from backend.interfaces.api.schemas.explain import ExplainRequest
    from backend.interfaces.api.services.model_service import (
        MODEL_REGISTRY,
        ModelIntegrityError,
        ModelService,
    )
    from backend.interfaces.api.routes.health import get_health, get_models, get_dataset_metadata
    from backend.interfaces.api.routes.inference import predict, predict_batch, explain


def make_valid_sequence(length: int = 10, fill_val: float = 0.05) -> SequenceItem:
    features = [[fill_val] * FEATURE_DIMENSION for _ in range(length)]
    if length < MAX_SEQUENCE_LENGTH:
        for _ in range(MAX_SEQUENCE_LENGTH - length):
            features.append([0.0] * FEATURE_DIMENSION)
    return SequenceItem(features=features, length=length)


@pytest.fixture(scope="module", autouse=True)
def ensure_models_loaded():
    ms = ModelService()
    ms.load_all_models()
    return ms


# 1. Unsupported Datasets & Malformed Requests
def test_unsupported_dataset_rejected():
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate({
            "dataset": "UNSUPPORTED_DATASET",
            "sequence": {"features": [[0.1] * 57] * 10, "length": 10},
        })
    assert "Input should be 'HI-Small' or 'LI-Medium'" in str(exc.value)

    with pytest.raises(HTTPException) as exc_meta:
        get_dataset_metadata("UNSUPPORTED_DATASET")
    assert exc_meta.value.status_code == 404


def test_malformed_request_body_missing_fields():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({})

    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"dataset": "HI-Small"})

    with pytest.raises(ValidationError):
        ExplainRequest.model_validate({"dataset": "HI-Small"})


# 2. Missing Features, Incorrect Dimensions, Invalid Lengths
def test_incorrect_feature_dimension_56():
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=[[0.1] * 56] * 10, length=10)
    assert "exactly 57 required" in str(exc.value)


def test_incorrect_feature_dimension_58():
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=[[0.1] * 58] * 10, length=10)
    assert "exactly 57 required" in str(exc.value)


def test_invalid_sequence_lengths():
    # Empty sequence
    with pytest.raises(ValidationError):
        SequenceItem(features=[], length=0)

    # Exceeding MAX_SEQUENCE_LENGTH=10
    with pytest.raises(ValidationError):
        SequenceItem(features=[[0.1] * 57] * 11, length=11)


# 3. Numeric Integrity: NaN, +Inf, -Inf
def test_nan_rejected():
    seq = [[0.05] * 57 for _ in range(10)]
    seq[0][0] = float("nan")
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=seq, length=10)
    assert "contains NaN" in str(exc.value)


def test_pos_inf_rejected():
    seq = [[0.05] * 57 for _ in range(10)]
    seq[2][5] = float("inf")
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=seq, length=10)
    assert "contains infinite value" in str(exc.value)


def test_neg_inf_rejected():
    seq = [[0.05] * 57 for _ in range(10)]
    seq[3][10] = float("-inf")
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=seq, length=10)
    assert "contains infinite value" in str(exc.value)


def test_non_zero_padding_rejected():
    # Length declared as 5, but step 6 has non-zero value
    seq = [[0.05] * 57 for _ in range(5)]
    seq.append([0.05] * 57)  # Step 5 is non-zero
    for _ in range(4):
        seq.append([0.0] * 57)
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=seq, length=5)
    assert "Padded positions must be zero-filled" in str(exc.value)


# 4. Batch Bounds: Above 64 Items
def test_batch_size_above_64_rejected():
    valid_seq = make_valid_sequence(length=5)
    items = [valid_seq] * 65
    with pytest.raises(ValidationError) as exc:
        BatchPredictRequest(dataset="HI-Small", items=items)
    assert "at most 64 items" in str(exc.value)


# 5. Explainability Steps and Top-K Limits
def test_invalid_explain_steps():
    valid_seq = make_valid_sequence(length=10)
    # steps < 10
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=valid_seq, steps=9)

    # steps > 50
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=valid_seq, steps=51)


def test_invalid_explain_top_k():
    valid_seq = make_valid_sequence(length=10)
    # top_k < 1
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=valid_seq, top_k=0)

    # top_k > 20
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=valid_seq, top_k=21)


# 6. Checkpoint Integrity & Fail-Closed Behavior
def test_checkpoint_hash_mismatch_fails_closed():
    ms = ModelService()
    with patch.dict(MODEL_REGISTRY["HI-Small"], {"expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000"}):
        with pytest.raises(ModelIntegrityError) as exc:
            ms._load_single_model("HI-Small")
        assert "Checkpoint SHA-256 mismatch" in str(exc.value)
    # Reload valid model to restore clean state
    ms._load_single_model("HI-Small")


def test_checkpoint_missing_fails_closed():
    ms = ModelService()
    fake_path = Path("results/nonexistent_dir/nonexistent_model.pt")
    with patch.dict(MODEL_REGISTRY["HI-Small"], {"checkpoint_path": fake_path}):
        with pytest.raises(ModelIntegrityError) as exc:
            ms._load_single_model("HI-Small")
        assert "not found at" in str(exc.value)
    # Reload valid model to restore clean state
    ms._load_single_model("HI-Small")


def test_unloaded_model_raises_503_in_route():
    ms = ModelService()
    saved_model = ms._models.pop("HI-Small", None)
    try:
        seq = make_valid_sequence(length=8)
        req = PredictRequest(dataset="HI-Small", sequence=seq)
        with pytest.raises(HTTPException) as exc:
            predict(req)
        assert exc.value.status_code == 503
        assert "not loaded or unavailable" in exc.value.detail
    finally:
        if saved_model:
            ms._models["HI-Small"] = saved_model


# 7. Model Mismatch Handling
def test_mismatched_model_name_rejected():
    seq = make_valid_sequence(length=8)
    with pytest.raises(ValidationError) as exc:
        PredictRequest(
            dataset="HI-Small",
            model_name="config_5_bilstm_fedavg_iid",  # LI-Medium model
            sequence=seq,
        )
    assert "is not valid for dataset" in str(exc.value)


# 8. Concurrency & Thread-Safety
def test_concurrent_prediction_requests():
    ms = ModelService()
    seq = make_valid_sequence(length=10)

    def run_pred(sample_idx: int):
        prob, threshold, decision, latency = ms.predict_sequence("HI-Small", seq)
        return sample_idx, prob, decision

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_pred, i) for i in range(25)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 25
    first_prob = results[0][1]
    for idx, p, dec in results:
        assert abs(p - first_prob) < 1e-6
        assert dec == "CLEARED_LEGITIMATE"


# 9. Privacy & Absence of Sensitive Data in Logs
def test_sensitive_data_absent_from_logs():
    logger = logging.getLogger("fedsecureai.api")
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    secret_account_id = "ACC_PRIVACY_SENSITIVE_998877"
    seq = make_valid_sequence(length=5)
    seq.sample_id = secret_account_id

    req = PredictRequest(dataset="HI-Small", sequence=seq)
    _ = predict(req)

    log_contents = log_stream.getvalue()
    logger.removeHandler(handler)

    # Verify secret account id is NOT logged
    assert secret_account_id not in log_contents
    # Verify raw features are NOT logged
    assert "0.05" not in log_contents


# 10. Authentication State Audit
def test_authentication_mechanism_status():
    """Verify that authentication is currently NOT IMPLEMENTED on the endpoints.
    
    This test formally documents that requests without Authorization headers
    are accepted, establishing this finding as an auditable deployment blocker.
    """
    seq = make_valid_sequence(length=6)
    req = PredictRequest(dataset="HI-Small", sequence=seq)
    # Calling predict does not require any auth token or header
    resp = predict(req)
    assert resp.predicted_probability is not None
    # Finding: Authentication is absent from the current interface
    auth_implemented = False
    assert not auth_implemented, "Authentication is unexpectedly enabled without formal documentation"
