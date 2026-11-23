"""Unit tests for Phase 13 Pydantic schemas."""

import math
import pytest
from pydantic import ValidationError

try:
    from interfaces.api.schemas import (
        SequenceItem,
        PredictRequest,
        PredictResponse,
        BatchPredictRequest,
        BatchPredictResponse,
        ExplainRequest,
        ExplainResponse,
        HealthResponse,
        ModelMetadataResponse,
        DatasetMetadataResponse,
        FEATURE_DIMENSION,
        MAX_SEQUENCE_LENGTH,
        MAX_BATCH_SIZE,
    )
except ModuleNotFoundError:
    from backend.interfaces.api.schemas import (
        SequenceItem,
        PredictRequest,
        PredictResponse,
        BatchPredictRequest,
        BatchPredictResponse,
        ExplainRequest,
        ExplainResponse,
        HealthResponse,
        ModelMetadataResponse,
        DatasetMetadataResponse,
        FEATURE_DIMENSION,
        MAX_SEQUENCE_LENGTH,
        MAX_BATCH_SIZE,
    )

def make_valid_sequence(length=10, pad_to=10):
    seq = [[1.0] * FEATURE_DIMENSION for _ in range(length)]
    if pad_to > length:
        for _ in range(pad_to - length):
            seq.append([0.0] * FEATURE_DIMENSION)
    return seq

# 1. Valid SequenceItem creation
def test_valid_sequence_item():
    seq = make_valid_sequence(length=5, pad_to=10)
    item = SequenceItem(features=seq, length=5, sample_id="S001")
    assert item.length == 5
    assert len(item.features) == 10
    assert item.sample_id == "S001"

    # Inferred length when not provided
    seq_unpadded = make_valid_sequence(length=7, pad_to=7)
    item2 = SequenceItem(features=seq_unpadded)
    assert item2.length == 7

# 2. Rejection of invalid feature dimension
def test_sequence_item_invalid_dimension():
    # 56 features instead of 57
    bad_seq = [[1.0] * (FEATURE_DIMENSION - 1)]
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=bad_seq)
    assert "exactly 57 required" in str(exc.value)

# 3. Rejection of sequence longer than 10
def test_sequence_item_too_long():
    too_long = [[1.0] * FEATURE_DIMENSION for _ in range(11)]
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=too_long)
    assert "exceeds maximum allowed window" in str(exc.value)

# 4. Rejection of NaN and Infinite values
def test_sequence_item_nan_and_inf():
    nan_seq = make_valid_sequence(length=2)
    nan_seq[0][0] = float("nan")
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=nan_seq)
    assert "contains NaN" in str(exc.value)

    inf_seq = make_valid_sequence(length=2)
    inf_seq[1][5] = float("inf")
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=inf_seq)
    assert "contains infinite value" in str(exc.value)

# 5. Rejection of non-zero padding values
def test_sequence_item_non_zero_padding():
    seq = make_valid_sequence(length=5, pad_to=10)
    # Put non-zero value at step 6 (index 6, which is > length 5)
    seq[6][10] = 0.5
    with pytest.raises(ValidationError) as exc:
        SequenceItem(features=seq, length=5)
    assert "contains non-zero values" in str(exc.value)

# 6. PredictRequest model validation
def test_predict_request_validation():
    seq = make_valid_sequence(length=10)
    item = SequenceItem(features=seq)
    
    # Valid default model assignment
    req_hi = PredictRequest(dataset="HI-Small", sequence=item)
    assert req_hi.model_name == "hi_small_lstm_fedavg_iid_seed42"

    req_li = PredictRequest(dataset="LI-Medium", sequence=item)
    assert req_li.model_name == "config_5_bilstm_fedavg_iid"

    # Rejection of mismatched model
    with pytest.raises(ValidationError) as exc:
        PredictRequest(dataset="HI-Small", model_name="config_5_bilstm_fedavg_iid", sequence=item)
    assert "not valid for dataset 'HI-Small'" in str(exc.value)

# 7. BatchPredictRequest batch size bounds
def test_batch_predict_request_bounds():
    seq = make_valid_sequence(length=5)
    item = SequenceItem(features=seq)

    # Empty batch rejected
    with pytest.raises(ValidationError):
        BatchPredictRequest(dataset="HI-Small", items=[])

    # Valid batch within limit
    valid_batch = [item] * 10
    req = BatchPredictRequest(dataset="HI-Small", items=valid_batch)
    assert len(req.items) == 10

    # Over 64 rejected
    too_large = [item] * 65
    with pytest.raises(ValidationError) as exc:
        BatchPredictRequest(dataset="HI-Small", items=too_large)
    assert "List should have at most 64 items" in str(exc.value)

# 8. ExplainRequest validation bounds
def test_explain_request_bounds():
    seq = make_valid_sequence(length=10)
    item = SequenceItem(features=seq)

    req = ExplainRequest(dataset="HI-Small", sequence=item, steps=25, top_k=5)
    assert req.steps == 25
    assert req.top_k == 5

    # Steps < 10 rejected
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=item, steps=5)

    # Steps > 50 rejected
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=item, steps=60)

    # top_k > 20 rejected
    with pytest.raises(ValidationError):
        ExplainRequest(dataset="HI-Small", sequence=item, top_k=25)

# 9. Health and Metadata responses
def test_metadata_responses():
    health = HealthResponse(
        status="ok",
        app_name="FedSecureAI Inference Service",
        app_version="1.0.0",
        loaded_models=["hi_small_lstm_fedavg_iid_seed42"],
        device="cpu",
        readiness="ready",
        timestamp="2026-10-26T23:00:00Z"
    )
    assert health.status == "ok"
    assert health.readiness == "ready"

    meta = ModelMetadataResponse(
        dataset="HI-Small",
        model_name="hi_small_lstm_fedavg_iid_seed42",
        architecture="LSTM",
        parameter_count=31553,
        sequence_length=10,
        feature_dimension=57,
        frozen_threshold=0.3377837836742401,
        checkpoint_verified=True,
        supported_explanation_methods=["integrated_gradients"]
    )
    assert meta.parameter_count == 31553
    assert meta.supported_explanation_methods == ["integrated_gradients"]
