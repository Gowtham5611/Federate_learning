"""Unit tests for Phase 13 ModelService."""

import pytest
import numpy as np
import torch

try:
    from interfaces.api.services.model_service import ModelService, MODEL_REGISTRY, ModelIntegrityError
    from interfaces.api.schemas.common import SequenceItem, FEATURE_DIMENSION, MAX_SEQUENCE_LENGTH
except ModuleNotFoundError:
    from backend.interfaces.api.services.model_service import ModelService, MODEL_REGISTRY, ModelIntegrityError
    from backend.interfaces.api.schemas.common import SequenceItem, FEATURE_DIMENSION, MAX_SEQUENCE_LENGTH


@pytest.fixture(scope="module")
def model_service():
    service = ModelService()
    service.load_all_models()
    return service


def test_model_loading_and_readiness(model_service):
    assert model_service.is_ready() is True
    loaded = model_service.get_loaded_models()
    assert "hi_small_lstm_fedavg_iid_seed42" in loaded
    assert "config_5_bilstm_fedavg_iid" in loaded


def test_model_metadata(model_service):
    meta_hi = model_service.get_model_metadata("HI-Small")
    assert meta_hi["architecture"] == "lstm"
    assert meta_hi["parameter_count"] == 31553
    assert meta_hi["frozen_threshold"] == 0.3377837836742401
    assert meta_hi["checkpoint_verified"] is True

    meta_li = model_service.get_model_metadata("LI-Medium")
    assert meta_li["architecture"] == "bilstm"
    assert meta_li["parameter_count"] == 63105
    assert meta_li["frozen_threshold"] == 0.2384829968214035
    assert meta_li["checkpoint_verified"] is True


def test_single_prediction_hi_small(model_service):
    # Create zero sequence
    seq = [[0.0] * FEATURE_DIMENSION for _ in range(MAX_SEQUENCE_LENGTH)]
    item = SequenceItem(features=seq, length=10)
    
    prob, thresh, dec, lat = model_service.predict_sequence("HI-Small", item)
    assert 0.0 <= prob <= 1.0
    assert thresh == 0.3377837836742401
    assert dec in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE")
    assert lat > 0.0


def test_single_prediction_li_medium(model_service):
    seq = [[0.0] * FEATURE_DIMENSION for _ in range(MAX_SEQUENCE_LENGTH)]
    item = SequenceItem(features=seq, length=10)
    
    prob, thresh, dec, lat = model_service.predict_sequence("LI-Medium", item)
    assert 0.0 <= prob <= 1.0
    assert thresh == 0.2384829968214035
    assert dec in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE")
    assert lat > 0.0


def test_batch_prediction_bounded(model_service):
    seq = [[0.0] * FEATURE_DIMENSION for _ in range(MAX_SEQUENCE_LENGTH)]
    items = [SequenceItem(features=seq, length=10, sample_id=f"ITEM_{i}") for i in range(16)]
    
    results, batch_lat = model_service.predict_batch("HI-Small", items)
    assert len(results) == 16
    assert batch_lat > 0.0
    for i, res in enumerate(results):
        assert res.index == i
        assert res.sample_id == f"ITEM_{i}"
        assert res.success is True
        assert 0.0 <= res.predicted_probability <= 1.0
        assert res.decision_threshold == 0.3377837836742401

