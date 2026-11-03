"""Pydantic v2 request and response schemas for FedSecureAI FastAPI service."""

from .common import (
    DecisionLabel,
    ExplanationMethod,
    MAX_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    FEATURE_DIMENSION,
    FROZEN_THRESHOLDS,
    SUPPORTED_MODELS_BY_DATASET,
    SequenceItem,
    SupportedDataset,
)
from .explain import (
    DriverAttribution,
    ExplainRequest,
    ExplainResponse,
)
from .health import (
    DatasetMetadataResponse,
    HealthResponse,
    ModelMetadataResponse,
)
from .predict import (
    BatchPredictItemResult,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

__all__ = [
    "DecisionLabel",
    "ExplanationMethod",
    "MAX_BATCH_SIZE",
    "MAX_SEQUENCE_LENGTH",
    "FEATURE_DIMENSION",
    "FROZEN_THRESHOLDS",
    "SUPPORTED_MODELS_BY_DATASET",
    "SequenceItem",
    "SupportedDataset",
    "HealthResponse",
    "ModelMetadataResponse",
    "DatasetMetadataResponse",
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "BatchPredictItemResult",
    "ExplainRequest",
    "ExplainResponse",
    "DriverAttribution",
]
