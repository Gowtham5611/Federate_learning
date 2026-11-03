"""Schemas for health, service status, and model metadata endpoints."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness and readiness status of the inference service."""
    status: str = Field(..., description="Service status ('ok' or 'degraded').")
    app_name: str = Field("FedSecureAI Inference Service", description="Application name.")
    app_version: str = Field("1.0.0", description="Application version.")
    loaded_models: List[str] = Field(..., description="List of verified and loaded model identifiers.")
    device: str = Field("cpu", description="PyTorch execution device.")
    readiness: str = Field(..., description="Readiness state ('ready' or 'initializing').")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ModelMetadataResponse(BaseModel):
    """Metadata describing a single verified model."""
    dataset: str = Field(..., description="Target AML dataset ('HI-Small' or 'LI-Medium').")
    model_name: str = Field(..., description="Authoritative model identifier.")
    architecture: str = Field(..., description="Recurrent neural network architecture.")
    parameter_count: int = Field(..., description="Number of trainable weights.")
    sequence_length: int = Field(10, description="Temporal window capacity.")
    feature_dimension: int = Field(57, description="Expanded feature dimension.")
    frozen_threshold: float = Field(..., description="Frozen operational decision threshold.")
    checkpoint_verified: bool = Field(True, description="Whether checkpoint SHA-256 is verified.")
    supported_explanation_methods: List[str] = Field(
        default=["integrated_gradients"],
        description="Supported explainability algorithms.",
    )


class DatasetMetadataResponse(BaseModel):
    """Detailed feature schema and preprocessing requirements for a dataset."""
    dataset: str = Field(..., description="Dataset name.")
    feature_columns: List[str] = Field(..., description="List of all 57 expanded model feature names in exact order.")
    raw_feature_count: int = Field(23, description="Count of raw transaction features before one-hot expansion.")
    expanded_feature_count: int = Field(57, description="Total float32 features handed to model.")
    sequence_length: int = Field(10, description="Max sequence length.")
    frozen_threshold: float = Field(..., description="Validation-calibrated decision threshold.")
    active_model_name: str = Field(..., description="Champion model associated with this dataset.")
    preprocessing_version: str = Field("1.0.0", description="Preprocessing pipeline version.")

