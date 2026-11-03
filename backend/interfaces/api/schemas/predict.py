"""Schemas for single and bounded batch transaction prediction endpoints."""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from .common import (
    DecisionLabel,
    MAX_BATCH_SIZE,
    SequenceItem,
    SupportedDataset,
    SUPPORTED_MODELS_BY_DATASET,
)


class PredictRequest(BaseModel):
    """Request payload for single sequence AML prediction.

    Attributes:
        dataset: Target dataset ('HI-Small' or 'LI-Medium').
        model_name: Optional model identifier. If provided, must match dataset champion.
        sequence: Validated sequence of shape (L, 57).
    """
    dataset: SupportedDataset = Field(
        ...,
        description="Dataset identifier ('HI-Small' or 'LI-Medium').",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional explicit model name. Defaults to the verified champion for the dataset.",
    )
    sequence: SequenceItem = Field(
        ...,
        description="Single sequence item containing up to 10 transaction steps.",
    )

    @model_validator(mode="after")
    def validate_model_selection(self) -> PredictRequest:
        expected_model = SUPPORTED_MODELS_BY_DATASET[self.dataset]
        if self.model_name is not None and self.model_name != expected_model:
            raise ValueError(
                f"Model '{self.model_name}' is not valid for dataset '{self.dataset}'. Expected '{expected_model}'."
            )
        if self.model_name is None:
            self.model_name = expected_model
        return self


class PredictResponse(BaseModel):
    """Response payload for single sequence AML prediction."""
    request_id: str = Field(..., description="Unique non-sensitive request identifier.")
    dataset: str = Field(..., description="Dataset evaluated.")
    model_name: str = Field(..., description="Model identifier used for inference.")
    predicted_probability: float = Field(
        ...,
        description="Laundering probability in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    decision_threshold: float = Field(..., description="Frozen operational classification threshold.")
    decision: DecisionLabel = Field(
        ...,
        description="Classification verdict: 'FLAGGED_LAUNDERING' or 'CLEARED_LEGITIMATE'.",
    )
    sample_id: str | None = Field(None, description="Client-provided sample correlation ID if supplied.")
    sequence_length: int = Field(..., description="Causal transaction history length used in prediction.")
    inference_time_ms: float = Field(..., description="Model forward-pass latency in milliseconds.")


class BatchPredictRequest(BaseModel):
    """Request payload for bounded batch AML prediction (max 64 items)."""
    dataset: SupportedDataset = Field(
        ...,
        description="Dataset identifier ('HI-Small' or 'LI-Medium').",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model identifier. Defaults to dataset champion.",
    )
    items: List[SequenceItem] = Field(
        ...,
        description=f"Batch of transaction sequences. Max {MAX_BATCH_SIZE} items.",
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )

    @model_validator(mode="after")
    def validate_model_selection(self) -> BatchPredictRequest:
        expected_model = SUPPORTED_MODELS_BY_DATASET[self.dataset]
        if self.model_name is not None and self.model_name != expected_model:
            raise ValueError(
                f"Model '{self.model_name}' is not valid for dataset '{self.dataset}'. Expected '{expected_model}'."
            )
        if self.model_name is None:
            self.model_name = expected_model
        return self


class BatchPredictItemResult(BaseModel):
    """Individual item result within a batch prediction response."""
    index: int = Field(..., description="Zero-based index of item in submitted batch.")
    sample_id: str | None = Field(None, description="Client correlation ID if supplied.")
    success: bool = Field(..., description="Whether inference succeeded for this item.")
    predicted_probability: float | None = Field(None, description="Laundering probability if successful.")
    decision_threshold: float | None = Field(None, description="Decision threshold applied.")
    decision: DecisionLabel | None = Field(None, description="Classification verdict if successful.")
    error: str | None = Field(None, description="Sanitized error description if failed.")


class BatchPredictResponse(BaseModel):
    """Response payload for bounded batch AML prediction."""
    request_id: str = Field(..., description="Unique batch request identifier.")
    dataset: str = Field(..., description="Dataset evaluated.")
    model_name: str = Field(..., description="Model identifier used for inference.")
    total_items: int = Field(..., description="Total items submitted in batch.")
    successful_items: int = Field(..., description="Items successfully evaluated.")
    failed_items: int = Field(..., description="Items that failed validation or evaluation.")
    predictions: List[BatchPredictItemResult] = Field(..., description="Individual item predictions in order.")
    batch_inference_time_ms: float = Field(..., description="Total batch forward-pass time in milliseconds.")

