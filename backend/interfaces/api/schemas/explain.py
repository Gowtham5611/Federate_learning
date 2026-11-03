"""Schemas for Integrated Gradients transaction explanation endpoint."""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, model_validator

from .common import (
    DecisionLabel,
    ExplanationMethod,
    SequenceItem,
    SupportedDataset,
    SUPPORTED_MODELS_BY_DATASET,
)


class DriverAttribution(BaseModel):
    """Individual feature attribution driver."""
    feature_name: str = Field(..., description="Feature column identifier.")
    attribution: float = Field(..., description="Integrated Gradients signed attribution score.")
    feature_group: str = Field(..., description="Semantic macro feature group.")
    temporal_position: int | None = Field(None, description="Sequence window position if step-specific.")


class ExplainRequest(BaseModel):
    """Request payload for Integrated Gradients feature attribution.

    Attributes:
        dataset: Target dataset ('HI-Small' or 'LI-Medium').
        model_name: Model identifier (defaults to champion).
        sequence: Single sequence window of shape (L, 57).
        steps: Number of Riemann integration steps (10 <= steps <= 50, default 25).
        top_k: Top positive and negative drivers to extract (1 <= top_k <= 20, default 5).
    """
    dataset: SupportedDataset = Field(
        ...,
        description="Dataset identifier ('HI-Small' or 'LI-Medium').",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model identifier. Defaults to dataset champion.",
    )
    sequence: SequenceItem = Field(
        ...,
        description="Single sequence to explain.",
    )
    steps: int = Field(
        default=25,
        ge=10,
        le=50,
        description="Number of Riemann path integration steps (bounded 10..50).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top positive and negative drivers to return.",
    )

    @model_validator(mode="after")
    def validate_model_selection(self) -> ExplainRequest:
        expected_model = SUPPORTED_MODELS_BY_DATASET[self.dataset]
        if self.model_name is not None and self.model_name != expected_model:
            raise ValueError(
                f"Model '{self.model_name}' is not valid for dataset '{self.dataset}'. Expected '{expected_model}'."
            )
        if self.model_name is None:
            self.model_name = expected_model
        return self


class ExplainResponse(BaseModel):
    """Response payload for Integrated Gradients explanation."""
    request_id: str = Field(..., description="Unique explanation request identifier.")
    dataset: str = Field(..., description="Dataset evaluated.")
    model_name: str = Field(..., description="Model identifier used for attribution.")
    method: ExplanationMethod = Field(
        "integrated_gradients",
        description="Explainability algorithm: strictly 'integrated_gradients'.",
    )
    baseline_type: str = Field(
        ...,
        description="Description of the reference baseline distribution used.",
    )
    steps: int = Field(..., description="Number of integration steps computed.")
    predicted_probability: float = Field(..., description="Model predicted laundering probability.")
    decision_threshold: float = Field(..., description="Decision threshold applied.")
    decision: DecisionLabel = Field(..., description="Classification verdict.")
    top_positive_drivers: List[DriverAttribution] = Field(
        ...,
        description="Top features pushing probability towards laundering alert.",
    )
    top_negative_drivers: List[DriverAttribution] = Field(
        ...,
        description="Top features dampening risk towards legitimate clearance.",
    )
    completeness_residual: float = Field(
        ...,
        description="Absolute completeness residual |sum(IG) - (F(x) - F(x0))|.",
    )
    explanation_time_ms: float = Field(..., description="Attribution calculation latency in milliseconds.")

