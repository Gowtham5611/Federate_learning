"""Schemas for Integrated Gradients transaction explanation endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal
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
    """Request payload for model-aware feature attribution (Integrated Gradients or Tree SHAP).

    Attributes:
        dataset: Target dataset ('HI-Small' or 'LI-Medium').
        model_name: Optional model identifier.
        sequence: Single sequence window of shape (L, 57) for recurrent models.
        tabular_features: Dict or vector of 23 engineered features for tabular fallback.
        prediction_id: Optional cached prediction ID to look up features on demand.
        row_id: Optional 1-based row ID within cached prediction batch.
        transaction_id: Optional transaction ID within cached prediction batch.
        steps: Number of Riemann integration steps (10 <= steps <= 50, default 25).
        top_k: Top positive and negative drivers to extract (1 <= top_k <= 20, default 5).
    """
    dataset: SupportedDataset = Field(
        default="HI-Small",
        description="Dataset identifier ('HI-Small' or 'LI-Medium').",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model identifier. Defaults to dataset champion or tabular fallback.",
    )
    sequence: SequenceItem | None = Field(
        default=None,
        description="Single sequence to explain (for recurrent models).",
    )
    tabular_features: Dict[str, float] | List[float] | None = Field(
        default=None,
        description="Tabular feature vector or mapping of 23 tabular features (for LightGBM fallback).",
    )
    prediction_id: str | None = Field(
        default=None,
        description="Cached prediction batch identifier from CSV evaluation.",
    )
    transaction_id: str | None = Field(
        default=None,
        description="Transaction ID within cached prediction batch.",
    )
    row_id: int | None = Field(
        default=None,
        description="1-based row index within cached prediction batch.",
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
    threshold: float | None = Field(
        default=None,
        description="Active decision threshold override.",
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> ExplainRequest:
        has_seq = self.sequence is not None and getattr(self.sequence, "features", None) is not None and len(self.sequence.features) > 0
        has_tab = self.tabular_features is not None
        has_lookup = bool(self.prediction_id and (self.row_id is not None or self.transaction_id is not None))

        if not (has_seq or has_tab or has_lookup):
            if self.sequence is not None:
                raise ValueError("Sequence features cannot be empty. Must provide at least 1 feature timestep.")
            raise ValueError("Must provide either 'sequence', 'tabular_features', or 'prediction_id' with 'row_id'/'transaction_id'.")

        if self.model_name is None:
            if has_tab or (not has_seq and has_lookup):
                self.model_name = "lightgbm_tabular_fallback"
            else:
                self.model_name = SUPPORTED_MODELS_BY_DATASET.get(self.dataset, "centralized_lstm")

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
    is_history_sufficient: bool = Field(default=True, description="Whether sequence has sufficient account history.")
    history_warning: str | None = Field(default=None, description="Warning if transaction sequence has insufficient history depth (<3 steps).")

