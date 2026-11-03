"""Common enumerations and validation utilities for FedSecureAI API schemas."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

SupportedDataset = Literal["HI-Small", "LI-Medium"]
DecisionLabel = Literal["FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE"]
ExplanationMethod = Literal["integrated_gradients"]

SUPPORTED_MODELS_BY_DATASET: dict[str, str] = {
    "HI-Small": "hi_small_lstm_fedavg_iid_seed42",
    "LI-Medium": "config_5_bilstm_fedavg_iid",
}

try:
    from config.threshold_config import get_canonical_threshold
except ModuleNotFoundError:
    try:
        from backend.config.threshold_config import get_canonical_threshold
    except ModuleNotFoundError:
        def get_canonical_threshold(d: str) -> float:
            return 0.337784 if "hi" in d.lower() else 0.238483

FROZEN_THRESHOLDS: dict[str, float] = {
    "HI-Small": get_canonical_threshold("HI-Small"),
    "LI-Medium": get_canonical_threshold("LI-Medium"),
}

FEATURE_DIMENSION: int = 57
MAX_SEQUENCE_LENGTH: int = 10
MAX_BATCH_SIZE: int = 64


def validate_finite_floats(values: List[float], field_name: str) -> None:
    """Ensure every float in the list is finite and not NaN."""
    for idx, val in enumerate(values):
        if not isinstance(val, (int, float)):
            raise ValueError(f"{field_name} at index {idx} must be a numeric float, got {type(val).__name__}")
        if math.isnan(val):
            raise ValueError(f"{field_name} at index {idx} contains NaN, which is prohibited.")
        if math.isinf(val):
            raise ValueError(f"{field_name} at index {idx} contains infinite value, which is prohibited.")


class SequenceItem(BaseModel):
    """Represents one transaction sequence window for AML classification.

    Attributes:
        features: 2D array of shape (sequence_length, 57), where 1 <= sequence_length <= 10.
        length: True causal length (1 <= length <= 10). If omitted, inferred from len(features).
        sample_id: Optional client-side non-sensitive identifier for request correlation.
    """
    features: List[List[float]] = Field(
        ...,
        description="Transaction sequence features of shape (L, 57) where 1 <= L <= 10.",
    )
    length: int | None = Field(
        default=None,
        description="True sequence length (1 <= L <= 10). If None, inferred as len(features).",
        ge=1,
        le=10,
    )
    sample_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional non-sensitive transaction correlation identifier.",
    )

    @field_validator("features")
    @classmethod
    def validate_features_shape_and_values(cls, val: List[List[float]]) -> List[List[float]]:
        if not val:
            raise ValueError("Features sequence cannot be empty.")
        if len(val) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence length {len(val)} exceeds maximum allowed window of {MAX_SEQUENCE_LENGTH}.")
        
        for step_idx, step_features in enumerate(val):
            if len(step_features) != FEATURE_DIMENSION:
                raise ValueError(
                    f"Time step {step_idx} has {len(step_features)} features; exactly {FEATURE_DIMENSION} required."
                )
            validate_finite_floats(step_features, f"Step {step_idx}")

        return val

    @model_validator(mode="after")
    def validate_and_normalize_length(self) -> SequenceItem:
        if self.length is None:
            self.length = len(self.features)
        elif self.length > len(self.features):
            raise ValueError(
                f"Declared length {self.length} cannot exceed feature steps provided ({len(self.features)})."
            )
        
        # Verify right-padding: if padded to length 10, steps >= length must be all zeros
        if len(self.features) > self.length:
            for step_idx in range(self.length, len(self.features)):
                pad_step = self.features[step_idx]
                if any(v != 0.0 for v in pad_step):
                    raise ValueError(
                        f"Step {step_idx} is beyond declared sequence length {self.length} but contains non-zero values. "
                        "Padded positions must be zero-filled."
                    )
        return self

