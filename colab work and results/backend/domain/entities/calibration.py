"""Domain entities and value objects for probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CalibrationMethod(StrEnum):
    """Supported probability calibration methods."""

    NONE = "none"
    TEMPERATURE = "temperature_scaling"
    PLATT = "platt_scaling"
    ISOTONIC = "isotonic_regression"


@dataclass(frozen=True)
class CalibrationParameters:
    """Fitted calibration parameters for reproducibility and serialization."""

    method: CalibrationMethod
    temperature: float | None = None
    platt_a: float | None = None
    platt_b: float | None = None
    isotonic_x_min: float | None = None
    isotonic_x_max: float | None = None
    isotonic_thresholds_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "method": self.method.value,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_x_min": self.isotonic_x_min,
            "isotonic_x_max": self.isotonic_x_max,
            "isotonic_thresholds_count": self.isotonic_thresholds_count,
        }

