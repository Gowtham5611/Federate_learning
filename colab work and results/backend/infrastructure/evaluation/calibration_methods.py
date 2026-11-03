"""Calibration algorithms and estimators: Temperature, Platt, and Isotonic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from domain.entities.calibration import CalibrationMethod, CalibrationParameters

logger = logging.getLogger(__name__)

PROBABILITY_EPSILON = 1e-12


def clip_probabilities(
    p: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip probabilities to (eps, 1 - eps) to prevent log(0) and logit overflows."""
    clipped = np.clip(
        np.asarray(p, dtype=np.float64),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return np.asarray(clipped, dtype=np.float64)


def probabilities_to_logits(
    p: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert probabilities to numerically stable logits."""
    p_clipped = clip_probabilities(p)
    return np.log(p_clipped / (1.0 - p_clipped))


def logits_to_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float32]:
    """Convert logits to probabilities via sigmoid with overflow protection."""
    clipped_logits = np.clip(logits, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-clipped_logits))
    return np.asarray(probs, dtype=np.float32)


class BaseCalibrator(ABC):
    """Abstract base class for all post-processing probability calibrators."""

    @property
    @abstractmethod
    def method(self) -> CalibrationMethod:
        """Calibration method name."""
        ...

    @abstractmethod
    def fit(
        self,
        y_true: NDArray[np.float32],
        p_pred: NDArray[np.float32],
    ) -> BaseCalibrator:
        """Fit calibration mapping using validation data only."""
        ...

    @abstractmethod
    def calibrate(self, p_pred: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply fitted calibration mapping to uncalibrated probabilities."""
        ...

    @abstractmethod
    def get_parameters(self) -> CalibrationParameters:
        """Return fitted parameter snapshot."""
        ...


class IdentityCalibrator(BaseCalibrator):
    """Uncalibrated baseline calibrator (returns probabilities unchanged)."""

    @property
    def method(self) -> CalibrationMethod:
        """Return NONE method."""
        return CalibrationMethod.NONE

    def fit(
        self,
        y_true: NDArray[np.float32],
        p_pred: NDArray[np.float32],
    ) -> IdentityCalibrator:
        """No-op fitting for baseline."""
        del y_true, p_pred
        return self

    def calibrate(self, p_pred: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return uncalibrated input unchanged."""
        return np.asarray(p_pred, dtype=np.float32)

    def get_parameters(self) -> CalibrationParameters:
        """Return empty parameters."""
        return CalibrationParameters(method=CalibrationMethod.NONE)


class TemperatureScalingCalibrator(BaseCalibrator):
    """Calibrates probabilities by scaling logits by a single learned temperature T > 0."""

    def __init__(self) -> None:
        """Initialize temperature to 1.0."""
        self._temperature: float = 1.0

    @property
    def method(self) -> CalibrationMethod:
        """Return TEMPERATURE method."""
        return CalibrationMethod.TEMPERATURE

    @property
    def temperature(self) -> float:
        """Return learned temperature."""
        return self._temperature

    def fit(
        self,
        y_true: NDArray[np.float32],
        p_pred: NDArray[np.float32],
    ) -> TemperatureScalingCalibrator:
        """Fit temperature T by minimizing negative log-likelihood on validation data."""
        y = np.asarray(y_true, dtype=np.float64)
        logits = probabilities_to_logits(p_pred)

        def nll_loss(param: NDArray[np.float64]) -> float:
            temp = float(param[0])
            scaled_logits = np.clip(logits / temp, -50.0, 50.0)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs_clipped = np.clip(probs, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
            loss = -np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped))
            return float(loss)

        res = scipy.optimize.minimize(
            nll_loss,
            x0=np.array([1.0]),
            bounds=[(1e-3, 50.0)],
            method="L-BFGS-B",
        )
        self._temperature = float(res.x[0])
        logger.info(
            "Temperature Scaling fitted: T=%.4f (converged=%s)",
            self._temperature,
            res.success,
        )
        return self

    def calibrate(self, p_pred: NDArray[np.float32]) -> NDArray[np.float32]:
        """Scale logits by temperature and return calibrated probabilities."""
        logits = probabilities_to_logits(p_pred)
        scaled_logits = logits / self._temperature
        return logits_to_probabilities(scaled_logits)

    def get_parameters(self) -> CalibrationParameters:
        """Return fitted temperature parameter."""
        return CalibrationParameters(
            method=CalibrationMethod.TEMPERATURE,
            temperature=self._temperature,
        )


class PlattScalingCalibrator(BaseCalibrator):
    """Logistic calibration: fits an affine transformation a * logit + b on logits."""

    def __init__(self) -> None:
        """Initialize parameters a=1.0 and b=0.0."""
        self._a: float = 1.0
        self._b: float = 0.0

    @property
    def method(self) -> CalibrationMethod:
        """Return PLATT method."""
        return CalibrationMethod.PLATT

    @property
    def a(self) -> float:
        """Return slope parameter a."""
        return self._a

    @property
    def b(self) -> float:
        """Return intercept parameter b."""
        return self._b

    def fit(
        self,
        y_true: NDArray[np.float32],
        p_pred: NDArray[np.float32],
    ) -> PlattScalingCalibrator:
        """Fit a and b by minimizing negative log-likelihood on validation data."""
        y = np.asarray(y_true, dtype=np.float64)
        logits = probabilities_to_logits(p_pred)

        def nll_loss(params: NDArray[np.float64]) -> float:
            a, b = float(params[0]), float(params[1])
            transformed = np.clip(a * logits + b, -50.0, 50.0)
            probs = 1.0 / (1.0 + np.exp(-transformed))
            probs_clipped = np.clip(probs, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
            loss = -np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped))
            return float(loss)

        res = scipy.optimize.minimize(
            nll_loss,
            x0=np.array([1.0, 0.0]),
            bounds=[(1e-4, 50.0), (-50.0, 50.0)],
            method="L-BFGS-B",
        )
        self._a = float(res.x[0])
        self._b = float(res.x[1])
        logger.info(
            "Platt Scaling fitted: a=%.4f, b=%.4f (converged=%s)",
            self._a,
            self._b,
            res.success,
        )
        return self

    def calibrate(self, p_pred: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply affine transformation to logits and map back via sigmoid."""
        logits = probabilities_to_logits(p_pred)
        transformed = self._a * logits + self._b
        return logits_to_probabilities(transformed)

    def get_parameters(self) -> CalibrationParameters:
        """Return fitted Platt scaling parameters."""
        return CalibrationParameters(
            method=CalibrationMethod.PLATT,
            platt_a=self._a,
            platt_b=self._b,
        )


class IsotonicRegressionCalibrator(BaseCalibrator):
    """Non-parametric monotonic step function calibration using Isotonic Regression."""

    def __init__(self) -> None:
        """Initialize calibrator with uninstantiated model."""
        self._model: IsotonicRegression | None = None

    @property
    def method(self) -> CalibrationMethod:
        """Return ISOTONIC method."""
        return CalibrationMethod.ISOTONIC

    def fit(
        self,
        y_true: NDArray[np.float32],
        p_pred: NDArray[np.float32],
    ) -> IsotonicRegressionCalibrator:
        """Fit isotonic regression on validation probabilities and labels."""
        y = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(p_pred, dtype=np.float64)

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p, y)
        self._model = ir
        logger.info("Isotonic Regression fitted: threshold count=%d", len(ir.X_thresholds_))
        return self

    def calibrate(self, p_pred: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict calibrated probabilities using fitted isotonic step function."""
        if self._model is None:
            raise RuntimeError("IsotonicRegressionCalibrator must be fitted before calibrate()")
        p = np.asarray(p_pred, dtype=np.float64)
        calibrated = self._model.predict(p)
        calibrated = np.clip(np.nan_to_num(calibrated, nan=0.0), 0.0, 1.0)
        return np.asarray(calibrated, dtype=np.float32)

    def get_parameters(self) -> CalibrationParameters:
        """Return fitted isotonic parameters."""
        if self._model is None:
            return CalibrationParameters(method=CalibrationMethod.ISOTONIC)
        return CalibrationParameters(
            method=CalibrationMethod.ISOTONIC,
            isotonic_x_min=float(self._model.X_min_),
            isotonic_x_max=float(self._model.X_max_),
            isotonic_thresholds_count=len(self._model.X_thresholds_),
        )


def build_calibrator(method: CalibrationMethod) -> BaseCalibrator:
    """Factory creating uncalibrated calibrator instance for a given method."""
    if method == CalibrationMethod.NONE:
        return IdentityCalibrator()
    if method == CalibrationMethod.TEMPERATURE:
        return TemperatureScalingCalibrator()
    if method == CalibrationMethod.PLATT:
        return PlattScalingCalibrator()
    if method == CalibrationMethod.ISOTONIC:
        return IsotonicRegressionCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")
