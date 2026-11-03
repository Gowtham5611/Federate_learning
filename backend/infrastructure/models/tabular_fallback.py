"""Tabular Fallback Model using LightGBM for FedSecureAI.

Provides an explicit fallback path for evaluating single-transaction rows or cold-start
accounts lacking multi-step sequence history.
CRITICAL CONSTRAINT: This model is used ONLY upon explicit user opt-in (fallback_model='tabular').
It is NEVER silently mixed or substituted into recurrent sequence models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "baselines" / "lightgbm_baseline.txt"

FEATURE_COLS = [
    "amount_paid_log", "amount_received_log", "amount_mismatch",
    "same_currency", "same_bank", "self_transfer", "payment_format_code",
    "payment_currency_code", "receiving_currency_code", "minute_of_day_sin",
    "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "is_night", "sender_prior_count_log", "sender_prior_amount_log",
    "sender_seconds_since_prior_log", "sender_has_prior",
    "receiver_prior_count_log", "receiver_prior_amount_log",
    "receiver_seconds_since_prior_log", "receiver_has_prior",
]


class TabularFallbackModel:
    """Explicit tabular fallback estimator for zero-history / single-transaction rows."""

    MODEL_ID = "lightgbm_tabular_fallback"
    DISPLAY_NAME = "Fallback: no sequence history (LightGBM)"

    def __init__(self, model_file: Optional[Path] = None) -> None:
        self.model_file = Path(model_file) if model_file else MODEL_PATH
        self.booster = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_file.exists():
            logger.warning("Tabular fallback model file not found at %s", self.model_file)
            return

        try:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(self.model_file))
            logger.info("Successfully loaded TabularFallbackModel from %s", self.model_file)
        except Exception as e:
            logger.error("Failed to load TabularFallbackModel: %s", e)

    @property
    def is_available(self) -> bool:
        return self.booster is not None

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Predict laundering probabilities from tabular features.
        
        Args:
            df_features: DataFrame containing the 23 engineered tabular features.
            
        Returns:
            1D numpy array of probabilities.
        """
        if self.booster is None:
            self._load_model()
            if self.booster is None:
                raise RuntimeError(f"Tabular fallback model is unavailable at {self.model_file}")

        # Ensure all columns present
        missing = [col for col in FEATURE_COLS if col not in df_features.columns]
        if missing:
            raise ValueError(f"Missing required tabular features: {missing}")

        X = df_features[FEATURE_COLS].to_numpy(dtype=np.float32)
        probs = self.booster.predict(X)
        return probs.astype(np.float64)
