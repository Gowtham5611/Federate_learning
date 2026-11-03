"""Explainability service for FedSecureAI FastAPI layer using Integrated Gradients."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid

import numpy as np
import torch

try:
    from infrastructure.explainability.integrated_gradients import SequenceIntegratedGradients
    from infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
    from interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from interfaces.api.schemas.explain import DriverAttribution, ExplainRequest, ExplainResponse
except ModuleNotFoundError:
    from backend.infrastructure.explainability.integrated_gradients import SequenceIntegratedGradients
    from backend.infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
    from backend.interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from backend.interfaces.api.schemas.explain import DriverAttribution, ExplainRequest, ExplainResponse
from .model_service import ModelService

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[4]


def _get_feature_group(name: str) -> str:
    """Categorize feature name into a semantic macro feature group."""
    if "amount" in name or "same_" in name or "self_" in name:
        return "Transaction Attributes"
    elif "format" in name:
        return "Payment Format"
    elif "currency" in name:
        return "Currency Rails"
    elif "minute" in name or "day_" in name or "weekend" in name or "night" in name:
        return "Temporal Dynamics"
    elif "sender_" in name:
        return "Sender Causal History"
    elif "receiver_" in name:
        return "Receiver Causal History"
    return "Other"


class ExplanationService:
    """Service executing Integrated Gradients with bounded compute limits."""

    _instance: ExplanationService | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ExplanationService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._feature_names: Dict[str, List[str]] = {}
        self._explainers: Dict[str, SequenceIntegratedGradients] = {}
        self._explain_lock = threading.Lock()
        self._initialized = True

    def _get_names(self, dataset: str) -> List[str]:
        if dataset not in self._feature_names:
            data_dir_env = os.getenv("FEDSECUREAI_DATA_DIR")
            if data_dir_env:
                data_dir = Path(data_dir_env) / f"processed/ibm_aml/{dataset}"
                if not data_dir.exists():
                    data_dir = Path(data_dir_env) / dataset
            else:
                data_dir = BASE_DIR / f"data/processed/ibm_aml/{dataset}"
            meta = ProcessedFeatureMetadata.read(data_dir)
            expansions = {nominal.column: nominal for nominal in meta.nominal_columns}
            names = []
            for col in meta.feature_columns:
                if col in expansions:
                    nom = expansions[col]
                    for i in range(nom.vocabulary_size):
                        names.append(f"{col}__cat_{i}")
                else:
                    names.append(col)
            self._feature_names[dataset] = names
        return self._feature_names[dataset]

    def _get_explainer(self, dataset: str) -> SequenceIntegratedGradients:
        if dataset not in self._explainers:
            model_service = ModelService()
            model = model_service.get_model(dataset)
            self._explainers[dataset] = SequenceIntegratedGradients(model, target="logit")
        return self._explainers[dataset]

    def explain_sequence(self, request: ExplainRequest) -> ExplainResponse:
        """Compute Integrated Gradients attributions for a single request sequence."""
        dataset = request.dataset
        seq_item = request.sequence
        steps = request.steps
        top_k = request.top_k
        model_name = request.model_name

        names = self._get_names(dataset)
        threshold = FROZEN_THRESHOLDS[dataset]

        # Prepare (1, 10, 57) float tensor
        feat_arr = np.zeros((1, MAX_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        seq_len = seq_item.length or len(seq_item.features)
        for t_idx, step in enumerate(seq_item.features):
            feat_arr[0, t_idx, :] = np.array(step, dtype=np.float32)

        x_tensor = torch.from_numpy(feat_arr)
        l_tensor = torch.tensor([seq_len], dtype=torch.int64)

        t0 = time.perf_counter()
        with self._explain_lock:
            explainer = self._get_explainer(dataset)
            # Default zero baseline with length masking
            attributions, diag = explainer.attribute(x_tensor, l_tensor, steps=steps)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Model predicted probability
        logit = float(diag["f_input"][0])
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        decision = "FLAGGED_LAUNDERING" if prob >= threshold else "CLEARED_LEGITIMATE"
        abs_res = float(diag["abs_residuals"][0])

        # Aggregate attribution across time steps: (57,)
        feature_attrs = attributions[0].sum(dim=0).cpu().numpy()

        # Extract top positive (pushing toward alert) and top negative (pushing toward clearance)
        pos_sorted_indices = np.argsort(-feature_attrs)
        neg_sorted_indices = np.argsort(feature_attrs)

        top_pos: List[DriverAttribution] = []
        for idx in pos_sorted_indices:
            val = float(feature_attrs[idx])
            if len(top_pos) >= top_k:
                break
            fname = names[idx]
            top_pos.append(
                DriverAttribution(
                    feature_name=fname,
                    attribution=val,
                    feature_group=_get_feature_group(fname),
                )
            )

        top_neg: List[DriverAttribution] = []
        for idx in neg_sorted_indices:
            val = float(feature_attrs[idx])
            if len(top_neg) >= top_k:
                break
            fname = names[idx]
            top_neg.append(
                DriverAttribution(
                    feature_name=fname,
                    attribution=val,
                    feature_group=_get_feature_group(fname),
                )
            )

        is_hist_suff = bool(seq_len >= 3)
        hist_warn = None if is_hist_suff else (
            f"Insufficient account sequence history ({seq_len}/10 steps). Attributions reflect isolated static "
            "features and padding priors rather than learned sequential laundering dynamics."
        )

        return ExplainResponse(
            request_id=f"exp_{uuid.uuid4().hex[:12]}",
            dataset=dataset,
            model_name=model_name or "champion",
            method="integrated_gradients",
            baseline_type="zero_sequence_with_length_masking",
            steps=steps,
            predicted_probability=prob,
            decision_threshold=threshold,
            decision=decision,
            top_positive_drivers=top_pos,
            top_negative_drivers=top_neg,
            completeness_residual=abs_res,
            explanation_time_ms=latency_ms,
            is_history_sufficient=is_hist_suff,
            history_warning=hist_warn,
        )
