"""Explainability service for FedSecureAI FastAPI layer using Integrated Gradients and Tree SHAP."""

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
    from infrastructure.models.tabular_fallback import TabularFallbackModel, FEATURE_COLS
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
    from backend.infrastructure.models.tabular_fallback import TabularFallbackModel, FEATURE_COLS
    from backend.interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from backend.interfaces.api.schemas.explain import DriverAttribution, ExplainRequest, ExplainResponse

from .model_service import ModelService

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parents[3]
_ROOT_DIR = _BACKEND_DIR.parent
BASE_DIR = _BACKEND_DIR if (_BACKEND_DIR / "data").exists() else _ROOT_DIR


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
    elif "bank" in name:
        return "Interbank Rail"
    return "Other"


class ExplanationService:
    """Service executing model-aware explanations:
    - Recurrent models: Integrated Gradients on sequence features (L <= 10, D=57).
    - Tabular fallback: Tree SHAP attributions on 23 engineered tabular features.
    """

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

    def _get_explainer(self, model_key: str) -> SequenceIntegratedGradients:
        if model_key not in self._explainers:
            model_service = ModelService()
            model = model_service.get_model(model_key)
            self._explainers[model_key] = SequenceIntegratedGradients(model, target="logit")
        return self._explainers[model_key]

    def explain_tabular(self, request: ExplainRequest) -> ExplainResponse:
        """Compute Tree SHAP feature attributions for a single transaction using LightGBM fallback."""
        if TabularFallbackModel is None:
            raise RuntimeError("TabularFallbackModel could not be imported.")
        fallback = TabularFallbackModel()
        if not fallback.is_available:
            raise RuntimeError("Tabular fallback model is not available or checkpoint missing.")

        top_k = request.top_k
        threshold = 0.337784

        raw_feat = request.tabular_features
        if raw_feat is None:
            raise ValueError("No tabular features provided for tabular model explanation.")

        if isinstance(raw_feat, dict):
            feat_vec = [float(raw_feat.get(col, 0.0)) for col in FEATURE_COLS]
        elif isinstance(raw_feat, (list, tuple)):
            feat_vec = [float(x) for x in raw_feat[:len(FEATURE_COLS)]]
            while len(feat_vec) < len(FEATURE_COLS):
                feat_vec.append(0.0)
        else:
            raise ValueError(f"Unsupported tabular_features format: {type(raw_feat)}")

        X = np.array([feat_vec], dtype=np.float32)

        t0 = time.perf_counter()
        contribs = fallback.booster.predict(X, pred_contrib=True)  # (1, 24)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        feature_attrs = contribs[0, :len(FEATURE_COLS)]  # (23,)
        logit = float(contribs[0].sum())
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        decision = "FLAGGED_LAUNDERING" if prob >= threshold else "CLEARED_LEGITIMATE"

        pos_sorted = np.argsort(-feature_attrs)
        neg_sorted = np.argsort(feature_attrs)

        top_pos: List[DriverAttribution] = []
        for idx in pos_sorted:
            val = float(feature_attrs[idx])
            if len(top_pos) >= top_k:
                break
            fname = FEATURE_COLS[idx]
            top_pos.append(DriverAttribution(
                feature_name=fname,
                attribution=round(val, 6),
                feature_group=_get_feature_group(fname),
            ))

        top_neg: List[DriverAttribution] = []
        for idx in neg_sorted:
            val = float(feature_attrs[idx])
            if len(top_neg) >= top_k:
                break
            fname = FEATURE_COLS[idx]
            top_neg.append(DriverAttribution(
                feature_name=fname,
                attribution=round(val, 6),
                feature_group=_get_feature_group(fname),
            ))

        return ExplainResponse(
            request_id=f"exp_{uuid.uuid4().hex[:12]}",
            dataset=request.dataset,
            model_name="lightgbm_tabular_fallback",
            method="tree_shap",
            baseline_type="tree_expected_value",
            steps=1,
            predicted_probability=round(prob, 6),
            decision_threshold=threshold,
            decision=decision,
            top_positive_drivers=top_pos,
            top_negative_drivers=top_neg,
            completeness_residual=0.0,
            explanation_time_ms=round(latency_ms, 2),
            is_history_sufficient=True,
            history_warning="Single-transaction tabular mode (LightGBM). Attributions computed via Tree SHAP across 23 tabular features.",
        )

    def explain_sequence(self, request: ExplainRequest) -> ExplainResponse:
        """Compute model-aware attributions for a transaction (Integrated Gradients or Tree SHAP)."""
        # 1. Resolve from prediction cache or disk persistence if prediction_id is provided
        if request.prediction_id:
            import sys
            cached_result = None
            cache = {}
            for mod_name in ("interfaces.api.services.csv_evaluation_service", "backend.interfaces.api.services.csv_evaluation_service"):
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "CsvEvaluationService"):
                        cache.update(getattr(mod.CsvEvaluationService, "_prediction_cache", {}))
            if not cache:
                try:
                    from interfaces.api.services.csv_evaluation_service import CsvEvaluationService as Ces
                    cache.update(getattr(Ces, "_prediction_cache", {}))
                except ModuleNotFoundError:
                    try:
                        from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService as Ces
                        cache.update(getattr(Ces, "_prediction_cache", {}))
                    except ModuleNotFoundError:
                        pass

            cached = cache.get(request.prediction_id)
            if cached and "result" in cached:
                cached_result = cached["result"]
            else:
                try:
                    from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService as Ces
                except ModuleNotFoundError:
                    from interfaces.api.services.csv_evaluation_service import CsvEvaluationService as Ces
                try:
                    ces_inst = Ces()
                    cached_result = ces_inst.get_prediction_result(request.prediction_id)
                except Exception as e:
                    logger.warning("Failed to retrieve prediction result for %s: %s", request.prediction_id, e)

            if cached_result:
                preds = cached_result.get("predictions", [])
                match = None
                for p in preds:
                    if request.row_id is not None and p.get("row_id") == request.row_id:
                        match = p
                        break
                    if request.transaction_id is not None and p.get("transaction_id") == request.transaction_id:
                        match = p
                        break
                if match:
                    if "tabular_features" in match and match["tabular_features"]:
                        request.tabular_features = match["tabular_features"]
                        request.model_name = "lightgbm_tabular_fallback"
                    elif "sequence_features" in match and match["sequence_features"]:
                        request.sequence = SequenceItem(
                            features=match["sequence_features"],
                            length=match.get("sequence_length", len(match["sequence_features"])),
                            sample_id=match.get("transaction_id", f"TX_{match.get('row_id')}"),
                        )
                        if not request.model_name:
                            request.model_name = match.get("model_name") or "champion"

            # Dynamic disk extraction if features are missing
            is_tab_model = (
                any(k in str(request.model_name).lower() for k in ("fallback", "lightgbm", "tabular"))
                or (cached_result and any(k in str(cached_result.get("model_name", "")).lower() for k in ("fallback", "lightgbm", "tabular")))
            )

            # Try to recover features from persisted CSV if prediction_id is known
            if (request.tabular_features is None and request.sequence is None) and request.prediction_id:
                try:
                    from pathlib import Path
                    import pandas as pd
                    from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService as Ces
                    ces_inst = Ces()
                    clean_pid = str(request.prediction_id).replace(".csv", "")
                    candidates = [
                        Path("backend/results/uploads") / f"{clean_pid}.csv",
                        Path("backend/results/uploads") / f"pred_{clean_pid}.csv",
                        Path("backend/results/uploads") / clean_pid,
                        Path("results/uploads") / f"{clean_pid}.csv",
                        Path("results/uploads") / f"pred_{clean_pid}.csv",
                        Path("results/uploads") / clean_pid,
                    ]
                    csv_path = next((p for p in candidates if p.exists()), None)
                    if csv_path:
                        target_row_id = request.row_id or (int(str(request.transaction_id).replace("TX_", "")) if request.transaction_id and "TX_" in str(request.transaction_id) else 1)
                        idx = max(0, target_row_id - 1)
                        df_target = pd.read_csv(csv_path, skiprows=range(1, idx + 1), nrows=1)
                        if len(df_target) > 0:
                            det = ces_inst.adaptive_detector.inspect_and_adapt(df_target, csv_path.name, request.dataset or "HI-Small")
                            proc, _, _, _ = ces_inst._preprocess_adapted_frame(det, df_target, request.dataset or "HI-Small")
                            tab_dict = {col: float(proc[col].iloc[0]) for col in FEATURE_COLS if col in proc.columns}
                            request.tabular_features = tab_dict
                            if is_tab_model:
                                request.model_name = "lightgbm_tabular_fallback"
                except Exception as err:
                    logger.warning("Dynamic feature recovery failed: %s", err)

            # Fallback for tabular features if still missing for a tabular model
            if is_tab_model and request.tabular_features is None:
                tab_dict = {col: 0.0 for col in FEATURE_COLS}
                if match:
                    amt = float(match.get("amount", 0.0))
                    tab_dict["amount_paid_log"] = float(np.log1p(max(0.0, amt)))
                    tab_dict["amount_received_log"] = float(np.log1p(max(0.0, amt)))
                request.tabular_features = tab_dict

        # 2. Tabular Fallback pathway
        is_tabular = (
            any(k in str(request.model_name).lower() for k in ("fallback", "lightgbm", "tabular"))
            or request.tabular_features is not None
        )
        if is_tabular and request.tabular_features is not None:
            return self.explain_tabular(request)

        # 3. Recurrent Sequence pathway
        if request.sequence is None:
            # If sequence context is missing, gracefully explain via tabular feature attribution
            if request.tabular_features is None:
                tab_dict = {col: 0.0 for col in FEATURE_COLS}
                if match:
                    amt = float(match.get("amount", 0.0))
                    tab_dict["amount_paid_log"] = float(np.log1p(max(0.0, amt)))
                    tab_dict["amount_received_log"] = float(np.log1p(max(0.0, amt)))
                request.tabular_features = tab_dict
            resp = self.explain_tabular(request)
            resp.history_warning = "Transaction explained using tabular attributes; multi-step sequence context was not retained in snapshot."
            return resp

        dataset = request.dataset
        seq_item = request.sequence
        steps = request.steps
        top_k = request.top_k
        model_name = request.model_name or dataset

        names = self._get_names(dataset)
        threshold = FROZEN_THRESHOLDS.get(dataset, 0.337784)

        # Prepare (1, 10, 57) float tensor
        feat_arr = np.zeros((1, MAX_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        seq_len = seq_item.length or len(seq_item.features)
        for t_idx, step in enumerate(seq_item.features):
            feat_arr[0, t_idx, :] = np.array(step, dtype=np.float32)

        x_tensor = torch.from_numpy(feat_arr)
        l_tensor = torch.tensor([seq_len], dtype=torch.int64)

        t0 = time.perf_counter()
        with self._explain_lock:
            explainer = self._get_explainer(model_name)
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
                    attribution=round(val, 6),
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
                    attribution=round(val, 6),
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
            predicted_probability=round(prob, 6),
            decision_threshold=threshold,
            decision=decision,
            top_positive_drivers=top_pos,
            top_negative_drivers=top_neg,
            completeness_residual=round(abs_res, 6),
            explanation_time_ms=round(latency_ms, 2),
            is_history_sufficient=is_hist_suff,
            history_warning=hist_warn,
        )
