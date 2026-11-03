"""Service for validating, preprocessing, evaluating, and training on transaction CSV files.

Supports:
- Universal CSV reading (UTF-8, UTF-8 BOM, Latin-1, comma, semicolon, tab, pipe delimiters).
- IBM AML Raw format (11 columns), Processed format (27 columns), synthetic/aliased schemas.
- Built-in federated sequence models (HI-Small LSTM, LI-Medium BiLSTM) and dynamic models.
- Deep schema inspection, feature compatibility reporting, and user column mapping.
- Supervised AML metric calculation and risk categorization (Low, Medium, High).
- In-memory prediction caching and downloadable CSV generation.
- Mode B: Dynamic model training on custom labeled datasets.
"""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

try:
    from evaluation.metrics import evaluate as run_robust_evaluate
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from evaluation.metrics import evaluate as run_robust_evaluate

try:
    from config.threshold_config import get_canonical_threshold
except ModuleNotFoundError:
    from backend.config.threshold_config import get_canonical_threshold

from domain.enums.split_name import SplitName
from infrastructure.data.features.account_history import AccountHistoryBuilder
from infrastructure.data.features.temporal_features import TemporalFeatureBuilder
from infrastructure.data.features.transaction_features import TransactionFeatureBuilder
from infrastructure.data.preprocessors.aml_preprocessor import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_FROM_BANK,
    COL_IS_LAUNDERING,
    COL_PAYMENT_CURRENCY,
    COL_PAYMENT_FORMAT,
    COL_RECEIVING_CURRENCY,
    COL_TIMESTAMP,
    COL_TO_ACCOUNT,
    COL_TO_BANK,
    AmlPreprocessor,
)
from infrastructure.data.preprocessors.categorical_encoder import OrdinalCategoricalEncoder
from infrastructure.data.preprocessors.running_stats import RunningStandardizer
from infrastructure.data.readers.universal_csv_reader import CsvMetadata, UniversalCsvReader
from infrastructure.data.schemas.adaptive_schema_detector import (
    AdaptiveSchemaDetector,
    DetectionStatus,
    FormatCategory,
    SchemaDetectionResult,
)
from infrastructure.data.schemas.ibm_aml_schema import CURRENCIES, PAYMENT_FORMATS
from infrastructure.data.schemas.universal_schema_detector import (
    FeatureCompatibilityReport,
    UniversalSchemaDetector,
    UniversalSchemaReport,
)
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
    build_previous_row_index,
)
from infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
from infrastructure.models.dynamic_model_trainer import DynamicModelTrainer
from infrastructure.models.registry.universal_model_registry import (
    ModelSchemaMetadata,
    UniversalModelRegistry,
)
from interfaces.api.schemas.common import (
    FEATURE_DIMENSION,
    FROZEN_THRESHOLDS,
    MAX_SEQUENCE_LENGTH,
    SUPPORTED_MODELS_BY_DATASET,
)
from interfaces.api.services.model_service import ModelService

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[4]
UPLOAD_DIR = BASE_DIR / "results" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_CSV_BYTES = 250 * 1024 * 1024  # 250 MB
DEFAULT_ROW_LIMIT = 1000000
MAX_ROW_LIMIT = 1000000


class CsvEvaluationService:
    """End-to-end service for inspecting, evaluating, and training on transaction CSVs."""

    _instance: Optional[CsvEvaluationService] = None
    _prediction_cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> CsvEvaluationService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self) -> None:
        self.csv_reader = UniversalCsvReader()
        self.model_service = ModelService()
        self.adaptive_detector = AdaptiveSchemaDetector()
        self.universal_detector = UniversalSchemaDetector()
        self.model_registry = UniversalModelRegistry()
        self.dynamic_trainer = DynamicModelTrainer()

    def inspect_csv_bytes(
        self,
        csv_bytes: bytes,
        filename: str = "upload.csv",
    ) -> Dict[str, Any]:
        """Perform universal structural and semantic inspection on CSV bytes."""
        if not csv_bytes or len(csv_bytes.strip()) == 0:
            return {
                "is_empty": True,
                "user_message": "Uploaded file is empty (0 bytes). Please upload a valid CSV file.",
                "total_rows": 0,
                "total_columns": 0,
                "column_names": [],
                "detected_delimiter": ",",
                "detected_encoding": "utf-8",
                "has_target_label": False,
                "can_construct_sequences": False,
                "schema_report": None,
                "metadata": None,
            }

        df, meta = self.csv_reader.read_dataframe(csv_bytes, nrows=500)
        schema_report = self.universal_detector.inspect_dataframe(df, filename=filename)
        schema_type = "Transaction Sequence" if schema_report.can_construct_sequences else "Tabular Dataset"

        return {
            # Canonical unified fields matching frontend expectations
            "delimiter": meta.detected_delimiter,
            "encoding": meta.detected_encoding,
            "num_rows": meta.total_rows,
            "num_cols": meta.total_columns,
            "column_names": meta.column_names,
            "preview_rows": meta.sample_rows,
            "target_column_name": schema_report.target_column_name,
            "has_ground_truth": schema_report.has_target_label,
            "detected_schema_type": schema_type,
            "can_construct_sequences": schema_report.can_construct_sequences,
            "column_types": meta.dtypes,
            "semantic_mappings": schema_report.detected_concepts,
            # Backward-compatibility legacy fields
            "is_empty": meta.is_empty,
            "filename": filename,
            "total_rows": meta.total_rows,
            "total_columns": meta.total_columns,
            "detected_delimiter": meta.detected_delimiter,
            "detected_encoding": meta.detected_encoding,
            "dtypes": meta.dtypes,
            "missing_counts": meta.missing_counts,
            "duplicate_count": meta.duplicate_count,
            "has_target_label": schema_report.has_target_label,
            "is_binary_target": schema_report.is_binary_target,
            "detected_concepts": schema_report.detected_concepts,
            "sample_rows": meta.sample_rows,
            "warnings": meta.warnings + schema_report.warnings,
            "schema_report": schema_report.to_dict(),
        }

    def check_compatibility(
        self,
        csv_bytes: bytes,
        model_id: Optional[str] = None,
        filename: str = "upload.csv",
        user_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Check compatibility between uploaded CSV and selected model, plus all available models."""
        df, _ = self.csv_reader.read_dataframe(csv_bytes, nrows=200)
        schema_report = self.universal_detector.inspect_dataframe(df, filename=filename)

        primary_model_id = model_id if (model_id and model_id != "auto") else "HI-Small"
        comp_report = self.universal_detector.check_compatibility(
            schema_report=schema_report,
            model_id=primary_model_id,
            user_mappings=user_mappings,
        )

        # Build multi-model compatibility results list for dashboard
        all_models = self.model_registry.list_models()
        if not all_models:
            all_models = [
                {"model_id": "HI-Small", "model_name": "HI-Small Champion (Federated LSTM)"},
                {"model_id": "LI-Medium", "model_name": "LI-Medium Champion (Federated BiLSTM)"},
            ]

        results_list = []
        for m in all_models:
            mid = m.get("model_id") or "HI-Small"
            if mid == primary_model_id:
                m_comp = comp_report
            else:
                m_comp = self.universal_detector.check_compatibility(
                    schema_report=schema_report,
                    model_id=mid,
                    user_mappings=user_mappings,
                )

            if m_comp.compatible:
                status_str = "COMPATIBLE"
                score = 1.0
            elif m_comp.allows_direct_prediction:
                status_str = "ADAPTABLE"
                score = 0.8
            elif m_comp.allows_training_new_model:
                status_str = "ADAPTABLE"
                score = 0.5
            else:
                status_str = "INCOMPATIBLE"
                score = 0.1

            actions = list(m_comp.missing_required_features)
            if m_comp.suggested_action and m_comp.suggested_action not in actions:
                actions.append(m_comp.suggested_action)

            results_list.append({
                "model_id": mid,
                "status": status_str,
                "missing_features": m_comp.missing_required_features,
                "extra_features": m_comp.extra_unused_features,
                "actions_needed": actions,
                "compatibility_score": score,
                "summary": m_comp.suggested_action or f"Status: {status_str}",
            })

        out = comp_report.to_dict()
        out["results"] = results_list
        out["status"] = "success"
        return out

    def validate_csv_bytes(
        self,
        csv_bytes: bytes,
        filename: str = "upload.csv",
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate CSV schema for backward compatibility with /evaluate/validate-schema."""
        if not csv_bytes or len(csv_bytes.strip()) == 0:
            return {
                "status": DetectionStatus.UNSUPPORTED.value,
                "user_message": "Uploaded file is empty (0 bytes). Please upload a valid CSV file.",
                "format_category": FormatCategory.UNKNOWN.value,
                "target_pipeline": "HI-Small",
                "decision_threshold": FROZEN_THRESHOLDS["HI-Small"],
                "has_labels": False,
                "field_mappings": {},
                "missing_required_fields": [],
                "conflicting_fields": {},
                "excluded_leakage_fields": [],
                "audit_notes": [],
            }

        try:
            df, meta = self.csv_reader.read_dataframe(csv_bytes, nrows=100)
        except Exception as e:
            return {
                "status": DetectionStatus.UNSUPPORTED.value,
                "user_message": f"Failed to parse CSV file: {str(e)}",
                "format_category": FormatCategory.UNKNOWN.value,
                "target_pipeline": "HI-Small",
                "decision_threshold": FROZEN_THRESHOLDS["HI-Small"],
                "has_labels": False,
                "field_mappings": {},
                "missing_required_fields": [],
                "conflicting_fields": {},
                "excluded_leakage_fields": [],
                "audit_notes": [],
            }

        detection_result = self.adaptive_detector.inspect_and_adapt(
            df=df,
            filename=filename,
            requested_dataset=dataset,
        )
        return detection_result.to_dict()

    def _auto_detect_model(
        self,
        df: pd.DataFrame,
        filename: str = "upload.csv",
        user_mappings: Optional[Dict[str, str]] = None,
    ) -> str:
        """Inspect schema compatibility across all registered models and select the best compatible match."""
        # 1. Test adaptation for authoritative federated models (HI-Small / LI-Medium)
        detection = self.adaptive_detector.inspect_and_adapt(
            df=df,
            filename=filename,
            requested_dataset=None,
        )
        if detection.status == DetectionStatus.COMPATIBLE:
            return detection.target_pipeline

        # 2. If detection is not COMPATIBLE, raise informative detection message
        raise ValueError(detection.user_message)

    def evaluate_csv_bytes(
        self,
        csv_bytes: bytes,
        dataset: Optional[str] = None,
        filename: str = "upload.csv",
        row_limit: Optional[int] = None,
        model_id: Optional[str] = None,
        user_mappings: Optional[Dict[str, str]] = None,
        tenant_id: str = "default_tenant",
        enable_persistent_history: bool = False,
        threshold: Optional[float] = None,
        fallback_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate, adapt, preprocess, and evaluate a CSV file provided as bytes."""
        t0 = time.perf_counter()

        if not csv_bytes or len(csv_bytes.strip()) == 0:
            raise ValueError("Uploaded file is empty (0 bytes). Please upload a valid CSV file.")

        if len(csv_bytes) > MAX_CSV_BYTES:
            raise ValueError(
                f"File size exceeds limit of {MAX_CSV_BYTES // (1024 * 1024)} MB (got {len(csv_bytes)} bytes)."
            )

        bounded_limit = min(max(1, row_limit or DEFAULT_ROW_LIMIT), MAX_ROW_LIMIT)

        # 1. Parse CSV using Universal Reader
        try:
            df, meta = self.csv_reader.read_dataframe(csv_bytes, nrows=bounded_limit)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

        if df.empty or len(df) == 0:
            raise ValueError("Uploaded CSV contains header only with zero data rows.")

        # Check chronological order, time span, and unique sender ratio BEFORE sorting
        is_chronological = True
        time_span_str = "N/A"
        ts_candidates = [c for c in df.columns if str(c).strip().lower() in ("timestamp", "time", "date", "datetime")]
        if ts_candidates:
            ts_col = ts_candidates[0]
            try:
                raw_ts = pd.to_datetime(df[ts_col], errors="coerce")
                valid_ts = raw_ts.dropna()
                if len(valid_ts) > 1:
                    is_chronological = bool(raw_ts.is_monotonic_increasing)
                    t_min = valid_ts.min()
                    t_max = valid_ts.max()
                    delta = t_max - t_min
                    days = delta.total_seconds() / 86400.0
                    if days >= 1.0:
                        time_span_str = f"{days:.1f} days ({t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')})"
                    else:
                        hours = delta.total_seconds() / 3600.0
                        time_span_str = f"{hours:.1f} hours"
                if not raw_ts.is_monotonic_increasing:
                    df = df.sort_values(by=ts_col, ascending=True).reset_index(drop=True)
            except Exception:
                try:
                    df = df.sort_values(by=ts_col, ascending=True).reset_index(drop=True)
                except Exception:
                    pass

        total_uploaded_rows = len(df)
        sender_cols = [c for c in df.columns if str(c).strip().lower() in ("from_account", "account", "from account", "sender", "originator", "from_acc")]
        if sender_cols:
            unique_senders = int(df[sender_cols[0]].nunique())
            unique_sender_ratio = float(unique_senders / total_uploaded_rows) if total_uploaded_rows > 0 else 1.0
        else:
            unique_sender_ratio = 1.0

        # 2. Resolve Model Target
        target_model_id = model_id or dataset
        if not target_model_id or target_model_id == "auto":
            target_model_id = self._auto_detect_model(df, filename=filename, user_mappings=user_mappings)

        model_meta = self.model_registry.get_model_metadata(target_model_id)

        # Fallback to dataset mapping if direct model not in registry
        if not model_meta and dataset:
            model_meta = self.model_registry.get_model_metadata(dataset)

        if not model_meta:
            model_meta = self.model_registry.get_model_metadata("HI-Small")
            target_model_id = "HI-Small"

        if not model_meta:
            raise ValueError(f"Unable to resolve model metadata for '{target_model_id}'.")

        # 3. Branch: Explicit Tabular Fallback vs. Dynamic Model vs. Built-in Sequence Model
        if fallback_model == "tabular" or target_model_id in ("lightgbm_tabular_fallback", "tabular_fallback"):
            return self._evaluate_tabular_fallback(
                df=df,
                meta=meta,
                filename=filename,
                t0=t0,
                threshold=threshold,
                user_mappings=user_mappings,
                tenant_id=tenant_id,
            )

        if not model_meta.is_builtin:
            return self._evaluate_dynamic_model(
                df=df,
                meta=meta,
                model_meta=model_meta,
                filename=filename,
                t0=t0,
            )

        # Built-in Sequence Models (Federated / Centralized)
        target_dataset = model_meta.dataset_name

        # Schema adaptation using AdaptiveSchemaDetector
        detection = self.adaptive_detector.inspect_and_adapt(
            df=df,
            filename=filename,
            requested_dataset=target_dataset,
        )

        if detection.status != DetectionStatus.COMPATIBLE:
            raise ValueError(detection.user_message)

        target_dataset = detection.target_pipeline
        canonical_thr = get_canonical_threshold(target_dataset)
        resolved_threshold = (
            threshold
            if threshold is not None
            else (model_meta.frozen_threshold if model_meta.frozen_threshold else canonical_thr)
        )
        if (model_meta.model_id in ("HI-Small", "LI-Medium", "auto", "default") or model_meta.model_id in SUPPORTED_MODELS_BY_DATASET.values()) and target_dataset in SUPPORTED_MODELS_BY_DATASET:
            model_name = SUPPORTED_MODELS_BY_DATASET[target_dataset]
        else:
            model_name = model_meta.model_name or SUPPORTED_MODELS_BY_DATASET.get(target_dataset, "FedSecureAI Lead Model")

        # Preprocess to Model-Ready Processed Features
        processed_df, has_labels, raw_amounts, builder_accounts = self._preprocess_adapted_frame(
            detection=detection,
            orig_df=df,
            target_dataset=target_dataset,
            tenant_id=tenant_id,
            enable_persistent_history=enable_persistent_history,
        )

        # Sequence Construction
        batch_features, batch_lengths, final_df, diag = self._construct_sequences(
            processed_df, target_dataset
        )

        # Resolve PyTorch model instance
        model = self.model_service.get_model(model_meta.model_id)

        # Device selection & PyTorch batch inference (batch_size=4096)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        batch_size = 4096
        total_samples = len(batch_features)
        probs_list = []

        with torch.inference_mode():
            for i in range(0, total_samples, batch_size):
                b_feat = torch.from_numpy(batch_features[i : i + batch_size]).to(device)
                b_len = torch.tensor(batch_lengths[i : i + batch_size], dtype=torch.int64, device=device)
                logits = model(b_feat, b_len).view(-1)
                b_probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(b_probs)

        probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=float)

        if isinstance(probs, float) or probs.ndim == 0:
            probs = np.array([float(probs)])

        probs = probs.astype(float)
        if not np.all(np.isfinite(probs)):
            raise ValueError("Sequence model inference produced non-finite (NaN or Inf) probabilities.")
        probs = np.clip(probs, 0.0, 1.0)

        # Label-aware Metrics calculation (strictly None for unlabeled data)
        eval_metrics: Optional[Dict[str, Any]] = None
        if has_labels and "is_laundering" in final_df.columns:
            labels = final_df["is_laundering"].to_numpy(dtype=np.int64)
            eval_metrics = self._calculate_metrics(labels, probs, resolved_threshold)

        # Prior-shift detection and logit-adjusted probabilities
        pi_train = 0.00177 if "hi" in str(target_dataset).lower() else 0.00057
        prior_shift_warning: Optional[str] = None
        supports_logit_shift = False
        logit_shift_delta = 0.0
        p_adjusted = probs.copy()

        if eval_metrics is not None and eval_metrics.get("prevalence") is not None:
            sample_prev = float(eval_metrics["prevalence"])
            if sample_prev > 0.005 or (sample_prev > 0 and abs(sample_prev - pi_train) / pi_train > 2.0):
                prior_shift_warning = (
                    f"Prior shift detected: sample positive prevalence ({sample_prev:.2%}) diverges "
                    f"from training prevalence ({pi_train:.2%}). Raw recurrent sequence models output lower probabilities "
                    "under elevated prior distribution. Optional prior-adjusted probabilities are provided."
                )
                if 0.0 < sample_prev < 1.0 and 0.0 < pi_train < 1.0:
                    target_log_odds = np.log(sample_prev / (1.0 - sample_prev))
                    train_log_odds = np.log(pi_train / (1.0 - pi_train))
                    logit_shift_delta = float(target_log_odds - train_log_odds)
                    eps = 1e-12
                    p_clipped = np.clip(probs, eps, 1.0 - eps)
                    raw_logits = np.log(p_clipped / (1.0 - p_clipped))
                    shifted_logits = raw_logits + logit_shift_delta
                    p_adjusted = 1.0 / (1.0 + np.exp(-shifted_logits))
                    supports_logit_shift = True

        # Build itemized prediction records
        predictions: List[Dict[str, Any]] = []
        flagged_count = 0

        # Preserve row-by-row mapping
        for idx in range(len(final_df)):
            p = float(probs[idx])
            p_adj_val = float(p_adjusted[idx]) if supports_logit_shift else p
            lbl = int(final_df["is_laundering"].iloc[idx]) if has_labels else None
            ts_str = str(final_df["timestamp"].iloc[idx])
            from_acc = str(final_df["from_account"].iloc[idx])
            to_acc = str(final_df["to_account"].iloc[idx])

            amount = (
                float(raw_amounts.get(idx, 0.0))
                if raw_amounts
                else float(np.expm1(final_df["amount_paid_log"].iloc[idx]))
            )

            is_suspicious = p >= resolved_threshold
            if is_suspicious:
                flagged_count += 1

            seq_len_val = int(batch_lengths[idx])
            prior_count_val = max(0, seq_len_val - 1)
            is_history_usable = bool(prior_count_val >= 3)

            if is_suspicious:
                status_label = "SUSPICIOUS"
                prediction_status = "FLAGGED_LAUNDERING"
                decision = "FLAGGED_LAUNDERING"
            else:
                if is_history_usable:
                    status_label = "LEGITIMATE"
                    prediction_status = "CLEARED_LEGITIMATE"
                    decision = "CLEARED_LEGITIMATE"
                else:
                    status_label = "LOW_CONFIDENCE_NO_HISTORY"
                    prediction_status = "LOW_CONFIDENCE_NO_HISTORY"
                    decision = "LOW_CONFIDENCE_NO_HISTORY"

            actual_label = None
            if has_labels and lbl is not None:
                actual_label = "SUSPICIOUS" if lbl == 1 else "LEGITIMATE"

            # Risk categorization
            if p >= resolved_threshold:
                risk_category = "High Risk"
            elif p >= resolved_threshold * 0.5:
                risk_category = "Medium Risk"
            else:
                risk_category = "Low Risk"

            predictions.append({
                "row_id": idx + 1,
                "transaction_id": f"TX_{idx + 1:05d}",
                "timestamp": ts_str,
                "from_account": from_acc,
                "to_account": to_acc,
                "amount": round(amount, 2),
                "predicted_probability": round(p, 6),
                "risk_score": round(p * 100.0, 2),
                "adjusted_probability": round(p_adj_val, 6) if supports_logit_shift else round(p, 6),
                "adjusted_risk_score": round(p_adj_val * 100.0, 2) if supports_logit_shift else round(p * 100.0, 2),
                "adjusted_decision": ("SUSPICIOUS" if p_adj_val >= resolved_threshold else "LEGITIMATE") if supports_logit_shift else status_label,
                "predicted_label": status_label,
                "prediction_status": prediction_status,
                "actual_label": actual_label,
                "decision": decision,
                "status": status_label,
                "risk_category": risk_category,
                "model_name": model_name,
                "classification_threshold": resolved_threshold,
                "legacy_decision": "FLAGGED_LAUNDERING" if is_suspicious else ("CLEARED_LEGITIMATE" if is_history_usable else "LOW_CONFIDENCE_NO_HISTORY"),
                "decision_threshold": resolved_threshold,
                "ground_truth_label": lbl,
                "sequence_length": seq_len_val,
                "valid_history_count": prior_count_val,
                "padding_ratio": round((MAX_SEQUENCE_LENGTH - seq_len_val) / float(MAX_SEQUENCE_LENGTH), 4),
                "unknown_category_ratio": round(float(diag["unknown_category_counts"][idx] / diag["total_nominal_cols"]), 4),
                "missing_feature_ratio": round(float(diag["missing_feature_counts"][idx] / diag["total_cols"]), 4),
                "is_history_sufficient": is_history_usable,
                "usable_history_note": None if is_history_usable else "Insufficient historical context (< 3 prior transactions for this account in slice).",
                "sequence_features": np.round(batch_features[idx], 6).tolist(),
            })

        # Verification check: output row count equals input row count
        if len(predictions) != len(final_df):
            logger.warning(
                "Prediction length mismatch: %d predictions for %d final rows",
                len(predictions),
                len(final_df),
            )

        # Risk distribution histogram
        risk_distribution = self._calculate_risk_distribution(probs)
        execution_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Generate unique prediction ID and cache for CSV download
        prediction_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        download_csv_content = self._generate_downloadable_csv(
            predictions, orig_df=df, model_name=model_name, threshold=resolved_threshold
        )

        avg_seq_len = float(np.mean(batch_lengths)) if len(batch_lengths) > 0 else 1.0
        single_tx_count = int(np.sum(batch_lengths == 1))
        single_tx_ratio = round(float(single_tx_count / len(batch_lengths)), 4) if len(batch_lengths) > 0 else 1.0
        mean_pad_ratio = round(float(np.mean([(MAX_SEQUENCE_LENGTH - l) / float(MAX_SEQUENCE_LENGTH) for l in batch_lengths])), 4) if len(batch_lengths) > 0 else 0.90
        pct_rows_ge_3_prior = round(float(np.mean((batch_lengths - 1) >= 3) * 100.0), 2) if len(batch_lengths) > 0 else 0.0
        insufficient_history_warning = bool(avg_seq_len < 3.0 or mean_pad_ratio >= 0.70 or pct_rows_ge_3_prior < 50.0)

        sample_prev = float(eval_metrics["prevalence"]) if eval_metrics and eval_metrics.get("prevalence") is not None else None
        positives_cnt = int(eval_metrics["positive_samples"]) if eval_metrics and eval_metrics.get("positive_samples") is not None else None

        # Build Task 1 Validity Banner
        validity_flags: List[str] = []
        validity_reasons: List[str] = []

        if pct_rows_ge_3_prior < 50.0:
            validity_flags.append("LOW_HISTORY")
            validity_reasons.append(
                f"Low Account History: only {pct_rows_ge_3_prior:.1f}% of rows have >=3 prior transactions "
                f"(mean sequence depth: {avg_seq_len:.1f}/10). Recurrent sequence models require historical "
                "temporal patterns to evaluate laundering velocity; single-transaction inputs produce conservative baseline probabilities."
            )

        if sample_prev is not None and (sample_prev > 0.05 or sample_prev > (5.0 * pi_train)):
            validity_flags.append("PREVALENCE_MISMATCH")
            validity_reasons.append(
                f"Prevalence Mismatch: positive laundering rate ({sample_prev:.2%}) is >5x higher than "
                f"the training distribution ({pi_train:.2%}). Raw model probabilities were calibrated for operational rare-class priors."
            )

        if (not is_chronological) or (unique_sender_ratio > 0.9):
            validity_flags.append("NOT_CONTIGUOUS")
            reason_parts = []
            if not is_chronological:
                reason_parts.append("non-chronological timestamps")
            if unique_sender_ratio > 0.9:
                reason_parts.append(f"{unique_sender_ratio:.1%} unique senders (near zero repeat accounts)")
            validity_reasons.append(
                f"Non-Contiguous Slice: {' and '.join(reason_parts)}. This file reflects scattered independent rows rather than contiguous chronological account activity."
            )

        if has_labels and positives_cnt is not None and positives_cnt < 30:
            validity_flags.append("SMALL_SAMPLE")
            validity_reasons.append(
                f"Small Positive Sample: file contains only {positives_cnt} positive cases (<30). Metric confidence intervals and alert rates have high statistical variance."
            )

        is_valid_slice = (pct_rows_ge_3_prior >= 50.0 and is_chronological and unique_sender_ratio <= 0.9)
        if is_valid_slice and (not validity_flags or validity_flags == ["SMALL_SAMPLE"]):
            banner_status = "VALID"
        elif "LOW_HISTORY" in validity_flags or "NOT_CONTIGUOUS" in validity_flags or "PREVALENCE_MISMATCH" in validity_flags:
            banner_status = "INVALID"
        else:
            banner_status = "WARNING"

        validity_banner = {
            "status": banner_status,
            "is_valid": is_valid_slice,
            "flags": validity_flags,
            "reasons": validity_reasons,
            "disclaimer": (
                "Results on LOW_HISTORY or NOT_CONTIGUOUS files are not representative of model quality. "
                "For accurate evaluation matching the training distribution, use contiguous test slices from data/demo_slices/."
            ),
            "diagnostics": {
                "n_rows": total_uploaded_rows,
                "prevalence": sample_prev,
                "positives_count": positives_cnt,
                "unique_sender_ratio": round(unique_sender_ratio, 4),
                "pct_single_tx_rows": round(single_tx_ratio * 100.0, 2),
                "mean_history_len": round(avg_seq_len, 2),
                "pct_rows_with_ge_3_prior_tx": pct_rows_ge_3_prior,
                "is_chronological": is_chronological,
                "time_span": time_span_str,
            },
        }

        suggested_models: List[Dict[str, Any]] = []
        if insufficient_history_warning:
            for dm in self.model_registry.list_dynamic_models():
                suggested_models.append({
                    "model_id": dm.model_id,
                    "model_name": dm.model_name,
                    "model_type": dm.model_type,
                    "reason": "Tabular model optimized for single-transaction or history-poor evaluation.",
                })

        sequence_context_diagnostics = {
            "average_sequence_length": round(avg_seq_len, 2),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "single_transaction_count": single_tx_count,
            "single_transaction_ratio": single_tx_ratio,
            "mean_padding_ratio": mean_pad_ratio,
            "insufficient_history_warning": insufficient_history_warning,
            "operational_context_status": "INSUFFICIENT_HISTORY_TRUNCATED" if insufficient_history_warning else "HISTORICALLY_SUPPORTED",
            "persistent_history_enabled": enable_persistent_history,
            "tenant_id": tenant_id if enable_persistent_history else None,
            "historical_accounts_tracked": builder_accounts,
            "warning_message": (
                f"Sequence history is truncated for this dataset (mean history length: {avg_seq_len:.1f}/10, "
                f"{single_tx_ratio * 100.0:.1f}% single-transaction rows). Deep recurrent sequence models rely on "
                f"multi-step temporal patterns and transaction velocity. Under single-transaction inputs, predictions reflect "
                f"conservative baseline transaction priors (~0.001 - 0.015) and remain below the standard sequence threshold ({resolved_threshold:.4f}). "
                f"To evaluate isolated transactions without historical context, consider training or selecting a tabular model via Mode B."
            ) if insufficient_history_warning else None,
            "suggested_models": suggested_models,
        }

        pct_sufficient_history = round(float(np.mean(batch_lengths >= 3) * 100.0), 2) if len(batch_lengths) > 0 else 0.0
        sequence_diagnostics = {
            "mean_history_len": round(avg_seq_len, 2),
            "pct_single_tx_rows": round(single_tx_ratio * 100.0, 2),
            "pct_sufficient_history": pct_sufficient_history,
            "pct_rows_with_ge_3_prior_tx": pct_rows_ge_3_prior,
            "insufficient_history": insufficient_history_warning,
            "warning": (
                "Insufficient transaction history: recurrent sequence models require multi-step history per account. "
                "<50% of accounts have >= 3 prior transactions in this slice. Tabular fallback model is recommended."
                if insufficient_history_warning else None
            ),
            "recommended_fallback": "tabular_lightgbm" if insufficient_history_warning else None,
        }

        prior_shift_diagnostics = {
            "training_prevalence": pi_train,
            "sample_prevalence": eval_metrics.get("prevalence") if eval_metrics else None,
            "prior_shift_warning": prior_shift_warning,
            "supports_logit_shift": supports_logit_shift,
            "logit_shift_delta": round(logit_shift_delta, 4),
        }

        result = {
            "prediction_id": prediction_id,
            "download_id": prediction_id,
            "download_url": f"/api/v1/predict/download/{prediction_id}",
            "status": "success",
            "filename": filename,
            "dataset": target_dataset,
            "model_id": model_meta.model_id,
            "model_name": model_name,
            "decision_threshold": resolved_threshold,
            "total_rows": total_uploaded_rows,
            "evaluated_sequences": len(predictions),
            "distinct_accounts": int(final_df["from_account"].nunique()),
            "flagged_laundering_count": flagged_count,
            "cleared_count": len(predictions) - flagged_count,
            "risk_distribution": risk_distribution,
            "has_labels": has_labels,
            "evaluation_metrics": eval_metrics,
            "validity_banner": validity_banner,
            "sequence_context_diagnostics": sequence_context_diagnostics,
            "sequence_diagnostics": sequence_diagnostics,
            "prior_shift_diagnostics": prior_shift_diagnostics,
            "suggested_models": suggested_models,
            "persistent_history_enabled": enable_persistent_history,
            "predictions": predictions,
            "execution_time_ms": execution_time_ms,
            "schema_status": detection.status.value,
            "schema_message": detection.user_message,
            "detected_format": detection.format_category.value,
            "field_mappings": detection.field_mappings,
            "excluded_leakage_fields": detection.excluded_leakage_fields,
            "audit_notes": detection.audit_notes,
            "detected_delimiter": meta.detected_delimiter,
            "detected_encoding": meta.detected_encoding,
        }

        # 1. Update in-memory cache
        self._prediction_cache[prediction_id] = {
            "result": result,
            "csv_content": download_csv_content,
            "filename": f"predictions_{Path(filename).stem}.csv",
            "created_at": time.time(),
            "is_expired": False,
        }

        # 2. Persist to results/uploads/<prediction_id>.csv and metadata JSON
        csv_file_path = UPLOAD_DIR / f"{prediction_id}.csv"
        meta_file_path = UPLOAD_DIR / f"{prediction_id}_meta.json"
        try:
            csv_file_path.write_text(download_csv_content, encoding="utf-8")
            meta_payload = {
                "prediction_id": prediction_id,
                "filename": f"predictions_{Path(filename).stem}.csv",
                "created_at": time.time(),
                "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "row_count": len(predictions),
                "model_id": model_meta.model_id,
                "dataset": target_dataset,
                "decision_threshold": resolved_threshold,
                "is_expired": False,
            }
            meta_file_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to write prediction upload artifact to %s: %s", csv_file_path, e)

        return result

    def _evaluate_tabular_fallback(
        self,
        df: pd.DataFrame,
        meta: CsvMetadata,
        filename: str,
        t0: float,
        threshold: Optional[float] = None,
        user_mappings: Optional[Dict[str, str]] = None,
        tenant_id: str = "default_tenant",
    ) -> Dict[str, Any]:
        """Execute inference against the explicit tabular LightGBM fallback model.

        CRITICAL: Used ONLY when explicitly opted in via fallback_model='tabular'
        or model_id='lightgbm_tabular_fallback'. Never silently substituted for sequence models.
        """
        try:
            from backend.infrastructure.models.tabular_fallback import TabularFallbackModel, FEATURE_COLS
        except ModuleNotFoundError:
            from infrastructure.models.tabular_fallback import TabularFallbackModel, FEATURE_COLS

        fallback_model = TabularFallbackModel()
        if not fallback_model.is_available:
            raise ValueError("Tabular fallback model is not available or could not be loaded from models/baselines/lightgbm_baseline.txt.")

        eff_threshold = threshold if threshold is not None else 0.337784
        total_rows = len(df)

        # Check if df already has the 23 preprocessed feature columns
        has_all_features = all(col in df.columns for col in FEATURE_COLS)
        if has_all_features:
            features_df = df[FEATURE_COLS]
        else:
            detection = self.adaptive_detector.inspect_and_adapt(
                df=df,
                filename=filename,
                requested_dataset="HI-Small",
            )
            processed_df, has_labels, raw_amounts, _ = self._preprocess_adapted_frame(
                detection=detection,
                orig_df=df,
                target_dataset="HI-Small",
                tenant_id=tenant_id,
                enable_persistent_history=False,
            )
            features_df = processed_df[FEATURE_COLS]

        probs = fallback_model.predict_proba(features_df)
        probs = np.clip(probs, 0.0, 1.0)

        # Check for ground truth labels
        has_labels = False
        eval_metrics = None
        target_col = None
        for cand in ["is_laundering", "Is Laundering", "Is_Laundering", "label", "target"]:
            if cand in df.columns:
                target_col = cand
                break

        if target_col is not None:
            s = pd.to_numeric(df[target_col], errors="coerce")
            if not s.isna().any() and set(s.unique()).issubset({0, 1}):
                has_labels = True
                eval_metrics = self._calculate_metrics(s.to_numpy(dtype=int), probs, eff_threshold)

        predictions: List[Dict[str, Any]] = []
        flagged_count = 0

        for idx in range(total_rows):
            p = float(probs[idx])
            is_suspicious = p >= eff_threshold
            if is_suspicious:
                flagged_count += 1

            status_label = "SUSPICIOUS" if is_suspicious else "LEGITIMATE"
            actual_label = None
            lbl = None
            if has_labels and target_col in df.columns:
                lbl = int(df[target_col].iloc[idx])
                actual_label = "SUSPICIOUS" if lbl == 1 else "LEGITIMATE"

            if p >= eff_threshold:
                risk_cat = "High Risk"
            elif p >= eff_threshold * 0.5:
                risk_cat = "Medium Risk"
            else:
                risk_cat = "Low Risk"

            row_dict = df.iloc[idx].to_dict()
            pred_status = "FLAGGED_LAUNDERING" if is_suspicious else "CLEARED_LEGITIMATE"

            predictions.append({
                "row_id": idx + 1,
                "transaction_id": str(row_dict.get("transaction_id", row_dict.get("id", f"TX_{idx + 1:05d}"))),
                "timestamp": str(row_dict.get("timestamp", row_dict.get("Timestamp", row_dict.get("date", "N/A")))),
                "from_account": str(row_dict.get("from_account", row_dict.get("Account", row_dict.get("sender", "N/A")))),
                "to_account": str(row_dict.get("to_account", row_dict.get("Account.1", row_dict.get("receiver", "N/A")))),
                "amount": float(row_dict.get("amount", row_dict.get("Amount Paid", row_dict.get("amount_paid", 0.0)))),
                "predicted_probability": round(p, 6),
                "risk_score": round(p * 100.0, 2),
                "predicted_label": status_label,
                "prediction_status": pred_status,
                "actual_label": actual_label,
                "decision": status_label,
                "status": status_label,
                "risk_category": risk_cat,
                "decision_threshold": eff_threshold,
                "ground_truth_label": lbl,
                "sequence_length": 1,
                "valid_history_count": 0,
                "padding_ratio": 0.0,
                "unknown_category_ratio": 0.0,
                "missing_feature_ratio": 0.0,
                "is_history_sufficient": True,
                "usable_history_note": "Evaluated via Tabular Fallback model (LightGBM). Does not require multi-step sequence history.",
            })

        prediction_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        download_csv_content = self._generate_downloadable_csv(predictions, orig_df=df)

        result = {
            "prediction_id": prediction_id,
            "download_id": prediction_id,
            "download_url": f"/api/v1/predict/download/{prediction_id}",
            "status": "success",
            "filename": filename,
            "dataset": "HI-Small",
            "model_id": "lightgbm_tabular_fallback",
            "model_name": "Fallback: no sequence history (LightGBM)",
            "model_type": "tabular_fallback",
            "is_sequence_model": False,
            "fallback_model_used": True,
            "disclaimer": "Evaluated using LightGBM tabular fallback model. Designed for single-transaction or history-poor evaluation without recurrent sequence context.",
            "decision_threshold": eff_threshold,
            "total_rows": total_rows,
            "evaluated_sequences": len(predictions),
            "distinct_accounts": total_rows,
            "flagged_laundering_count": flagged_count,
            "cleared_count": total_rows - flagged_count,
            "predictions": predictions,
            "has_ground_truth": has_labels,
            "evaluation_metrics": eval_metrics,
            "threshold_metrics": eval_metrics,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

        self._prediction_cache[prediction_id] = {
            "csv_content": download_csv_content,
            "filename": f"predictions_{Path(filename).stem}_{prediction_id}.csv",
            "total_rows": total_rows,
            "flagged_count": flagged_count,
            "created_at": time.time(),
        }

        # Persist to disk
        try:
            csv_file_path = UPLOAD_DIR / f"{prediction_id}.csv"
            meta_file_path = UPLOAD_DIR / f"{prediction_id}_meta.json"
            with open(csv_file_path, "w", encoding="utf-8") as f:
                f.write(download_csv_content)
            with open(meta_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "prediction_id": prediction_id,
                    "filename": f"predictions_{Path(filename).stem}_{prediction_id}.csv",
                    "total_rows": total_rows,
                    "flagged_count": flagged_count,
                    "created_at": time.time(),
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to write prediction upload artifact: %s", e)

        return result

    def _evaluate_dynamic_model(
        self,
        df: pd.DataFrame,
        meta: CsvMetadata,
        model_meta: ModelSchemaMetadata,
        filename: str,
        t0: float,
    ) -> Dict[str, Any]:
        """Execute inference against a user-trained dynamic tabular model."""
        artifacts = self.model_registry.load_dynamic_model_artifacts(model_meta.model_id)
        if not artifacts or "model" not in artifacts:
            raise ValueError(f"Model artifacts for dynamic model '{model_meta.model_id}' could not be loaded.")

        pipeline = artifacts["model"]
        threshold = model_meta.frozen_threshold

        # Predict positive laundering probabilities
        try:
            raw_probs = pipeline.predict_proba(df)
        except Exception as e:
            raise ValueError(f"Inference failed on dynamic model '{model_meta.model_name}': {str(e)}")

        probs = np.asarray(raw_probs, dtype=float)
        if probs.ndim == 2:
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            else:
                probs = probs[:, 0]
        elif probs.ndim == 0:
            probs = np.array([float(probs)])

        if not np.all(np.isfinite(probs)):
            raise ValueError("Dynamic model inference produced non-finite (NaN or Inf) probabilities.")
        probs = np.clip(probs, 0.0, 1.0)

        total_rows = len(df)

        # Check for labels (strictly None if unlabeled)
        has_labels = False
        eval_metrics = None
        target_col = model_meta.target_column
        if target_col in df.columns:
            s = pd.to_numeric(df[target_col], errors="coerce")
            if not s.isna().any() and set(s.unique()).issubset({0, 1}):
                has_labels = True
                eval_metrics = self._calculate_metrics(s.to_numpy(dtype=int), probs, threshold)

        predictions: List[Dict[str, Any]] = []
        flagged_count = 0

        for idx in range(total_rows):
            p = float(probs[idx])
            is_suspicious = p >= threshold
            if is_suspicious:
                flagged_count += 1

            status_label = "SUSPICIOUS" if is_suspicious else "LEGITIMATE"
            actual_label = None
            lbl = None
            if has_labels and target_col in df.columns:
                lbl = int(df[target_col].iloc[idx])
                actual_label = "SUSPICIOUS" if lbl == 1 else "LEGITIMATE"

            # Risk category
            if p >= threshold:
                risk_cat = "High Risk"
            elif p >= threshold * 0.5:
                risk_cat = "Medium Risk"
            else:
                risk_cat = "Low Risk"

            # Extract identifier values if available
            row_dict = df.iloc[idx].to_dict()

            predictions.append({
                "row_id": idx + 1,
                "transaction_id": str(row_dict.get("transaction_id", row_dict.get("id", f"TX_{idx + 1:05d}"))),
                "timestamp": str(row_dict.get("timestamp", row_dict.get("date", "N/A"))),
                "from_account": str(row_dict.get("from_account", row_dict.get("sender", "N/A"))),
                "to_account": str(row_dict.get("to_account", row_dict.get("receiver", "N/A"))),
                "amount": float(row_dict.get("amount", row_dict.get("amount_paid", 0.0))),
                "predicted_probability": round(p, 6),
                "risk_score": round(p * 100.0, 2),
                "predicted_label": status_label,
                "prediction_status": "FLAGGED_LAUNDERING" if is_suspicious else "CLEARED_LEGITIMATE",
                "actual_label": actual_label,
                "decision": status_label,
                "status": status_label,
                "risk_category": risk_cat,
                "decision_threshold": threshold,
                "ground_truth_label": lbl,
                "sequence_length": 1,
                "valid_history_count": 0,
                "padding_ratio": 0.0,
                "unknown_category_ratio": 0.0,
                "missing_feature_ratio": 0.0,
                "is_history_sufficient": True,
            })

        prediction_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        download_csv_content = self._generate_downloadable_csv(predictions, orig_df=df)

        result = {
            "prediction_id": prediction_id,
            "download_id": prediction_id,
            "download_url": f"/api/v1/predict/download/{prediction_id}",
            "status": "success",
            "filename": filename,
            "dataset": model_meta.dataset_name,
            "model_id": model_meta.model_id,
            "model_name": model_meta.model_name,
            "decision_threshold": threshold,
            "total_rows": total_rows,
            "evaluated_sequences": len(predictions),
            "distinct_accounts": total_rows,
            "flagged_laundering_count": flagged_count,
            "cleared_count": total_rows - flagged_count,
            "risk_distribution": self._calculate_risk_distribution(probs),
            "has_labels": has_labels,
            "evaluation_metrics": eval_metrics,
            "sequence_context_diagnostics": {
                "average_sequence_length": 1.0,
                "max_sequence_length": 1,
                "single_transaction_count": total_rows,
                "single_transaction_ratio": 1.0,
                "mean_padding_ratio": 0.0,
                "insufficient_history_warning": False,
                "operational_context_status": "HISTORICALLY_SUPPORTED",
                "warning_message": None,
            },
            "predictions": predictions,
            "execution_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "schema_status": "COMPATIBLE",
            "schema_message": f"Evaluated using custom model '{model_meta.model_name}'.",
            "detected_format": "CUSTOM_DYNAMIC_SCHEMA",
            "field_mappings": {col: col for col in model_meta.feature_columns if col in df.columns},
            "excluded_leakage_fields": [target_col] if has_labels else [],
            "audit_notes": [f"Evaluated on {total_rows} tabular records with zero sequence truncation."],
            "detected_delimiter": meta.detected_delimiter,
            "detected_encoding": meta.detected_encoding,
        }

        self._prediction_cache[prediction_id] = {
            "result": result,
            "csv_content": download_csv_content,
            "filename": f"predictions_{Path(filename).stem}.csv",
            "created_at": time.time(),
        }

        return result

    def train_custom_model_from_csv(
        self,
        csv_bytes: bytes,
        model_name: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        model_type: str = "HistGradientBoosting",
        filename: str = "train.csv",
    ) -> Dict[str, Any]:
        """Train a new custom AML model from an uploaded labeled CSV file."""
        df, _ = self.csv_reader.read_dataframe(csv_bytes)
        if df.empty or len(df) == 0:
            raise ValueError("Uploaded training CSV contains no rows.")

        meta, metrics = self.dynamic_trainer.train_custom_model(
            df=df,
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
            model_type=model_type,
        )

        return {
            "status": "success",
            "model_id": meta.model_id,
            "model_name": meta.model_name,
            "model_type": meta.model_type,
            "target_column": meta.target_column,
            "feature_columns": meta.feature_columns,
            "input_dimension": len(meta.feature_columns),
            "frozen_threshold": meta.frozen_threshold,
            "training_metrics": metrics,
            "message": f"Successfully trained and registered custom model '{model_name}'.",
        }

    def get_prediction_download(self, prediction_id: str) -> Optional[Tuple[str, str]]:
        """Retrieve downloadable CSV content and filename from cache or disk persistence."""
        # 1. Check in-memory cache
        cached = self._prediction_cache.get(prediction_id)
        if cached:
            if cached.get("is_expired"):
                return "EXPIRED", cached.get("filename", f"{prediction_id}.csv")
            return cached["csv_content"], cached["filename"]

        # 2. Check disk persistence in results/uploads/
        csv_file = UPLOAD_DIR / f"{prediction_id}.csv"
        meta_file = UPLOAD_DIR / f"{prediction_id}_meta.json"

        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("is_expired"):
                    return "EXPIRED", meta.get("filename", f"{prediction_id}.csv")
            except Exception:
                pass

        if csv_file.exists():
            try:
                csv_content = csv_file.read_text(encoding="utf-8")
                filename = f"predictions_{prediction_id}.csv"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        filename = meta.get("filename", filename)
                    except Exception:
                        pass
                # Repopulate in-memory cache
                self._prediction_cache[prediction_id] = {
                    "csv_content": csv_content,
                    "filename": filename,
                    "created_at": time.time(),
                    "is_expired": False,
                }
                return csv_content, filename
            except Exception as e:
                logger.error("Failed to read persisted CSV %s: %s", csv_file, e)
                return None

        return None

    def expire_prediction_download(self, prediction_id: str) -> bool:
        """Mark a prediction download as deliberately expired (HTTP 410)."""
        if prediction_id in self._prediction_cache:
            self._prediction_cache[prediction_id]["is_expired"] = True

        meta_file = UPLOAD_DIR / f"{prediction_id}_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                meta["is_expired"] = True
                meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                return True
            except Exception:
                pass
        elif (UPLOAD_DIR / f"{prediction_id}.csv").exists():
            meta = {
                "prediction_id": prediction_id,
                "is_expired": True,
                "expired_at": time.time(),
            }
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return True

        return False

    def _generate_downloadable_csv(
        self,
        predictions: List[Dict[str, Any]],
        orig_df: pd.DataFrame,
        model_name: str = "FedSecureAI Lead Model",
        threshold: float = 0.5,
    ) -> str:
        """Create a clean CSV export preserving original records with predictions appended."""
        export_df = orig_df.copy()
        pred_records = []
        for p in predictions:
            rec = {
                "predicted_probability": p["predicted_probability"],
                "predicted_label": p["predicted_label"],
                "prediction_status": p.get("prediction_status", p["predicted_label"]),
                "model_name": p.get("model_name", model_name),
                "classification_threshold": p.get("classification_threshold", threshold),
                "risk_category": p.get("risk_category", "Low Risk"),
                "sequence_length": p.get("sequence_length", 1),
                "is_history_sufficient": p.get("is_history_sufficient", True),
                # Legacy compatibility headers
                "PRED_PROBABILITY": p["predicted_probability"],
                "PRED_LABEL": p["predicted_label"],
                "PRED_DECISION_THRESHOLD": p.get("classification_threshold", threshold),
            }
            if "is_laundering" not in export_df.columns and p.get("ground_truth_label") is not None:
                rec["is_laundering"] = p["ground_truth_label"]
            pred_records.append(rec)

        pred_sub = pd.DataFrame(pred_records)

        # Concatenate side-by-side preserving original row order and identifiers
        combined = pd.concat([export_df.reset_index(drop=True), pred_sub], axis=1)
        buf = io.StringIO()
        combined.to_csv(buf, index=False)
        return buf.getvalue()

    def evaluate_multi_model_csv(
        self,
        csv_bytes: bytes,
        model_ids: Optional[List[str]] = None,
        filename: str = "upload.csv",
        threshold: Optional[float] = None,
        row_limit: Optional[int] = None,
        user_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate identical preprocessed transaction sequences across multiple selected models."""
        t0 = time.perf_counter()
        if not csv_bytes or len(csv_bytes.strip()) == 0:
            raise ValueError("Uploaded file is empty (0 bytes). Please upload a valid CSV file.")

        bounded_limit = min(max(1, row_limit or DEFAULT_ROW_LIMIT), MAX_ROW_LIMIT)
        df, meta = self.csv_reader.read_dataframe(csv_bytes, nrows=bounded_limit)
        if df.empty or len(df) == 0:
            raise ValueError("Uploaded CSV contains header only with zero data rows.")

        target_dataset = "HI-Small"
        detection = self.adaptive_detector.inspect_and_adapt(
            df=df, filename=filename, requested_dataset=target_dataset
        )
        if detection.status != DetectionStatus.COMPATIBLE:
            raise ValueError(detection.user_message)

        target_dataset = detection.target_pipeline
        processed_df, has_labels, raw_amounts, _ = self._preprocess_adapted_frame(
            detection=detection, orig_df=df, target_dataset=target_dataset
        )
        batch_features, batch_lengths, final_df, _ = self._construct_sequences(
            processed_df, target_dataset
        )

        labels = (
            final_df["is_laundering"].to_numpy(dtype=np.int64)
            if has_labels and "is_laundering" in final_df.columns
            else None
        )

        target_model_ids = model_ids
        if not target_model_ids or len(target_model_ids) == 0:
            target_model_ids = [
                "HI-Small",
                "centralized_lstm",
                "centralized_gru",
                "centralized_bilstm",
                "hi_small_gru_fedavg_iid_seed42",
                "hi_small_bilstm_fedavg_iid_seed42",
                "hi_small_lstm_fedprox_mu0.0001_iid_seed42",
                "hi_small_gru_fedprox_mu0.0001_iid_seed42",
                "hi_small_bilstm_fedprox_mu0.0001_iid_seed42",
                "hi_small_lstm_fedavg_seed42",
                "hi_small_gru_fedavg_seed42",
                "hi_small_bilstm_fedavg_seed42",
                "hi_small_lstm_fedprox_mu0.0001_seed42",
                "ensemble_hybrid",
            ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        comparison_results = []
        batch_size = 4096
        total_samples = len(batch_features)

        for mid in target_model_ids:
            model_meta = self.model_registry.get_model_metadata(mid)
            if not model_meta:
                continue

            try:
                model = self.model_service.get_model(model_meta.model_id)
                model = model.to(device)
                model.eval()

                probs_list = []
                with torch.inference_mode():
                    for i in range(0, total_samples, batch_size):
                        b_feat = torch.from_numpy(batch_features[i : i + batch_size]).to(device)
                        b_len = torch.tensor(batch_lengths[i : i + batch_size], dtype=torch.int64, device=device)
                        logits = model(b_feat, b_len).view(-1)
                        b_probs = torch.sigmoid(logits).cpu().numpy()
                        probs_list.append(b_probs)

                probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=float)
                eff_threshold = threshold if threshold is not None else model_meta.frozen_threshold

                metrics = self._calculate_metrics(labels, probs, eff_threshold) if labels is not None else None
                preds = (probs >= eff_threshold).astype(int)
                flagged = int(np.sum(preds == 1))

                comparison_results.append({
                    "model_id": model_meta.model_id,
                    "model_name": model_meta.model_name,
                    "model_type": model_meta.model_type,
                    "dataset_name": model_meta.dataset_name,
                    "decision_threshold": eff_threshold,
                    "total_samples": total_samples,
                    "flagged_count": flagged,
                    "cleared_count": total_samples - flagged,
                    "laundering_percentage": round(float(flagged / total_samples * 100.0), 2) if total_samples > 0 else 0.0,
                    "mean_probability": round(float(np.mean(probs)), 6) if total_samples > 0 else 0.0,
                    "max_probability": round(float(np.max(probs)), 6) if total_samples > 0 else 0.0,
                    "min_probability": round(float(np.min(probs)), 6) if total_samples > 0 else 0.0,
                    "metrics": metrics,
                    # Flat convenience fields for table display
                    "pr_auc": metrics["pr_auc"] if metrics else None,
                    "roc_auc": metrics["roc_auc"] if metrics else None,
                    "precision": metrics["precision"] if metrics else None,
                    "recall": metrics["recall"] if metrics else None,
                    "f1_score": metrics["f1_score"] if metrics else None,
                    "accuracy": metrics["accuracy"] if metrics else None,
                    "specificity": metrics["specificity"] if metrics else None,
                    "balanced_accuracy": metrics.get("balanced_accuracy") if metrics else None,
                    "mcc": metrics.get("mcc") if metrics else None,
                    "cohen_kappa": metrics.get("cohen_kappa") if metrics else None,
                    "brier_score": metrics["brier_score"] if metrics else None,
                })
            except Exception as e:
                logger.warning("Failed to evaluate model '%s': %s", mid, e)
                continue

        execution_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "status": "success",
            "filename": filename,
            "total_transactions": len(final_df),
            "has_labels": has_labels,
            "models_evaluated_count": len(comparison_results),
            "execution_time_ms": execution_time_ms,
            "models": comparison_results,
        }

    def _calculate_risk_distribution(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate 10 decile histogram bins for prediction probabilities."""
        bins = np.linspace(0.0, 1.0, 11)
        bin_counts, _ = np.histogram(probs, bins=bins)
        risk_distribution = []
        for b_idx in range(10):
            low = round(float(bins[b_idx]), 1)
            high = round(float(bins[b_idx + 1]), 1)
            cnt = int(bin_counts[b_idx])
            risk_distribution.append({
                "bin_range": f"{low:.1f}-{high:.1f}",
                "lower": low,
                "upper": high,
                "count": cnt,
                "percentage": round(cnt / len(probs) * 100.0, 2) if len(probs) > 0 else 0.0,
            })
        return risk_distribution

    def _preprocess_adapted_frame(
        self,
        detection: SchemaDetectionResult,
        orig_df: pd.DataFrame,
        target_dataset: str,
        tenant_id: str = "default_tenant",
        enable_persistent_history: bool = False,
    ) -> Tuple[pd.DataFrame, bool, Dict[int, float], int]:
        """Preprocess adapted dataframe using the dataset's fitted preprocessors."""
        has_labels = detection.has_labels
        raw_amounts: Dict[int, float] = {}

        if detection.format_category == FormatCategory.PROCESSED_FEATURES:
            return detection.mapped_df.copy(), has_labels, raw_amounts, 0

        mapped_df = detection.mapped_df.copy()

        amt_col = detection.field_mappings.get("amount_paid", COL_AMOUNT_PAID)
        if amt_col in orig_df.columns:
            for idx, val in enumerate(pd.to_numeric(orig_df[amt_col], errors="coerce")):
                raw_amounts[idx] = float(val) if pd.notna(val) else 0.0
        elif COL_AMOUNT_PAID in mapped_df.columns:
            for idx, val in enumerate(pd.to_numeric(mapped_df[COL_AMOUNT_PAID], errors="coerce")):
                raw_amounts[idx] = float(val) if pd.notna(val) else 0.0

        meta_dir = BASE_DIR / f"data/processed/ibm_aml/{target_dataset}/metadata"
        if not meta_dir.exists():
            raise ValueError(f"Metadata directory for dataset '{target_dataset}' not found at {meta_dir}")

        transformers_path = meta_dir / "transformers.json"
        tf_dict = json.loads(transformers_path.read_text(encoding="utf-8"))

        encoder = OrdinalCategoricalEncoder.from_mapping(
            {c: d["mapping"] for c, d in tf_dict["categorical_encoding"]["columns"].items()}
        )
        standardizer = RunningStandardizer.from_parameters(tf_dict["scaling_parameters"])

        builder = AccountHistoryBuilder()
        if enable_persistent_history:
            try:
                from infrastructure.data.features.account_history_store import get_account_history_store
            except ModuleNotFoundError:
                from backend.infrastructure.data.features.account_history_store import get_account_history_store
            store = get_account_history_store()
            profiles = store.get_all_profiles(tenant_id)
            if profiles:
                builder.import_profiles(profiles)

        preprocessor = AmlPreprocessor(history_builder=builder)
        preprocessor._encoder = encoder
        preprocessor._standardizer = standardizer

        cleaned, removed = preprocessor.clean(mapped_df)
        if len(cleaned) == 0:
            raise ValueError(
                f"All uploaded rows were invalid or removed by data cleaning rules: {dict(removed)}"
            )

        sorted_df = preprocessor.sort_chronologically(cleaned)
        engineered = preprocessor.engineer(sorted_df)
        transformed = preprocessor.transform(engineered)

        if enable_persistent_history:
            updated_profiles = builder.export_profiles()
            store.save_profiles(tenant_id, updated_profiles)

        return transformed, has_labels, raw_amounts, builder.distinct_accounts

    def _construct_sequences(
        self, df: pd.DataFrame, dataset: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """Construct (N, 10, 57) sequences from processed dataframe with diagnostics."""
        meta_dir = BASE_DIR / f"data/processed/ibm_aml/{dataset}"
        meta = ProcessedFeatureMetadata.read(meta_dir)
        nominal_map = {nom.column: nom for nom in meta.nominal_columns}
        width = meta.feature_dimension(one_hot_nominal_codes=True)

        rows = len(df)
        features = np.zeros((rows, width), dtype=np.float32)

        unknown_cat_counts = np.zeros(rows, dtype=np.float32)
        total_nominal_cols = max(1, len(meta.nominal_columns))

        offset = 0
        for col in meta.feature_columns:
            if col in nominal_map:
                nom = nominal_map[col]
                codes = df[col].to_numpy(dtype=np.int64)
                for i, c in enumerate(codes):
                    if 0 <= c < nom.vocabulary_size:
                        features[i, offset + c] = 1.0
                    else:
                        unknown_cat_counts[i] += 1.0
                offset += nom.vocabulary_size
            else:
                features[:, offset] = df[col].to_numpy(dtype=np.float32)
                offset += 1

        accounts = df["from_account"]
        codes, _ = pd.factorize(accounts, sort=False)
        previous_row = build_previous_row_index(codes.astype(np.int64))

        targets = (
            df["is_laundering"].to_numpy(dtype=np.float32)
            if "is_laundering" in df.columns
            else np.zeros(rows, dtype=np.float32)
        )

        seq_dataset = AccountSequenceDataset(
            split=SplitName.TEST,
            features=features,
            targets=targets,
            previous_row=previous_row,
            sequence_length=MAX_SEQUENCE_LENGTH,
            stride=1,
        )

        batch = seq_dataset._assemble(seq_dataset.target_rows)

        missing_counts = df.isna().sum(axis=1).to_numpy(dtype=np.float32)
        total_cols = float(max(1, len(df.columns)))

        row_diagnostics = {
            "unknown_category_counts": unknown_cat_counts,
            "total_nominal_cols": total_nominal_cols,
            "missing_feature_counts": missing_counts,
            "total_cols": total_cols,
        }

        return batch.features, batch.lengths, df, row_diagnostics

    def _calculate_metrics(
        self, labels: np.ndarray, probs: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive supervised AML metrics using decision threshold."""
        labels = np.asarray(labels, dtype=int)
        probs = np.asarray(probs, dtype=float)
        if not np.all(np.isfinite(probs)):
            raise ValueError("Evaluation metrics calculation received non-finite probabilities.")
        probs = np.clip(probs, 0.0, 1.0)

        preds = (probs >= threshold).astype(int)

        tp = int(np.sum((labels == 1) & (preds == 1)))
        fp = int(np.sum((labels == 0) & (preds == 1)))
        tn = int(np.sum((labels == 0) & (preds == 0)))
        fn = int(np.sum((labels == 1) & (preds == 0)))

        total = len(labels)
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # When zero positive predictions are made, precision and F-scores are undefined (0/0)
        zero_alerts_note: Optional[str] = None
        if (tp + fp) == 0:
            precision: Optional[float] = None
            f1: Optional[float] = None
            f2: Optional[float] = None
            zero_alerts_note = (
                "Zero transactions exceeded decision threshold (0 alerts generated). "
                "Precision and F-scores are mathematically undefined (0/0) when no positive predictions are made."
            )
        else:
            precision = float(tp / (tp + fp))
            rec_denom = tp + fn
            rec_val = float(tp / rec_denom) if rec_denom > 0 else 0.0
            
            p_val = precision
            f1_denom = p_val + rec_val
            f1 = float(2.0 * p_val * rec_val / f1_denom) if f1_denom > 0 else 0.0
            f2_denom = 4.0 * p_val + rec_val
            f2 = float(5.0 * p_val * rec_val / f2_denom) if f2_denom > 0 else 0.0

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        pr_auc: Optional[float] = None
        roc_auc: Optional[float] = None
        pr_curve_points: List[Dict[str, float]] = []
        roc_curve_points: List[Dict[str, float]] = []
        edge_case_note: Optional[str] = None

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            try:
                pr_auc = float(average_precision_score(labels, probs))
            except Exception:
                pr_auc = None

            try:
                roc_auc = float(roc_auc_score(labels, probs))
            except Exception:
                roc_auc = None

            eval_thresholds = np.linspace(0.01, 0.99, 50)
            for eth in eval_thresholds:
                t_preds = (probs >= eth).astype(int)
                t_tp = int(np.sum((labels == 1) & (t_preds == 1)))
                t_fp = int(np.sum((labels == 0) & (t_preds == 1)))
                t_fn = int(np.sum((labels == 1) & (t_preds == 0)))
                t_tn = int(np.sum((labels == 0) & (t_preds == 0)))

                t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 1.0
                t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
                t_fpr = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0.0

                pr_curve_points.append({
                    "threshold": round(float(eth), 4),
                    "recall": round(float(t_rec), 4),
                    "precision": round(float(t_prec), 4),
                })
                roc_curve_points.append({
                    "threshold": round(float(eth), 4),
                    "fpr": round(float(t_fpr), 4),
                    "tpr": round(float(t_rec), 4),
                })
        else:
            single_val = int(unique_labels[0])
            val_name = "LEGITIMATE (0)" if single_val == 0 else "LAUNDERING (1)"
            edge_case_note = (
                f"Evaluation dataset contains only a single class: {val_name}. "
                "ROC-AUC, PR-AUC, and curves require both positive and negative samples."
            )

        if edge_case_note is None and zero_alerts_note is not None:
            edge_case_note = zero_alerts_note

        positive_count = int(np.sum(labels == 1))
        negative_count = int(np.sum(labels == 0))
        brier = float(np.mean((probs - labels) ** 2)) if total > 0 else 0.0
        alerts = float((tp + fp) / total * 10000.0) if total > 0 else 0.0

        # Detailed rate and specificity calculations
        sensitivity: float = recall
        specificity: Optional[float] = float(tn / (tn + fp)) if (tn + fp) > 0 else (1.0 if fp == 0 else 0.0)
        balanced_accuracy: Optional[float] = float(0.5 * (sensitivity + specificity)) if specificity is not None else sensitivity

        # Matthews Correlation Coefficient (MCC)
        mcc_num = float(tp * tn - fp * fn)
        mcc_den = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        mcc: Optional[float] = float(mcc_num / mcc_den) if mcc_den > 0 else 0.0

        # Cohen's Kappa
        p0 = float(tp + tn) / float(total) if total > 0 else 0.0
        pe = float((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / float(total ** 2) if total > 0 else 0.0
        cohen_kappa: Optional[float] = float((p0 - pe) / (1.0 - pe)) if (1.0 - pe) > 0 else 0.0

        false_positive_rate: Optional[float] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        false_negative_rate: Optional[float] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        false_discovery_rate: Optional[float] = float(fp / (fp + tp)) if (fp + tp) > 0 else (None if (tp + fp) == 0 else 0.0)
        negative_predictive_value: Optional[float] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        alert_count: int = tp + fp
        alert_rate: float = float(alert_count / total) if total > 0 else 0.0
        mean_prob: float = float(np.mean(probs)) if total > 0 else 0.0
        max_prob: float = float(np.max(probs)) if total > 0 else 0.0
        min_prob: float = float(np.min(probs)) if total > 0 else 0.0

        log_loss_val: Optional[float] = None
        if len(unique_labels) > 1:
            try:
                eps_probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
                log_loss_val = float(log_loss(labels, eps_probs))
            except Exception:
                log_loss_val = None

        ece_val: Optional[float] = None
        mce_val: Optional[float] = None
        if len(unique_labels) > 1:
            try:
                from infrastructure.evaluation.calibration_metrics import compute_ece
                ece_res, mce_res, _ = compute_ece(labels, probs, n_bins=10)
                ece_val = round(float(ece_res), 6)
                mce_val = round(float(mce_res), 6)
            except Exception:
                pass

        # Compute robust bootstrap metrics, lift, and recall at alert budgets
        try:
            n_boot_samples = 100 if total > 10000 else 300
            robust_res = run_robust_evaluate(labels, probs, thr=threshold, n_boot=n_boot_samples)
        except Exception as e:
            logger.warning("Robust metrics computation fallback: %s", e)
            robust_res = {}

        return {
            "total_samples": total,
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "prevalence": robust_res.get("prevalence", float(positive_count / total) if total > 0 else 0.0),
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6),
            "sensitivity": round(sensitivity, 6),
            "specificity": round(specificity, 6) if specificity is not None else None,
            "balanced_accuracy": round(balanced_accuracy, 6) if balanced_accuracy is not None else None,
            "f1_score": round(f1, 6) if f1 is not None else None,
            "f2_score": round(f2, 6) if f2 is not None else None,
            "mcc": round(mcc, 6) if mcc is not None else None,
            "cohen_kappa": round(cohen_kappa, 6) if cohen_kappa is not None else None,
            "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
            "average_precision": round(pr_auc, 6) if pr_auc is not None else None,
            "pr_auc_ci_lower": robust_res.get("pr_auc_ci_lower"),
            "pr_auc_ci_upper": robust_res.get("pr_auc_ci_upper"),
            "pr_auc_lift": robust_res.get("pr_auc_lift"),
            "random_baseline_pr_auc": robust_res.get("random_baseline_pr_auc"),
            "recall_at_alerts": robust_res.get("recall_at_alerts"),
            "evaluation_warnings": robust_res.get("warnings", []),
            "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
            "log_loss": round(log_loss_val, 6) if log_loss_val is not None else None,
            "brier_score": round(brier, 6),
            "expected_calibration_error": ece_val,
            "maximum_calibration_error": mce_val,
            "ece": ece_val,
            "mce": mce_val,
            "false_positive_rate": round(false_positive_rate, 6) if false_positive_rate is not None else None,
            "false_negative_rate": round(false_negative_rate, 6) if false_negative_rate is not None else None,
            "false_discovery_rate": round(false_discovery_rate, 6) if false_discovery_rate is not None else None,
            "negative_predictive_value": round(negative_predictive_value, 6) if negative_predictive_value is not None else None,
            "alert_count": alert_count,
            "alert_rate": round(alert_rate, 6),
            "alerts_per_10k": round(alerts, 2),
            "mean_predicted_probability": round(mean_prob, 6),
            "max_predicted_probability": round(max_prob, 6),
            "min_predicted_probability": round(min_prob, 6),
            "zero_alerts_note": zero_alerts_note,
            "ranking_vs_classification": {
                "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
                "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
                "interpretation": "Ranking metrics (PR-AUC / ROC-AUC) evaluate relative transaction risk discrimination across all possible decision boundaries, whereas Precision/Recall/F1 evaluate alert generation at the operational threshold."
            },
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "pr_curve": pr_curve_points,
            "roc_curve": roc_curve_points,
            "edge_case_note": edge_case_note,
            "single_class_note": edge_case_note,
        }

    def analyze_validation_thresholds(
        self,
        dataset: str = "HI-Small",
        objective: str = "max_f1",
        target_recall: float = 0.80,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Perform threshold analysis strictly on validation partition (test set remains sequestered)."""
        clean_name = dataset.replace("-", "_")
        npz_path = BASE_DIR / f"results/phase7/{clean_name}/validation_predictions/iid_validation_predictions.npz"
        if not npz_path.exists():
            npz_path = BASE_DIR / f"results/phase7/{dataset}/validation_predictions/iid_validation_predictions.npz"

        if not npz_path.exists():
            raise ValueError(f"Validation predictions for dataset '{dataset}' not found at {npz_path}")

        data = np.load(npz_path)
        probs = data["probabilities"].astype(float)
        labels = data["labels"].astype(int)

        frozen_val = float(FROZEN_THRESHOLDS.get(dataset, 0.337784))

        if candidate_thresholds is None:
            # Candidate thresholds spanning 0.02 to 0.98 + exact frozen threshold
            grid = [round(float(x), 4) for x in np.linspace(0.02, 0.98, 49)]
            grid.append(round(frozen_val, 6))
            candidates = sorted(list(set(grid)))
        else:
            candidates = sorted(candidate_thresholds)

        total = len(labels)
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        pos_probs = np.sort(probs[pos_mask])
        neg_probs = np.sort(probs[neg_mask])

        total_pos = len(pos_probs)
        total_neg = len(neg_probs)

        evaluations: List[Dict[str, Any]] = []
        best_candidate: Optional[Dict[str, Any]] = None
        best_score = -float("inf")

        for th in candidates:
            # Vectorized O(log N) calculation via np.searchsorted
            tp = total_pos - int(np.searchsorted(pos_probs, th))
            fp = total_neg - int(np.searchsorted(neg_probs, th))
            fn = total_pos - tp
            tn = total_neg - fp

            prec = tp / (tp + fp) if (tp + fp) > 0 else None
            rec = tp / total_pos if total_pos > 0 else 0.0

            p_val = prec if prec is not None else 0.0
            f1 = (2.0 * p_val * rec) / (p_val + rec) if (p_val + rec) > 0 else 0.0
            f2 = (5.0 * p_val * rec) / (4.0 * p_val + rec) if (4.0 * p_val + rec) > 0 else 0.0

            alerts_per_10k = ((tp + fp) / total) * 10000.0 if total > 0 else 0.0
            workload = tp + fp
            cost = (cost_fp * fp) + (cost_fn * fn)

            row = {
                "threshold": float(th),
                "precision": round(prec, 6) if prec is not None else None,
                "recall": round(rec, 6),
                "f1_score": round(f1, 6) if prec is not None else None,
                "f2_score": round(f2, 6) if prec is not None else None,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "alerts_per_10k": round(alerts_per_10k, 2),
                "investigation_workload": workload,
                "cost": round(cost, 2),
            }
            evaluations.append(row)

            if objective == "max_f1":
                score = f1
            elif objective == "max_f2":
                score = f2
            elif objective == "target_recall":
                if rec >= target_recall:
                    score = -alerts_per_10k
                else:
                    score = -100000.0 + rec
            elif objective == "cost_sensitive":
                score = -cost
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_candidate = row

        return {
            "dataset": dataset,
            "validation_samples_count": total,
            "validation_positive_count": total_pos,
            "validation_negative_count": total_neg,
            "objective": objective,
            "frozen_production_threshold": frozen_val,
            "optimal_threshold": best_candidate["threshold"] if best_candidate else frozen_val,
            "optimal_metrics": best_candidate,
            "candidates_count": len(evaluations),
            "candidates": evaluations,
            "methodology_note": (
                "Validation-only threshold analysis. "
                "Final test partition was not accessed and remains sequestered."
            ),
        }

