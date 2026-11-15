"""Universal Schema Detector and Feature Compatibility Checker for FedSecureAI.

Detects semantic financial transaction concepts:
- Transaction ID, Account IDs (Sender / Receiver), Timestamp, Amount, Currency,
  Transaction Type, Merchant, Country, Target / Label, and feature types.
Verifies compatibility against selected models (recurrent sequence vs. tabular)
and returns actionable reports with suggested mappings and training options.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from infrastructure.models.registry.universal_model_registry import (
    ModelSchemaMetadata,
    UniversalModelRegistry,
)

logger = logging.getLogger(__name__)


# Standard Synonyms for AML concepts
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "transaction_id": [
        r"^id$", r"^txn_?id$", r"^trans(action)?_?id$", r"^reference(_?id)?$",
        r"^step$", r"^tx_?hash$", r"^invoice_?id$", r"^seq_?id$"
    ],
    "from_account": [
        r"^from_?(acc(ount)?)?$", r"^sender(_?acc(ount)?)?$", r"^nameorig$",
        r"^source(_?acc(ount)?)?$", r"^originator(_?acc(ount)?)?$", r"^payer(_?acc(ount)?)?$",
        r"^from_?account_?id$", r"^src_?acc$", r"^acc(ount)?$"
    ],
    "to_account": [
        r"^to_?(acc(ount)?)?$", r"^receiver(_?acc(ount)?)?$", r"^namedest$",
        r"^destination(_?acc(ount)?)?$", r"^beneficiary(_?acc(ount)?)?$", r"^payee(_?acc(ount)?)?$",
        r"^to_?account_?id$", r"^dest_?acc$", r"^account\.1$", r"^acc(ount)?_?1$"
    ],
    "timestamp": [
        r"^timestamp$", r"^trans(action)?_?time$", r"^trans(action)?_?date$", r"^datetime$", r"^date(_?time)?$",
        r"^time$", r"^txn_?date$", r"^created_?at$", r"^trade_?date$", r"^step$"
    ],
    "amount": [
        r"^amount(_?paid)?$", r"^transaction_?amount$", r"^trans_?amount$",
        r"^amount_?received$", r"^amount(_?paid|_?received)?_?log$", r"^amt$", r"^value$", r"^sum$", r"^payment_?amount$", r"^total_?amount$"
    ],
    "currency": [
        r"^currency$", r"^payment_?currency$", r"^receiving_?currency$",
        r"^curr$", r"^trans_?currency$", r"^iso_?curr(ency)?$", r"^currency_?iso$", r"^ccy$"
    ],
    "transaction_type": [
        r"^type$", r"^payment_?format$", r"^format$", r"^transaction_?type$",
        r"^trans_?type$", r"^txn_?type$", r"^action$", r"^operation$", r"^method$", r"^channel$",
        r"^payment_?rail$", r"^rail$"
    ],
    "merchant": [
        r"^merchant(_?id)?$", r"^vendor(_?id)?$", r"^retailer$", r"^store(_?id)?$", r"^seller$"
    ],
    "country": [
        r"^country$", r"^origin_?country$", r"^dest(ination)?_?country$",
        r"^source_?country$", r"^nation$"
    ],
    "label": [
        r"^is_?laundering$", r"^is_?fraud$", r"^fraud$", r"^fraud_?indicator$", r"^suspicious$",
        r"^label$", r"^target$", r"^class$", r"^is_?illicit$", r"^sar$", r"^flag$"
    ],
}


@dataclass
class ColumnProfile:
    """Statistical and semantic profile of a single dataframe column."""

    name: str
    inferred_type: str  # "numerical", "categorical", "datetime", "boolean", "text"
    inferred_concept: Optional[str]  # e.g. "amount", "from_account", "label"
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: List[Any]


@dataclass
class UniversalSchemaReport:
    """Comprehensive schema analysis of an uploaded CSV."""

    filename: str
    total_rows: int
    total_columns: int
    column_profiles: Dict[str, ColumnProfile]
    detected_concepts: Dict[str, str]  # concept -> column_name
    has_target_label: bool
    target_column_name: Optional[str]
    is_binary_target: bool
    positive_label_count: int
    negative_label_count: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    datetime_columns: List[str]
    id_columns: List[str]
    can_construct_sequences: bool
    missing_sequence_concepts: List[str]
    warnings: List[str] = field(default_factory=list)
    routing_decision: str = "ALLOW_RECURRENT_MODELS"
    recommended_model_id: str = "hi_small_lstm_fedavg_iid_seed42"
    recommended_model_name: str = "HI-Small Champion (Federated LSTM FedAvg IID)"
    routing_reason: str = ""
    unique_sender_ratio: float = 1.0
    average_sequence_depth: float = 1.0
    sequence_sufficiency_status: str = "SUFFICIENT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        res = asdict(self)
        return res


@dataclass
class FeatureCompatibilityReport:
    """Compatibility verdict between an uploaded CSV and a chosen model."""

    compatible: bool
    model_id: str
    model_name: str
    model_type: str
    is_sequence_model: bool
    expected_features_count: int
    uploaded_features_count: int
    missing_required_features: List[str]
    extra_unused_features: List[str]
    type_mismatches: List[str]
    warnings: List[str]
    suggested_action: str
    suggested_mappings: Dict[str, str]
    allows_direct_prediction: bool
    allows_training_new_model: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)


class UniversalSchemaDetector:
    """Analyzes CSV structure, infers financial semantics, and evaluates model compatibility."""

    def __init__(self) -> None:
        self.model_registry = UniversalModelRegistry()

    @staticmethod
    def _clean_header(h: str) -> str:
        """Normalize header string for regex matching."""
        return str(h).strip().lower().replace(" ", "_").replace("-", "_")

    def inspect_dataframe(
        self,
        df: pd.DataFrame,
        filename: str = "upload.csv",
    ) -> UniversalSchemaReport:
        """Perform deep semantic inspection on a DataFrame."""
        total_rows = len(df)
        total_columns = len(df.columns)
        column_profiles: Dict[str, ColumnProfile] = {}
        detected_concepts: Dict[str, str] = {}
        warnings: List[str] = []

        numerical_cols: List[str] = []
        categorical_cols: List[str] = []
        boolean_cols: List[str] = []
        datetime_cols: List[str] = []
        id_cols: List[str] = []

        norm_headers = {c: self._clean_header(c) for c in df.columns}

        # 1. Match concepts using regex rules
        for concept, patterns in CONCEPT_PATTERNS.items():
            for orig_col, norm_col in norm_headers.items():
                if any(re.match(pat, norm_col) for pat in patterns):
                    if concept not in detected_concepts:
                        detected_concepts[concept] = orig_col
                        break

        # Disambiguate account columns if duplicate or Account.1
        if "from_account" in detected_concepts and "to_account" not in detected_concepts:
            for orig_col in df.columns:
                if orig_col != detected_concepts["from_account"]:
                    norm = norm_headers[orig_col]
                    if "account" in norm or "dest" in norm or "receiver" in norm:
                        detected_concepts["to_account"] = orig_col
                        break

        # 2. Inspect each column
        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round(missing_count / total_rows * 100.0, 2) if total_rows > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))

            # Sample values
            sample_vals = series.dropna().head(5).tolist()

            # Determine concept
            concept = next((k for k, v in detected_concepts.items() if v == col), None)

            # Inferred type detection
            inferred_type = "categorical"
            if pd.api.types.is_bool_dtype(series):
                inferred_type = "boolean"
                boolean_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                # Check if it is a boolean 0/1 indicator
                unique_vals = set(series.dropna().unique())
                if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
                    inferred_type = "boolean"
                    boolean_cols.append(col)
                else:
                    inferred_type = "numerical"
                    numerical_cols.append(col)
            else:
                # Check if string parses to datetime
                is_dt = False
                if concept == "timestamp" or "date" in norm_headers[col] or "time" in norm_headers[col]:
                    try:
                        parsed = pd.to_datetime(series.dropna().head(20), errors="coerce")
                        if parsed.notna().sum() >= min(len(parsed), 5):
                            is_dt = True
                    except Exception:
                        is_dt = False

                if is_dt:
                    inferred_type = "datetime"
                    datetime_cols.append(col)
                else:
                    inferred_type = "categorical"
                    categorical_cols.append(col)

            # Check if identifier column
            if concept in ("transaction_id", "from_account", "to_account") or unique_count >= total_rows * 0.9:
                if inferred_type != "numerical" and col not in id_cols:
                    id_cols.append(col)

            column_profiles[col] = ColumnProfile(
                name=col,
                inferred_type=inferred_type,
                inferred_concept=concept,
                missing_count=missing_count,
                missing_pct=missing_pct,
                unique_count=unique_count,
                sample_values=sample_vals,
            )

        # 3. Target label validation
        has_target_label = "label" in detected_concepts
        target_col = detected_concepts.get("label")
        is_binary_target = False
        pos_count = 0
        neg_count = 0

        if has_target_label and target_col is not None:
            lbl_series = pd.to_numeric(df[target_col], errors="coerce")
            unique_lbls = set(lbl_series.dropna().unique())
            if unique_lbls.issubset({0, 1}):
                is_binary_target = True
                pos_count = int((lbl_series == 1).sum())
                neg_count = int((lbl_series == 0).sum())
            else:
                warnings.append(f"Target column '{target_col}' contains non-binary values ({unique_lbls}).")

        # 4. Sequence constructibility check
        missing_seq_concepts: List[str] = []
        for req in ["timestamp", "from_account", "to_account", "amount"]:
            if req not in detected_concepts:
                missing_seq_concepts.append(req)

        can_construct_sequences = len(missing_seq_concepts) == 0

        # 5. Model Routing Strategy & Sequence History Depth Analysis
        from_acc_col = detected_concepts.get("from_account")
        unique_senders = int(df[from_acc_col].nunique()) if (from_acc_col and from_acc_col in df.columns) else total_rows
        unique_sender_ratio = round(float(unique_senders / max(1, total_rows)), 4)
        avg_sequence_depth = round(float(total_rows / max(1, unique_senders)), 2)

        tabular_feature_names = {
            "amount_paid_log", "amount_received_log", "amount_mismatch",
            "same_currency", "same_bank", "self_transfer", "payment_format_code",
            "payment_currency_code", "receiving_currency_code", "minute_of_day_sin",
            "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
            "is_night", "sender_prior_count_log", "sender_prior_amount_log",
            "sender_seconds_since_prior_log", "sender_has_prior",
            "receiver_prior_count_log", "receiver_prior_amount_log",
            "receiver_seconds_since_prior_log", "receiver_has_prior",
        }
        has_direct_tabular_features = tabular_feature_names.issubset(set(df.columns))

        has_prior_history = (
            "sender_has_prior" in df.columns and float(df["sender_has_prior"].mean()) >= 0.5
        )

        if not can_construct_sequences and not has_direct_tabular_features:
            routing_decision = "REQUEST_MISSING_FEATURES"
            recommended_model_id = ""
            recommended_model_name = "None (Features Missing)"
            sequence_sufficiency_status = "MISSING_FEATURES"
            routing_reason = (
                f"Missing critical transaction features ({', '.join(missing_seq_concepts)}). "
                "Please provide column mappings or upload a dataset with timestamp, sender account, receiver account, and amount."
            )
        elif total_rows >= 100 and not has_prior_history and (unique_sender_ratio > 0.92 or avg_sequence_depth < 1.15):
            routing_decision = "RECOMMEND_TABULAR_FALLBACK"
            recommended_model_id = "lightgbm_tabular_fallback"
            recommended_model_name = "Fallback: no sequence history (LightGBM)"
            sequence_sufficiency_status = "INSUFFICIENT_COLD_START"
            routing_reason = (
                f"Insufficient historical sequence context detected ({unique_sender_ratio * 100:.1f}% unique senders, "
                f"avg depth {avg_sequence_depth:.1f} tx/sender). Recurrent sequence models (LSTM/GRU/BiLSTM) require multi-step "
                "temporal history per account and will produce conservative baseline probabilities without historical bursts. "
                "The validated Tabular Fallback Model (LightGBM) is recommended for single-transaction scoring."
            )
        else:
            routing_decision = "ALLOW_RECURRENT_MODELS"
            if "li_medium" in filename.lower():
                recommended_model_id = "config_5_bilstm_fedavg_iid"
                recommended_model_name = "LI-Medium Champion (Federated BiLSTM FedAvg IID)"
            else:
                recommended_model_id = "hi_small_lstm_fedavg_iid_seed42"
                recommended_model_name = "HI-Small Champion (Federated LSTM FedAvg IID)"
            sequence_sufficiency_status = "SUFFICIENT"
            routing_reason = (
                f"Sufficient temporal sequence context detected ({unique_sender_ratio * 100:.1f}% unique senders, "
                f"avg depth {avg_sequence_depth:.1f} transactions per sender). Compatible with recurrent sequence models "
                "(FedAvg LSTM, GRU, BiLSTM)."
            )

        return UniversalSchemaReport(
            filename=filename,
            total_rows=total_rows,
            total_columns=total_columns,
            column_profiles=column_profiles,
            detected_concepts=detected_concepts,
            has_target_label=has_target_label,
            target_column_name=target_col,
            is_binary_target=is_binary_target,
            positive_label_count=pos_count,
            negative_label_count=neg_count,
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            boolean_columns=boolean_cols,
            datetime_columns=datetime_cols,
            id_columns=id_cols,
            can_construct_sequences=can_construct_sequences,
            missing_sequence_concepts=missing_seq_concepts,
            warnings=warnings,
            routing_decision=routing_decision,
            recommended_model_id=recommended_model_id,
            recommended_model_name=recommended_model_name,
            routing_reason=routing_reason,
            unique_sender_ratio=unique_sender_ratio,
            average_sequence_depth=avg_sequence_depth,
            sequence_sufficiency_status=sequence_sufficiency_status,
        )

    def check_compatibility(
        self,
        schema_report: UniversalSchemaReport,
        model_id: str,
        user_mappings: Optional[Dict[str, str]] = None,
    ) -> FeatureCompatibilityReport:
        """Check compatibility between an inspected CSV and a specific registered model."""
        meta = self.model_registry.get_model_metadata(model_id)
        if not meta:
            return FeatureCompatibilityReport(
                compatible=False,
                model_id=model_id,
                model_name=f"Unknown ({model_id})",
                model_type="UNKNOWN",
                is_sequence_model=False,
                expected_features_count=0,
                uploaded_features_count=schema_report.total_columns,
                missing_required_features=[],
                extra_unused_features=list(schema_report.column_profiles.keys()),
                type_mismatches=[],
                warnings=[f"Model '{model_id}' was not found in the model registry."],
                suggested_action="Select a valid registered model from the models list.",
                suggested_mappings={},
                allows_direct_prediction=False,
                allows_training_new_model=schema_report.has_target_label and schema_report.is_binary_target,
            )

        active_mappings = dict(schema_report.detected_concepts)
        if user_mappings:
            active_mappings.update(user_mappings)

        warnings: List[str] = list(schema_report.warnings)
        missing_required: List[str] = []
        extra_unused: List[str] = []
        type_mismatches: List[str] = []

        uploaded_cols = set(schema_report.column_profiles.keys())
        expected_cols = set(meta.feature_columns)

        # 1. Sequence Model Requirements
        if meta.is_sequence_model:
            for req in meta.required_sequence_fields:
                is_present = req in active_mappings or (
                    req == "amount" and ("amount_paid_log" in uploaded_cols or "amount_received_log" in uploaded_cols)
                )
                if not is_present:
                    missing_required.append(req)

        # Check if direct processed format
        is_direct_processed = expected_cols.issubset(uploaded_cols)

        compatible = False
        suggested_action = ""

        if is_direct_processed:
            compatible = True
            suggested_action = "Dataset directly matches model feature dimensions. Ready for prediction."
        elif meta.model_id == "lightgbm_tabular_fallback" and (is_direct_processed or schema_report.can_construct_sequences):
            compatible = True
            suggested_action = (
                "Tabular model is fully compatible with single-transaction and multi-transaction records. "
                "Does not require sequence history."
            )
        elif meta.is_sequence_model and len(missing_required) == 0:
            compatible = True
            if schema_report.sequence_sufficiency_status == "INSUFFICIENT_COLD_START":
                suggested_action = (
                    f"Compatible schema detected, but file lacks account history ({schema_report.average_sequence_depth:.1f} tx/sender). "
                    f"Recurrent {meta.model_type} will evaluate single-transaction windows with low prior probabilities. "
                    "Consider using Tabular Fallback (LightGBM)."
                )
            else:
                suggested_action = (
                    "Compatible transaction stream detected. Fields will be automatically mapped "
                    "and preprocessed into sequence windows for inference."
                )
        elif not meta.is_sequence_model and not is_direct_processed:
            # Check overlap
            missing = list(expected_cols - uploaded_cols)
            if len(missing) <= len(expected_cols) * 0.3:
                compatible = False
                suggested_action = (
                    f"Missing {len(missing)} features. Please provide column mappings or train a new model."
                )
            else:
                compatible = False
                suggested_action = (
                    "Schema is incompatible with the selected model. "
                    "You can train a new custom model if this dataset contains labels."
                )
        else:
            compatible = False
            suggested_action = (
                f"Missing required sequence fields ({', '.join(missing_required)}). "
                "Map compatible columns or train a new model on this dataset."
            )

        allows_training = schema_report.has_target_label and schema_report.is_binary_target

        return FeatureCompatibilityReport(
            compatible=compatible,
            model_id=meta.model_id,
            model_name=meta.model_name,
            model_type=meta.model_type,
            is_sequence_model=meta.is_sequence_model,
            expected_features_count=len(meta.feature_columns),
            uploaded_features_count=schema_report.total_columns,
            missing_required_features=missing_required,
            extra_unused_features=list(uploaded_cols - expected_cols) if is_direct_processed else [],
            type_mismatches=type_mismatches,
            warnings=warnings,
            suggested_action=suggested_action,
            suggested_mappings=active_mappings,
            allows_direct_prediction=compatible,
            allows_training_new_model=allows_training,
        )


