"""Adaptive schema detector and field mapping layer for AML evaluation.

Inspects incoming transaction CSVs, normalizes column names, detects known formats
(IBM AML Raw, Processed Features, Synthetic, and Aliased schemas), prevents target
leakage, verifies model/pipeline compatibility, and maps fields to the canonical
representation required by the authoritative inference engine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from infrastructure.data.schemas.ibm_aml_schema import (
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
    CURRENCIES,
    PAYMENT_FORMATS,
    RAW_COLUMN_ORDER,
)

logger = logging.getLogger(__name__)


class DetectionStatus(str, Enum):
    """User-facing validation status categories."""

    COMPATIBLE = "COMPATIBLE"
    PARTIALLY_COMPATIBLE = "PARTIALLY_COMPATIBLE"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class FormatCategory(str, Enum):
    """Internal structural categorization of the transaction CSV."""

    IBM_AML_RAW = "IBM_AML_RAW"
    PROCESSED_FEATURES = "PROCESSED_FEATURES"
    SYNTHETIC_TRANSACTIONS = "SYNTHETIC_TRANSACTIONS"
    ALIASED_TRANSACTIONS = "ALIASED_TRANSACTIONS"
    UNKNOWN = "UNKNOWN"


@dataclass
class SchemaDetectionResult:
    """Detailed result of schema detection, mapping, and compatibility verification."""

    status: DetectionStatus
    user_message: str
    format_category: FormatCategory
    target_pipeline: str  # "HI-Small" or "LI-Medium"
    decision_threshold: float
    mapped_df: Optional[pd.DataFrame] = None
    has_labels: bool = False
    ground_truth_labels: Optional[pd.Series] = None
    field_mappings: Dict[str, str] = field(default_factory=dict)
    missing_required_fields: List[str] = field(default_factory=list)
    conflicting_fields: Dict[str, List[str]] = field(default_factory=dict)
    excluded_leakage_fields: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to serializable dictionary."""
        return {
            "status": self.status.value,
            "user_message": self.user_message,
            "format_category": self.format_category.value,
            "target_pipeline": self.target_pipeline,
            "decision_threshold": self.decision_threshold,
            "has_labels": self.has_labels,
            "field_mappings": self.field_mappings,
            "missing_required_fields": self.missing_required_fields,
            "conflicting_fields": self.conflicting_fields,
            "excluded_leakage_fields": self.excluded_leakage_fields,
            "audit_notes": self.audit_notes,
        }


# Canonical required columns for full processed features (26 columns + identifiers)
PROCESSED_REQUIRED_COLS = [
    "timestamp",
    "from_account",
    "to_account",
    "amount_paid_log",
    "amount_received_log",
    "amount_mismatch",
    "same_currency",
    "same_bank",
    "self_transfer",
    "payment_format_code",
    "payment_currency_code",
    "receiving_currency_code",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_night",
    "sender_prior_count_log",
    "sender_prior_amount_log",
    "sender_seconds_since_prior_log",
    "sender_has_prior",
    "receiver_prior_count_log",
    "receiver_prior_amount_log",
    "receiver_seconds_since_prior_log",
    "receiver_has_prior",
]

# Frozen decision thresholds
FROZEN_THRESHOLDS = {
    "HI-Small": 0.3377837836742401,
    "LI-Medium": 0.2384829968214035,
}

# Synonyms for transaction type / payment format mapping into IBM AML vocabulary
PAYMENT_FORMAT_SYNONYMS = {
    # Wire / Transfer
    "wire": "Wire",
    "wire_transfer": "Wire",
    "wire transfer": "Wire",
    "transfer": "Wire",
    "bank_transfer": "Wire",
    "electronic": "Wire",
    "sepa": "Wire",
    "swift": "Wire",
    "interbank": "Wire",
    # ACH / Direct Debit
    "ach": "ACH",
    "direct_debit": "ACH",
    "direct debit": "ACH",
    "debit": "ACH",
    "standing_order": "ACH",
    "clearing": "ACH",
    # Credit Card / Card / POS
    "credit_card": "Credit Card",
    "credit card": "Credit Card",
    "card": "Credit Card",
    "debit_card": "Credit Card",
    "pos": "Credit Card",
    "payment": "Credit Card",
    "online": "Credit Card",
    "merchant": "Credit Card",
    # Cash / ATM / In / Out
    "cash": "Cash",
    "cash_out": "Cash",
    "cash out": "Cash",
    "cash_in": "Cash",
    "cash in": "Cash",
    "atm": "Cash",
    "deposit": "Cash",
    "withdrawal": "Cash",
    # Cheque / Check
    "cheque": "Cheque",
    "check": "Cheque",
    "draft": "Cheque",
    "bank_draft": "Cheque",
    # Bitcoin / Crypto
    "bitcoin": "Bitcoin",
    "crypto": "Bitcoin",
    "cryptocurrency": "Bitcoin",
    "btc": "Bitcoin",
    "blockchain": "Bitcoin",
    # Reinvestment
    "reinvestment": "Reinvestment",
    "dividend": "Reinvestment",
    "interest": "Reinvestment",
}

# Synonyms for currency mapping into IBM AML vocabulary
CURRENCY_SYNONYMS = {
    "usd": "US Dollar",
    "us dollar": "US Dollar",
    "dollar": "US Dollar",
    "$": "US Dollar",
    "eur": "Euro",
    "euro": "Euro",
    "gbp": "UK Pound",
    "pound": "UK Pound",
    "uk pound": "UK Pound",
    "inr": "Rupee",
    "rupee": "Rupee",
    "cad": "Canadian Dollar",
    "canadian dollar": "Canadian Dollar",
    "aud": "Australian Dollar",
    "australian dollar": "Australian Dollar",
    "chf": "Swiss Franc",
    "swiss franc": "Swiss Franc",
    "jpy": "Yen",
    "yen": "Yen",
    "cny": "Yuan",
    "yuan": "Yuan",
    "rmb": "Yuan",
    "brl": "Brazil Real",
    "brazil real": "Brazil Real",
    "rub": "Ruble",
    "ruble": "Ruble",
    "ils": "Shekel",
    "shekel": "Shekel",
    "sar": "Saudi Riyal",
    "saudi riyal": "Saudi Riyal",
    "mxn": "Mexican Peso",
    "peso": "Mexican Peso",
    "mexican peso": "Mexican Peso",
    "btc": "Bitcoin",
    "bitcoin": "Bitcoin",
}

# Candidate aliases for each canonical field
# Key: canonical name, Value: set of normalized regex or string patterns
CANONICAL_ALIASES: Dict[str, List[str]] = {
    "timestamp": [
        "timestamp",
        "trans_time",
        "trans_date",
        "transaction_date",
        "transaction_time",
        "trans_date_trans_time",
        "date_time",
        "datetime",
        "date",
        "time",
        "tx_time",
        "tx_date",
        "txn_time",
        "created_at",
        "step",
    ],
    "from_account": [
        "account",
        "from_account",
        "from_acc",
        "sender_account",
        "sender_acc",
        "source_account",
        "source_acc",
        "originator",
        "originator_account",
        "orig_account",
        "nameorig",
        "sender",
    ],
    "to_account": [
        "account_1",
        "account.1",
        "to_account",
        "to_acc",
        "receiver_account",
        "receiver_acc",
        "destination_account",
        "dest_account",
        "dest_acc",
        "namedest",
        "beneficiary",
        "beneficiary_account",
        "receiver",
    ],
    "from_bank": [
        "from_bank",
        "sender_bank",
        "source_bank",
        "orig_bank",
        "sending_bank",
        "frombank",
    ],
    "to_bank": [
        "to_bank",
        "receiver_bank",
        "destination_bank",
        "dest_bank",
        "receiving_bank",
        "tobank",
    ],
    "amount_paid": [
        "amount_paid",
        "amount",
        "transaction_amount",
        "trans_amount",
        "amt",
        "value",
        "payment_amount",
        "amount_transferred",
    ],
    "amount_received": [
        "amount_received",
        "received_amount",
        "rcv_amount",
    ],
    "payment_currency": [
        "payment_currency",
        "currency",
        "transaction_currency",
        "curr",
        "trans_currency",
        "source_currency",
        "ccy",
        "currency_iso",
    ],
    "receiving_currency": [
        "receiving_currency",
        "dest_currency",
        "destination_currency",
    ],
    "payment_format": [
        "payment_format",
        "transaction_type",
        "transaction_type_name",
        "trans_type",
        "type",
        "channel",
        "format",
        "method",
        "payment_type",
        "payment_rail",
        "rail",
    ],
    "label": [
        "is_laundering",
        "laundering",
        "target",
        "label",
        "is_fraud",
        "isfraud",
        "fraud",
        "fraud_indicator",
        "is_sar",
        "class",
    ],
    "transaction_id": [
        "transaction_id",
        "trans_id",
        "tx_id",
        "txn_id",
        "id",
        "reference_number",
    ],
}

# Explicit target leakage and post-investigation outcome fields
TARGET_LEAKAGE_FIELDS = {
    "structuring_flag",
    "structuring",
    "sar_filed",
    "sar_reported",
    "fraud_confirmed",
    "investigation_closed",
    "chargeback",
    "alert_status",
    "analyst_verdict",
    "post_investigation_risk",
    "suspicious_flag",
    "confirmed_laundering",
    "case_closed",
    "investigation_result",
    "target_derived",
}


def normalize_col(name: str) -> str:
    """Normalize column header: lowercase, strip, punctuation to underscore."""
    clean = str(name).strip().lower()
    # Replace dots, spaces, hyphens with underscores
    clean = re.sub(r"[\s\.\-]+", "_", clean)
    # Remove any characters that are not alphanumeric or underscore
    clean = re.sub(r"[^\w]", "", clean)
    return clean


class AdaptiveSchemaDetector:
    """Detects, maps, and validates CSV schemas against authoritative inference pipelines."""

    def __init__(self) -> None:
        pass

    def inspect_and_adapt(
        self,
        df: pd.DataFrame,
        filename: str = "upload.csv",
        requested_dataset: Optional[str] = None,
    ) -> SchemaDetectionResult:
        """Inspect DataFrame headers and data rows, detect schema, and adapt."""
        if df is None or df.empty or len(df) == 0:
            return SchemaDetectionResult(
                status=DetectionStatus.UNSUPPORTED,
                user_message="This CSV cannot be evaluated: uploaded file contains zero data rows.",
                format_category=FormatCategory.UNKNOWN,
                target_pipeline="HI-Small",
                decision_threshold=FROZEN_THRESHOLDS["HI-Small"],
            )

        cols = list(df.columns)
        norm_to_orig: Dict[str, str] = {normalize_col(c): c for c in cols}
        norm_cols = list(norm_to_orig.keys())

        # 1. Pipeline Selection & Disambiguation
        target_pipeline, pipeline_error = self._resolve_target_pipeline(
            filename=filename,
            norm_cols=norm_cols,
            requested_dataset=requested_dataset,
        )
        if pipeline_error:
            return SchemaDetectionResult(
                status=DetectionStatus.AMBIGUOUS,
                user_message=pipeline_error,
                format_category=FormatCategory.UNKNOWN,
                target_pipeline="HI-Small",
                decision_threshold=FROZEN_THRESHOLDS["HI-Small"],
            )

        decision_threshold = FROZEN_THRESHOLDS[target_pipeline]

        # 2. Check for Processed Features Format (26+ features)
        if all(normalize_col(pc) in norm_cols for pc in PROCESSED_REQUIRED_COLS):
            return self._handle_processed_format(
                df=df,
                norm_to_orig=norm_to_orig,
                target_pipeline=target_pipeline,
                decision_threshold=decision_threshold,
            )

        # 3. Check for Data Leakage Fields
        detected_leakage: List[str] = []
        for nc, orig in norm_to_orig.items():
            if nc in TARGET_LEAKAGE_FIELDS:
                detected_leakage.append(orig)

        # 4. Check for Column Ambiguities / Conflicting Mappings
        field_matches: Dict[str, List[str]] = {}
        for canonical, aliases in CANONICAL_ALIASES.items():
            matches = []
            for alias in aliases:
                norm_alias = normalize_col(alias)
                if norm_alias in norm_to_orig:
                    matches.append(norm_to_orig[norm_alias])
            if matches:
                # Remove duplicates while preserving order
                unique_matches = list(dict.fromkeys(matches))
                field_matches[canonical] = unique_matches

        # Check for conflicts where multiple distinct columns map to the same canonical concept
        conflicts: Dict[str, List[str]] = {}
        # We allow amount_paid and amount_received to share "amount" if only one exists,
        # but if two distinct columns match sender_account or receiver_account, that is ambiguous.
        for canon in ["timestamp", "from_account", "to_account", "amount_paid"]:
            matched = field_matches.get(canon, [])
            if len(matched) > 1:
                # Special case: pandas Account and Account.1
                if canon == "from_account" and set(matched) == {"Account", "Account.1"}:
                    field_matches["from_account"] = ["Account"]
                    field_matches["to_account"] = ["Account.1"]
                    continue
                # If distinct column names have distinct non-identical data, flag conflict
                is_conflicting = False
                first_s = df[matched[0]]
                for other_col in matched[1:]:
                    if not first_s.equals(df[other_col]):
                        is_conflicting = True
                        break
                if is_conflicting:
                    conflicts[canon] = matched

        if conflicts:
            conflict_desc = ", ".join(f"{k}: {v}" for k, v in conflicts.items())
            return SchemaDetectionResult(
                status=DetectionStatus.AMBIGUOUS,
                user_message=f"Multiple possible field mappings were detected ({conflict_desc}). Evaluation cannot continue until the schema is resolved.",
                format_category=FormatCategory.UNKNOWN,
                target_pipeline=target_pipeline,
                decision_threshold=decision_threshold,
                conflicting_fields=conflicts,
                excluded_leakage_fields=detected_leakage,
            )

        # 5. Check Required Fields for Raw / Synthetic Adaptation
        # Minimum required to reconstruct an AML transaction:
        # timestamp, from_account, to_account, amount (or amount_paid)
        has_time = "timestamp" in field_matches
        has_from = "from_account" in field_matches
        has_to = "to_account" in field_matches
        # Special check: if only one "account" column is found, check if pandas de-duplicated
        if has_from and not has_to:
            if "Account.1" in df.columns:
                has_to = True
                field_matches["to_account"] = ["Account.1"]
            elif "account_1" in norm_to_orig:
                has_to = True
                field_matches["to_account"] = [norm_to_orig["account_1"]]

        has_amount = "amount_paid" in field_matches or "amount_received" in field_matches

        # If zero transaction fields are present, it is completely unsupported
        if not has_time and not has_from and not has_to and not has_amount:
            return SchemaDetectionResult(
                status=DetectionStatus.UNSUPPORTED,
                user_message="This CSV cannot be evaluated using the currently available trained models. Missing required columns: timestamp, sender_account, receiver_account, amount.",
                format_category=FormatCategory.UNKNOWN,
                target_pipeline=target_pipeline,
                decision_threshold=decision_threshold,
                excluded_leakage_fields=detected_leakage,
            )

        # Check for missing required fields
        missing_req: List[str] = []
        if not has_time:
            missing_req.append("timestamp")
        if not has_from:
            missing_req.append("from_account")
        if not has_to:
            missing_req.append("to_account")
        if not has_amount:
            missing_req.append("amount")

        if missing_req:
            return SchemaDetectionResult(
                status=DetectionStatus.PARTIALLY_COMPATIBLE,
                user_message=f"Some fields could not be mapped. Please review the missing fields: {', '.join(missing_req)}.",
                format_category=FormatCategory.UNKNOWN,
                target_pipeline=target_pipeline,
                decision_threshold=decision_threshold,
                missing_required_fields=missing_req,
                excluded_leakage_fields=detected_leakage,
            )

        # 6. Adapt and Construct Canonical 11-Column Frame
        return self._adapt_raw_or_synthetic_stream(
            df=df,
            field_matches=field_matches,
            detected_leakage=detected_leakage,
            target_pipeline=target_pipeline,
            decision_threshold=decision_threshold,
        )

    def _resolve_target_pipeline(
        self,
        filename: str,
        norm_cols: List[str],
        requested_dataset: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Resolve pipeline compatibility between HI-Small and LI-Medium."""
        if requested_dataset:
            if requested_dataset not in ("HI-Small", "LI-Medium"):
                return "HI-Small", f"Unsupported pipeline '{requested_dataset}'. Supported options: 'HI-Small' or 'LI-Medium'."
            return requested_dataset, None

        fname_lower = filename.lower()
        is_li = "li_medium" in fname_lower or "li-medium" in fname_lower or "limedium" in fname_lower
        is_hi = "hi_small" in fname_lower or "hi-small" in fname_lower or "hismall" in fname_lower

        if is_li and is_hi:
            return "HI-Small", "Filename contains conflicting indicators for both HI-Small and LI-Medium pipelines."

        if is_li:
            return "LI-Medium", None

        # Default to production champion pipeline
        return "HI-Small", None

    def _handle_processed_format(
        self,
        df: pd.DataFrame,
        norm_to_orig: Dict[str, str],
        target_pipeline: str,
        decision_threshold: float,
    ) -> SchemaDetectionResult:
        """Process already-engineered 26/27 column features."""
        mapped_df = pd.DataFrame()
        field_mappings = {}

        for req in PROCESSED_REQUIRED_COLS:
            orig = norm_to_orig[normalize_col(req)]
            mapped_df[req] = df[orig]
            field_mappings[req] = orig

        # Check for label
        has_labels = False
        lbl_series: Optional[pd.Series] = None
        for cand in ["is_laundering", "target", "label", "class"]:
            nc = normalize_col(cand)
            if nc in norm_to_orig:
                orig_lbl = norm_to_orig[nc]
                s = pd.to_numeric(df[orig_lbl], errors="coerce")
                if not s.isna().any() and set(s.unique()).issubset({0, 1}):
                    has_labels = True
                    lbl_series = s.astype("int8")
                    mapped_df["is_laundering"] = lbl_series
                    field_mappings["is_laundering"] = orig_lbl
                    break

        return SchemaDetectionResult(
            status=DetectionStatus.COMPATIBLE,
            user_message="CSV format recognized. Required fields mapped successfully.",
            format_category=FormatCategory.PROCESSED_FEATURES,
            target_pipeline=target_pipeline,
            decision_threshold=decision_threshold,
            mapped_df=mapped_df,
            has_labels=has_labels,
            ground_truth_labels=lbl_series,
            field_mappings=field_mappings,
            audit_notes=["Processed feature schema verified directly with 26 input dimensions."],
        )

    def _adapt_raw_or_synthetic_stream(
        self,
        df: pd.DataFrame,
        field_matches: Dict[str, List[str]],
        detected_leakage: List[str],
        target_pipeline: str,
        decision_threshold: float,
    ) -> SchemaDetectionResult:
        """Adapt arbitrary raw or synthetic stream to the 11 verified IBM AML columns."""
        field_mappings: Dict[str, str] = {}
        audit_notes: List[str] = []
        n_rows = len(df)

        mapped_df = pd.DataFrame(index=range(n_rows))

        # 1. Timestamp adaptation
        time_col = field_matches["timestamp"][0]
        field_mappings["timestamp"] = time_col
        raw_time_series = df[time_col]

        # Check if numeric step (e.g. PaySim hour 1, 2, ...)
        if pd.api.types.is_numeric_dtype(raw_time_series):
            base_dt = pd.Timestamp("2022-09-01 00:00:00")
            parsed_dts = base_dt + pd.to_timedelta(raw_time_series, unit="h")
            mapped_df[COL_TIMESTAMP] = parsed_dts.dt.strftime("%Y/%m/%d %H:%M")
            audit_notes.append(f"Mapped numeric step in '{time_col}' to chronological timestamps.")
        else:
            try:
                # Try parsing with format='mixed', utc=True
                parsed_dts = pd.to_datetime(raw_time_series, format="mixed", utc=True, errors="coerce")
                if parsed_dts.isna().sum() > len(parsed_dts) * 0.5:
                    # Fallback to standard parse
                    parsed_dts = pd.to_datetime(raw_time_series, errors="coerce")
                mapped_df[COL_TIMESTAMP] = parsed_dts.dt.strftime("%Y/%m/%d %H:%M")
            except Exception:
                parsed_dts = pd.to_datetime(raw_time_series, errors="coerce")
                mapped_df[COL_TIMESTAMP] = parsed_dts.dt.strftime("%Y/%m/%d %H:%M")

        # 2. Account identifiers
        from_col = field_matches["from_account"][0]
        to_col = field_matches["to_account"][0]
        field_mappings["from_account"] = from_col
        field_mappings["to_account"] = to_col
        mapped_df[COL_FROM_ACCOUNT] = df[from_col].astype(str)
        mapped_df[COL_TO_ACCOUNT] = df[to_col].astype(str)

        # 3. Bank identifiers (default to deterministic hash if absent or missing)
        derived_fb = mapped_df[COL_FROM_ACCOUNT].apply(
            lambda a: str(abs(hash(str(a))) % 500 + 1)
        )
        if "from_bank" in field_matches:
            from_bank_col = field_matches["from_bank"][0]
            field_mappings["from_bank"] = from_bank_col
            s_fb = df[from_bank_col].fillna("").astype(str).str.strip()
            mapped_df[COL_FROM_BANK] = np.where(
                s_fb.isin(["", "nan", "None", "null", "NaN", "0.0"]),
                derived_fb,
                s_fb,
            )
        else:
            mapped_df[COL_FROM_BANK] = derived_fb
            audit_notes.append("Derived sending bank identifier from account topology.")

        derived_tb = mapped_df[COL_TO_ACCOUNT].apply(
            lambda a: str(abs(hash(str(a))) % 500 + 1)
        )
        if "to_bank" in field_matches:
            to_bank_col = field_matches["to_bank"][0]
            field_mappings["to_bank"] = to_bank_col
            s_tb = df[to_bank_col].fillna("").astype(str).str.strip()
            mapped_df[COL_TO_BANK] = np.where(
                s_tb.isin(["", "nan", "None", "null", "NaN", "0.0"]),
                derived_tb,
                s_tb,
            )
        else:
            mapped_df[COL_TO_BANK] = derived_tb
            audit_notes.append("Derived receiving bank identifier from account topology.")

        # 4. Amounts (paid and received)
        amount_paid_col = (
            field_matches["amount_paid"][0]
            if "amount_paid" in field_matches
            else field_matches["amount_received"][0]
        )
        field_mappings["amount_paid"] = amount_paid_col
        paid_numeric = pd.to_numeric(df[amount_paid_col], errors="coerce").fillna(0.0)
        mapped_df[COL_AMOUNT_PAID] = paid_numeric

        if "amount_received" in field_matches:
            amount_rcv_col = field_matches["amount_received"][0]
            field_mappings["amount_received"] = amount_rcv_col
            mapped_df[COL_AMOUNT_RECEIVED] = pd.to_numeric(df[amount_rcv_col], errors="coerce").fillna(0.0)
        else:
            # Default receiving amount to paid amount
            mapped_df[COL_AMOUNT_RECEIVED] = paid_numeric
            audit_notes.append(f"Derived 'Amount Received' from transaction amount '{amount_paid_col}'.")

        # 5. Currencies
        if "payment_currency" in field_matches:
            pay_curr_col = field_matches["payment_currency"][0]
            field_mappings["payment_currency"] = pay_curr_col
            raw_curr = df[pay_curr_col].fillna("").astype(str).str.strip().str.lower()
            mapped_curr = raw_curr.map(CURRENCY_SYNONYMS)
            mapped_df[COL_PAYMENT_CURRENCY] = mapped_curr.fillna("US Dollar").replace({"": "US Dollar", "nan": "US Dollar"})
        else:
            mapped_df[COL_PAYMENT_CURRENCY] = "US Dollar"
            audit_notes.append("Defaulted payment currency to US Dollar.")

        if "receiving_currency" in field_matches:
            rcv_curr_col = field_matches["receiving_currency"][0]
            field_mappings["receiving_currency"] = rcv_curr_col
            raw_curr = df[rcv_curr_col].fillna("").astype(str).str.strip().str.lower()
            mapped_curr = raw_curr.map(CURRENCY_SYNONYMS)
            mapped_df[COL_RECEIVING_CURRENCY] = mapped_curr.fillna("US Dollar").replace({"": "US Dollar", "nan": "US Dollar"})
        else:
            mapped_df[COL_RECEIVING_CURRENCY] = mapped_df[COL_PAYMENT_CURRENCY]

        # 6. Payment Format / Transaction Type
        if "payment_format" in field_matches:
            fmt_col = field_matches["payment_format"][0]
            field_mappings["payment_format"] = fmt_col
            raw_types = df[fmt_col].fillna("").astype(str).str.strip().str.lower()
            mapped_types = raw_types.map(PAYMENT_FORMAT_SYNONYMS)
            mapped_df[COL_PAYMENT_FORMAT] = mapped_types.fillna("Wire").replace({"": "Wire", "nan": "Wire"})
        else:
            mapped_df[COL_PAYMENT_FORMAT] = "Wire"
            audit_notes.append("Defaulted transaction payment format to Wire transfer.")

        # 7. Ground Truth Label (Separated strictly from features!)
        has_labels = False
        lbl_series: Optional[pd.Series] = None
        if "label" in field_matches:
            lbl_col = field_matches["label"][0]
            field_mappings["label"] = lbl_col
            s = pd.to_numeric(df[lbl_col], errors="coerce")
            if not s.isna().any() and set(s.unique()).issubset({0, 1}):
                has_labels = True
                lbl_series = s.astype("int8")
                mapped_df[COL_IS_LAUNDERING] = lbl_series
                audit_notes.append(f"Detected binary ground-truth labels in column '{lbl_col}'.")
            else:
                mapped_df[COL_IS_LAUNDERING] = 0
                audit_notes.append(f"Ignored non-binary or invalid label column '{lbl_col}'.")
        else:
            mapped_df[COL_IS_LAUNDERING] = 0

        # 8. Excluded Target Leakage Fields
        if detected_leakage:
            for lk in detected_leakage:
                audit_notes.append(f"Target-derived field '{lk}' detected and excluded from model inputs to prevent data leakage.")

        # Reorder to strict raw column order
        mapped_df = mapped_df[list(RAW_COLUMN_ORDER)]

        # Determine format category
        orig_cols_set = set(df.columns)
        is_exact_ibm = set(RAW_COLUMN_ORDER).issubset(orig_cols_set)
        cat = FormatCategory.IBM_AML_RAW if is_exact_ibm else FormatCategory.ALIASED_TRANSACTIONS

        return SchemaDetectionResult(
            status=DetectionStatus.COMPATIBLE,
            user_message="CSV format recognized. Required fields mapped successfully.",
            format_category=cat,
            target_pipeline=target_pipeline,
            decision_threshold=decision_threshold,
            mapped_df=mapped_df,
            has_labels=has_labels,
            ground_truth_labels=lbl_series,
            field_mappings=field_mappings,
            excluded_leakage_fields=detected_leakage,
            audit_notes=audit_notes,
        )
