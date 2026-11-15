"""AML Typology Analysis Module (Phase N).

Separates probabilistic model risk scoring from deterministic rule-based typology indicators:
1. Model Prediction: Continuous risk score in [0.0, 1.0] from neural sequence model.
2. Rule-Based Typology Indicator: Explicit behavioral pattern match (Structuring, Rapid Movement, Circular Flow).
3. Analyst Adjudication: Human verdict field for audit compliance.

Strict Rules:
- A high model risk score does NOT automatically mean a confirmed typology.
- Typology indicators are only flagged when underlying transaction evidence satisfies rule preconditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class TypologyFlag:
    typology_name: str
    is_detected: bool
    confidence_level: str  # "HIGH", "MEDIUM", "LOW", "NONE"
    evidence_summary: str
    rule_thresholds_applied: Dict[str, Any]
    limitations: str


@dataclass
class TransactionTypologyProfile:
    tx_id: str
    from_account: str
    to_account: str
    amount: float
    timestamp: str
    model_risk_score: float
    model_decision: str  # "FLAGGED_LAUNDERING" / "CLEARED_LEGITIMATE"
    detected_typologies: List[TypologyFlag]
    has_any_rule_flag: bool
    overall_suspicion_tier: str  # "HIGH_PRIORITY_INVESTIGATION", "MEDIUM_REVIEW", "LOW_RISK"
    analyst_verdict: Optional[str] = None  # None until analyst review


class AmlTypologyAnalyzer:
    """Evaluates transaction streams against canonical financial crime typologies."""

    def __init__(
        self,
        structuring_min_amount: float = 8500.0,
        structuring_max_amount: float = 9999.0,
        structuring_window_hours: int = 24,
        rapid_movement_window_hours: int = 2,
        rapid_movement_retention_max_pct: float = 0.20,
        velocity_burst_count: int = 4,
        velocity_burst_minutes: int = 15,
    ):
        self.structuring_min = structuring_min_amount
        self.structuring_max = structuring_max_amount
        self.structuring_window = timedelta(hours=structuring_window_hours)
        self.rapid_window = timedelta(hours=rapid_movement_window_hours)
        self.rapid_retention_max = rapid_movement_retention_max_pct
        self.velocity_burst = velocity_burst_count
        self.velocity_window = timedelta(minutes=velocity_burst_minutes)

    def analyze_transaction(
        self,
        tx_row: Dict[str, Any],
        account_history: List[Dict[str, Any]],
        model_risk_score: float,
        model_threshold: float = 0.337784,
    ) -> TransactionTypologyProfile:
        """Evaluate a single transaction against rule-based typology indicators using historical records."""
        tx_id = str(tx_row.get("tx_id", tx_row.get("row_id", "tx_unknown")))
        from_acc = str(tx_row.get("from_account", "acc_unknown"))
        to_acc = str(tx_row.get("to_account", "acc_unknown"))
        amount = float(tx_row.get("amount", 0.0))
        ts = pd.to_datetime(tx_row.get("timestamp"))

        # Filter history strictly at or before current timestamp
        valid_hist = [
            h for h in account_history
            if pd.to_datetime(h.get("timestamp")) <= ts and h.get("tx_id") != tx_id
        ]

        flags: List[TypologyFlag] = []

        # ----------------------------------------------------------------------
        # 1. Structuring / Smurfing Rule
        # ----------------------------------------------------------------------
        # Pattern: Amounts just below the $10,000 regulatory reporting threshold within window
        is_struct_current = self.structuring_min <= amount <= self.structuring_max
        recent_struct_txs = [
            h for h in valid_hist
            if h.get("from_account") == from_acc
            and self.structuring_min <= float(h.get("amount", 0.0)) <= self.structuring_max
            and ts - pd.to_datetime(h.get("timestamp")) <= self.structuring_window
        ]
        struct_detected = is_struct_current and (len(recent_struct_txs) >= 1)
        flags.append(
            TypologyFlag(
                typology_name="Structuring / Smurfing",
                is_detected=struct_detected,
                confidence_level="HIGH" if struct_detected and len(recent_struct_txs) >= 2 else "MEDIUM" if struct_detected else "NONE",
                evidence_summary=(
                    f"Current amount ${amount:,.2f} with {len(recent_struct_txs)} other sub-threshold transactions "
                    f"in last {self.structuring_window}."
                    if struct_detected
                    else "No structuring pattern detected."
                ),
                rule_thresholds_applied={
                    "sub_threshold_range": f"${self.structuring_min}-${self.structuring_max}",
                    "window": str(self.structuring_window),
                },
                limitations="Legitimate commercial payments may naturally fall within this range.",
            )
        )

        # ----------------------------------------------------------------------
        # 2. Rapid Fund Movement / Pass-Through Flow
        # ----------------------------------------------------------------------
        # Pattern: Sender recently received funds within 2h and is now forwarding >= 80% out
        recent_incoming = [
            h for h in valid_hist
            if h.get("to_account") == from_acc
            and ts - pd.to_datetime(h.get("timestamp")) <= self.rapid_window
        ]
        sum_in = sum(float(h.get("amount", 0.0)) for h in recent_incoming)
        rapid_detected = sum_in > 0 and amount >= (sum_in * (1.0 - self.rapid_retention_max))
        flags.append(
            TypologyFlag(
                typology_name="Rapid Fund Movement (Pass-Through)",
                is_detected=rapid_detected,
                confidence_level="HIGH" if rapid_detected and amount > 10000 else "MEDIUM" if rapid_detected else "NONE",
                evidence_summary=(
                    f"Account received ${sum_in:,.2f} in past {self.rapid_window} and forwarded ${amount:,.2f} "
                    f"({(amount / sum_in)*100:.1f}% outbound flow)."
                    if rapid_detected
                    else "Normal balance retention observed."
                ),
                rule_thresholds_applied={
                    "window": str(self.rapid_window),
                    "retention_threshold_pct": f"{(1 - self.rapid_retention_max)*100}%",
                },
                limitations="Payroll and escrow aggregators legitimately pass through funds quickly.",
            )
        )

        # ----------------------------------------------------------------------
        # 3. High Velocity Transaction Burst
        # ----------------------------------------------------------------------
        # Pattern: Unusually high transaction frequency within short window
        burst_txs = [
            h for h in valid_hist
            if h.get("from_account") == from_acc
            and ts - pd.to_datetime(h.get("timestamp")) <= self.velocity_window
        ]
        burst_detected = len(burst_txs) + 1 >= self.velocity_burst
        flags.append(
            TypologyFlag(
                typology_name="Unusual High Velocity Burst",
                is_detected=burst_detected,
                confidence_level="MEDIUM" if burst_detected else "NONE",
                evidence_summary=(
                    f"{len(burst_txs) + 1} transactions executed in past {self.velocity_window}."
                    if burst_detected
                    else "Transaction velocity within standard baseline."
                ),
                rule_thresholds_applied={
                    "burst_count": self.velocity_burst,
                    "window": str(self.velocity_window),
                },
                limitations="Automated merchant POS or API integrations can generate high velocity normally.",
            )
        )

        # ----------------------------------------------------------------------
        # 4. Immediate Round-Trip / Circular Flow Indicator
        # ----------------------------------------------------------------------
        # Pattern: Funds moving between reciprocal accounts (To -> From -> To)
        reciprocal_incoming = [
            h for h in valid_hist
            if h.get("from_account") == to_acc
            and h.get("to_account") == from_acc
            and ts - pd.to_datetime(h.get("timestamp")) <= timedelta(hours=48)
        ]
        round_trip_detected = len(reciprocal_incoming) > 0
        flags.append(
            TypologyFlag(
                typology_name="Circular Flow / Round-Trip",
                is_detected=round_trip_detected,
                confidence_level="HIGH" if round_trip_detected else "NONE",
                evidence_summary=(
                    f"Reciprocal flow detected: {to_acc} sent funds to {from_acc} within 48h prior."
                    if round_trip_detected
                    else "No circular reciprocal transactions detected."
                ),
                rule_thresholds_applied={"window": "48 hours"},
                limitations="Legitimate refunds, supplier adjustments, or inter-subsidiary balancing.",
            )
        )

        has_any_flag = any(f.is_detected for f in flags)
        is_model_flagged = model_risk_score >= model_threshold

        # Composite priority tiering
        if is_model_flagged and has_any_flag:
            tier = "HIGH_PRIORITY_INVESTIGATION"
        elif is_model_flagged or has_any_flag:
            tier = "MEDIUM_REVIEW"
        else:
            tier = "LOW_RISK"

        return TransactionTypologyProfile(
            tx_id=tx_id,
            from_account=from_acc,
            to_account=to_acc,
            amount=round(amount, 2),
            timestamp=str(ts),
            model_risk_score=round(model_risk_score, 4),
            model_decision="FLAGGED_LAUNDERING" if is_model_flagged else "CLEARED_LEGITIMATE",
            detected_typologies=[f for f in flags if f.is_detected],
            has_any_rule_flag=has_any_flag,
            overall_suspicion_tier=tier,
            analyst_verdict=None,
        )

