"""Graph-Based AML Feature Engineering Module (Phase D).

Computes strictly causal (time-aware, zero-leakage) graph features for anti-money laundering:
- Account Activity (in-degree, out-degree, degree ratio, unique counterparties, net flow, avg/max amounts)
- Temporal Velocity (sliding windows: 5m, 1h, 24h transaction counts, sums, velocity acceleration)
- Network Typologies (2-hop paths, circular flow / round-trip detection A->B->A, A->B->C->A, counterparty concentration)

Strict Leakage Guarantee:
For any transaction at time t, only historical edges with timestamp <= t are queried.
No future edges or post-split records are ever accessible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransactionEdge:
    tx_id: str
    from_account: str
    to_account: str
    amount: float
    timestamp: pd.Timestamp
    currency: str = "USD"
    payment_format: str = "ACH"


@dataclass
class GraphFeatureVector:
    tx_id: str
    # Account Activity Features (Sender & Receiver)
    sender_in_degree: int = 0
    sender_out_degree: int = 0
    sender_unique_in_counterparties: int = 0
    sender_unique_out_counterparties: int = 0
    sender_net_flow: float = 0.0
    sender_avg_out_amount: float = 0.0
    sender_max_out_amount: float = 0.0
    
    receiver_in_degree: int = 0
    receiver_out_degree: int = 0
    receiver_unique_in_counterparties: int = 0
    receiver_unique_out_counterparties: int = 0
    receiver_net_flow: float = 0.0
    receiver_avg_in_amount: float = 0.0
    receiver_max_in_amount: float = 0.0

    # Temporal Velocity Features (Last 5m, 1h, 24h)
    sender_tx_count_5m: int = 0
    sender_tx_count_1h: int = 0
    sender_tx_count_24h: int = 0
    sender_amount_sum_5m: float = 0.0
    sender_amount_sum_1h: float = 0.0
    sender_amount_sum_24h: float = 0.0
    sender_distinct_counterparties_24h: int = 0

    receiver_tx_count_5m: int = 0
    receiver_tx_count_1h: int = 0
    receiver_tx_count_24h: int = 0
    receiver_amount_sum_5m: float = 0.0
    receiver_amount_sum_1h: float = 0.0
    receiver_amount_sum_24h: float = 0.0

    # Network Typology Indicators (Round-Trip / Circular Flow)
    is_immediate_round_trip: int = 0      # A -> B within 24h of B -> A
    is_circular_flow_3hop: int = 0        # A -> B -> C -> A within 48h
    rapid_pass_through_ratio: float = 0.0 # outgoing / incoming within 1 hour
    counterparty_concentration_hhi: float = 0.0 # Herfindahl-Hirschman Index of counterparties


class TimeAwareAmlGraph:
    """Scalable, chronological graph representation for zero-leakage AML feature extraction.
    
    Transactions are indexed chronologically. Queries at time t only evaluate edges with timestamp <= t.
    """

    def __init__(self, max_history_window: timedelta = timedelta(days=7)):
        self.max_history_window = max_history_window
        # Account -> chronological list of outgoing edges
        self._out_edges: Dict[str, List[TransactionEdge]] = defaultdict(list)
        # Account -> chronological list of incoming edges
        self._in_edges: Dict[str, List[TransactionEdge]] = defaultdict(list)
        # Global edge count
        self._total_edges = 0

    def add_transaction(
        self,
        tx_id: str,
        from_account: str,
        to_account: str,
        amount: float,
        timestamp: Any,
        currency: str = "USD",
        payment_format: str = "ACH",
    ) -> TransactionEdge:
        """Add a transaction edge to the graph in chronological sequence."""
        ts = pd.to_datetime(timestamp)
        edge = TransactionEdge(
            tx_id=str(tx_id),
            from_account=str(from_account),
            to_account=str(to_account),
            amount=float(amount),
            timestamp=ts,
            currency=str(currency),
            payment_format=str(payment_format),
        )
        self._out_edges[from_account].append(edge)
        self._in_edges[to_account].append(edge)
        self._total_edges += 1
        return edge

    def extract_features_for_transaction(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        timestamp: Any,
        tx_id: str = "current_tx",
    ) -> GraphFeatureVector:
        """Extract graph features for a transaction occurring at timestamp t.
        
        Strict causal restriction: only queries edges strictly before or up to this timestamp.
        """
        ts = pd.to_datetime(timestamp)
        cutoff_5m = ts - timedelta(minutes=5)
        cutoff_1h = ts - timedelta(hours=1)
        cutoff_24h = ts - timedelta(hours=24)
        cutoff_48h = ts - timedelta(hours=48)

        # 1. Sender Historical Outgoing & Incoming (prior to ts)
        sender_out = [e for e in self._out_edges.get(from_account, []) if e.timestamp <= ts]
        sender_in = [e for e in self._in_edges.get(from_account, []) if e.timestamp <= ts]

        sender_out_amounts = [e.amount for e in sender_out]
        sender_in_amounts = [e.amount for e in sender_in]

        sender_in_deg = len(sender_in)
        sender_out_deg = len(sender_out)
        sender_uniq_in = len(set(e.from_account for e in sender_in))
        sender_uniq_out = len(set(e.to_account for e in sender_out))
        sender_net_flow = sum(sender_in_amounts) - sum(sender_out_amounts)
        sender_avg_out = float(np.mean(sender_out_amounts)) if sender_out_amounts else 0.0
        sender_max_out = float(np.max(sender_out_amounts)) if sender_out_amounts else 0.0

        # Sender Velocities
        s_out_5m = [e for e in sender_out if e.timestamp >= cutoff_5m]
        s_out_1h = [e for e in sender_out if e.timestamp >= cutoff_1h]
        s_out_24h = [e for e in sender_out if e.timestamp >= cutoff_24h]

        sender_tx_count_5m = len(s_out_5m)
        sender_tx_count_1h = len(s_out_1h)
        sender_tx_count_24h = len(s_out_24h)
        sender_amount_sum_5m = sum(e.amount for e in s_out_5m)
        sender_amount_sum_1h = sum(e.amount for e in s_out_1h)
        sender_amount_sum_24h = sum(e.amount for e in s_out_24h)
        sender_distinct_cp_24h = len(set(e.to_account for e in s_out_24h))

        # 2. Receiver Historical Incoming & Outgoing (prior to ts)
        receiver_in = [e for e in self._in_edges.get(to_account, []) if e.timestamp <= ts]
        receiver_out = [e for e in self._out_edges.get(to_account, []) if e.timestamp <= ts]

        receiver_in_amounts = [e.amount for e in receiver_in]
        receiver_out_amounts = [e.amount for e in receiver_out]

        receiver_in_deg = len(receiver_in)
        receiver_out_deg = len(receiver_out)
        receiver_uniq_in = len(set(e.from_account for e in receiver_in))
        receiver_uniq_out = len(set(e.to_account for e in receiver_out))
        receiver_net_flow = sum(receiver_in_amounts) - sum(receiver_out_amounts)
        receiver_avg_in = float(np.mean(receiver_in_amounts)) if receiver_in_amounts else 0.0
        receiver_max_in = float(np.max(receiver_in_amounts)) if receiver_in_amounts else 0.0

        r_in_5m = [e for e in receiver_in if e.timestamp >= cutoff_5m]
        r_in_1h = [e for e in receiver_in if e.timestamp >= cutoff_1h]
        r_in_24h = [e for e in receiver_in if e.timestamp >= cutoff_24h]

        receiver_tx_count_5m = len(r_in_5m)
        receiver_tx_count_1h = len(r_in_1h)
        receiver_tx_count_24h = len(r_in_24h)
        receiver_amount_sum_5m = sum(e.amount for e in r_in_5m)
        receiver_amount_sum_1h = sum(e.amount for e in r_in_1h)
        receiver_amount_sum_24h = sum(e.amount for e in r_in_24h)

        # 3. Typology Detection: Immediate Round Trip (B -> A within last 24h)
        # Check if receiver has recently sent funds back to sender
        b_to_a_edges = [
            e for e in self._out_edges.get(to_account, [])
            if e.to_account == from_account and cutoff_24h <= e.timestamp <= ts
        ]
        is_immediate_round_trip = 1 if len(b_to_a_edges) > 0 else 0

        # 4. Circular Flow: 3-hop check (A -> B -> C -> A within last 48h)
        is_circular_3hop = 0
        if not is_immediate_round_trip:
            # Check counterparties C that receiver has sent to
            receiver_out_48h = [
                e for e in self._out_edges.get(to_account, [])
                if cutoff_48h <= e.timestamp <= ts and e.to_account != from_account
            ]
            for edge_bc in receiver_out_48h:
                c_account = edge_bc.to_account
                # Check if C sent to A
                c_to_a_edges = [
                    e for e in self._out_edges.get(c_account, [])
                    if e.to_account == from_account and edge_bc.timestamp <= e.timestamp <= ts
                ]
                if c_to_a_edges:
                    is_circular_3hop = 1
                    break

        # 5. Rapid Pass-Through Ratio (Funds received by sender in last 1h immediately paid out)
        s_in_1h_amount = sum(e.amount for e in sender_in if e.timestamp >= cutoff_1h)
        rapid_pass_through_ratio = (
            min(10.0, float(amount / max(1.0, s_in_1h_amount))) if s_in_1h_amount > 0 else 0.0
        )

        # 6. Herfindahl-Hirschman Index (Counterparty Concentration for sender)
        hhi = 0.0
        if sender_out:
            cp_counts: Dict[str, int] = defaultdict(int)
            for e in sender_out:
                cp_counts[e.to_account] += 1
            total_out_tx = len(sender_out)
            hhi = sum((count / total_out_tx) ** 2 for count in cp_counts.values())

        return GraphFeatureVector(
            tx_id=tx_id,
            sender_in_degree=sender_in_deg,
            sender_out_degree=sender_out_deg,
            sender_unique_in_counterparties=sender_uniq_in,
            sender_unique_out_counterparties=sender_uniq_out,
            sender_net_flow=round(sender_net_flow, 2),
            sender_avg_out_amount=round(sender_avg_out, 2),
            sender_max_out_amount=round(sender_max_out, 2),
            receiver_in_degree=receiver_in_deg,
            receiver_out_degree=receiver_out_deg,
            receiver_unique_in_counterparties=receiver_uniq_in,
            receiver_unique_out_counterparties=receiver_uniq_out,
            receiver_net_flow=round(receiver_net_flow, 2),
            receiver_avg_in_amount=round(receiver_avg_in, 2),
            receiver_max_in_amount=round(receiver_max_in, 2),
            sender_tx_count_5m=sender_tx_count_5m,
            sender_tx_count_1h=sender_tx_count_1h,
            sender_tx_count_24h=sender_tx_count_24h,
            sender_amount_sum_5m=round(sender_amount_sum_5m, 2),
            sender_amount_sum_1h=round(sender_amount_sum_1h, 2),
            sender_amount_sum_24h=round(sender_amount_sum_24h, 2),
            sender_distinct_counterparties_24h=sender_distinct_cp_24h,
            receiver_tx_count_5m=receiver_tx_count_5m,
            receiver_tx_count_1h=receiver_tx_count_1h,
            receiver_tx_count_24h=receiver_tx_count_24h,
            receiver_amount_sum_5m=round(receiver_amount_sum_5m, 2),
            receiver_amount_sum_1h=round(receiver_amount_sum_1h, 2),
            receiver_amount_sum_24h=round(receiver_amount_sum_24h, 2),
            is_immediate_round_trip=is_immediate_round_trip,
            is_circular_flow_3hop=is_circular_3hop,
            rapid_pass_through_ratio=round(rapid_pass_through_ratio, 4),
            counterparty_concentration_hhi=round(hhi, 4),
        )


class CausalGraphFeatureExtractor:
    """Batch-oriented transformer that sequentially extracts graph features without temporal leakage."""

    def __init__(self):
        self.graph = TimeAwareAmlGraph()

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        from_col: str = "from_account",
        to_col: str = "to_account",
        amount_col: str = "amount",
        time_col: str = "timestamp",
        tx_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Sort transactions chronologically and compute causal graph features for every row."""
        if len(df) == 0:
            return pd.DataFrame()

        # Preserve original index for output row-order alignment
        work_df = df.copy()
        work_df["_orig_idx"] = np.arange(len(work_df))
        work_df[time_col] = pd.to_datetime(work_df[time_col])
        work_df = work_df.sort_values(by=[time_col, "_orig_idx"]).reset_index(drop=True)

        features: List[Dict[str, Any]] = []

        for idx, row in work_df.iterrows():
            tx_id = str(row[tx_id_col]) if tx_id_col and tx_id_col in row else f"tx_{idx}"
            from_acc = str(row[from_col])
            to_acc = str(row[to_col])
            amt = float(row[amount_col])
            ts = row[time_col]

            # 1. Extract features using history strictly prior to this transaction
            feat = self.graph.extract_features_for_transaction(
                from_account=from_acc,
                to_account=to_acc,
                amount=amt,
                timestamp=ts,
                tx_id=tx_id,
            )
            feat_dict = feat.__dict__.copy()
            feat_dict["_orig_idx"] = row["_orig_idx"]
            features.append(feat_dict)

            # 2. Add current transaction to graph for future transactions
            self.graph.add_transaction(
                tx_id=tx_id,
                from_account=from_acc,
                to_account=to_acc,
                amount=amt,
                timestamp=ts,
            )

        # Restore original row ordering
        feat_df = pd.DataFrame(features)
        feat_df = feat_df.sort_values(by="_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)
        return feat_df

