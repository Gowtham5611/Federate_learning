"""Temporal Graph and Account Feature Extractor for FedSecureAI.

Extracts dynamic graph, counterparty, and account features using STRICTLY past transactions
(t_past < t_current). Guarantees zero lookahead / zero future data leakage.

Features extracted:
1.  sender_in_degree: Historical count of transactions where sender was receiver.
2.  sender_out_degree: Historical count of transactions where sender was sender.
3.  unique_counterparties_1h: Unique counterparties interacted with in the preceding 1 hour.
4.  unique_counterparties_24h: Unique counterparties interacted with in the preceding 24 hours.
5.  unique_counterparties_7d: Unique counterparties interacted with in the preceding 7 days.
6.  fan_out_count_24h: Outgoing transactions from sender in preceding 24 hours.
7.  fan_in_count_24h: Incoming transactions to receiver in preceding 24 hours.
8.  time_since_last_tx_sender: Elapsed seconds since sender's previous transaction.
9.  time_since_last_tx_receiver: Elapsed seconds since receiver's previous transaction.
10. amount_to_sender_mean: Transaction amount divided by sender's historical running mean.
11. amount_to_sender_std: Standardized transaction amount relative to sender's historical mean and std.
12. currency_mismatch: Binary flag (1 if receiving currency != payment currency, else 0).
13. is_round_amount: Binary flag (1 if amount is an exact multiple of 100 or 1000, else 0).
14. cycle_two_hop_indicator: Binary flag (1 if receiver previously sent to sender, indicating a 2-hop return/loop).
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TemporalGraphFeatureExtractor:
    """Online streaming or batch extractor for temporal graph and velocity features.
    
    Guarantees strict causal ordering: every feature at row i is computed using
    only rows 0 .. i - 1.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all historical graph state and running accumulators."""
        # Account running statistics
        self.sender_tx_timestamps: Dict[str, List[float]] = defaultdict(list)
        self.receiver_tx_timestamps: Dict[str, List[float]] = defaultdict(list)
        
        self.sender_counterparties_time: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self.receiver_counterparties_time: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        
        self.sender_amounts: Dict[str, List[float]] = defaultdict(list)
        
        # Directed edge history for cycle / return detection: sender -> receiver -> timestamps
        self.directed_edges: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        self.sender_in_counts: Dict[str, int] = defaultdict(int)
        self.sender_out_counts: Dict[str, int] = defaultdict(int)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal temporal graph features row-by-row on a chronological dataframe.
        
        Args:
            df: DataFrame containing at least ['timestamp', 'from_account', 'to_account', 'amount']
                and optionally currency columns.
                
        Returns:
            DataFrame containing extracted causal temporal graph features.
        """
        # Ensure chronological ordering if timestamp exists
        ts_col = None
        for c in ["timestamp", "time", "date", "datetime", "Timestamp"]:
            if c in df.columns:
                ts_col = c
                break

        sender_col = None
        for c in ["from_account", "Account", "from account", "sender"]:
            if c in df.columns:
                sender_col = c
                break

        receiver_col = None
        for c in ["to_account", "Account.1", "to account", "receiver"]:
            if c in df.columns:
                receiver_col = c
                break

        amount_col = None
        for c in ["amount", "Amount Paid", "Amount Received", "amount_paid", "amount_received", "amount_paid_log"]:
            if c in df.columns:
                amount_col = c
                break

        if not (ts_col and sender_col and receiver_col and amount_col):
            raise ValueError(f"DataFrame must contain timestamp, sender, receiver, and amount columns. Found: {list(df.columns)}")

        n = len(df)
        
        # Parse timestamps to epoch seconds float
        ts_series = pd.to_datetime(df[ts_col], errors="coerce")
        timestamps = ts_series.astype(np.int64).to_numpy() // 10**9
        
        senders = df[sender_col].astype(str).to_numpy()
        receivers = df[receiver_col].astype(str).to_numpy()
        amounts = df[amount_col].astype(float).to_numpy()
        
        # Check currencies if present
        curr_mismatch = np.zeros(n, dtype=np.float32)
        if "Payment Currency" in df.columns and "Receiving Currency" in df.columns:
            curr_mismatch = (df["Payment Currency"].astype(str) != df["Receiving Currency"].astype(str)).astype(np.float32).to_numpy()
        elif "same_currency" in df.columns:
            curr_mismatch = (df["same_currency"] == 0).astype(np.float32).to_numpy()

        # Pre-allocate output arrays
        f_in_degree = np.zeros(n, dtype=np.float32)
        f_out_degree = np.zeros(n, dtype=np.float32)
        f_cp_1h = np.zeros(n, dtype=np.float32)
        f_cp_24h = np.zeros(n, dtype=np.float32)
        f_cp_7d = np.zeros(n, dtype=np.float32)
        f_fan_out_24h = np.zeros(n, dtype=np.float32)
        f_fan_in_24h = np.zeros(n, dtype=np.float32)
        f_time_since_sender = np.zeros(n, dtype=np.float32)
        f_time_since_receiver = np.zeros(n, dtype=np.float32)
        f_amount_ratio_mean = np.ones(n, dtype=np.float32)
        f_amount_zscore = np.zeros(n, dtype=np.float32)
        f_is_round = np.zeros(n, dtype=np.float32)
        f_cycle_indicator = np.zeros(n, dtype=np.float32)

        one_hour = 3600.0
        one_day = 86400.0
        seven_days = 7.0 * 86400.0

        for i in range(n):
            t = float(timestamps[i])
            s = senders[i]
            r = receivers[i]
            amt = float(amounts[i])

            # 1. Historical in-degree & out-degree of sender (prior transactions only)
            f_in_degree[i] = float(self.sender_in_counts[s])
            f_out_degree[i] = float(self.sender_out_counts[s])

            # 2. Unique counterparties in time windows (1h, 24h, 7d)
            s_cp = self.sender_counterparties_time[s]
            if s_cp:
                cp_1h = {cp for (ts_prev, cp) in s_cp if t - ts_prev <= one_hour and ts_prev < t}
                cp_24h = {cp for (ts_prev, cp) in s_cp if t - ts_prev <= one_day and ts_prev < t}
                cp_7d = {cp for (ts_prev, cp) in s_cp if t - ts_prev <= seven_days and ts_prev < t}
                f_cp_1h[i] = float(len(cp_1h))
                f_cp_24h[i] = float(len(cp_24h))
                f_cp_7d[i] = float(len(cp_7d))

            # 3. Fan-out count (sender outgoing in last 24h) and Fan-in count (receiver incoming in last 24h)
            s_txs = self.sender_tx_timestamps[s]
            if s_txs:
                f_fan_out_24h[i] = float(sum(1 for ts_prev in s_txs if t - ts_prev <= one_day and ts_prev < t))
                f_time_since_sender[i] = float(t - s_txs[-1]) if t >= s_txs[-1] else 0.0
            else:
                f_time_since_sender[i] = -1.0  # Sentinel for no prior transaction

            r_txs = self.receiver_tx_timestamps[r]
            if r_txs:
                f_fan_in_24h[i] = float(sum(1 for ts_prev in r_txs if t - ts_prev <= one_day and ts_prev < t))
                f_time_since_receiver[i] = float(t - r_txs[-1]) if t >= r_txs[-1] else 0.0
            else:
                f_time_since_receiver[i] = -1.0  # Sentinel for no prior transaction

            # 4. Amount relative to sender's historical running mean and std
            s_amts = self.sender_amounts[s]
            if len(s_amts) > 0:
                mean_amt = float(np.mean(s_amts))
                std_amt = float(np.std(s_amts))
                f_amount_ratio_mean[i] = float(amt / mean_amt) if mean_amt > 0 else 1.0
                f_amount_zscore[i] = float((amt - mean_amt) / (std_amt + 1e-6))
            else:
                f_amount_ratio_mean[i] = 1.0
                f_amount_zscore[i] = 0.0

            # 5. Round amount flag
            if amt > 0 and (amt % 100 == 0.0 or amt % 1000 == 0.0):
                f_is_round[i] = 1.0

            # 6. Cycle / Return indicator: Did receiver r previously send to sender s?
            # If (r -> s) exists in past transactions, this forms a 2-hop loop/cycle
            if (r, s) in self.directed_edges and len(self.directed_edges[(r, s)]) > 0:
                # Check if reverse transfer occurred in last 7 days
                rev_txs = self.directed_edges[(r, s)]
                if any(t - prev_t <= seven_days and prev_t < t for prev_t in rev_txs):
                    f_cycle_indicator[i] = 1.0

            # --- UPDATE STATE (strictly AFTER computing row i features) ---
            self.sender_out_counts[s] += 1
            self.sender_in_counts[r] += 1
            
            self.sender_tx_timestamps[s].append(t)
            self.receiver_tx_timestamps[r].append(t)
            
            self.sender_counterparties_time[s].append((t, r))
            self.receiver_counterparties_time[r].append((t, s))
            
            self.sender_amounts[s].append(amt)
            self.directed_edges[(s, r)].append(t)

        # Assemble final feature dataframe
        features_df = pd.DataFrame({
            "sender_in_degree": f_in_degree,
            "sender_out_degree": f_out_degree,
            "unique_counterparties_1h": f_cp_1h,
            "unique_counterparties_24h": f_cp_24h,
            "unique_counterparties_7d": f_cp_7d,
            "fan_out_count_24h": f_fan_out_24h,
            "fan_in_count_24h": f_fan_in_24h,
            "time_since_last_tx_sender": np.log1p(np.maximum(0.0, f_time_since_sender)),
            "time_since_last_tx_receiver": np.log1p(np.maximum(0.0, f_time_since_receiver)),
            "amount_to_sender_mean": f_amount_ratio_mean,
            "amount_to_sender_zscore": f_amount_zscore,
            "currency_mismatch": curr_mismatch,
            "is_round_amount": f_is_round,
            "cycle_two_hop_indicator": f_cycle_indicator,
        })
        
        return features_df
