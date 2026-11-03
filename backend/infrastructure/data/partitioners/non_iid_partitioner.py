"""Deterministic typology-based Non-IID data partitioner for federated learning.

Partitions an AML training dataset across 3 heterogeneous client domains:
- client_1: Retail Giant
- client_2: Offshore / Crypto Exchange
- client_3: Regional Credit Union

Enforces atomic account-level partitioning so that all transactions belonging
to the same originating account remain on the same client, guaranteeing that
temporal sequences never cross client boundaries.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from application.dto.federated import ClientPartitionStats, NonIIDPartitionReport

logger = logging.getLogger(__name__)

CLIENT_1_ID = "client_1"
CLIENT_1_TYPOLOGY = "Retail Giant"

CLIENT_2_ID = "client_2"
CLIENT_2_TYPOLOGY = "Offshore / Crypto Exchange"

CLIENT_3_ID = "client_3"
CLIENT_3_TYPOLOGY = "Regional Credit Union"

CLIENT_DEFINITIONS: tuple[tuple[str, str], ...] = (
    (CLIENT_1_ID, CLIENT_1_TYPOLOGY),
    (CLIENT_2_ID, CLIENT_2_TYPOLOGY),
    (CLIENT_3_ID, CLIENT_3_TYPOLOGY),
)


class NonIIDPartitionerError(ValueError):
    """Raised when a Non-IID partition fails validation invariants."""


class NonIIDPartitioner:
    """Partitions tabular transactions into 3 typology-based Non-IID client subsets.

    Partitioning is performed at the originating account (`from_account`) level to
    guarantee sequence integrity. Feature mappings are grounded in the verified
    IBM AML dataset preprocessing schema:
    - Retail: Credit Card (format=4), Cash (format=2)
    - Offshore / Crypto: Bitcoin format/currency (code=1), Cross-currency (same_currency=0), Wire (format=6)
    - Regional CU: Cheque (format=3), ACH (format=0), Reinvestment (format=5), Intra-bank (same_bank=1)
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize partitioner.

        Args:
            seed: Random seed for deterministic tie-breaking on mixed-typology accounts.
        """
        self._seed = seed
        self._n_clients = 3

    @property
    def n_clients(self) -> int:
        """Number of client partitions (fixed to 3)."""
        return self._n_clients

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    def partition(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "is_laundering",
        account_column: str = "from_account",
        client_ids: Sequence[str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], NonIIDPartitionReport]:
        """Partition a DataFrame across 3 Non-IID clients by account typology.

        Args:
            frame: Input training DataFrame.
            target_column: Binary label column name ('is_laundering').
            account_column: Originating account column name ('from_account').
            client_ids: Optional custom client IDs; defaults to ('client_1', 'client_2', 'client_3').

        Returns:
            Tuple of (dict mapping client_id to client DataFrame, NonIIDPartitionReport).

        Raises:
            NonIIDPartitionerError: If partitioning fails validation checks.
        """
        if client_ids is None:
            cids = [c[0] for c in CLIENT_DEFINITIONS]
        else:
            cids = list(client_ids)
            if len(cids) != self._n_clients:
                raise ValueError(
                    f"client_ids length ({len(cids)}) must match n_clients ({self._n_clients})"
                )

        total_rows = len(frame)
        if total_rows < self._n_clients:
            raise NonIIDPartitionerError(
                f"Cannot partition {total_rows} row(s) into {self._n_clients} clients: "
                "not enough samples."
            )

        if target_column not in frame.columns:
            raise NonIIDPartitionerError(
                f"Target column '{target_column}' not found in DataFrame columns: {list(frame.columns)}"
            )

        if account_column not in frame.columns:
            raise NonIIDPartitionerError(
                f"Account column '{account_column}' not found in DataFrame columns: {list(frame.columns)}"
            )

        # 1. Identify transaction-level feature flags
        is_crypto_wire_fx = self._extract_crypto_wire_fx_mask(frame)
        is_retail = self._extract_retail_mask(frame, is_crypto_wire_fx)
        is_cu = self._extract_cu_mask(frame, is_crypto_wire_fx, is_retail)

        # 2. Build account-level typology profiles
        accts = frame[account_column].to_numpy()
        unique_accts, inverse_indices = np.unique(accts, return_inverse=True)
        n_unique = len(unique_accts)

        # Aggregate counts per unique account using bincount
        crypto_counts = np.bincount(inverse_indices, weights=is_crypto_wire_fx.astype(int), minlength=n_unique)
        retail_counts = np.bincount(inverse_indices, weights=is_retail.astype(int), minlength=n_unique)
        cu_counts = np.bincount(inverse_indices, weights=is_cu.astype(int), minlength=n_unique)

        # 3. Assign each unique account to exactly one client
        rng = np.random.default_rng(self._seed)
        acct_client_assignment = np.zeros(n_unique, dtype=int)  # 0: client_1, 1: client_2, 2: client_3

        for i in range(n_unique):
            c_cnt = crypto_counts[i]
            r_cnt = retail_counts[i]
            u_cnt = cu_counts[i]

            # In AML, any presence of crypto, cross-currency FX, or wires defines Offshore/Crypto
            if c_cnt > 0:
                acct_client_assignment[i] = 1  # client_2: Offshore / Crypto Exchange
            elif r_cnt > u_cnt:
                acct_client_assignment[i] = 0  # client_1: Retail Giant
            elif u_cnt > r_cnt:
                acct_client_assignment[i] = 2  # client_3: Regional Credit Union
            else:
                # Seeded tie-breaking between client_1 (Retail) and client_3 (Credit Union)
                acct_client_assignment[i] = 0 if rng.random() < 0.5 else 2

        # 4. Map account assignments back to transaction rows
        row_assignments = acct_client_assignment[inverse_indices]

        # 5. Build client DataFrames in original chronological order
        partitions: dict[str, pd.DataFrame] = {}
        client_stats: list[ClientPartitionStats] = []
        targets = frame[target_column].to_numpy()
        total_positives = int(np.sum(targets == 1))

        typology_names = [c[1] for c in CLIENT_DEFINITIONS]

        for k, cid in enumerate(cids):
            row_indices = np.flatnonzero(row_assignments == k)
            client_df = frame.iloc[row_indices].copy().reset_index(drop=True)
            partitions[cid] = client_df

            c_samples = len(client_df)
            c_pos = int(np.sum(client_df[target_column].to_numpy() == 1)) if c_samples > 0 else 0
            c_neg = c_samples - c_pos
            c_rate = c_pos / c_samples if c_samples > 0 else 0.0
            c_accts = int(np.sum(acct_client_assignment == k))

            # Feature distribution metrics on client partition
            feat_dist: dict[str, float] = {}
            if c_samples > 0:
                feat_dist["retail_payment_share"] = float(np.mean(self._extract_retail_mask(client_df)))
                feat_dist["crypto_wire_fx_share"] = float(np.mean(self._extract_crypto_wire_fx_mask(client_df)))
                feat_dist["cu_payment_share"] = float(np.mean(self._extract_cu_mask(client_df)))
                if "same_bank" in client_df.columns:
                    feat_dist["same_bank_share"] = float(np.mean(client_df["same_bank"].to_numpy() == 1))

            client_stats.append(
                ClientPartitionStats(
                    client_id=cid,
                    sample_count=c_samples,
                    positive_count=c_pos,
                    negative_count=c_neg,
                    positive_rate=c_rate,
                    distribution="non_iid",
                    seed=self._seed,
                    typology=typology_names[k],
                    account_count=c_accts,
                    feature_distributions=feat_dist,
                )
            )

        report = NonIIDPartitionReport(
            clients=tuple(client_stats),
            total_samples=total_rows,
            total_positives=total_positives,
            overall_positive_rate=(
                total_positives / total_rows if total_rows > 0 else 0.0
            ),
            seed=self._seed,
            partition_method="typology_rule_v1",
        )

        self.validate_partition(
            frame,
            partitions,
            report,
            target_column=target_column,
            account_column=account_column,
        )

        logger.info(
            "Non-IID typology partition complete: %d rows across %d clients with seed %d. "
            "Client sizes: %s. Overall positive rate: %.6f%%",
            total_rows,
            self._n_clients,
            self._seed,
            [len(df) for df in partitions.values()],
            report.overall_positive_rate * 100.0,
        )

        return partitions, report

    def _extract_crypto_wire_fx_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Identify transactions characteristic of Offshore / Crypto Exchange."""
        mask = np.zeros(len(frame), dtype=bool)

        if "payment_format_code" in frame.columns:
            fmt = frame["payment_format_code"].to_numpy()
            mask |= (fmt == 1) | (fmt == 6)  # 1: Bitcoin, 6: Wire

        if "payment_currency_code" in frame.columns:
            mask |= frame["payment_currency_code"].to_numpy() == 1  # 1: Bitcoin

        if "receiving_currency_code" in frame.columns:
            mask |= frame["receiving_currency_code"].to_numpy() == 1  # 1: Bitcoin

        if "same_currency" in frame.columns:
            mask |= frame["same_currency"].to_numpy() == 0  # 0: Cross-currency FX

        return mask

    def _extract_retail_mask(
        self, frame: pd.DataFrame, exclude_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Identify transactions characteristic of Retail Giant."""
        mask = np.zeros(len(frame), dtype=bool)
        if "payment_format_code" in frame.columns:
            fmt = frame["payment_format_code"].to_numpy()
            mask |= (fmt == 2) | (fmt == 4)  # 2: Cash, 4: Credit Card

        if exclude_mask is not None:
            mask &= ~exclude_mask

        return mask

    def _extract_cu_mask(
        self,
        frame: pd.DataFrame,
        exclude_crypto: np.ndarray | None = None,
        exclude_retail: np.ndarray | None = None,
    ) -> np.ndarray:
        """Identify transactions characteristic of Regional Credit Union."""
        mask = np.zeros(len(frame), dtype=bool)
        if "payment_format_code" in frame.columns:
            fmt = frame["payment_format_code"].to_numpy()
            mask |= (fmt == 0) | (fmt == 3) | (fmt == 5)  # 0: ACH, 3: Cheque, 5: Reinvestment

        if "same_bank" in frame.columns:
            mask |= frame["same_bank"].to_numpy() == 1  # 1: Intra-bank local transfer

        if exclude_crypto is not None:
            mask &= ~exclude_crypto
        if exclude_retail is not None:
            mask &= ~exclude_retail

        return mask

    def validate_partition(
        self,
        original_frame: pd.DataFrame,
        partitions: dict[str, pd.DataFrame],
        report: NonIIDPartitionReport,
        *,
        target_column: str = "is_laundering",
        account_column: str = "from_account",
    ) -> None:
        """Validate all required Non-IID partition invariants.

        Raises:
            NonIIDPartitionerError: If any invariant is violated.
        """
        # 1. Exactly 3 clients exist
        if len(partitions) != self._n_clients:
            raise NonIIDPartitionerError(
                f"Expected exactly {self._n_clients} client partitions, got {len(partitions)}"
            )

        # 2. Every client contains training samples
        for cid, df in partitions.items():
            if len(df) == 0:
                raise NonIIDPartitionerError(f"Client '{cid}' partition is empty")

        # 3. Row conservation: total partition rows equals original training set
        total_partition_rows = sum(len(df) for df in partitions.values())
        if total_partition_rows != len(original_frame):
            raise NonIIDPartitionerError(
                f"Total partition rows ({total_partition_rows}) != original rows ({len(original_frame)})"
            )

        # 4. Account atomicity: No account appears across multiple clients
        client_accounts: list[set[Any]] = [
            set(df[account_column].unique()) for df in partitions.values()
        ]
        for i in range(len(client_accounts)):
            for j in range(i + 1, len(client_accounts)):
                overlap = client_accounts[i].intersection(client_accounts[j])
                if overlap:
                    cids = list(partitions.keys())
                    raise NonIIDPartitionerError(
                        f"Account overlap between '{cids[i]}' and '{cids[j]}': {len(overlap)} shared accounts"
                    )

        # 5. Label conservation: overall positive count equals original positive count
        original_positives = int(np.sum(original_frame[target_column].to_numpy() == 1))
        partition_positives = sum(
            int(np.sum(df[target_column].to_numpy() == 1)) for df in partitions.values()
        )
        if partition_positives != original_positives:
            raise NonIIDPartitionerError(
                f"Partition positive count ({partition_positives}) != original positive count ({original_positives})"
            )

        # 6. Chronological ordering preserved within each client
        if "timestamp" in original_frame.columns:
            for cid, df in partitions.items():
                if len(df) <= 1:
                    continue
                stamps = df["timestamp"]
                if stamps.is_monotonic_increasing:
                    continue
                parsed = pd.to_datetime(stamps, errors="coerce")
                if not parsed.is_monotonic_increasing:
                    raise NonIIDPartitionerError(
                        f"Client '{cid}' rows are not in chronological order"
                    )

        # 7. Heterogeneity verification
        if len(original_frame) > 1000:
            pos_rates = [c.positive_rate for c in report.clients]
            # Verify that client positive rates or feature distributions are not identical (genuine Non-IID)
            retail_shares = [
                c.feature_distributions.get("retail_payment_share", 0.0)
                for c in report.clients
                if c.feature_distributions
            ]
            crypto_shares = [
                c.feature_distributions.get("crypto_wire_fx_share", 0.0)
                for c in report.clients
                if c.feature_distributions
            ]
            if len(set(pos_rates)) <= 1 and (max(retail_shares) - min(retail_shares) < 1e-4):
                raise NonIIDPartitionerError(
                    f"Partitions show no measurable heterogeneity across clients: {pos_rates}"
                )
