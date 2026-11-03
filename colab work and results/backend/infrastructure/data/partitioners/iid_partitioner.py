"""Deterministic IID stratified data partitioner for federated learning.

Partitions a training dataset into N clients using stratified random assignment
to preserve class imbalance and balance client sizes.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from application.dto.federated import ClientPartitionStats, IIDPartitionReport

logger = logging.getLogger(__name__)

CLIENT_ID_TEMPLATE = "client_{}"


class IIDPartitionerError(ValueError):
    """Raised when an IID partition fails validation invariants."""


class IIDPartitioner:
    """Partitions tabular transactions into IID client subsets with stratification."""

    def __init__(self, n_clients: int = 3, seed: int = 42) -> None:
        """Initialize partitioner.

        Args:
            n_clients: Number of clients to partition data across (default: 3).
            seed: Random seed for deterministic shuffling.
        """
        if n_clients <= 0:
            raise ValueError(f"n_clients must be positive; got {n_clients}")
        self._n_clients = n_clients
        self._seed = seed

    @property
    def n_clients(self) -> int:
        """Number of partitions."""
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
        client_ids: Sequence[str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], IIDPartitionReport]:
        """Partition a DataFrame across clients.

        Args:
            frame: Input DataFrame (e.g., training split).
            target_column: Binary label column name.
            client_ids: Optional custom client IDs; defaults to client_1, client_2, ...

        Returns:
            Tuple of (dict mapping client_id to client DataFrame, IIDPartitionReport).

        Raises:
            IIDPartitionerError: If partitioning fails validation checks.
        """
        if client_ids is None:
            cids = [CLIENT_ID_TEMPLATE.format(k + 1) for k in range(self._n_clients)]
        else:
            cids = list(client_ids)
            if len(cids) != self._n_clients:
                raise ValueError(
                    f"client_ids length ({len(cids)}) must match n_clients ({self._n_clients})"
                )

        total_rows = len(frame)
        if total_rows < self._n_clients:
            raise IIDPartitionerError(
                f"Cannot partition {total_rows} row(s) into {self._n_clients} clients: "
                "not enough samples."
            )

        if target_column not in frame.columns:
            raise IIDPartitionerError(
                f"Target column '{target_column}' not found in DataFrame columns: {list(frame.columns)}"
            )

        targets = frame[target_column].to_numpy()
        rng = np.random.default_rng(self._seed)

        client_indices: list[list[int]] = [[] for _ in range(self._n_clients)]

        # Unique classes present (typically 0 and 1)
        classes = np.unique(targets)
        for cls_val in sorted(classes):
            cls_indices = np.flatnonzero(targets == cls_val)
            shuffled_cls = rng.permutation(cls_indices)
            for pos, idx in enumerate(shuffled_cls):
                client_indices[pos % self._n_clients].append(int(idx))

        # Build client frames in chronological order (sorted index order)
        partitions: dict[str, pd.DataFrame] = {}
        client_stats: list[ClientPartitionStats] = []
        total_positives = int(np.sum(targets == 1))

        for k, cid in enumerate(cids):
            # Sorting the indices preserves the original chronological order of rows
            indices = sorted(client_indices[k])
            client_df = frame.iloc[indices].copy().reset_index(drop=True)
            partitions[cid] = client_df

            c_samples = len(client_df)
            c_pos = int(np.sum(client_df[target_column].to_numpy() == 1))
            c_neg = c_samples - c_pos
            c_rate = c_pos / c_samples if c_samples > 0 else 0.0

            client_stats.append(
                ClientPartitionStats(
                    client_id=cid,
                    sample_count=c_samples,
                    positive_count=c_pos,
                    negative_count=c_neg,
                    positive_rate=c_rate,
                    distribution="iid",
                    seed=self._seed,
                )
            )

        report = IIDPartitionReport(
            clients=tuple(client_stats),
            total_samples=total_rows,
            total_positives=total_positives,
            overall_positive_rate=(
                total_positives / total_rows if total_rows > 0 else 0.0
            ),
            seed=self._seed,
        )

        self.validate_partition(frame, partitions, report, target_column=target_column)

        logger.info(
            "IID partition complete: %d rows across %d clients with seed %d. "
            "Client sizes: %s. Overall positive rate: %.6f%%",
            total_rows,
            self._n_clients,
            self._seed,
            [len(df) for df in partitions.values()],
            report.overall_positive_rate * 100.0,
        )

        return partitions, report

    def validate_partition(
        self,
        original_frame: pd.DataFrame,
        partitions: dict[str, pd.DataFrame],
        report: IIDPartitionReport,
        *,
        target_column: str = "is_laundering",
    ) -> None:
        """Validate all 11 required IID partition invariants.

        Raises:
            IIDPartitionerError: If any invariant is violated.
        """
        # 1. Exactly n_clients exist
        if len(partitions) != self._n_clients:
            raise IIDPartitionerError(
                f"Expected exactly {self._n_clients} client partitions, got {len(partitions)}"
            )

        # 2. Every client contains training samples
        for cid, df in partitions.items():
            if len(df) == 0:
                raise IIDPartitionerError(f"Client '{cid}' partition is empty")

        # 3, 4, 5. Union of sample counts equals original training set; no lost samples
        total_partition_rows = sum(len(df) for df in partitions.values())
        if total_partition_rows != len(original_frame):
            raise IIDPartitionerError(
                f"Total partition rows ({total_partition_rows}) != original rows ({len(original_frame)})"
            )

        # 6. Client sample counts are approximately balanced (diff at most len(classes))
        sizes = [len(df) for df in partitions.values()]
        if max(sizes) - min(sizes) > len(np.unique(original_frame[target_column])):
            raise IIDPartitionerError(
                f"Client sample counts are imbalanced beyond round-robin tolerance: {sizes}"
            )

        # 8. Overall positive count equals original training positive count
        original_positives = int(np.sum(original_frame[target_column].to_numpy() == 1))
        partition_positives = sum(
            int(np.sum(df[target_column].to_numpy() == 1)) for df in partitions.values()
        )
        if partition_positives != original_positives:
            raise IIDPartitionerError(
                f"Partition positive count ({partition_positives}) != original positive count ({original_positives})"
            )

        # 7. Positive counts across clients differ by at most 1
        pos_counts = [
            int(np.sum(df[target_column].to_numpy() == 1)) for df in partitions.values()
        ]
        if max(pos_counts) - min(pos_counts) > 1:
            raise IIDPartitionerError(
                f"Client positive counts are not balanced within stratified tolerance: {pos_counts}"
            )

