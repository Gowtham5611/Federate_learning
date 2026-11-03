"""Unit tests for the deterministic IID stratified partitioner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infrastructure.data.partitioners.iid_partitioner import (
    IIDPartitioner,
    IIDPartitionerError,
)


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """Create a synthetic AML dataset with realistic severe class imbalance."""
    rng = np.random.default_rng(42)
    n_samples = 3000
    # 0.1% positives ~ 3 positives in 3000 samples
    n_positives = 6
    labels = np.zeros(n_samples, dtype=int)
    pos_indices = rng.choice(n_samples, size=n_positives, replace=False)
    labels[pos_indices] = 1

    return pd.DataFrame(
        {
            "timestamp": [f"2026-01-01 {i:05d}" for i in range(n_samples)],
            "from_account": [f"acc_{rng.integers(0, 100)}" for _ in range(n_samples)],
            "to_account": [f"acc_{rng.integers(0, 100)}" for _ in range(n_samples)],
            "amount": rng.uniform(10.0, 10000.0, size=n_samples),
            "is_laundering": labels,
        }
    )


def test_exactly_three_clients_created(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)

    assert len(partitions) == 3
    assert set(partitions.keys()) == {"client_1", "client_2", "client_3"}
    assert len(report.clients) == 3


def test_deterministic_partition_with_seed(synthetic_training_data: pd.DataFrame) -> None:
    p1 = IIDPartitioner(n_clients=3, seed=42)
    partitions1, _ = p1.partition(synthetic_training_data)

    p2 = IIDPartitioner(n_clients=3, seed=42)
    partitions2, _ = p2.partition(synthetic_training_data)

    for cid in ("client_1", "client_2", "client_3"):
        pd.testing.assert_frame_equal(partitions1[cid], partitions2[cid])


def test_different_seed_changes_partition(synthetic_training_data: pd.DataFrame) -> None:
    p1 = IIDPartitioner(n_clients=3, seed=42)
    partitions1, _ = p1.partition(synthetic_training_data)

    p2 = IIDPartitioner(n_clients=3, seed=99)
    partitions2, _ = p2.partition(synthetic_training_data)

    # Different seeds should assign different rows
    assert not partitions1["client_1"].equals(partitions2["client_1"])


def test_no_duplicate_or_missing_samples(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    partitions, _ = partitioner.partition(synthetic_training_data)

    # Check row counts sum exactly to original
    total_partition_rows = sum(len(df) for df in partitions.values())
    assert total_partition_rows == len(synthetic_training_data)

    # Check timestamps across partitions are mutually disjoint
    all_timestamps: set[str] = set()
    for df in partitions.values():
        stamps = set(df["timestamp"])
        assert len(stamps & all_timestamps) == 0
        all_timestamps.update(stamps)

    assert len(all_timestamps) == len(synthetic_training_data)


def test_approximately_balanced_client_sizes(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    partitions, _ = partitioner.partition(synthetic_training_data)

    sizes = [len(df) for df in partitions.values()]
    assert max(sizes) - min(sizes) <= 2  # at most 1 diff per class
    for s in sizes:
        assert s == 1000


def test_similar_class_proportions_and_exact_positive_count(
    synthetic_training_data: pd.DataFrame,
) -> None:
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)

    # Total positives was 6, so each client should get exactly 2
    for _cid, df in partitions.items():
        pos = int(df["is_laundering"].sum())
        assert pos == 2

    assert report.total_positives == 6
    assert sum(c.positive_count for c in report.clients) == 6


def test_rows_preserve_chronological_order(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    partitions, _ = partitioner.partition(synthetic_training_data)

    for _cid, df in partitions.items():
        stamps = df["timestamp"].tolist()
        assert stamps == sorted(stamps)


def test_raises_on_too_few_rows() -> None:
    tiny_df = pd.DataFrame({"is_laundering": [0, 1]})
    partitioner = IIDPartitioner(n_clients=3, seed=42)
    with pytest.raises(IIDPartitionerError, match="Cannot partition"):
        partitioner.partition(tiny_df)

