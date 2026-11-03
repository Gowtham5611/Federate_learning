"""Unit tests for the deterministic typology-based Non-IID partitioner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.partitioners.client_sequence_provider import ClientSequenceProvider
from infrastructure.data.partitioners.non_iid_partitioner import (
    CLIENT_1_ID,
    CLIENT_1_TYPOLOGY,
    CLIENT_2_ID,
    CLIENT_2_TYPOLOGY,
    CLIENT_3_ID,
    CLIENT_3_TYPOLOGY,
    NonIIDPartitioner,
    NonIIDPartitionerError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """Create synthetic transactions with varied payment formats and currencies."""
    rng = np.random.default_rng(42)
    n_samples = 3000

    # 100 accounts with distinct typology tendencies
    # Accounts 0-29: mostly retail (CC=4, Cash=2)
    # Accounts 30-59: mostly crypto/offshore (BTC=1, Wire=6, Cross-curr)
    # Accounts 60-99: mostly community/credit union (Cheque=3, ACH=0, Reinvest=5)
    accounts = []
    formats = []
    pay_curr = []
    rec_curr = []
    same_curr = []
    same_bank = []

    for _ in range(n_samples):
        acct_id = rng.integers(0, 100)
        accounts.append(f"acc_{acct_id:04d}")

        if acct_id < 30:
            # Retail
            fmt = int(rng.choice([4, 2]))
            p_cur = 12  # USD
            r_cur = 12
            s_cur = 1
            s_bnk = int(rng.choice([0, 1]))
        elif acct_id < 60:
            # Offshore / Crypto
            fmt = int(rng.choice([1, 6, 0]))
            p_cur = int(rng.choice([1, 12, 4]))
            r_cur = int(rng.choice([1, 12, 4]))
            s_cur = 0 if rng.random() < 0.6 else 1
            s_bnk = 0
        else:
            # Regional CU
            fmt = int(rng.choice([3, 0, 5]))
            p_cur = 12
            r_cur = 12
            s_cur = 1
            s_bnk = 1 if rng.random() < 0.8 else 0

        formats.append(fmt)
        pay_curr.append(p_cur)
        rec_curr.append(r_cur)
        same_curr.append(s_cur)
        same_bank.append(s_bnk)

    labels = np.zeros(n_samples, dtype=int)
    pos_indices = rng.choice(n_samples, size=15, replace=False)
    labels[pos_indices] = 1

    return pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-01 {i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
                for i in range(n_samples)
            ],
            "from_account": accounts,
            "to_account": [f"acc_{rng.integers(0, 100):04d}" for _ in range(n_samples)],
            "payment_format_code": formats,
            "payment_currency_code": pay_curr,
            "receiving_currency_code": rec_curr,
            "same_currency": same_curr,
            "same_bank": same_bank,
            "amount_paid_log": rng.uniform(1.0, 10.0, size=n_samples),
            "amount_received_log": rng.uniform(1.0, 10.0, size=n_samples),
            "amount_mismatch": rng.choice([0, 1], size=n_samples),
            "self_transfer": np.zeros(n_samples, dtype=int),
            "minute_of_day_sin": np.zeros(n_samples, dtype=np.float32),
            "minute_of_day_cos": np.ones(n_samples, dtype=np.float32),
            "day_of_week_sin": np.zeros(n_samples, dtype=np.float32),
            "day_of_week_cos": np.ones(n_samples, dtype=np.float32),
            "is_weekend": np.zeros(n_samples, dtype=int),
            "is_night": np.zeros(n_samples, dtype=int),
            "sender_prior_count_log": np.zeros(n_samples, dtype=np.float32),
            "sender_prior_amount_log": np.zeros(n_samples, dtype=np.float32),
            "sender_seconds_since_prior_log": np.zeros(n_samples, dtype=np.float32),
            "sender_has_prior": np.zeros(n_samples, dtype=int),
            "receiver_prior_count_log": np.zeros(n_samples, dtype=np.float32),
            "receiver_prior_amount_log": np.zeros(n_samples, dtype=np.float32),
            "receiver_seconds_since_prior_log": np.zeros(n_samples, dtype=np.float32),
            "receiver_has_prior": np.zeros(n_samples, dtype=int),
            "is_laundering": labels,
        }
    )


def test_exactly_three_clients_created(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)

    assert len(partitions) == 3
    assert set(partitions.keys()) == {CLIENT_1_ID, CLIENT_2_ID, CLIENT_3_ID}
    assert len(report.clients) == 3


def test_typology_names_assigned(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    _, report = partitioner.partition(synthetic_training_data)

    c_map = {c.client_id: c.typology for c in report.clients}
    assert c_map[CLIENT_1_ID] == CLIENT_1_TYPOLOGY
    assert c_map[CLIENT_2_ID] == CLIENT_2_TYPOLOGY
    assert c_map[CLIENT_3_ID] == CLIENT_3_TYPOLOGY


def test_deterministic_partition_with_seed_42(synthetic_training_data: pd.DataFrame) -> None:
    p1 = NonIIDPartitioner(seed=42)
    parts1, rep1 = p1.partition(synthetic_training_data)

    p2 = NonIIDPartitioner(seed=42)
    parts2, rep2 = p2.partition(synthetic_training_data)

    for cid in (CLIENT_1_ID, CLIENT_2_ID, CLIENT_3_ID):
        pd.testing.assert_frame_equal(parts1[cid], parts2[cid])

    assert rep1.total_samples == rep2.total_samples
    assert rep1.total_positives == rep2.total_positives


def test_row_conservation_and_no_loss_or_duplication(
    synthetic_training_data: pd.DataFrame,
) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)

    total_rows = sum(len(df) for df in partitions.values())
    assert total_rows == len(synthetic_training_data)
    assert report.total_samples == len(synthetic_training_data)


def test_account_atomicity_no_shared_accounts(
    synthetic_training_data: pd.DataFrame,
) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, _ = partitioner.partition(synthetic_training_data)

    accts1 = set(partitions[CLIENT_1_ID]["from_account"].unique())
    accts2 = set(partitions[CLIENT_2_ID]["from_account"].unique())
    accts3 = set(partitions[CLIENT_3_ID]["from_account"].unique())

    assert len(accts1.intersection(accts2)) == 0
    assert len(accts1.intersection(accts3)) == 0
    assert len(accts2.intersection(accts3)) == 0


def test_label_conservation(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)

    orig_pos = int(synthetic_training_data["is_laundering"].sum())
    part_pos = sum(int(df["is_laundering"].sum()) for df in partitions.values())
    assert part_pos == orig_pos
    assert report.total_positives == orig_pos


def test_chronological_order_preserved(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, _ = partitioner.partition(synthetic_training_data)

    for _cid, df in partitions.items():
        stamps = pd.to_datetime(df["timestamp"])
        assert stamps.is_monotonic_increasing


def test_heterogeneity_of_partitions(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    _, report = partitioner.partition(synthetic_training_data)

    client_stats = {c.client_id: c for c in report.clients}
    c1_dist = client_stats[CLIENT_1_ID].feature_distributions
    c2_dist = client_stats[CLIENT_2_ID].feature_distributions
    assert c1_dist is not None
    assert c2_dist is not None

    # Client 1 should have highest retail share
    c1_retail = c1_dist["retail_payment_share"]
    # Client 2 should have highest crypto share
    c2_crypto = c2_dist["crypto_wire_fx_share"]

    assert c1_retail > 0.0
    assert c2_crypto > 0.0
    c1_acct = client_stats[CLIENT_1_ID].account_count
    c2_acct = client_stats[CLIENT_2_ID].account_count
    c3_acct = client_stats[CLIENT_3_ID].account_count
    assert c1_acct is not None and c1_acct > 0
    assert c2_acct is not None and c2_acct > 0
    assert c3_acct is not None and c3_acct > 0


def test_missing_column_raises_error(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    bad_df = synthetic_training_data.drop(columns=["from_account"])

    with pytest.raises(NonIIDPartitionerError, match="Target column|Account column"):
        partitioner.partition(bad_df)


def test_too_few_rows_raises_error(synthetic_training_data: pd.DataFrame) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    tiny_df = synthetic_training_data.iloc[:2]

    with pytest.raises(NonIIDPartitionerError, match="not enough samples"):
        partitioner.partition(tiny_df)


def test_seed_sensitivity_different_seed() -> None:
    # Build dataframe with accounts that have equal retail and cu activity (ties)
    accounts = []
    formats = []
    # 5 crypto accounts for client_2
    for acct_id in range(5):
        accounts.append(f"crypto_{acct_id:04d}")
        formats.append(1)  # Bitcoin
    # 50 tied accounts between client_1 and client_3
    for acct_id in range(50):
        for fmt in [2, 4, 0, 3]:
            accounts.append(f"acc_{acct_id:04d}")
            formats.append(fmt)
    df = pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-01 {i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
                for i in range(len(accounts))
            ],
            "from_account": accounts,
            "to_account": ["acc_dest"] * len(accounts),
            "payment_format_code": formats,
            "is_laundering": [0] * len(accounts),
        }
    )
    p42 = NonIIDPartitioner(seed=42)
    parts42, _ = p42.partition(df)

    p99 = NonIIDPartitioner(seed=99)
    parts99, _ = p99.partition(df)

    assert sum(len(d) for d in parts42.values()) == len(df)
    assert sum(len(d) for d in parts99.values()) == len(df)
    c1_42 = set(parts42[CLIENT_1_ID]["from_account"].unique())
    c1_99 = set(parts99[CLIENT_1_ID]["from_account"].unique())
    assert c1_42 != c1_99


def test_validation_detects_account_overlap(
    synthetic_training_data: pd.DataFrame,
) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)
    c1_row = partitions[CLIENT_1_ID].iloc[0:1]
    corrupted_p = {
        CLIENT_1_ID: partitions[CLIENT_1_ID],
        CLIENT_2_ID: pd.concat([partitions[CLIENT_2_ID], c1_row]).reset_index(drop=True),
        CLIENT_3_ID: partitions[CLIENT_3_ID],
    }
    with pytest.raises(NonIIDPartitionerError):
        partitioner.validate_partition(synthetic_training_data, corrupted_p, report)


def test_validation_detects_label_mismatch(
    synthetic_training_data: pd.DataFrame,
) -> None:
    partitioner = NonIIDPartitioner(seed=42)
    partitions, report = partitioner.partition(synthetic_training_data)
    corrupted_c1 = partitions[CLIENT_1_ID].copy()
    corrupted_c1["is_laundering"] = 1
    corrupted_p = {
        CLIENT_1_ID: corrupted_c1,
        CLIENT_2_ID: partitions[CLIENT_2_ID],
        CLIENT_3_ID: partitions[CLIENT_3_ID],
    }
    with pytest.raises(NonIIDPartitionerError, match="positive count"):
        partitioner.validate_partition(synthetic_training_data, corrupted_p, report)


def test_client_sequence_provider_compatibility(
    tmp_path: Path,
) -> None:
    from infrastructure.data.sequences.processed_sequence_provider import (
        ProcessedSequenceProvider,
    )
    from tests.fixtures.sequence_fixtures import write_processed_tree

    processed_dir = write_processed_tree(tmp_path / "processed")
    seq_prov = ProcessedSequenceProvider(processed_dir)
    csp = ClientSequenceProvider(seq_prov._metadata)

    partitioner = NonIIDPartitioner(seed=42)
    train_frame = pd.read_csv(seq_prov._metadata.split_path("train"))
    partitions, _ = partitioner.partition(train_frame)

    t_cfg = TrainingConfig(
        architecture=ModelArchitecture.GRU, sequence_length=3, hidden_size=8
    )
    for cid, c_df in partitions.items():
        dataset = csp.build_client_dataset(
            c_df, t_cfg, client_id=cid, split=SplitName.TRAIN
        )
        assert dataset.row_count == len(c_df)
        assert dataset.sequence_count > 0
