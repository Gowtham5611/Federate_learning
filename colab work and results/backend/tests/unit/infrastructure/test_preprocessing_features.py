"""Tests for the three feature builders, with leakage as the central concern.

The builders divide cleanly by risk. The transaction features are row-local and
cannot leak. The temporal features are derived from a single timestamp and
cannot leak either — but the *set* of them matters, because a monotonic
"days since start" feature would be a leak of a different kind, and the builder
records why it is absent. The account-history builder is the one that could
genuinely look forward, so most of this file is about it.

The property tested for the history builder is stronger than "the numbers look
right": chunking must not change the answer. If feeding rows in three chunks
produced different history from feeding them in one, the memory-bounded pipeline
would be quietly computing something other than what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame, Series

from infrastructure.data.features.account_history import (
    NO_PRIOR_SECONDS,
    AccountHistoryBuilder,
)
from infrastructure.data.features.temporal_features import (
    EPOCH_SECONDS_COLUMN,
    MINUTES_PER_DAY,
    NIGHT_ENDS_AT_HOUR,
    PARSED_TIMESTAMP_COLUMN,
    TemporalFeatureBuilder,
    parse_timestamps,
)
from infrastructure.data.features.transaction_features import TransactionFeatureBuilder
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_FROM_BANK,
    COL_PAYMENT_CURRENCY,
    COL_RECEIVING_CURRENCY,
    COL_TO_ACCOUNT,
    COL_TO_BANK,
    TIMESTAMP_FORMAT,
)

pytestmark = pytest.mark.unit


# --- Transaction features ------------------------------------------------------


def _transaction_frame(**overrides: object) -> DataFrame:
    """One synthetic row, with any field overridable."""
    row: dict[str, object] = {
        COL_AMOUNT_PAID: 100.0,
        COL_AMOUNT_RECEIVED: 100.0,
        COL_PAYMENT_CURRENCY: "US Dollar",
        COL_RECEIVING_CURRENCY: "US Dollar",
        COL_FROM_BANK: "010",
        COL_TO_BANK: "011",
        COL_FROM_ACCOUNT: "TEST00001",
        COL_TO_ACCOUNT: "TEST00002",
    }
    row.update(overrides)
    return DataFrame([row])


def _feature(name: str, **overrides: object) -> float:
    built = TransactionFeatureBuilder().apply(_transaction_frame(**overrides))
    return float(built[name].iloc[0])


def test_the_transaction_builder_produces_its_six_declared_columns() -> None:
    built = TransactionFeatureBuilder().apply(_transaction_frame())

    for column in TransactionFeatureBuilder().produced_columns:
        assert column in built.columns


def test_amounts_are_log1p_transformed_not_clipped() -> None:
    """Twelve orders of magnitude were measured; the largest are the signal."""
    assert _feature("amount_paid_log", **{COL_AMOUNT_PAID: 1_000_000_000_000.0}) == (
        pytest.approx(np.log1p(1_000_000_000_000.0), rel=1e-6)
    )


def test_a_very_large_amount_survives_rather_than_being_trimmed() -> None:
    """Rule 5: unusual values in AML may be meaningful, so nothing is dropped."""
    frame = _transaction_frame(**{COL_AMOUNT_PAID: 1.0463e12})

    built = TransactionFeatureBuilder().apply(frame)

    assert len(built) == 1
    assert np.isfinite(built["amount_paid_log"].iloc[0])


def test_a_tiny_amount_stays_finite() -> None:
    """The verified minimum is 1e-06; log1p keeps it defined and near zero."""
    assert _feature("amount_paid_log", **{COL_AMOUNT_PAID: 1e-06}) == pytest.approx(
        1e-06, abs=1e-09
    )


def test_the_relationship_flags_read_the_two_sides_of_the_row() -> None:
    assert _feature("amount_mismatch") == 0
    assert _feature("amount_mismatch", **{COL_AMOUNT_RECEIVED: 99.0}) == 1
    assert _feature("same_currency") == 1
    assert _feature("same_currency", **{COL_RECEIVING_CURRENCY: "Euro"}) == 1 - 1
    assert _feature("same_bank") == 0
    assert _feature("same_bank", **{COL_TO_BANK: "010"}) == 1
    assert _feature("self_transfer") == 0
    assert _feature("self_transfer", **{COL_TO_ACCOUNT: "TEST00001"}) == 1


def test_bank_identifiers_are_compared_as_text_not_as_numbers() -> None:
    """The columns are read as strings so a chunk boundary cannot change the type.

    ``'010'`` and ``10`` are the same bank; if one chunk inferred int64 the flag
    would flip for no reason.
    """
    assert _feature("same_bank", **{COL_FROM_BANK: "010", COL_TO_BANK: "010"}) == 1


def test_the_builder_leaves_the_input_frame_alone() -> None:
    """Chunks are reused by the caller; in-place mutation would corrupt them."""
    frame = _transaction_frame()

    TransactionFeatureBuilder().apply(frame)

    assert "amount_paid_log" not in frame.columns


def test_the_transaction_builder_records_what_it_refused_to_build() -> None:
    """Rule 6: a feature that cannot be built safely is documented, not dropped."""
    rejected = TransactionFeatureBuilder().rejected_features

    assert "amount_difference" in rejected
    assert "currencies" in rejected["amount_difference"]
    assert "bank_identifier_encoded" in rejected


# --- Temporal features ---------------------------------------------------------


def _temporal_frame(stamps: list[str]) -> DataFrame:
    frame = DataFrame({"Timestamp": stamps})
    frame[PARSED_TIMESTAMP_COLUMN] = parse_timestamps(frame["Timestamp"], TIMESTAMP_FORMAT)
    return frame


def test_midnight_and_the_minute_before_it_are_adjacent_in_feature_space() -> None:
    """The whole reason for the cyclical encoding.

    As raw minute-of-day these two are 1439 apart, the widest possible gap.
    """
    built = TemporalFeatureBuilder().apply(
        _temporal_frame(["2020/01/01 00:00", "2020/01/01 23:59", "2020/01/01 12:00"])
    )

    def point(index: int) -> np.ndarray:
        return np.array(
            [built["minute_of_day_sin"].iloc[index], built["minute_of_day_cos"].iloc[index]]
        )

    near = float(np.linalg.norm(point(0) - point(1)))
    far = float(np.linalg.norm(point(0) - point(2)))

    assert near < far


def test_the_cyclical_pairs_stay_on_the_unit_circle() -> None:
    """Which is why they are documented as needing no scaling."""
    built = TemporalFeatureBuilder().apply(
        _temporal_frame([f"2020/01/01 {hour:02d}:30" for hour in range(24)])
    )

    radius = built["minute_of_day_sin"] ** 2 + built["minute_of_day_cos"] ** 2

    assert list(radius) == pytest.approx([1.0] * 24, abs=1e-6)


def test_the_cycle_closes_after_exactly_one_day() -> None:
    assert MINUTES_PER_DAY == 1440


def test_the_weekend_flag_follows_the_calendar() -> None:
    """2020-01-04 is a Saturday, 2020-01-05 a Sunday, 2020-01-06 a Monday."""
    built = TemporalFeatureBuilder().apply(
        _temporal_frame(["2020/01/04 09:00", "2020/01/05 09:00", "2020/01/06 09:00"])
    )

    assert list(built["is_weekend"]) == [1, 1, 0]


def test_the_night_flag_uses_the_recorded_cutoff() -> None:
    built = TemporalFeatureBuilder().apply(
        _temporal_frame(["2020/01/01 05:59", "2020/01/01 06:00"])
    )

    assert NIGHT_ENDS_AT_HOUR == 6
    assert list(built["is_night"]) == [1, 0]


def test_a_row_index_that_is_not_a_range_survives_the_encoding() -> None:
    """Day partitions are selected by boolean mask, so indices arrive with gaps."""
    frame = _temporal_frame(["2020/01/01 00:00", "2020/01/01 06:00"])
    frame.index = [11, 93]

    built = TemporalFeatureBuilder().apply(frame)

    assert list(built.index) == [11, 93]
    assert not built["minute_of_day_sin"].isna().any()


def test_no_feature_counts_days_from_the_start_of_the_dataset() -> None:
    """A monotone index whose test range never occurs in training.

    It is not a leak of the future into the past, but it makes every test metric
    a measure of extrapolation, so it is refused with the reason recorded.
    """
    builder = TemporalFeatureBuilder()

    assert "days_since_dataset_start" not in builder.produced_columns
    assert "days_since_dataset_start" in builder.rejected_features


def test_the_forward_looking_temporal_feature_is_refused_by_name() -> None:
    """Rule 6, exactly: time-to-next needs a later row to describe an earlier one."""
    rejected = TemporalFeatureBuilder().rejected_features

    assert "time_to_next_transaction" in rejected
    assert "future information" in rejected["time_to_next_transaction"]


# --- Account history: the causal builder ---------------------------------------


def _history_frame(rows: list[tuple[str, str, float, float]]) -> DataFrame:
    """Rows as ``(from_account, to_account, amount, epoch_seconds)``."""
    return DataFrame(
        {
            COL_FROM_ACCOUNT: [row[0] for row in rows],
            COL_TO_ACCOUNT: [row[1] for row in rows],
            COL_AMOUNT_PAID: [row[2] for row in rows],
            COL_AMOUNT_RECEIVED: [row[2] for row in rows],
            EPOCH_SECONDS_COLUMN: [row[3] for row in rows],
        }
    )


def test_the_first_transaction_of_an_account_has_no_prior_history() -> None:
    """The base case of the causal rule: row 0 can see nothing."""
    built = AccountHistoryBuilder().apply(_history_frame([("A", "B", 50.0, 0.0)]))

    assert built["sender_prior_count_log"].iloc[0] == pytest.approx(np.log1p(0.0))
    assert built["sender_has_prior"].iloc[0] == 0
    assert built["sender_seconds_since_prior_log"].iloc[0] == pytest.approx(
        np.log1p(NO_PRIOR_SECONDS)
    )


def test_prior_counts_exclude_the_row_they_describe() -> None:
    """The exclusive prefix. An inclusive count would leak the row into itself."""
    built = AccountHistoryBuilder().apply(
        _history_frame(
            [("A", "B", 10.0, 0.0), ("A", "C", 20.0, 60.0), ("A", "D", 30.0, 120.0)]
        )
    )

    assert list(built["sender_prior_count_log"]) == pytest.approx(
        [np.log1p(0.0), np.log1p(1.0), np.log1p(2.0)], rel=1e-6
    )


def test_prior_amounts_exclude_the_rows_own_amount() -> None:
    built = AccountHistoryBuilder().apply(
        _history_frame(
            [("A", "B", 10.0, 0.0), ("A", "C", 20.0, 60.0), ("A", "D", 30.0, 120.0)]
        )
    )

    assert list(built["sender_prior_amount_log"]) == pytest.approx(
        [np.log1p(0.0), np.log1p(10.0), np.log1p(30.0)], rel=1e-6
    )


def test_elapsed_time_is_measured_back_to_the_previous_transaction() -> None:
    built = AccountHistoryBuilder().apply(
        _history_frame([("A", "B", 10.0, 0.0), ("A", "C", 20.0, 300.0)])
    )

    assert built["sender_seconds_since_prior_log"].iloc[1] == pytest.approx(
        np.log1p(300.0), rel=1e-6
    )
    assert list(built["sender_has_prior"]) == [0, 1]


def test_a_later_row_never_changes_an_earlier_rows_features() -> None:
    """The leakage test stated directly: truncating the future changes nothing.

    If any feature were computed over the whole frame rather than over the rows
    before each one, appending rows would rewrite earlier values.
    """
    rows = [("A", "B", 10.0, 0.0), ("A", "C", 20.0, 60.0)]
    tracked = "sender_prior_amount_log"

    short = AccountHistoryBuilder().apply(_history_frame(rows))
    long = AccountHistoryBuilder().apply(
        _history_frame([*rows, ("A", "D", 999_999.0, 120.0)])
    )

    assert list(short[tracked]) == pytest.approx(list(long[tracked][:2]))


def test_chunking_does_not_change_the_history_features() -> None:
    """The property that makes the pipeline exact rather than approximate.

    One builder fed three chunks must produce what one builder fed the whole
    frame produces. Anything else means the processed dataset depends on the
    chunk size.
    """
    rows = [
        ("A", "B", 10.0, 0.0),
        ("B", "C", 20.0, 60.0),
        ("A", "C", 30.0, 120.0),
        ("C", "A", 40.0, 180.0),
        ("A", "B", 50.0, 240.0),
        ("B", "A", 60.0, 300.0),
    ]
    tracked = list(AccountHistoryBuilder().produced_columns)

    whole = AccountHistoryBuilder().apply(_history_frame(rows))

    chunked = AccountHistoryBuilder()
    pieces = [
        chunked.apply(_history_frame(rows[start : start + 2]))
        for start in range(0, len(rows), 2)
    ]

    for column in tracked:
        expected = list(whole[column])
        actual = [value for piece in pieces for value in piece[column]]
        assert actual == pytest.approx(expected), column


def test_history_carries_across_chunk_boundaries() -> None:
    """The specific case the chunk-equivalence test covers in aggregate."""
    builder = AccountHistoryBuilder()
    builder.apply(_history_frame([("A", "B", 10.0, 0.0)]))

    second = builder.apply(_history_frame([("A", "C", 20.0, 60.0)]))

    assert second["sender_has_prior"].iloc[0] == 1
    assert second["sender_prior_amount_log"].iloc[0] == pytest.approx(np.log1p(10.0))


def test_sending_and_receiving_histories_are_kept_apart() -> None:
    """An account that has only received has no sending history, and vice versa."""
    builder = AccountHistoryBuilder()
    builder.apply(_history_frame([("A", "B", 10.0, 0.0)]))

    later = builder.apply(_history_frame([("B", "A", 20.0, 60.0)]))

    assert later["sender_has_prior"].iloc[0] == 0  # B has never sent
    assert later["receiver_has_prior"].iloc[0] == 0  # A has never received


def test_a_self_transfer_updates_both_sides() -> None:
    builder = AccountHistoryBuilder()
    builder.apply(_history_frame([("A", "A", 10.0, 0.0)]))

    later = builder.apply(_history_frame([("A", "A", 20.0, 60.0)]))

    assert later["sender_has_prior"].iloc[0] == 1
    assert later["receiver_has_prior"].iloc[0] == 1


def test_rows_sharing_a_timestamp_do_not_produce_negative_elapsed_time() -> None:
    """Minute resolution means ties are common; a negative gap would be nonsense."""
    built = AccountHistoryBuilder().apply(
        _history_frame([("A", "B", 10.0, 60.0), ("A", "C", 20.0, 60.0)])
    )

    assert built["sender_seconds_since_prior_log"].iloc[1] == pytest.approx(
        np.log1p(0.0)
    )
    assert (built["sender_seconds_since_prior_log"] >= 0).all()


def test_the_state_grows_with_accounts_not_with_rows() -> None:
    """What keeps the builder memory-safe over five million rows."""
    builder = AccountHistoryBuilder()
    builder.apply(
        _history_frame([("A", "B", 1.0, float(index)) for index in range(500)])
    )

    assert builder.distinct_accounts == 2


def test_the_account_index_extends_past_its_initial_capacity() -> None:
    """More than 1024 accounts must grow the arrays, not overflow them."""
    builder = AccountHistoryBuilder()
    builder.apply(
        _history_frame(
            [(f"A{index:05d}", f"B{index:05d}", 1.0, float(index)) for index in range(1500)]
        )
    )

    assert builder.distinct_accounts == 3000

    # And the state survived the growth: a repeat of an early account still
    # reports its prior history rather than a zeroed slot.
    repeat = builder.apply(_history_frame([("A00000", "Z", 1.0, 9999.0)]))
    assert repeat["sender_has_prior"].iloc[0] == 1


def test_an_empty_chunk_leaves_the_history_state_unchanged() -> None:
    builder = AccountHistoryBuilder()
    builder.apply(_history_frame([("A", "B", 10.0, 0.0)]))
    empty = builder.apply(
        DataFrame(
            {
                COL_FROM_ACCOUNT: Series(dtype="object"),
                COL_TO_ACCOUNT: Series(dtype="object"),
                COL_AMOUNT_PAID: Series(dtype="float64"),
                COL_AMOUNT_RECEIVED: Series(dtype="float64"),
                EPOCH_SECONDS_COLUMN: Series(dtype="float64"),
            }
        )
    )

    assert len(empty) == 0
    assert builder.distinct_accounts == 2


def test_the_history_builder_records_the_features_it_refused() -> None:
    """Including the two that would have leaked, with the reason each time."""
    rejected = AccountHistoryBuilder().rejected_features

    assert "future information" in rejected["account_total_transaction_count"]
    assert "leaks the target" in rejected["is_fraud_ring_member"]
    assert "counterparty_pair_prior_count" in rejected


def test_the_repr_reports_state_size_and_no_account_identifiers() -> None:
    """Account identifiers are the dataset's personal data; not for logs."""
    builder = AccountHistoryBuilder()
    builder.apply(_history_frame([("SECRET123", "B", 10.0, 0.0)]))

    assert "SECRET123" not in repr(builder)
    assert "2 account(s)" in repr(builder)
