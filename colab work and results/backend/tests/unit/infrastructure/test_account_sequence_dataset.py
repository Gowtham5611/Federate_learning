"""Tests for :mod:`infrastructure.data.sequences.account_sequence_dataset`.

The dataset is where the project's temporal claim either holds or does not, so
these tests check the claim directly rather than checking that a 3-D tensor came
out. Three properties matter:

1. **No future leakage.** Every row in a window is at a strictly lower index than
   its target, so nothing after the classified transaction can influence it.
2. **One account per window.** Every row in a window belongs to the same account
   as the target.
3. **No boundary crossing.** A window built from one split's rows cannot reach
   into another split, because the index only covers those rows.

The fixture is deliberately tiny — a handful of rows over two or three accounts —
so the expected windows can be written out by hand and read.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from domain.enums.split_name import SplitName
from infrastructure.data.sequences.account_sequence_dataset import (
    NO_PREVIOUS_ROW,
    AccountSequenceDataset,
    build_previous_row_index,
)

pytestmark = pytest.mark.unit


def build(
    account_codes: list[int],
    *,
    labels: list[float] | None = None,
    sequence_length: int = 3,
    stride: int = 1,
    feature_dimension: int = 2,
    split: SplitName = SplitName.TRAIN,
) -> AccountSequenceDataset:
    """Assemble a dataset whose features encode their own row number.

    Feature row ``i`` is ``[i, i * 10]``, which makes a window's contents readable
    at a glance: the first column of a feature vector *is* the row it came from.
    """
    rows = len(account_codes)
    features = np.zeros((rows, feature_dimension), dtype=np.float32)
    features[:, 0] = np.arange(rows, dtype=np.float32)
    if feature_dimension > 1:
        features[:, 1] = np.arange(rows, dtype=np.float32) * 10.0
    targets = np.array(
        labels if labels is not None else [0.0] * rows, dtype=np.float32
    )
    return AccountSequenceDataset(
        split=split,
        features=features,
        targets=targets,
        previous_row=build_previous_row_index(
            np.array(account_codes, dtype=np.int64)
        ),
        sequence_length=sequence_length,
        stride=stride,
    )


def only_batch(dataset: AccountSequenceDataset) -> tuple[
    NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]
]:
    """Collect the whole dataset as one unshuffled batch."""
    batches = list(dataset.batches(dataset.sequence_count or 1, shuffle=False))
    assert len(batches) == 1
    batch = batches[0]
    return batch.features, batch.lengths, batch.labels, batch.row_indices


# --- The previous-row index --------------------------------------------------


def test_first_transaction_of_each_account_has_no_predecessor() -> None:
    index = build_previous_row_index(np.array([0, 1, 0, 1], dtype=np.int64))

    assert index[0] == NO_PREVIOUS_ROW
    assert index[1] == NO_PREVIOUS_ROW


def test_index_links_each_row_to_the_same_accounts_previous_row() -> None:
    # Rows:      0    1    2    3    4    5
    # Accounts:  A    B    A    A    B    C
    index = build_previous_row_index(np.array([0, 1, 0, 0, 1, 2], dtype=np.int64))

    assert index.tolist() == [
        NO_PREVIOUS_ROW,  # A's first
        NO_PREVIOUS_ROW,  # B's first
        0,  # A: back to row 0
        2,  # A: back to row 2
        1,  # B: back to row 1
        NO_PREVIOUS_ROW,  # C's first
    ]


def test_index_never_points_forward() -> None:
    codes = np.array([3, 1, 3, 2, 1, 3, 2, 2], dtype=np.int64)
    index = build_previous_row_index(codes)

    for row, predecessor in enumerate(index.tolist()):
        assert predecessor < row


def test_index_of_an_empty_split_is_empty() -> None:
    index = build_previous_row_index(np.empty(0, dtype=np.int64))

    assert index.shape == (0,)


# --- Window contents ---------------------------------------------------------


def test_window_is_the_accounts_own_history_oldest_first() -> None:
    # One account, five transactions, window of three.
    dataset = build([0, 0, 0, 0, 0], sequence_length=3)
    features, lengths, _, row_indices = only_batch(dataset)

    assert row_indices.tolist() == [0, 1, 2, 3, 4]
    assert lengths.tolist() == [1, 2, 3, 3, 3]
    # Column 0 of each feature vector is the row it came from, so the window for
    # row 4 reads: rows 2, 3, 4 - oldest first, target last.
    assert features[4, :, 0].tolist() == [2.0, 3.0, 4.0]
    assert features[3, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_short_history_is_right_padded_with_zeros() -> None:
    dataset = build([0, 0, 0], sequence_length=4)
    features, lengths, _, _ = only_batch(dataset)

    assert lengths.tolist() == [1, 2, 3]
    # The very first transaction has no history: one real element, then padding.
    assert features[0, :, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    # Row 1 has two real elements (rows 0 and 1) at the front, then zeros.
    assert features[1, :, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert features[1, :, 1].tolist() == [0.0, 10.0, 0.0, 0.0]


def test_padding_is_at_the_end_so_the_target_is_the_last_real_element() -> None:
    dataset = build([0, 0], sequence_length=5)
    features, lengths, _, row_indices = only_batch(dataset)

    for sequence, length, target in zip(
        features, lengths.tolist(), row_indices.tolist(), strict=True
    ):
        assert sequence[length - 1, 0] == pytest.approx(float(target))
        assert np.all(sequence[length:] == 0.0)


def test_no_element_of_a_window_comes_from_after_its_target() -> None:
    # Six accounts interleaved, so a naive "previous rows in the file" window
    # would certainly pick up other accounts' later transactions.
    codes = [i % 4 for i in range(24)]
    dataset = build(codes, sequence_length=6)
    features, lengths, _, row_indices = only_batch(dataset)

    for sequence, length, target in zip(
        features, lengths.tolist(), row_indices.tolist(), strict=True
    ):
        rows = sequence[:length, 0].tolist()
        assert all(row <= target for row in rows), (target, rows)
        # Strictly earlier, except for the target itself in the final position.
        assert all(row < target for row in rows[:-1])


def test_every_element_of_a_window_belongs_to_the_target_account() -> None:
    codes = [0, 1, 0, 2, 1, 0, 2, 1, 2, 0]
    dataset = build(codes, sequence_length=4)
    features, lengths, _, row_indices = only_batch(dataset)

    for sequence, length, target in zip(
        features, lengths.tolist(), row_indices.tolist(), strict=True
    ):
        rows = [int(value) for value in sequence[:length, 0].tolist()]
        assert {codes[row] for row in rows} == {codes[target]}


def test_window_is_ordered_oldest_to_newest() -> None:
    dataset = build([0] * 8, sequence_length=4)
    features, lengths, _, _ = only_batch(dataset)

    for sequence, length in zip(features, lengths.tolist(), strict=True):
        rows = sequence[:length, 0].tolist()
        assert rows == sorted(rows)


def test_a_split_cannot_reach_into_another_split() -> None:
    # The validation dataset is built from validation rows alone; the largest row
    # a window can name is therefore its own split's last row.
    dataset = build([0] * 5, sequence_length=10, split=SplitName.VALIDATION)
    features, lengths, _, _ = only_batch(dataset)

    assert dataset.split is SplitName.VALIDATION
    for sequence, length in zip(features, lengths.tolist(), strict=True):
        assert max(sequence[:length, 0].tolist()) < 5.0


# --- Targets, labels and stride ----------------------------------------------


def test_target_is_the_last_transaction_in_the_window() -> None:
    labels = [0.0, 1.0, 0.0, 1.0]
    dataset = build([0, 0, 0, 0], labels=labels, sequence_length=3)
    features, lengths, batch_labels, row_indices = only_batch(dataset)

    assert batch_labels.tolist() == labels
    for sequence, length, target, label in zip(
        features, lengths.tolist(), row_indices.tolist(), batch_labels.tolist(),
        strict=True,
    ):
        assert sequence[length - 1, 0] == pytest.approx(float(target))
        assert label == labels[target]


def test_stride_one_emits_a_sequence_for_every_row() -> None:
    dataset = build([0, 1, 0, 1, 0], stride=1)

    assert dataset.sequence_count == 5
    assert dataset.row_count == 5
    assert dataset.target_rows.tolist() == [0, 1, 2, 3, 4]


def test_stride_greater_than_one_skips_targets() -> None:
    dataset = build([0] * 7, stride=3)

    assert dataset.row_count == 7
    assert dataset.sequence_count == 3
    assert dataset.target_rows.tolist() == [0, 3, 6]


def test_stride_discards_labelled_rows_and_the_counts_show_it() -> None:
    # Every row is positive, so a stride that drops rows visibly drops positives.
    dataset = build([0] * 6, labels=[1.0] * 6, stride=2)

    assert dataset.row_count == 6
    assert dataset.sequence_count == 3
    assert dataset.positive_count == 3


def test_skipped_rows_still_appear_inside_windows() -> None:
    # Stride affects which rows are classified, not which rows are visible: row 1
    # is not a target under stride 2 but is still part of row 2's history.
    dataset = build([0] * 4, stride=2, sequence_length=3)
    features, lengths, _, row_indices = only_batch(dataset)

    assert row_indices.tolist() == [0, 2]
    assert lengths.tolist() == [1, 3]
    assert features[1, :, 0].tolist() == [0.0, 1.0, 2.0]


def test_positive_count_counts_target_labels_only() -> None:
    dataset = build([0] * 5, labels=[1.0, 0.0, 0.0, 1.0, 0.0])

    assert dataset.positive_count == 2
    assert dataset.labels.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_history_lengths_measure_the_data_rather_than_assume_it() -> None:
    dataset = build([0, 1, 0, 0, 1], sequence_length=3)

    # Account 0: rows 0, 2, 3 -> histories 1, 2, 3.
    # Account 1: rows 1, 4    -> histories 1, 2.
    assert dataset.history_lengths().tolist() == [1, 1, 2, 3, 2]


# --- Batching ----------------------------------------------------------------


def test_batches_cover_every_sequence_exactly_once() -> None:
    dataset = build([i % 3 for i in range(10)])

    seen = [
        int(index)
        for batch in dataset.batches(4, shuffle=False)
        for index in batch.row_indices.tolist()
    ]
    assert sorted(seen) == list(range(10))
    assert len(seen) == dataset.sequence_count


def test_final_batch_is_smaller_rather_than_dropped() -> None:
    dataset = build([0] * 10)

    sizes = [len(batch) for batch in dataset.batches(4, shuffle=False)]
    assert sizes == [4, 4, 2]


def test_unshuffled_order_matches_the_labels_property() -> None:
    labels = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    dataset = build([0, 1, 0, 1, 0, 1, 0], labels=labels)

    streamed = [
        value
        for batch in dataset.batches(3, shuffle=False)
        for value in batch.labels.tolist()
    ]
    assert streamed == dataset.labels.tolist()


def test_shuffling_permutes_without_losing_or_duplicating() -> None:
    dataset = build([i % 4 for i in range(20)])

    shuffled = [
        int(index)
        for batch in dataset.batches(6, shuffle=True, seed=11)
        for index in batch.row_indices.tolist()
    ]
    assert sorted(shuffled) == list(range(20))
    assert shuffled != list(range(20))


def test_shuffling_is_reproducible_from_the_seed() -> None:
    dataset = build([i % 4 for i in range(20)])

    def order(seed: int) -> list[int]:
        return [
            int(index)
            for batch in dataset.batches(6, shuffle=True, seed=seed)
            for index in batch.row_indices.tolist()
        ]

    assert order(5) == order(5)
    assert order(5) != order(6)


def test_shuffling_keeps_each_window_matched_to_its_own_label() -> None:
    labels = [float(row % 2) for row in range(12)]
    dataset = build([row % 3 for row in range(12)], labels=labels)

    for batch in dataset.batches(5, shuffle=True, seed=3):
        for position, target in enumerate(batch.row_indices.tolist()):
            assert batch.labels[position] == pytest.approx(labels[target])
            length = int(batch.lengths[position])
            assert batch.features[position, length - 1, 0] == pytest.approx(
                float(target)
            )


def test_non_positive_batch_size_is_refused() -> None:
    dataset = build([0, 0, 0])

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(dataset.batches(0))


def test_batch_tensor_has_the_documented_shape() -> None:
    dataset = build([0] * 7, sequence_length=4, feature_dimension=3)

    batch = next(iter(dataset.batches(5, shuffle=False)))
    assert batch.features.shape == (5, 4, 3)
    assert batch.lengths.shape == (5,)
    assert batch.labels.shape == (5,)
    assert batch.features.dtype == np.float32


def test_every_length_is_at_least_one() -> None:
    # A packed sequence rejects a length of zero, and the target itself always
    # counts, so no window can be empty.
    dataset = build(list(range(15)), sequence_length=5)
    _, lengths, _, _ = only_batch(dataset)

    assert lengths.min() >= 1
    assert lengths.max() <= 5


# --- Construction guards -----------------------------------------------------


def test_mismatched_array_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same rows"):
        AccountSequenceDataset(
            split=SplitName.TRAIN,
            features=np.zeros((4, 2), dtype=np.float32),
            targets=np.zeros(3, dtype=np.float32),
            previous_row=np.full(4, NO_PREVIOUS_ROW, dtype=np.int64),
            sequence_length=3,
            stride=1,
        )


def test_feature_dimension_is_read_from_the_data() -> None:
    dataset = build([0, 0], feature_dimension=9)

    assert dataset.feature_dimension == 9
    assert dataset.sequence_length == 3


def test_an_empty_split_yields_no_batches() -> None:
    dataset = AccountSequenceDataset(
        split=SplitName.TEST,
        features=np.zeros((0, 5), dtype=np.float32),
        targets=np.zeros(0, dtype=np.float32),
        previous_row=np.zeros(0, dtype=np.int64),
        sequence_length=3,
        stride=1,
    )

    assert dataset.sequence_count == 0
    assert dataset.positive_count == 0
    assert list(dataset.batches(4)) == []
