"""Tests for the chronological split: boundaries, ties, and ordering.

The split is the single most consequential piece of this phase. A random split
of transaction data lets a model see the future of the accounts it is scored on,
and the resulting metrics look excellent and mean nothing. So these tests check
the properties directly rather than the arithmetic: rows never move backwards
across a boundary, a timestamp is never divided between two splits, and no split
is left empty.

Every fixture here is synthetic. The real dataset's measured properties — out of
timestamp order, 15,018 distinct timestamps — motivated the design, and the
fixtures reproduce the *shape* of those properties at a size that runs instantly.
"""

from __future__ import annotations

import pytest
from pandas import Series

from domain.enums.split_name import ORDERED_SPLITS, SplitName
from domain.exceptions.preprocessing import (
    ChronologyViolationError,
    InvalidSplitFractionsError,
    NotEnoughDataToSplitError,
)
from domain.value_objects.split_boundaries import TIE_POLICY, SplitBoundaries
from infrastructure.data.splitters.chronological_splitter import (
    MINIMUM_DISTINCT_TIMESTAMPS,
    ChronologicalSplitter,
    TimestampHistogram,
)

pytestmark = pytest.mark.unit


def _histogram(counts: dict[str, int]) -> TimestampHistogram:
    """Build a histogram by feeding it one series per timestamp."""
    histogram = TimestampHistogram()
    for stamp, rows in counts.items():
        histogram.update(Series([stamp] * rows))
    return histogram


def _even_histogram(distinct: int, per_stamp: int = 1) -> TimestampHistogram:
    """``distinct`` timestamps one minute apart, each carrying the same rows."""
    return _histogram(
        {f"2020/01/01 {index // 60:02d}:{index % 60:02d}": per_stamp
         for index in range(distinct)}
    )


# --- TimestampHistogram --------------------------------------------------------


def test_the_histogram_accumulates_across_chunks() -> None:
    """It is filled one chunk at a time, so partial updates must add up."""
    histogram = TimestampHistogram()
    histogram.update(Series(["2020/01/01 00:00", "2020/01/01 00:01"]))
    histogram.update(Series(["2020/01/01 00:00"]))

    assert histogram.total_rows == 3
    assert histogram.distinct_timestamps == 2
    assert histogram.rows_at("2020/01/01 00:00") == 2


def test_the_histogram_size_tracks_distinct_timestamps_not_rows() -> None:
    """This is the property that keeps the split memory-bounded.

    Ten thousand rows spanning five timestamps must cost five entries. If this
    ever became one entry per row the histogram would be as large as the file it
    was meant to summarise.
    """
    histogram = _histogram({f"2020/01/01 00:0{index}": 2_000 for index in range(5)})

    assert histogram.total_rows == 10_000
    assert histogram.distinct_timestamps == 5
    assert len(histogram.counts()) == 5


def test_timestamps_come_back_in_chronological_order_however_they_arrived() -> None:
    """The raw file is not time-ordered, so the histogram must impose the order."""
    histogram = TimestampHistogram()
    histogram.update(Series(["2020/01/03 09:00", "2020/01/01 23:59", "2020/01/02 00:00"]))

    assert histogram.sorted_timestamps() == [
        "2020/01/01 23:59",
        "2020/01/02 00:00",
        "2020/01/03 09:00",
    ]


def test_an_unseen_timestamp_reports_zero_rather_than_failing() -> None:
    assert TimestampHistogram().rows_at("2020/01/01 00:00") == 0


def test_an_empty_chunk_does_not_disturb_the_histogram() -> None:
    histogram = TimestampHistogram()
    histogram.update(Series(["2020/01/01 00:00"]))
    histogram.update(Series([], dtype="object"))

    assert histogram.total_rows == 1


# --- Requested fractions -------------------------------------------------------


def test_the_default_split_is_the_requested_sixty_twenty_twenty() -> None:
    assert ChronologicalSplitter().fractions == pytest.approx((0.60, 0.20, 0.20))


@pytest.mark.parametrize(
    ("train", "validation"),
    [
        (0.9, 0.1),  # leaves no test split
        (0.0, 0.5),  # empty train
        (0.5, 0.0),  # empty validation
        (0.8, 0.4),  # sums past one
        (-0.2, 0.5),  # negative
    ],
)
def test_fractions_that_cannot_leave_three_splits_are_refused(
    train: float, validation: float
) -> None:
    """Refused at construction, before any data has been read."""
    with pytest.raises(InvalidSplitFractionsError):
        ChronologicalSplitter(train_fraction=train, validation_fraction=validation)


# --- Computing boundaries ------------------------------------------------------


def test_an_even_dataset_splits_at_the_requested_proportions() -> None:
    """Ten equally sized timestamps make the intended cut visible by hand."""
    boundaries = ChronologicalSplitter().compute_boundaries(_even_histogram(10))

    # Six of ten rows at or before the train cut, eight at or before validation.
    assert boundaries.train_last == "2020/01/01 00:05"
    assert boundaries.validation_last == "2020/01/01 00:07"


def test_the_boundaries_are_derived_from_row_counts_not_timestamp_counts() -> None:
    """A heavy early timestamp must pull the cut earlier.

    The cut is not 60% of the distinct values: splitting on the midpoint of the
    *timestamps* would put 90 of 130 rows in training while claiming 60%.
    """
    histogram = _histogram(
        {
            "2020/01/01 00:00": 60,
            "2020/01/01 00:01": 10,
            "2020/01/01 00:02": 10,
            "2020/01/01 00:03": 10,
            "2020/01/01 00:04": 10,
        }
    )

    boundaries = ChronologicalSplitter().compute_boundaries(histogram)

    assert boundaries.train_last == "2020/01/01 00:00"


def test_every_split_keeps_at_least_one_timestamp_at_the_smallest_size() -> None:
    """Three timestamps is the documented minimum; it must actually work."""
    boundaries = ChronologicalSplitter().compute_boundaries(_even_histogram(3))

    assert boundaries.train_last == "2020/01/01 00:00"
    assert boundaries.validation_last == "2020/01/01 00:01"


def test_a_dominant_first_timestamp_still_leaves_room_for_the_later_splits() -> None:
    """Clamping earns its place here.

    The 60% target is met by the first timestamp alone, and an unclamped cut
    would swallow the whole dataset.
    """
    histogram = _histogram(
        {
            "2020/01/01 00:00": 900,
            "2020/01/01 00:01": 1,
            "2020/01/01 00:02": 1,
        }
    )

    boundaries = ChronologicalSplitter().compute_boundaries(histogram)

    assert boundaries.train_last == "2020/01/01 00:00"
    assert boundaries.validation_last == "2020/01/01 00:01"


def test_fewer_than_three_distinct_timestamps_is_refused_not_approximated() -> None:
    """Two timestamps cannot make three chronological splits, at any size."""
    histogram = _histogram({"2020/01/01 00:00": 500, "2020/01/01 00:01": 500})

    with pytest.raises(NotEnoughDataToSplitError) as raised:
        ChronologicalSplitter().compute_boundaries(histogram)

    assert raised.value.distinct_timestamps == 2
    assert raised.value.row_count == 1_000


def test_a_single_timestamp_however_many_rows_is_refused() -> None:
    with pytest.raises(NotEnoughDataToSplitError):
        ChronologicalSplitter().compute_boundaries(
            _histogram({"2020/01/01 00:00": 10_000})
        )


def test_an_empty_histogram_is_refused() -> None:
    with pytest.raises(NotEnoughDataToSplitError):
        ChronologicalSplitter().compute_boundaries(TimestampHistogram())


def test_the_documented_minimum_matches_the_behaviour() -> None:
    assert MINIMUM_DISTINCT_TIMESTAMPS == 3
    ChronologicalSplitter().compute_boundaries(
        _even_histogram(MINIMUM_DISTINCT_TIMESTAMPS)
    )


def test_computing_boundaries_twice_gives_the_same_answer() -> None:
    """Reproducibility starts here: nothing about the cut may be run-dependent."""
    splitter = ChronologicalSplitter()
    histogram = _even_histogram(50, per_stamp=3)

    assert splitter.compute_boundaries(histogram) == splitter.compute_boundaries(
        histogram
    )


def test_a_different_arrival_order_produces_identical_boundaries() -> None:
    """The real file arrives shuffled; the cut must not depend on that."""
    forward = TimestampHistogram()
    backward = TimestampHistogram()
    stamps = [f"2020/01/01 00:{index:02d}" for index in range(12)]
    forward.update(Series(stamps))
    backward.update(Series(list(reversed(stamps))))

    splitter = ChronologicalSplitter()

    assert splitter.compute_boundaries(forward) == splitter.compute_boundaries(backward)


# --- Assigning rows ------------------------------------------------------------


def _assign(stamps: list[str]) -> list[SplitName]:
    boundaries = SplitBoundaries(
        train_last="2020/01/01 00:05", validation_last="2020/01/01 00:07"
    )
    return list(ChronologicalSplitter.assign(Series(stamps), boundaries))


def test_rows_land_in_the_split_their_timestamp_selects() -> None:
    assigned = _assign(
        [
            "2020/01/01 00:00",
            "2020/01/01 00:05",
            "2020/01/01 00:06",
            "2020/01/01 00:07",
            "2020/01/01 00:08",
        ]
    )

    assert assigned == [
        SplitName.TRAIN,
        SplitName.TRAIN,
        SplitName.VALIDATION,
        SplitName.VALIDATION,
        SplitName.TEST,
    ]


def test_the_boundary_timestamp_belongs_to_the_earlier_split() -> None:
    """The documented tie policy, checked at both cuts."""
    assert TIE_POLICY == "whole_timestamp_to_earlier_split"
    assert _assign(["2020/01/01 00:05"]) == [SplitName.TRAIN]
    assert _assign(["2020/01/01 00:07"]) == [SplitName.VALIDATION]


def test_tied_rows_are_never_divided_between_two_splits() -> None:
    """The reason ties go whole: the same instant in two splits is a leak.

    Transactions sharing a timestamp are frequently the same laundering pattern.
    Splitting them would put one leg in training and the other in test.
    """
    assigned = _assign(["2020/01/01 00:05"] * 25)

    assert set(assigned) == {SplitName.TRAIN}


def test_assignment_does_not_depend_on_row_order() -> None:
    """Assignment reads the timestamp, never the position in the chunk."""
    stamps = ["2020/01/01 00:09", "2020/01/01 00:00", "2020/01/01 00:06"]

    assert _assign(stamps) == [SplitName.TEST, SplitName.TRAIN, SplitName.VALIDATION]
    assert _assign(list(reversed(stamps))) == [
        SplitName.VALIDATION,
        SplitName.TRAIN,
        SplitName.TEST,
    ]


def test_assignment_preserves_the_index_so_rows_can_be_selected_by_it() -> None:
    """The service uses ``engineered[labels == split]``; a reset index misaligns."""
    stamps = Series(
        ["2020/01/01 00:00", "2020/01/01 00:09"], index=[17, 42]
    )
    boundaries = SplitBoundaries(
        train_last="2020/01/01 00:05", validation_last="2020/01/01 00:07"
    )

    labels = ChronologicalSplitter.assign(stamps, boundaries)

    assert list(labels.index) == [17, 42]


def test_an_empty_chunk_assigns_to_nothing() -> None:
    assert _assign([]) == []


def test_every_row_reaches_exactly_one_split() -> None:
    """No row may be dropped by falling through the boundary comparisons."""
    stamps = [f"2020/01/01 00:{index:02d}" for index in range(12)]

    assigned = _assign(stamps)

    assert len(assigned) == len(stamps)
    assert set(assigned) <= set(ORDERED_SPLITS)


def test_computed_boundaries_and_assignment_agree_on_the_proportions() -> None:
    """End to end for the split alone: what was requested is what rows receive."""
    stamps = [f"2020/01/01 {index // 60:02d}:{index % 60:02d}" for index in range(100)]
    histogram = TimestampHistogram()
    histogram.update(Series(stamps))

    boundaries = ChronologicalSplitter().compute_boundaries(histogram)
    assigned = list(ChronologicalSplitter.assign(Series(stamps), boundaries))

    assert assigned.count(SplitName.TRAIN) == 60
    assert assigned.count(SplitName.VALIDATION) == 20
    assert assigned.count(SplitName.TEST) == 20


def test_the_assigned_splits_never_overlap_in_time() -> None:
    """The invariant the whole design exists for, checked on assigned rows.

    Stated as ``max(train) <= min(validation)`` and ``max(validation) <=
    min(test)`` rather than by comparing to the boundaries, so that a boundary
    that was itself wrong cannot make this pass.
    """
    stamps = [f"2020/01/01 {index // 60:02d}:{index % 60:02d}" for index in range(200)]
    histogram = TimestampHistogram()
    histogram.update(Series(stamps))
    boundaries = ChronologicalSplitter().compute_boundaries(histogram)
    labels = ChronologicalSplitter.assign(Series(stamps), boundaries)
    series = Series(stamps)

    windows = {
        split: (series[labels == split].min(), series[labels == split].max())
        for split in ORDERED_SPLITS
    }

    assert windows[SplitName.TRAIN][1] <= windows[SplitName.VALIDATION][0]
    assert windows[SplitName.VALIDATION][1] <= windows[SplitName.TEST][0]


# --- SplitBoundaries -----------------------------------------------------------


def test_boundaries_that_run_backwards_are_refused_at_construction() -> None:
    """An invalid cut cannot be carried far enough to produce output."""
    with pytest.raises(ChronologyViolationError):
        SplitBoundaries(
            train_last="2020/01/09 00:00", validation_last="2020/01/02 00:00"
        )


def test_boundaries_name_the_split_a_timestamp_belongs_to() -> None:
    boundaries = SplitBoundaries(
        train_last="2020/01/05 00:00", validation_last="2020/01/07 00:00"
    )

    assert boundaries.split_for("2020/01/01 00:00") is SplitName.TRAIN
    assert boundaries.split_for("2020/01/05 00:00") is SplitName.TRAIN
    assert boundaries.split_for("2020/01/06 00:00") is SplitName.VALIDATION
    assert boundaries.split_for("2020/01/07 00:00") is SplitName.VALIDATION
    assert boundaries.split_for("2020/01/08 00:00") is SplitName.TEST


def test_the_tie_policy_travels_with_the_boundaries() -> None:
    """It has to reach the metadata; a policy only in a docstring is unusable."""
    boundaries = SplitBoundaries(
        train_last="2020/01/05 00:00", validation_last="2020/01/07 00:00"
    )

    assert boundaries.tie_policy == TIE_POLICY
