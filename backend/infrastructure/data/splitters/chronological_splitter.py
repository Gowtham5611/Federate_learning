"""Chronological 60/20/20 splitting, computed from a timestamp histogram.

The obvious implementation — sort the whole file, take the first 60% of rows —
is unavailable here for two independent reasons, and both shaped this design.

**The file is not in time order.** Measured over the real dataset: reading it
top to bottom does *not* produce ascending timestamps. A positional split would
therefore not be a chronological split at all; it would scatter every period
across all three sets while looking perfectly reasonable in the code.

**Sorting five million rows in memory is not an option.** But the split does not
need the rows — it needs to know how many rows carry each distinct timestamp.
That histogram is small: the real file has 15,018 distinct timestamps, so the
counts fit in well under a megabyte and can be accumulated in a single chunked
pass. The boundaries are then found by walking the histogram in sorted order.

The result is an *exact* chronological split, not an approximation of one, at a
cost that does not grow with the number of rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from pandas import Series

from domain.enums.split_name import SplitName
from domain.exceptions.preprocessing import (
    InvalidSplitFractionsError,
    NotEnoughDataToSplitError,
)
from domain.value_objects.split_boundaries import SplitBoundaries

MINIMUM_DISTINCT_TIMESTAMPS = 3
"""One cut per boundary needs a distinct timestamp on each side of both cuts."""


class TimestampHistogram:
    """Counts rows per distinct timestamp, accumulated across chunks.

    Deliberately not a plain ``Counter`` on the caller's side: the class fixes
    the string form of the key, so the counts, the boundaries and the assignment
    all compare timestamps the same way.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._counts: Counter[str] = Counter()

    def update(self, timestamps: Series) -> None:
        """Fold one chunk's timestamps into the histogram."""
        counts = timestamps.astype(str).value_counts()
        self._counts.update({str(value): int(count) for value, count in counts.items()})

    @property
    def total_rows(self) -> int:
        """Rows counted so far."""
        return int(sum(self._counts.values()))

    @property
    def distinct_timestamps(self) -> int:
        """Distinct timestamps seen so far. This is what bounds memory."""
        return len(self._counts)

    def counts(self) -> Mapping[str, int]:
        """The raw histogram, for serialisation or inspection."""
        return dict(self._counts)

    def sorted_timestamps(self) -> list[str]:
        """Distinct timestamps in chronological (== lexicographic) order."""
        return sorted(self._counts)

    def rows_at(self, timestamp: str) -> int:
        """How many rows carry ``timestamp``."""
        return self._counts.get(timestamp, 0)


class ChronologicalSplitter:
    """Finds split boundaries from a histogram and assigns rows to splits."""

    def __init__(self, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> None:
        """Configure the requested proportions.

        Args:
            train_fraction: Target share of rows for training.
            validation_fraction: Target share for validation. The remainder
                becomes the test split.

        Raises:
            InvalidSplitFractionsError: The proportions cannot leave three
                non-empty splits.
        """
        if (
            train_fraction <= 0.0
            or validation_fraction <= 0.0
            or train_fraction + validation_fraction >= 1.0
        ):
            raise InvalidSplitFractionsError(train_fraction, validation_fraction)
        self._train_fraction = train_fraction
        self._validation_fraction = validation_fraction

    @property
    def fractions(self) -> tuple[float, float, float]:
        """Requested train, validation and test shares."""
        test = 1.0 - self._train_fraction - self._validation_fraction
        return self._train_fraction, self._validation_fraction, test

    def compute_boundaries(self, histogram: TimestampHistogram) -> SplitBoundaries:
        """Find the two timestamps that realise the requested proportions.

        Walks the sorted histogram accumulating rows, and cuts at the first
        timestamp whose cumulative total reaches each target. Because a whole
        timestamp is indivisible, the cut lands on or just past the target, and
        the realised proportions are reported rather than assumed.

        The chosen indices are then clamped so that each split keeps at least
        one distinct timestamp. Clamping matters only for small inputs; on the
        real dataset the targets fall far from either end.

        Args:
            histogram: Row counts per distinct timestamp, over the whole file.

        Returns:
            The two boundaries.

        Raises:
            NotEnoughDataToSplitError: Fewer than three distinct timestamps, or
                fewer than three rows.
        """
        ordered = histogram.sorted_timestamps()
        total = histogram.total_rows
        if len(ordered) < MINIMUM_DISTINCT_TIMESTAMPS or total < MINIMUM_DISTINCT_TIMESTAMPS:
            raise NotEnoughDataToSplitError(total, len(ordered))

        cumulative = list(_running_total(histogram.rows_at(stamp) for stamp in ordered))
        last_index = len(ordered) - 1

        train_index = _first_reaching(cumulative, total * self._train_fraction)
        train_index = min(train_index, last_index - 2)

        validation_target = total * (self._train_fraction + self._validation_fraction)
        validation_index = max(
            _first_reaching(cumulative, validation_target), train_index + 1
        )
        validation_index = min(validation_index, last_index - 1)

        return SplitBoundaries(
            train_last=ordered[train_index],
            validation_last=ordered[validation_index],
        )

    @staticmethod
    def assign(timestamps: Series, boundaries: SplitBoundaries) -> Series:
        """Label each row with the split it belongs to.

        Vectorised over the chunk, and comparing text directly: the dataset's
        timestamp format is fixed-width and zero-padded, so string comparison
        and chronological comparison coincide.

        Args:
            timestamps: Raw timestamp column.
            boundaries: The computed cuts.

        Returns:
            A series of :class:`~domain.enums.split_name.SplitName` values.
        """
        text = timestamps.astype(str)
        labels = Series(SplitName.TEST, index=text.index, dtype="object")
        labels[text <= boundaries.validation_last] = SplitName.VALIDATION
        labels[text <= boundaries.train_last] = SplitName.TRAIN
        return labels


def _running_total(values: Iterable[int]) -> Iterable[int]:
    """Yield the cumulative sum of ``values``."""
    total = 0
    for value in values:
        total += value
        yield total


def _first_reaching(cumulative: list[int], target: float) -> int:
    """Index of the first cumulative total that reaches ``target``.

    Returns the last index if nothing reaches it, which the caller clamps.
    """
    for index, total in enumerate(cumulative):
        if total >= target:
            return index
    return len(cumulative) - 1
