"""The three dataset splits, and the direction of time between them."""

from __future__ import annotations

from enum import StrEnum


class SplitName(StrEnum):
    """A chronological partition of the transaction stream.

    The order of the members is the order of time: every ``TRAIN`` transaction
    happens no later than every ``VALIDATION`` transaction, which happens no
    later than every ``TEST`` transaction. That ordering is the whole point of
    the split, so it is encoded here rather than left to each caller's
    convention.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @property
    def position(self) -> int:
        """Zero-based position in time. ``TRAIN`` is first."""
        return _ORDER[self]

    @property
    def is_fitting_split(self) -> bool:
        """Whether preprocessing statistics may be learned from this split.

        Only ``TRAIN``. Every category mapping, every mean and every standard
        deviation is fitted here and merely *applied* elsewhere; a helper that
        answers this question makes the rule checkable in one place instead of
        being restated at each call site.
        """
        return self is SplitName.TRAIN


_ORDER: dict[SplitName, int] = {
    SplitName.TRAIN: 0,
    SplitName.VALIDATION: 1,
    SplitName.TEST: 2,
}

ORDERED_SPLITS: tuple[SplitName, ...] = (
    SplitName.TRAIN,
    SplitName.VALIDATION,
    SplitName.TEST,
)
"""The splits in chronological order, for iteration."""
