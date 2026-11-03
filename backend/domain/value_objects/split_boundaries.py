"""The two timestamps that separate train from validation from test.

Stored as the *last* timestamp of the earlier split rather than the first of the
later one, because that is the form the assignment rule uses directly::

    timestamp <= train_last                      -> train
    train_last < timestamp <= validation_last    -> validation
    timestamp > validation_last                  -> test

Timestamps are kept as text in the dataset's own fixed-width format
(``YYYY/MM/DD HH:MM``), where lexicographic order *is* chronological order. That
keeps the boundary comparable without parsing and makes it identical in the
metadata artefact, in the log and in the code.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums.split_name import SplitName
from domain.exceptions.preprocessing import ChronologyViolationError

TIE_POLICY = "whole_timestamp_to_earlier_split"
"""How rows sharing a boundary timestamp are assigned.

The data has minute resolution, so many rows share a timestamp and the split
cannot fall between two of them without splitting a single instant across two
sets. Every row carrying the boundary timestamp therefore goes to the *earlier*
split. The consequence is that the realised proportions differ slightly from the
requested ones; the alternative — cutting mid-timestamp — would put transactions
from the same minute on both sides of a boundary that is supposed to represent
"everything the model had already seen".
"""


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Where the chronological cuts fall.

    Attributes:
        train_last: Last timestamp belonging to the training split.
        validation_last: Last timestamp belonging to the validation split.
    """

    train_last: str
    validation_last: str

    def __post_init__(self) -> None:
        """Reject boundaries that do not advance in time.

        Raises:
            ChronologyViolationError: The validation boundary is not strictly
                after the training boundary, which would make the validation
                split empty or overlapping.
        """
        if self.validation_last <= self.train_last:
            raise ChronologyViolationError(
                SplitName.TRAIN.value,
                SplitName.VALIDATION.value,
                (
                    f"the training boundary '{self.train_last}' is not before "
                    f"the validation boundary '{self.validation_last}'"
                ),
            )

    def split_for(self, timestamp: str) -> SplitName:
        """Which split a single timestamp belongs to."""
        if timestamp <= self.train_last:
            return SplitName.TRAIN
        if timestamp <= self.validation_last:
            return SplitName.VALIDATION
        return SplitName.TEST

    @property
    def tie_policy(self) -> str:
        """The documented rule for rows sharing a boundary timestamp."""
        return TIE_POLICY
