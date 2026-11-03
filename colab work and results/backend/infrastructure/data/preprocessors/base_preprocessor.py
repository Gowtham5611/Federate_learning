"""The contract every feature builder in this package satisfies.

A feature builder takes one chunk and returns it with columns added. It is
deliberately *not* allowed to filter rows, reorder them, or look at more than
the chunk it was given plus explicitly-carried causal state: those restrictions
are what make a chunked pipeline produce the same features a single-pass
in-memory pipeline would.

Two members exist beyond ``apply`` and both earn their place in the final
report. :attr:`produced_columns` is what the metadata artefact lists as the
feature set, so it cannot drift from what the code actually writes. :meth:`describe`
carries the per-feature rationale *and* the features that were considered and
rejected — a rejected feature with a recorded reason is a design decision; a
rejected feature with no record is indistinguishable from an oversight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame


class FeatureBuilder(ABC):
    """Adds a fixed, documented set of columns to a chunk."""

    @property
    @abstractmethod
    def produced_columns(self) -> tuple[str, ...]:
        """Names of the columns :meth:`apply` adds, in output order."""

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must already be present for :meth:`apply` to work.

        Defaults to none, for builders that derive everything from state.
        """
        return ()

    @property
    def rejected_features(self) -> dict[str, str]:
        """Features deliberately not built, mapped to why.

        Defaults to empty. Anything a reader might expect to find and does not
        belongs here rather than nowhere.
        """
        return {}

    @abstractmethod
    def apply(self, chunk: DataFrame) -> DataFrame:
        """Return a new frame: ``chunk`` plus :attr:`produced_columns`.

        Implementations must not mutate ``chunk``, must not drop or reorder its
        rows, and must not use any value from a row that comes later in the
        canonical chronological order.
        """

    def describe(self) -> dict[str, Any]:
        """Metadata for the preprocessing artefacts."""
        return {
            "builder": type(self).__name__,
            "produces": list(self.produced_columns),
            "requires": list(self.required_columns),
            "rejected": dict(self.rejected_features),
        }
