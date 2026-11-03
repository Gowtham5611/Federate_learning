"""Time-of-day and day-of-week features, encoded so that midnight is near 23:59.

The verified ``Timestamp`` column has minute resolution and spans 2022-09-01 to
2022-09-18 — eighteen days. That span is what governs which temporal features
are safe to build.

**Cyclical encoding.** Hour 23 and hour 0 are one hour apart, but as raw
integers they are the two extremes of the range. Every periodic quantity here is
therefore emitted as a sine/cosine pair, which puts adjacent times adjacent in
feature space and needs no scaling: the pairs are already in ``[-1, 1]``.

**What is deliberately absent.** A "days since the dataset began" feature is not
built, and the chronological split is the reason. Training covers days 0-5 and
test covers days 7-17, so the feature's *test values never occur in training*.
A model would be free to learn a trend over an index it can only extrapolate,
and every reported test metric would then be measuring extrapolation rather than
detection. See :attr:`TemporalFeatureBuilder.rejected_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from infrastructure.data.preprocessors.base_preprocessor import FeatureBuilder

MINUTES_PER_DAY = 24 * 60
DAYS_PER_WEEK = 7
NIGHT_ENDS_AT_HOUR = 6
"""Hours before this count as night. A convention, recorded so it is not silent."""

TIMESTAMP_COLUMN = "Timestamp"
PARSED_TIMESTAMP_COLUMN = "_parsed_timestamp"
"""Internal column holding the parsed datetime; never reaches the output."""

EPOCH_SECONDS_COLUMN = "_epoch_seconds"
"""Internal column holding integer seconds; consumed by the history builder."""


def parse_timestamps(values: Series, timestamp_format: str) -> Series:
    """Parse the raw timestamp column, marking unparseable values as missing.

    ``errors="coerce"`` rather than raising: an unparseable timestamp is a data
    problem to be counted in the cleaning ledger, not an exception that aborts a
    five-million-row run. The caller decides what to do with the resulting
    missing values.

    Args:
        values: Raw timestamp strings.
        timestamp_format: Expected ``strftime`` format. Passing it explicitly
            avoids pandas inferring a different format per chunk, which would
            make chunked parsing disagree with whole-file parsing.

    Returns:
        Parsed datetimes, ``NaT`` where parsing failed.
    """
    return pd.to_datetime(values, format=timestamp_format, errors="coerce")


class TemporalFeatureBuilder(FeatureBuilder):
    """Builds cyclical time-of-day and day-of-week features.

    Requires the parsed timestamp column produced by :func:`parse_timestamps`,
    so parsing happens once per chunk rather than once per feature.
    """

    @property
    def produced_columns(self) -> tuple[str, ...]:
        """The six temporal features, in output order."""
        return (
            "minute_of_day_sin",
            "minute_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_weekend",
            "is_night",
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The parsed timestamp column."""
        return (PARSED_TIMESTAMP_COLUMN,)

    @property
    def rejected_features(self) -> dict[str, str]:
        """Temporal features considered and not built, with the reason."""
        return {
            "days_since_dataset_start": (
                "A monotonically increasing index whose training range (days "
                "0-5) and test range (days 7-17) are disjoint under the "
                "chronological split. A model could fit a trend it can only "
                "extrapolate at test time, so every test metric would measure "
                "extrapolation instead of detection."
            ),
            "calendar_month": (
                "The verified data spans a single month (September 2022), so "
                "the feature is constant and carries no information."
            ),
            "time_to_next_transaction": (
                "Requires a later row to compute a value for an earlier one. "
                "That is future information by construction and is exactly the "
                "leakage this pipeline exists to prevent."
            ),
        }

    def apply(self, chunk: DataFrame) -> DataFrame:
        """Add the temporal features to a copy of ``chunk``."""
        result = chunk.copy()
        parsed = result[PARSED_TIMESTAMP_COLUMN]

        minute_of_day = parsed.dt.hour * 60 + parsed.dt.minute
        day_of_week = parsed.dt.dayofweek

        sin_minute, cos_minute = _cyclical(minute_of_day, MINUTES_PER_DAY)
        sin_day, cos_day = _cyclical(day_of_week, DAYS_PER_WEEK)

        result["minute_of_day_sin"] = sin_minute
        result["minute_of_day_cos"] = cos_minute
        result["day_of_week_sin"] = sin_day
        result["day_of_week_cos"] = cos_day
        # Saturday is 5 and Sunday is 6 in pandas' convention.
        result["is_weekend"] = (day_of_week >= 5).astype("int8")
        result["is_night"] = (parsed.dt.hour < NIGHT_ENDS_AT_HOUR).astype("int8")
        return result


def _cyclical(values: Series, period: int) -> tuple[Series, Series]:
    """Map a periodic integer onto the unit circle.

    Args:
        values: Periodic quantity, e.g. minute of day.
        period: Number of distinct positions in one cycle.

    Returns:
        The sine and cosine components, as ``float32``.
    """
    angle = 2.0 * np.pi * values.astype("float64") / period
    return (
        Series(np.sin(angle), index=values.index).astype("float32"),
        Series(np.cos(angle), index=values.index).astype("float32"),
    )
