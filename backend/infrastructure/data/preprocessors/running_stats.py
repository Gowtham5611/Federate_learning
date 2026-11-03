"""Streaming mean and standard deviation, fitted one chunk at a time.

Standardisation needs the mean and standard deviation of each column *before* it
can transform anything, which normally means holding the column in memory. This
accumulator does not: it merges each chunk's statistics into a running total
using Chan's parallel form of Welford's algorithm, so a 3-million-row column is
summarised from state proportional to the number of *columns*.

The naive alternative — accumulating :math:`\\sum x` and :math:`\\sum x^2` — is
smaller still but loses precision catastrophically when the mean is large
relative to the spread, which is exactly the regime of log-transformed monetary
amounts. Welford's form is used because the numbers here are financial.

The class enforces the project's leakage rule structurally: it accepts chunks
only until :meth:`freeze` is called, and it transforms only after. There is no
state in which it both learns and applies, so "fitted on train only" cannot be
violated by calling the methods in the wrong order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from domain.exceptions.preprocessing import (
    TransformerAlreadyFittedError,
    TransformerNotFittedError,
)

_NAME = "RunningStandardizer"

MINIMUM_STANDARD_DEVIATION = 1e-12
"""Below this, a column is treated as constant and left un-scaled.

Dividing by a near-zero standard deviation turns floating-point noise into
enormous feature values. A column with no variance in training carries no
information, so it is centred and left at zero rather than amplified.
"""


class RunningStandardizer:
    """Accumulates per-column mean and variance across chunks, then standardises.

    Attributes are exposed through :meth:`parameters` rather than as fields so
    that the fitted state has exactly one serialisation, which is what gets
    written to the metadata artefacts and re-read to reproduce a run.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        """Prepare an unfitted accumulator for a fixed set of columns.

        Args:
            columns: Columns to standardise, in output order.
        """
        self._columns = tuple(columns)
        self._count = np.zeros(len(self._columns), dtype=np.float64)
        self._mean = np.zeros(len(self._columns), dtype=np.float64)
        self._m2 = np.zeros(len(self._columns), dtype=np.float64)
        self._frozen = False

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns this accumulator covers, in order."""
        return self._columns

    @property
    def is_fitted(self) -> bool:
        """Whether fitting has been closed off and transforming is allowed."""
        return self._frozen

    def update(self, chunk: DataFrame) -> None:
        """Fold one chunk's statistics into the running totals.

        Only the training split may be passed here. Rows whose value is missing
        are skipped per column, so a column's count can differ from the chunk's
        row count.

        Args:
            chunk: Frame containing this accumulator's columns.

        Raises:
            TransformerAlreadyFittedError: :meth:`freeze` was already called.
            KeyError: A configured column is absent from the chunk.
        """
        if self._frozen:
            raise TransformerAlreadyFittedError(_NAME)

        for index, column in enumerate(self._columns):
            values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            batch_count = float(values.size)
            if batch_count == 0.0:
                continue

            batch_mean = float(values.mean())
            # ddof=0: this is the second central moment of the batch, which is
            # what Chan's merge expects, not a sample variance estimate.
            batch_m2 = float(((values - batch_mean) ** 2).sum())

            existing = self._count[index]
            total = existing + batch_count
            delta = batch_mean - self._mean[index]

            self._mean[index] += delta * (batch_count / total)
            self._m2[index] += batch_m2 + delta * delta * existing * batch_count / total
            self._count[index] = total

    def freeze(self) -> None:
        """Close fitting. After this the accumulator only transforms.

        Idempotent, so a caller that is not sure whether fitting finished can
        call it safely.
        """
        self._frozen = True

    def parameters(self) -> dict[str, dict[str, float]]:
        """The fitted mean and standard deviation per column.

        Returns:
            Column name to ``{"count", "mean", "std"}``.

        Raises:
            TransformerNotFittedError: Fitting has not been closed.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME, "report its parameters")
        return {
            column: {
                "count": float(self._count[index]),
                "mean": float(self._mean[index]),
                "std": float(self._standard_deviation(index)),
            }
            for index, column in enumerate(self._columns)
        }

    def transform(self, chunk: DataFrame) -> DataFrame:
        """Standardise this accumulator's columns in place on a copy.

        Args:
            chunk: Frame to transform. Not modified.

        Returns:
            A new frame with the configured columns centred and scaled.

        Raises:
            TransformerNotFittedError: Fitting has not been closed.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME)

        result = chunk.copy()
        for index, column in enumerate(self._columns):
            deviation = self._standard_deviation(index)
            values = pd.to_numeric(result[column], errors="coerce").astype("float64")
            centred = values - self._mean[index]
            result[column] = (
                centred if deviation < MINIMUM_STANDARD_DEVIATION else centred / deviation
            )
        return result

    @classmethod
    def from_parameters(cls, parameters: dict[str, dict[str, float]]) -> RunningStandardizer:
        """Rebuild a fitted accumulator from serialised parameters.

        Reproducing a run must not require re-reading the training data, so a
        standardiser can be restored from the metadata artefact alone.

        Args:
            parameters: As produced by :meth:`parameters`.

        Returns:
            A frozen accumulator that transforms identically to the original.
        """
        instance = cls(tuple(parameters))
        for index, column in enumerate(instance._columns):
            entry = parameters[column]
            instance._count[index] = entry.get("count", 0.0)
            instance._mean[index] = entry["mean"]
            standard_deviation = entry["std"]
            count = max(instance._count[index], 1.0)
            instance._m2[index] = standard_deviation * standard_deviation * count
        instance._frozen = True
        return instance

    def _standard_deviation(self, index: int) -> float:
        """Population standard deviation for one column.

        Population rather than sample: the training split is not a sample of a
        larger population here, it *is* the data the model sees, and the row
        counts are large enough that the distinction is numerically irrelevant
        while the definition needs to be pinned down for reproducibility.
        """
        count = self._count[index]
        if count <= 0.0:
            return 0.0
        return float(np.sqrt(self._m2[index] / count))

    def __repr__(self) -> str:
        """Show fitted state and column count, never the values themselves."""
        state = "fitted" if self._frozen else "unfitted"
        return f"{_NAME}({len(self._columns)} column(s), {state})"


def describe_log_transform(columns: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Record which columns received ``log1p`` and why.

    ``log1p`` rather than ``log``: the verified data has no non-positive amount,
    but derived counts and elapsed times legitimately reach zero, and
    ``log1p(0) == 0`` keeps those defined without inventing an epsilon.

    Args:
        columns: Columns the pipeline log-transforms before standardising.

    Returns:
        Per-column transformation metadata for the artefacts.
    """
    return {
        column: {
            "transform": "log1p",
            "reason": (
                "right-skewed non-negative quantity; log1p compresses the tail "
                "without discarding the large values that matter in AML"
            ),
            "applied_before": "standardization",
        }
        for column in columns
    }
