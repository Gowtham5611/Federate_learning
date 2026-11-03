"""Deterministic ordinal encoding for low-cardinality categorical columns.

Three of the verified columns are closed vocabularies — ``Receiving Currency``
and ``Payment Currency`` (15 values each) and ``Payment Format`` (7). Those are
encoded here. The account and bank identifier columns are **not**, and that
refusal is part of the design rather than an omission; see
:data:`HIGH_CARDINALITY_REFUSALS` for the counts that decided it.

Two properties matter more than the encoding scheme itself:

*Determinism.* Codes come from the **sorted** vocabulary, not from order of
first appearance. Two runs over the same training rows therefore agree even if
the rows arrive in a different chunk order, which order-of-appearance encoding
does not guarantee.

*Fitted on train only.* The vocabulary is learned from training chunks and then
frozen. A value that appears for the first time in validation or test is mapped
to :data:`UNKNOWN_CODE` rather than silently extending the vocabulary — the
extension would be a leak, and dropping the row would be worse, because an
unfamiliar currency is exactly the kind of thing an AML model should still see.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from domain.exceptions.preprocessing import (
    TransformerAlreadyFittedError,
    TransformerNotFittedError,
)

_NAME = "OrdinalCategoricalEncoder"

UNKNOWN_CODE = -1
"""Code for a value not present in the training vocabulary.

Negative so it can never collide with a real code, and constant so a downstream
model can learn a single "unseen category" behaviour instead of one per column.
"""

HIGH_CARDINALITY_REFUSALS: dict[str, str] = {
    "Account": (
        "515,080 distinct sending/receiving account identifiers were measured "
        "across the file. One-hot encoding would produce a matrix wider than "
        "the dataset is long, and ordinal codes would impose a false ordering "
        "on identifiers that have none. Account identity is used only to "
        "accumulate causal per-account history features, never as a model "
        "input column."
    ),
    "Account.1": (
        "Same identifier space as 'Account'; see that entry. Encoded nowhere, "
        "used only for receiver-side history."
    ),
    "From Bank": (
        "Bank identifiers range to 356,303. An integer bank id carries no "
        "magnitude - bank 300,000 is not larger than bank 3 - so it is neither "
        "scaled nor encoded. The relationship that does carry signal, whether "
        "sender and receiver share a bank, is kept as a boolean feature."
    ),
    "To Bank": (
        "Same identifier space as 'From Bank'; see that entry."
    ),
}
"""Columns deliberately left unencoded, each with the measured reason."""


class OrdinalCategoricalEncoder:
    """Learns a sorted vocabulary per column from training chunks, then maps.

    Like :class:`~infrastructure.data.preprocessors.running_stats.RunningStandardizer`,
    this is either learning or applying and never both: :meth:`update` raises
    once :meth:`freeze` has been called, and :meth:`transform` raises before it.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        """Prepare an unfitted encoder for a fixed set of columns.

        Args:
            columns: Columns to encode, in output order.
        """
        self._columns = tuple(columns)
        self._seen: dict[str, set[str]] = {column: set() for column in self._columns}
        self._mapping: dict[str, dict[str, int]] = {}
        self._frozen = False

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns this encoder covers, in order."""
        return self._columns

    @property
    def is_fitted(self) -> bool:
        """Whether the vocabulary is closed and mapping is allowed."""
        return self._frozen

    def update(self, chunk: DataFrame) -> None:
        """Add one training chunk's distinct values to the vocabulary.

        Args:
            chunk: Frame containing this encoder's columns.

        Raises:
            TransformerAlreadyFittedError: :meth:`freeze` was already called.
            KeyError: A configured column is absent from the chunk.
        """
        if self._frozen:
            raise TransformerAlreadyFittedError(_NAME)

        for column in self._columns:
            values = chunk[column].dropna().astype(str)
            self._seen[column].update(values.unique().tolist())

    def freeze(self) -> None:
        """Close the vocabulary and assign codes in sorted order.

        Idempotent. Sorting here rather than during :meth:`update` is what makes
        the codes independent of the order the chunks arrived in.
        """
        if self._frozen:
            return
        self._mapping = {
            column: {value: code for code, value in enumerate(sorted(self._seen[column]))}
            for column in self._columns
        }
        self._frozen = True

    def mapping(self) -> dict[str, dict[str, int]]:
        """The fitted value-to-code mapping per column.

        Raises:
            TransformerNotFittedError: The vocabulary is still open.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME, "report its mapping")
        return {column: dict(codes) for column, codes in self._mapping.items()}

    def vocabulary_size(self, column: str) -> int:
        """Number of distinct training values learned for ``column``.

        Raises:
            TransformerNotFittedError: The vocabulary is still open.
            KeyError: The column is not covered by this encoder.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME, "report its vocabulary size")
        return len(self._mapping[column])

    def transform(self, chunk: DataFrame, *, suffix: str = "_code") -> DataFrame:
        """Add an integer code column for each encoded column.

        The source columns are left in place; the caller decides which columns
        reach the output, and keeping both makes the encoding auditable against
        the raw value during development.

        Args:
            chunk: Frame to transform. Not modified.
            suffix: Appended to each source column name to form the code column.

        Returns:
            A new frame with one added code column per encoded column.

        Raises:
            TransformerNotFittedError: The vocabulary is still open.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME)

        result = chunk.copy()
        for column in self._columns:
            codes = self._mapping[column]
            mapped: Series = result[column].astype(str).map(codes)
            result[f"{column}{suffix}"] = (
                mapped.fillna(UNKNOWN_CODE).astype("int32")
            )
        return result

    def count_unknown(self, chunk: DataFrame) -> dict[str, int]:
        """Count values in ``chunk`` that are outside the fitted vocabulary.

        Reported rather than acted upon: an unseen category is a finding about
        the data, not a reason to drop rows.

        Raises:
            TransformerNotFittedError: The vocabulary is still open.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME, "count unknown values for")
        return {
            column: int((~chunk[column].astype(str).isin(self._mapping[column])).sum())
            for column in self._columns
        }

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, dict[str, int]]
    ) -> OrdinalCategoricalEncoder:
        """Rebuild a fitted encoder from a serialised mapping.

        Args:
            mapping: As produced by :meth:`mapping`.

        Returns:
            A frozen encoder that maps identically to the original.
        """
        instance = cls(tuple(mapping))
        instance._mapping = {column: dict(codes) for column, codes in mapping.items()}
        instance._seen = {column: set(codes) for column, codes in mapping.items()}
        instance._frozen = True
        return instance

    def describe(self) -> dict[str, Any]:
        """Encoding metadata for the artefacts, including the refusals.

        Raises:
            TransformerNotFittedError: The vocabulary is still open.
        """
        if not self._frozen:
            raise TransformerNotFittedError(_NAME, "describe")
        return {
            "scheme": "ordinal",
            "codes_assigned_in": "sorted vocabulary order (order-independent)",
            "unknown_code": UNKNOWN_CODE,
            "codes_are_nominal": True,
            "nominal_note": (
                "Codes identify categories and imply no ordering. Models that "
                "assume ordinal inputs must one-hot or embed these columns; "
                "that expansion is deferred to the modelling phase so the "
                "processed files stay compact and streamable."
            ),
            "columns": {
                column: {
                    "vocabulary_size": len(self._mapping[column]),
                    "mapping": dict(self._mapping[column]),
                }
                for column in self._columns
            },
            "not_encoded": dict(HIGH_CARDINALITY_REFUSALS),
        }

    def __repr__(self) -> str:
        """Show fitted state and column count."""
        state = "fitted" if self._frozen else "unfitted"
        return f"{_NAME}({len(self._columns)} column(s), {state})"


def encode_binary_target(values: Series, mapping: dict[str, int]) -> Series:
    """Map a raw label column onto its encoded values.

    Separate from the categorical encoder because the target is not a feature
    and its mapping is fixed by the verified data rather than learned from a
    split.

    Args:
        values: Raw label column.
        mapping: Raw value (as text) to encoded integer.

    Returns:
        Encoded labels as ``int8``, with unmapped values as ``UNKNOWN_CODE``.
    """
    mapped = values.astype(str).map(mapping)
    return pd.to_numeric(mapped, errors="coerce").fillna(UNKNOWN_CODE).astype("int8")
