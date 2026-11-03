"""Logical data types a tabular column may declare."""

from __future__ import annotations

from enum import StrEnum


class ColumnDataType(StrEnum):
    """The type a column's values are expected to conform to.

    These are *logical* types, not pandas dtypes. A CSV has no types of its own:
    every field arrives as text and is inferred by the parser, whose inference
    depends on the rows it happens to see in a chunk. Declaring the intended
    type separately means validation compares the data against the project's
    expectation rather than against whatever the parser guessed.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"

    @property
    def is_numeric(self) -> bool:
        """Whether values of this type support ordering comparisons.

        Determines which columns may declare a minimum/maximum bound.
        """
        return self in _NUMERIC


_NUMERIC: frozenset[ColumnDataType] = frozenset(
    {ColumnDataType.INTEGER, ColumnDataType.FLOAT}
)
