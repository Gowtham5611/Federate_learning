"""Closed set of validation finding categories."""

from __future__ import annotations

from enum import StrEnum


class ValidationIssueKind(StrEnum):
    """What kind of schema violation was found.

    The kind is what makes de-duplication possible. A chunk of 100 000 rows with
    a bad currency code in 4 000 of them must not produce 4 000 identical
    messages; it produces one finding of kind :attr:`VALUE_NOT_ALLOWED` for that
    column, carrying an occurrence count. ``(column, kind)`` is therefore the
    aggregation key, and every finding must classify itself.
    """

    MISSING_REQUIRED_COLUMN = "missing_required_column"
    UNEXPECTED_COLUMN = "unexpected_column"
    WRONG_TYPE = "wrong_type"
    NULL_IN_NON_NULLABLE = "null_in_non_nullable"
    VALUE_NOT_ALLOWED = "value_not_allowed"
    VALUE_OUT_OF_RANGE = "value_out_of_range"

    @property
    def sort_key(self) -> int:
        """Position used to order findings deterministically.

        Structural problems sort before value problems: a missing column
        explains the errors that follow it, so it should be read first.
        """
        return _ORDER[self]


_ORDER: dict[ValidationIssueKind, int] = {
    kind: index
    for index, kind in enumerate(
        (
            ValidationIssueKind.MISSING_REQUIRED_COLUMN,
            ValidationIssueKind.UNEXPECTED_COLUMN,
            ValidationIssueKind.WRONG_TYPE,
            ValidationIssueKind.NULL_IN_NON_NULLABLE,
            ValidationIssueKind.VALUE_NOT_ALLOWED,
            ValidationIssueKind.VALUE_OUT_OF_RANGE,
        )
    )
}
