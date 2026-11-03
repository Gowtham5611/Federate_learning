"""How a schema treats columns it does not declare."""

from __future__ import annotations

from enum import StrEnum


class UnexpectedColumnPolicy(StrEnum):
    """What to do about a column present in the data but absent from the schema.

    The policy is explicit rather than implied because the dangerous option is
    the silent one. Quietly discarding an undeclared column loses data without
    telling anyone; quietly accepting it lets an upstream format change pass
    unnoticed. Both are therefore refused as defaults — a schema must state
    which behaviour it intends.
    """

    REJECT = "reject"
    """An undeclared column is a validation error. Use when the source format is
    fixed and any deviation means the file is not what was expected."""

    ALLOW = "allow"
    """An undeclared column is permitted and reported as a warning. Use only
    when extra columns are a designed, tolerated extension."""
