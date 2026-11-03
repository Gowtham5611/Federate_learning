"""Severity of a single validation finding."""

from __future__ import annotations

from enum import StrEnum


class ValidationSeverity(StrEnum):
    """How serious a validation finding is.

    Only :attr:`ERROR` makes a chunk invalid. The distinction exists so that a
    tolerated deviation (an extra column under an ``ALLOW`` policy) can be
    reported without discarding otherwise usable data.
    """

    ERROR = "error"
    WARNING = "warning"

    @property
    def is_blocking(self) -> bool:
        """Whether a finding of this severity makes the chunk invalid."""
        return self is ValidationSeverity.ERROR
