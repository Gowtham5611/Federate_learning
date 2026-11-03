"""Kinds of remote dataset source."""

from __future__ import annotations

from enum import StrEnum


class DatasetSourceKind(StrEnum):
    """Where a dataset is fetched from.

    Keeping this as a closed enumeration lets the application layer route a
    download request to the right adapter without importing, or knowing
    anything about, a specific provider's client library.
    """

    KAGGLE = "kaggle"

    @property
    def label(self) -> str:
        """Human-readable form used in CLI and report output."""
        return _LABELS[self]

    @classmethod
    def parse(cls, value: str) -> DatasetSourceKind:
        """Parse a registry value case-insensitively.

        Raises:
            ValueError: If ``value`` is not a known source kind.
        """
        try:
            return cls(value.strip().lower())
        except ValueError:
            known = ", ".join(sorted(member.value for member in cls))
            raise ValueError(f"Unknown dataset source '{value}'. Known sources: {known}") from None


_LABELS: dict[DatasetSourceKind, str] = {
    DatasetSourceKind.KAGGLE: "Kaggle",
}
