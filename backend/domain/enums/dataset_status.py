"""Lifecycle status of a dataset within the project."""

from __future__ import annotations

from enum import StrEnum


class DatasetStatus(StrEnum):
    """Lifecycle state of a dataset.

    The value of each member is the token used in
    ``datasets/dataset_registry.yaml``, so the registry and the code share one
    vocabulary.

    A dataset being *configured* says nothing about whether its files exist on
    disk. Only :attr:`DOWNLOADED` and later states assert local presence, and
    even then the claim is re-checked against the filesystem rather than
    trusted; see :meth:`implies_local_files`.
    """

    CONFIGURED_NOT_DOWNLOADED = "configured_not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VALIDATED = "validated"
    PROCESSED = "processed"
    FAILED = "failed"

    @property
    def label(self) -> str:
        """Human-readable form used in CLI and report output."""
        return _LABELS[self]

    @property
    def implies_local_files(self) -> bool:
        """Whether this state asserts that the expected files exist locally.

        Used to detect a registry that disagrees with the filesystem — for
        example a `downloaded` entry whose files were deleted.
        """
        return self in _IMPLIES_LOCAL_FILES


_LABELS: dict[DatasetStatus, str] = {
    DatasetStatus.CONFIGURED_NOT_DOWNLOADED: "CONFIGURED - NOT DOWNLOADED",
    DatasetStatus.DOWNLOADING: "DOWNLOADING",
    DatasetStatus.DOWNLOADED: "DOWNLOADED",
    DatasetStatus.VALIDATED: "VALIDATED",
    DatasetStatus.PROCESSED: "PROCESSED",
    DatasetStatus.FAILED: "FAILED",
}

_IMPLIES_LOCAL_FILES: frozenset[DatasetStatus] = frozenset(
    {DatasetStatus.DOWNLOADED, DatasetStatus.VALIDATED, DatasetStatus.PROCESSED}
)
