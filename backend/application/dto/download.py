"""Download request and outcome DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain.enums.dataset_source import DatasetSourceKind


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Everything a downloader needs, with no reference to a provider SDK.

    Attributes:
        dataset_id: Project-internal identifier, for logging and errors.
        source: Which adapter should handle this request.
        remote_id: Provider-side identifier to fetch.
        destination: Absolute directory to place files in.
        expected_files: Specific filenames wanted. An adapter that can fetch
            individual files should fetch only these rather than the whole
            dataset — the IBM AML distribution is far larger than one variant.
        variant: Variant label, for logging.
        overwrite: Whether to replace files that already exist.
    """

    dataset_id: str
    source: DatasetSourceKind
    remote_id: str
    destination: Path
    expected_files: tuple[str, ...]
    variant: str
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """What a downloader actually delivered.

    Reported by the adapter and then independently verified against the
    filesystem by the caller — a downloader's own success report is not taken
    as proof.

    Attributes:
        destination: Directory files were written to.
        files: Filenames the adapter believes it delivered.
    """

    destination: Path
    files: tuple[str, ...]
