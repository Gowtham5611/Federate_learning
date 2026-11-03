"""Test doubles for the dataset downloader port.

None of these touch a network, a credential store, or the Kaggle SDK. Their
existence is what makes the dataset manager testable offline, which is the point
of depending on the port rather than the adapter.
"""

from __future__ import annotations

from pathlib import Path

from application.dto.download import DownloadOutcome, DownloadRequest
from application.ports.output.dataset_downloader import DatasetDownloaderPort
from domain.enums.dataset_source import DatasetSourceKind
from domain.exceptions.download import (
    AuthenticationError,
    DownloadError,
    MissingCredentialsError,
    NetworkError,
)


class RecordingDownloader(DatasetDownloaderPort):
    """Records each request and writes the expected files as small stubs.

    Simulates a successful transfer without any I/O beyond the destination.
    """

    def __init__(self, *, content: str = "stub", deliver: bool = True) -> None:
        """Configure whether the fake transfer actually produces files."""
        self.requests: list[DownloadRequest] = []
        self._content = content
        self._deliver = deliver

    @property
    def call_count(self) -> int:
        """How many times :meth:`download` was invoked."""
        return len(self.requests)

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Record the request and optionally create the expected files."""
        self.requests.append(request)
        request.destination.mkdir(parents=True, exist_ok=True)
        delivered: list[str] = []
        if self._deliver:
            for filename in request.expected_files:
                (request.destination / filename).write_text(self._content, encoding="utf-8")
                delivered.append(filename)
        return DownloadOutcome(destination=request.destination, files=tuple(delivered))


class ExplodingDownloader(DatasetDownloaderPort):
    """Fails the test if a download is ever attempted.

    Injected into metadata-only tests to prove that querying dataset information
    cannot trigger a transfer.
    """

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Always raise: reaching here means a metadata query hit the network."""
        raise AssertionError(
            f"download() must not be called during a metadata query "
            f"(dataset '{request.dataset_id}')"
        )


class FailingDownloader(DatasetDownloaderPort):
    """Raises a configured :class:`DownloadError` subclass."""

    def __init__(self, error: DownloadError | None = None) -> None:
        """Store the error to raise, defaulting to a network failure."""
        self.error = error or NetworkError("Kaggle", "simulated outage")
        self.call_count = 0

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Record the attempt, then fail."""
        self.call_count += 1
        raise self.error


def auth_failing_downloader() -> FailingDownloader:
    """A downloader whose credentials are rejected."""
    return FailingDownloader(AuthenticationError("Kaggle", "invalid token"))


def credentialless_downloader() -> FailingDownloader:
    """A downloader with no credentials available."""
    return FailingDownloader(
        MissingCredentialsError("Kaggle", ("KAGGLE_USERNAME", "KAGGLE_KEY"))
    )


class LyingDownloader(DatasetDownloaderPort):
    """Reports success without delivering anything.

    Exercises the manager's independent post-download verification.
    """

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Claim every expected file was delivered, but write nothing."""
        request.destination.mkdir(parents=True, exist_ok=True)
        return DownloadOutcome(
            destination=request.destination, files=request.expected_files
        )


def write_dataset_files(
    root: Path, filenames: tuple[str, ...], *, content: str = "stub"
) -> None:
    """Create non-empty stub files, simulating an already-downloaded dataset.

    These are test stubs standing in for *any* dataset file, not fabricated IBM
    AML data: content is a literal placeholder and no row, column or statistic is
    invented.
    """
    root.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (root / filename).write_text(content, encoding="utf-8")
