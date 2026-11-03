"""Shared base class for dataset downloader adapters.

This is *not* a second abstraction competing with
:class:`~application.ports.output.dataset_downloader.DatasetDownloaderPort`. The
port is the only abstraction callers depend on. This base class implements that
port once, as a template method covering the concerns every adapter shares —
destination validation, directory creation, and post-transfer verification —
and leaves exactly one hook, :meth:`_fetch`, for provider-specific transport.

A new provider therefore implements transport only, and cannot forget to verify
its own output.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

from application.dto.download import DownloadOutcome, DownloadRequest
from application.ports.output.dataset_downloader import DatasetDownloaderPort
from domain.exceptions.dataset import InvalidDatasetPathError
from domain.exceptions.download import IncompleteDownloadError

logger = logging.getLogger(__name__)


class BaseDatasetDownloader(DatasetDownloaderPort):
    """Template implementation of the downloader port."""

    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Validate the destination, delegate transport, then verify the result.

        Subclasses override :meth:`_fetch`, not this method.

        Raises:
            InvalidDatasetPathError: If the destination cannot be used.
            IncompleteDownloadError: If expected files are absent afterwards.
            DownloadError: Whatever :meth:`_fetch` raises for transport failures.
        """
        destination = self._prepare_destination(request.destination)

        logger.info(
            "Download started: id=%s source=%s variant=%s files=%d destination=%s",
            request.dataset_id,
            self.source.value,
            request.variant,
            len(request.expected_files),
            destination,
        )

        delivered = self._fetch(request, destination)

        missing = self._missing_files(destination, request.expected_files)
        if missing:
            raise IncompleteDownloadError(request.dataset_id, missing, str(destination))

        logger.info(
            "Download finished: id=%s delivered=%d destination=%s",
            request.dataset_id,
            len(delivered),
            destination,
        )
        return DownloadOutcome(destination=destination, files=delivered)

    @abstractmethod
    def _fetch(self, request: DownloadRequest, destination: Path) -> tuple[str, ...]:
        """Transfer the dataset's files into ``destination``.

        Implementations handle authentication, transport and any extraction, and
        translate provider errors into ``domain.exceptions.download`` types.

        Args:
            request: What to fetch.
            destination: Existing, writable directory to write into.

        Returns:
            The filenames delivered.
        """

    @staticmethod
    def _prepare_destination(destination: Path) -> Path:
        """Create the destination directory and confirm it is usable.

        Raises:
            InvalidDatasetPathError: If the path exists as a non-directory, or
                cannot be created or written to.
        """
        if destination.exists() and not destination.is_dir():
            raise InvalidDatasetPathError(destination, "exists but is not a directory")
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InvalidDatasetPathError(destination, f"could not be created ({exc})") from exc

        probe = destination / ".write_probe"
        try:
            probe.touch()
            probe.unlink()
        except OSError as exc:
            raise InvalidDatasetPathError(destination, f"is not writable ({exc})") from exc
        return destination

    @staticmethod
    def _missing_files(destination: Path, expected_files: tuple[str, ...]) -> tuple[str, ...]:
        """Return expected files that are absent or zero-length.

        Searched directly under ``destination`` first, then recursively, so an
        archive that extracted into a nested folder still counts as delivered. A
        zero-byte file counts as missing — a truncated transfer is not a success.
        """
        missing: list[str] = []
        for filename in expected_files:
            direct = destination / filename
            if direct.is_file() and direct.stat().st_size > 0:
                continue
            if any(c.is_file() and c.stat().st_size > 0 for c in destination.rglob(filename)):
                continue
            missing.append(filename)
        return tuple(missing)
