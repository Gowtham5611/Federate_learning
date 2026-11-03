"""Outbound port for fetching a dataset from its remote source."""

from __future__ import annotations

from abc import ABC, abstractmethod

from application.dto.download import DownloadOutcome, DownloadRequest
from domain.enums.dataset_source import DatasetSourceKind


class DatasetDownloaderPort(ABC):
    """The single downloader abstraction the application depends on.

    Concrete adapters live in ``infrastructure/data/downloaders/``. Provider
    credentials, authentication and transport mechanics are entirely the
    adapter's business — nothing above this port may reference them.

    Implementations must translate their client library's failures into the
    hierarchy in ``domain.exceptions.download`` so callers can distinguish a
    missing token from an unreachable network.
    """

    @property
    @abstractmethod
    def source(self) -> DatasetSourceKind:
        """The source kind this adapter handles."""

    @abstractmethod
    def download(self, request: DownloadRequest) -> DownloadOutcome:
        """Fetch the requested dataset files into ``request.destination``.

        Implementations must not decide *whether* a download is wanted; that is
        the caller's decision. Being called means a transfer was explicitly
        requested.

        Raises:
            MissingCredentialsError: Required credentials were not supplied.
            AuthenticationError: Credentials were supplied but rejected.
            NetworkError: The provider could not be reached.
            DownloadFailedError: The transfer did not complete.
            IncompleteDownloadError: Expected files are missing afterwards.
            InvalidDatasetPathError: The destination is unusable.
        """
