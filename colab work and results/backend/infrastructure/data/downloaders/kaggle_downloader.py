"""Kaggle download adapter.

Isolates every Kaggle-specific concern — credentials, authentication, the client
library, per-file downloads and archive extraction — behind
:class:`~application.ports.output.dataset_downloader.DatasetDownloaderPort`.
Nothing outside this module imports the Kaggle SDK or reads its credentials.
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Any

from application.dto.download import DownloadRequest
from domain.enums.dataset_source import DatasetSourceKind
from domain.exceptions.download import (
    AuthenticationError,
    DownloadFailedError,
    NetworkError,
)
from infrastructure.data.downloaders.base_downloader import BaseDatasetDownloader
from infrastructure.data.downloaders.credentials import (
    EnvironmentKaggleCredentialsProvider,
    KaggleCredentialsProvider,
)

logger = logging.getLogger(__name__)

# Substrings used to classify an opaque SDK exception into our error contract.
_AUTH_MARKERS = ("401", "403", "unauthorized", "forbidden", "authenticat", "credential")
_NETWORK_MARKERS = ("connection", "timeout", "timed out", "dns", "network", "unreachable")


class KaggleDownloader(BaseDatasetDownloader):
    """Downloads datasets from Kaggle.

    Constructing this adapter performs no I/O and requires no credentials, so it
    can be wired at startup and injected into the dataset manager even on a
    machine that has never seen a Kaggle token. Credentials are resolved only
    inside :meth:`_fetch`, i.e. only when a download is explicitly requested.
    """

    def __init__(
        self,
        credentials_provider: KaggleCredentialsProvider | None = None,
    ) -> None:
        """Wire the adapter to a credentials provider.

        Args:
            credentials_provider: Supplies credentials at download time.
                Defaults to reading the environment.
        """
        self._credentials = credentials_provider or EnvironmentKaggleCredentialsProvider()

    @property
    def source(self) -> DatasetSourceKind:
        """This adapter handles Kaggle."""
        return DatasetSourceKind.KAGGLE

    def _fetch(self, request: DownloadRequest, destination: Path) -> tuple[str, ...]:
        """Authenticate, fetch the requested files, and extract any archives."""
        api = self._authenticated_api()

        # Fetch only the files the selected variant needs. The IBM AML
        # distribution contains several size variants; pulling the whole dataset
        # to obtain one CSV would transfer far more than required.
        delivered: list[str] = []
        for filename in request.expected_files:
            logger.info(
                "Fetching '%s' from Kaggle dataset '%s'", filename, request.remote_id
            )
            try:
                api.dataset_download_file(
                    request.remote_id,
                    file_name=filename,
                    path=str(destination),
                    force=request.overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                raise self._classify(exc, request) from exc
            delivered.append(filename)

        self._extract_archives(destination)
        return tuple(delivered)

    def _authenticated_api(self) -> Any:
        """Return an authenticated Kaggle API client.

        The Kaggle package authenticates during import by reading environment
        variables, so credentials are placed in the environment immediately
        before the import. Importing lazily also means neither the test suite nor
        a metadata query needs the package installed.
        """
        credentials = self._credentials.get()

        # Injected into the process environment because that is the only
        # interface the Kaggle SDK offers for non-file credentials. The value is
        # already registered with the log redaction filter by the provider.
        os.environ["KAGGLE_USERNAME"] = credentials.username
        os.environ["KAGGLE_KEY"] = credentials.key

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise DownloadFailedError(
                "kaggle",
                "the 'kaggle' package is not installed; install requirements/base.txt",
            ) from exc

        api = KaggleApi()
        try:
            api.authenticate()
        except SystemExit as exc:
            # The Kaggle SDK prints setup guidance and calls `exit(1)` when it
            # can find no usable credentials. `SystemExit` derives from
            # BaseException, so without this clause it would terminate the host
            # process instead of surfacing through the port's error contract.
            raise AuthenticationError(
                "Kaggle",
                "the SDK found no usable credentials and aborted",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
            raise AuthenticationError("Kaggle", type(exc).__name__) from exc

        logger.info("Authenticated with Kaggle as user '%s'", credentials.username)
        return api

    @staticmethod
    def _extract_archives(destination: Path) -> None:
        """Extract and remove any ``.zip`` archives Kaggle delivered.

        Kaggle serves single files as zips when they exceed a size threshold, so
        an archive may appear in place of the requested CSV.
        """
        for archive in sorted(destination.glob("*.zip")):
            logger.info("Extracting archive '%s'", archive.name)
            try:
                with zipfile.ZipFile(archive) as zf:
                    KaggleDownloader._extract_members(zf, archive.name, destination)
            except (zipfile.BadZipFile, OSError) as exc:
                raise DownloadFailedError(
                    archive.name, f"archive could not be extracted ({exc})"
                ) from exc
            archive.unlink(missing_ok=True)

    @staticmethod
    def _extract_members(archive: zipfile.ZipFile, name: str, destination: Path) -> None:
        """Extract every member, refusing any that would escape ``destination``.

        A zip entry may contain ``../`` or an absolute path. Extracting it
        blindly would let a remote archive write anywhere on the filesystem, so
        each destination is resolved and checked first.
        """
        root = destination.resolve()
        for member in archive.namelist():
            target = (root / member).resolve()
            if target != root and root not in target.parents:
                raise DownloadFailedError(
                    name, f"archive entry '{member}' would extract outside '{root}'"
                )
        archive.extractall(destination)

    @staticmethod
    def _classify(exc: Exception, request: DownloadRequest) -> Exception:
        """Translate an opaque SDK exception into the port's error contract.

        The Kaggle SDK signals most failures as a generic exception carrying an
        HTTP status in its message, so classification is textual by necessity.
        Anything unrecognised becomes a plain download failure rather than being
        mislabelled.
        """
        text = str(exc).lower()
        if any(marker in text for marker in _AUTH_MARKERS):
            return AuthenticationError("Kaggle", type(exc).__name__)
        if any(marker in text for marker in _NETWORK_MARKERS):
            return NetworkError("Kaggle", type(exc).__name__)
        return DownloadFailedError(request.dataset_id, f"{type(exc).__name__}: {exc}")
