"""Tests for the base downloader template.

Verifies the shared behaviour every adapter inherits: destination preparation
and independent verification of what was actually delivered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.dto.download import DownloadRequest
from domain.enums.dataset_source import DatasetSourceKind
from domain.exceptions.dataset import InvalidDatasetPathError
from domain.exceptions.download import DownloadFailedError, IncompleteDownloadError
from infrastructure.data.downloaders.base_downloader import BaseDatasetDownloader

pytestmark = pytest.mark.unit

EXPECTED_FILE = "HI-Small_Trans.csv"


class StubDownloader(BaseDatasetDownloader):
    """Writes whatever files it is told to, so verification can be exercised."""

    def __init__(self, files_to_write: tuple[str, ...] = (EXPECTED_FILE,)) -> None:
        """Record which filenames the fake transfer should produce."""
        self.files_to_write = files_to_write
        self.fetch_calls = 0

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def _fetch(self, request: DownloadRequest, destination: Path) -> tuple[str, ...]:
        """Create the configured files and report them as delivered."""
        self.fetch_calls += 1
        for name in self.files_to_write:
            (destination / name).write_text("stub", encoding="utf-8")
        return request.expected_files


class RaisingDownloader(BaseDatasetDownloader):
    """Fails during transport."""

    @property
    def source(self) -> DatasetSourceKind:
        """Pretends to be the Kaggle adapter."""
        return DatasetSourceKind.KAGGLE

    def _fetch(self, request: DownloadRequest, destination: Path) -> tuple[str, ...]:
        """Always fail."""
        raise DownloadFailedError(request.dataset_id, "simulated transport failure")


def make_request(
    destination: Path, expected: tuple[str, ...] = (EXPECTED_FILE,)
) -> DownloadRequest:
    """Build a download request aimed at ``destination``."""
    return DownloadRequest(
        dataset_id="ibm_transactions_aml",
        source=DatasetSourceKind.KAGGLE,
        remote_id="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        destination=destination,
        expected_files=expected,
        variant="HI-Small",
    )


def test_successful_download_returns_outcome(tmp_path: Path) -> None:
    downloader = StubDownloader()
    outcome = downloader.download(make_request(tmp_path / "ibm_aml"))
    assert outcome.files == (EXPECTED_FILE,)
    assert (outcome.destination / EXPECTED_FILE).is_file()
    assert downloader.fetch_calls == 1


def test_destination_directory_is_created(tmp_path: Path) -> None:
    destination = tmp_path / "deeply" / "nested" / "ibm_aml"
    StubDownloader().download(make_request(destination))
    assert destination.is_dir()


def test_missing_delivered_file_is_reported_as_incomplete(tmp_path: Path) -> None:
    downloader = StubDownloader(files_to_write=())
    with pytest.raises(IncompleteDownloadError, match="reported success"):
        downloader.download(make_request(tmp_path / "ibm_aml"))


def test_zero_length_delivered_file_is_reported_as_incomplete(tmp_path: Path) -> None:
    destination = tmp_path / "ibm_aml"
    destination.mkdir()
    (destination / EXPECTED_FILE).touch()
    downloader = StubDownloader(files_to_write=())
    with pytest.raises(IncompleteDownloadError):
        downloader.download(make_request(destination))


def test_file_delivered_into_subdirectory_counts_as_present(tmp_path: Path) -> None:
    class NestingDownloader(StubDownloader):
        """Extracts into a nested folder, as an archive would."""

        def _fetch(self, request: DownloadRequest, destination: Path) -> tuple[str, ...]:
            nested = destination / "archive_root"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / EXPECTED_FILE).write_text("stub", encoding="utf-8")
            return request.expected_files

    outcome = NestingDownloader().download(make_request(tmp_path / "ibm_aml"))
    assert outcome.files == (EXPECTED_FILE,)


def test_destination_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    blocker = tmp_path / "ibm_aml"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(InvalidDatasetPathError, match="not a directory"):
        StubDownloader().download(make_request(blocker))


def test_transport_failure_propagates(tmp_path: Path) -> None:
    with pytest.raises(DownloadFailedError, match="simulated transport failure"):
        RaisingDownloader().download(make_request(tmp_path / "ibm_aml"))


def test_write_probe_is_cleaned_up(tmp_path: Path) -> None:
    destination = tmp_path / "ibm_aml"
    StubDownloader().download(make_request(destination))
    assert not (destination / ".write_probe").exists()
