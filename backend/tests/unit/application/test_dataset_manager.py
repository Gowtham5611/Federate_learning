"""Tests for the dataset manager.

Every test here runs with no network, no Kaggle credentials, and no real
dataset. That is enforced structurally: the manager only ever sees a fake
registry and a fake downloader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from application.ports.output.dataset_downloader import DatasetDownloaderPort
from application.services.dataset_manager import DatasetManager
from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.download import (
    AuthenticationError,
    IncompleteDownloadError,
    MissingCredentialsError,
    NetworkError,
)
from tests.fixtures.dataset_registry_doubles import (
    IBM_AML_KAGGLE_ID,
    InMemoryDatasetRegistry,
    make_descriptor,
)
from tests.fixtures.downloader_doubles import (
    ExplodingDownloader,
    FailingDownloader,
    LyingDownloader,
    RecordingDownloader,
    auth_failing_downloader,
    credentialless_downloader,
    write_dataset_files,
)

pytestmark = pytest.mark.unit

EXPECTED_FILE = "HI-Small_Trans.csv"
DATASET_NAME = "IBM Transactions for Anti-Money Laundering (AML)"


def build_manager(
    tmp_path: Path,
    downloader: DatasetDownloaderPort,
    **descriptor_kwargs: Any,
) -> DatasetManager:
    """Build a manager rooted at ``tmp_path`` with injected doubles."""
    registry = InMemoryDatasetRegistry((make_descriptor(**descriptor_kwargs),))
    return DatasetManager(
        registry=registry,
        downloader=downloader,
        project_root=tmp_path,
    )


# --- 1-8. Registry and descriptor are read correctly -------------------------


def test_registry_loads_and_exposes_dataset_id(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    assert manager.descriptor.dataset_id == "ibm_transactions_aml"


def test_dataset_name_is_correct(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    assert manager.get_dataset_info().name == DATASET_NAME


def test_source_is_kaggle(tmp_path: Path) -> None:
    info = build_manager(tmp_path, ExplodingDownloader()).get_dataset_info()
    assert info.source is DatasetSourceKind.KAGGLE
    assert info.source.label == "Kaggle"


def test_source_url_and_remote_id_are_correct(tmp_path: Path) -> None:
    info = build_manager(tmp_path, ExplodingDownloader()).get_dataset_info()
    assert info.remote_id == IBM_AML_KAGGLE_ID
    assert info.source_url.endswith(IBM_AML_KAGGLE_ID)


def test_selected_variant_is_hi_small(tmp_path: Path) -> None:
    assert build_manager(tmp_path, ExplodingDownloader()).get_dataset_info().selected_variant == (
        "HI-Small"
    )


def test_declared_status_is_configured_not_downloaded(tmp_path: Path) -> None:
    info = build_manager(tmp_path, ExplodingDownloader()).get_dataset_info()
    assert info.declared_status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED


def test_local_path_resolves_under_project_root(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    assert manager.get_local_path() == (tmp_path / "data" / "raw" / "ibm_aml").resolve()


# --- 9-10. Local presence is observed, not assumed ---------------------------


def test_missing_dataset_is_detected(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    info = manager.get_dataset_info()
    assert manager.is_dataset_available() is False
    assert info.missing_files == (EXPECTED_FILE,)
    assert info.effective_status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED


def test_existing_dataset_is_detected(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))
    info = manager.get_dataset_info()
    assert manager.is_dataset_available() is True
    assert info.present_files == (EXPECTED_FILE,)
    assert info.effective_status is DatasetStatus.DOWNLOADED


def test_empty_file_counts_as_missing(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    root = manager.get_local_path()
    root.mkdir(parents=True)
    (root / EXPECTED_FILE).touch()
    assert manager.is_dataset_available() is False


def test_file_nested_in_subdirectory_is_found(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    write_dataset_files(manager.get_local_path() / "nested", (EXPECTED_FILE,))
    assert manager.is_dataset_available() is True


def test_registry_claiming_downloaded_is_downgraded_when_files_absent(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader(), status=DatasetStatus.DOWNLOADED)
    info = manager.get_dataset_info()
    assert info.declared_status is DatasetStatus.DOWNLOADED
    assert info.effective_status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED
    assert info.registry_disagrees_with_disk is True


def test_processed_status_is_preserved_when_files_present(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader(), status=DatasetStatus.PROCESSED)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))
    assert manager.get_status() is DatasetStatus.PROCESSED


# --- 11-12. Download is explicit and never duplicated ------------------------


def test_existing_dataset_is_not_downloaded_again(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    manager = build_manager(tmp_path, downloader)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))

    info = manager.download_dataset()

    assert downloader.call_count == 0
    assert info.download_performed is False
    assert info.effective_status is DatasetStatus.DOWNLOADED


def test_downloader_is_called_when_download_explicitly_requested(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    manager = build_manager(tmp_path, downloader)

    info = manager.download_dataset()

    assert downloader.call_count == 1
    assert info.download_performed is True
    assert info.effective_status is DatasetStatus.DOWNLOADED


def test_force_redownloads_an_existing_dataset(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    manager = build_manager(tmp_path, downloader)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))

    manager.download_dataset(force=True)

    assert downloader.call_count == 1
    assert downloader.requests[0].overwrite is True


def test_download_request_carries_expected_fields(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    manager = build_manager(tmp_path, downloader)

    manager.download_dataset()

    request = downloader.requests[0]
    assert request.remote_id == IBM_AML_KAGGLE_ID
    assert request.expected_files == (EXPECTED_FILE,)
    assert request.variant == "HI-Small"
    assert request.destination == manager.get_local_path()
    assert request.source is DatasetSourceKind.KAGGLE


# --- 13-14. Failures are surfaced as typed, actionable errors ----------------


def test_network_failure_is_propagated(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, FailingDownloader())
    with pytest.raises(NetworkError, match="Could not reach Kaggle"):
        manager.download_dataset()


def test_authentication_failure_is_propagated(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, auth_failing_downloader())
    with pytest.raises(AuthenticationError, match="rejected the supplied credentials"):
        manager.download_dataset()


def test_missing_credentials_failure_is_propagated(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, credentialless_downloader())
    with pytest.raises(MissingCredentialsError, match="KAGGLE_USERNAME, KAGGLE_KEY"):
        manager.download_dataset()


def test_download_reporting_success_without_files_is_rejected(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, LyingDownloader())
    with pytest.raises(IncompleteDownloadError, match="reported success"):
        manager.download_dataset()


def test_failed_download_leaves_dataset_unavailable(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, FailingDownloader())
    with pytest.raises(NetworkError):
        manager.download_dataset()
    assert manager.is_dataset_available() is False
    assert manager.get_status() is DatasetStatus.CONFIGURED_NOT_DOWNLOADED


# --- 15. Metadata never requires network ------------------------------------


def test_metadata_operations_never_touch_the_downloader(tmp_path: Path) -> None:
    # ExplodingDownloader raises AssertionError if download() is ever reached.
    manager = build_manager(tmp_path, ExplodingDownloader())

    manager.get_dataset_info()
    manager.get_status()
    manager.get_local_path()
    assert manager.is_dataset_available() is False


def test_repeated_metadata_queries_read_the_registry_once(tmp_path: Path) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader())
    assert manager.descriptor is manager.descriptor


# --- 16. Logging discloses status, never secrets -----------------------------


def test_download_progress_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manager = build_manager(tmp_path, RecordingDownloader())
    with caplog.at_level(logging.INFO):
        manager.download_dataset()
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Download requested" in messages
    assert "Download completed" in messages


def test_skipped_download_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manager = build_manager(tmp_path, RecordingDownloader())
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))
    with caplog.at_level(logging.INFO):
        manager.download_dataset()
    assert "Download skipped" in " ".join(r.getMessage() for r in caplog.records)


def test_registry_disagreement_is_logged_as_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manager = build_manager(tmp_path, ExplodingDownloader(), status=DatasetStatus.DOWNLOADED)
    with caplog.at_level(logging.WARNING):
        manager.get_dataset_info()
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# --- 18. The port is genuinely substitutable --------------------------------


def test_manager_accepts_any_downloader_implementation(tmp_path: Path) -> None:
    for downloader in (RecordingDownloader(), FailingDownloader(), LyingDownloader()):
        manager = build_manager(tmp_path / type(downloader).__name__, downloader)
        assert manager.get_dataset_info().is_available is False


def test_explicit_dataset_id_overrides_primary(tmp_path: Path) -> None:
    registry = InMemoryDatasetRegistry(
        (make_descriptor(), make_descriptor(dataset_id="other_dataset")),
        primary_id="ibm_transactions_aml",
    )
    manager = DatasetManager(
        registry=registry,
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
        dataset_id="other_dataset",
    )
    assert manager.descriptor.dataset_id == "other_dataset"
