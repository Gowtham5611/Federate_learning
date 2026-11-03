"""Tests for the dataset CLI.

The CLI is exercised with an :class:`ExplodingDownloader`, which raises if
``download()`` is ever reached. That makes "``dataset-info`` never downloads" a
structural guarantee rather than a hopeful assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.services.dataset_manager import DatasetManager
from domain.exceptions.download import MissingCredentialsError
from interfaces.cli.app import build_parser, run
from tests.fixtures.dataset_registry_doubles import (
    IBM_AML_KAGGLE_ID,
    InMemoryDatasetRegistry,
    make_descriptor,
)
from tests.fixtures.downloader_doubles import (
    ExplodingDownloader,
    FailingDownloader,
    RecordingDownloader,
    credentialless_downloader,
    write_dataset_files,
)

pytestmark = pytest.mark.unit

EXPECTED_FILE = "HI-Small_Trans.csv"


def build_manager(tmp_path: Path) -> DatasetManager:
    """Build a manager with no network-capable collaborators."""
    return DatasetManager(
        registry=InMemoryDatasetRegistry((make_descriptor(),)),
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
    )


def test_dataset_info_prints_every_required_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run(["dataset-info"], build_manager(tmp_path))
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "FedSecureAI Dataset Information" in out
    for label in ("Dataset:", "Source:", "Dataset ID:", "Variant:", "Local Path:", "Status:"):
        assert label in out
    assert "IBM Transactions for Anti-Money Laundering (AML)" in out
    assert "Kaggle" in out
    assert IBM_AML_KAGGLE_ID in out
    assert "HI-Small" in out
    assert "data/raw/ibm_aml/" in out


def test_dataset_info_reports_not_downloaded_when_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["dataset-info"], build_manager(tmp_path))
    out = capsys.readouterr().out
    assert "CONFIGURED - NOT DOWNLOADED" in out
    assert "Not performed" in out
    assert EXPECTED_FILE in out


def test_dataset_info_reports_downloaded_when_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = build_manager(tmp_path)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))

    run(["dataset-info"], manager)
    out = capsys.readouterr().out

    assert "DOWNLOADED" in out
    assert "Expected files present locally (1):" in out


def test_dataset_info_never_downloads(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # ExplodingDownloader turns any transfer attempt into a test failure.
    run(["dataset-info"], build_manager(tmp_path))
    out = capsys.readouterr().out

    assert "This command never downloads data." in out
    assert not (tmp_path / "data" / "raw" / "ibm_aml" / EXPECTED_FILE).exists()


def test_dataset_info_creates_no_files_at_all(tmp_path: Path) -> None:
    run(["dataset-info"], build_manager(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_no_command_prints_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run([], build_manager(tmp_path))
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "dataset-info" in out
    assert "usage: fedsecureai" in out


def test_unknown_command_exits_with_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run(["train-everything"], build_manager(tmp_path))
    assert exc_info.value.code != 0


def test_parser_advertises_that_dataset_info_never_downloads() -> None:
    # argparse hard-wraps help text, so compare on normalised whitespace.
    help_text = " ".join(build_parser().format_help().split())
    assert "never downloads" in help_text


# --- dataset-download: the only command that transfers data --------------------


def build_downloading_manager(
    tmp_path: Path, downloader: RecordingDownloader | FailingDownloader
) -> DatasetManager:
    """A manager whose downloader simulates a transfer instead of performing one."""
    return DatasetManager(
        registry=InMemoryDatasetRegistry((make_descriptor(),)),
        downloader=downloader,
        project_root=tmp_path,
    )


def test_dataset_download_performs_the_transfer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    downloader = RecordingDownloader()
    manager = build_downloading_manager(tmp_path, downloader)

    exit_code = run(["dataset-download"], manager)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert downloader.call_count == 1
    assert "Download completed" in out
    assert (manager.get_local_path() / EXPECTED_FILE).is_file()


def test_dataset_download_reports_what_it_is_about_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The banner is the user's last chance to notice a misconfiguration."""
    run(["dataset-download"], build_downloading_manager(tmp_path, RecordingDownloader()))
    out = capsys.readouterr().out

    assert "FedSecureAI Dataset Download" in out
    assert IBM_AML_KAGGLE_ID in out
    assert "HI-Small" in out
    assert EXPECTED_FILE in out


def test_dataset_download_requests_the_approved_dataset_only(tmp_path: Path) -> None:
    downloader = RecordingDownloader()

    run(["dataset-download"], build_downloading_manager(tmp_path, downloader))

    request = downloader.requests[0]
    assert request.remote_id == IBM_AML_KAGGLE_ID
    assert request.variant == "HI-Small"
    assert request.expected_files == (EXPECTED_FILE,)


def test_dataset_download_skips_an_existing_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running the command must not re-transfer half a gigabyte by accident."""
    downloader = RecordingDownloader()
    manager = build_downloading_manager(tmp_path, downloader)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))

    exit_code = run(["dataset-download"], manager)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert downloader.call_count == 0
    assert "Nothing transferred" in out
    assert "--force" in out


def test_force_re_downloads_an_existing_copy(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    manager = build_downloading_manager(tmp_path, downloader)
    write_dataset_files(manager.get_local_path(), (EXPECTED_FILE,))

    run(["dataset-download", "--force"], manager)

    assert downloader.call_count == 1
    assert downloader.requests[0].overwrite is True


def test_without_force_the_request_does_not_overwrite(tmp_path: Path) -> None:
    downloader = RecordingDownloader()

    run(["dataset-download"], build_downloading_manager(tmp_path, downloader))

    assert downloader.requests[0].overwrite is False


def test_a_download_failure_propagates_to_the_entry_point(tmp_path: Path) -> None:
    """``main`` renders a :class:`FedSecureAIError` without a traceback."""
    manager = build_downloading_manager(tmp_path, credentialless_downloader())

    with pytest.raises(MissingCredentialsError):
        run(["dataset-download"], manager)


def test_the_download_output_cannot_disclose_a_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command prints configuration, never authentication material."""
    monkeypatch.setenv("KAGGLE_USERNAME", "example-user")
    monkeypatch.setenv("KAGGLE_KEY", "not-a-real-token-9f3b2c8a")

    run(["dataset-download"], build_downloading_manager(tmp_path, RecordingDownloader()))
    captured = capsys.readouterr()

    for forbidden in ("not-a-real-token-9f3b2c8a", "example-user"):
        assert forbidden not in captured.out
        assert forbidden not in captured.err


def test_the_parser_offers_dataset_download_and_force() -> None:
    help_text = " ".join(build_parser().format_help().split())
    assert "dataset-download" in help_text

    assert build_parser().parse_args(["dataset-download"]).force is False
    assert build_parser().parse_args(["dataset-download", "--force"]).force is True


def test_dataset_download_help_states_that_it_is_the_transferring_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dataset-download", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())

    assert "only command that transfers data" in help_text
    assert "KAGGLE_USERNAME" in help_text
    assert "never printed" in help_text
