"""Tests for the Kaggle adapter's non-network behaviour.

Nothing here contacts Kaggle. What is exercised is everything the adapter does
*around* the transfer: construction without credentials, credential resolution
ordering, archive extraction, and translation of opaque SDK exceptions into the
port's error contract.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from application.dto.download import DownloadRequest
from domain.enums.dataset_source import DatasetSourceKind
from domain.exceptions.download import (
    AuthenticationError,
    DownloadFailedError,
    MissingCredentialsError,
    NetworkError,
)
from infrastructure.data.downloaders.credentials import KAGGLE_KEY_ENV, KAGGLE_USERNAME_ENV
from infrastructure.data.downloaders.kaggle_downloader import KaggleDownloader

pytestmark = pytest.mark.unit

EXPECTED_FILE = "HI-Small_Trans.csv"
DOWNLOADER_LOGGER = "infrastructure.data.downloaders.kaggle_downloader"


def make_request(destination: Path) -> DownloadRequest:
    """Build a request for the approved dataset."""
    return DownloadRequest(
        dataset_id="ibm_transactions_aml",
        source=DatasetSourceKind.KAGGLE,
        remote_id="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        destination=destination,
        expected_files=(EXPECTED_FILE,),
        variant="HI-Small",
    )


# --- Construction is inert ----------------------------------------------------


def test_construction_requires_no_credentials_and_no_io() -> None:
    downloader = KaggleDownloader()
    assert downloader.source is DatasetSourceKind.KAGGLE


def test_missing_credentials_fail_before_any_network_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials are resolved first, so an unconfigured machine fails fast."""
    monkeypatch.delenv(KAGGLE_USERNAME_ENV, raising=False)
    monkeypatch.delenv(KAGGLE_KEY_ENV, raising=False)

    with pytest.raises(MissingCredentialsError, match="KAGGLE_USERNAME, KAGGLE_KEY"):
        KaggleDownloader().download(make_request(tmp_path / "ibm_aml"))


# --- Archive handling ---------------------------------------------------------


def test_archive_is_extracted_and_removed(tmp_path: Path) -> None:
    archive = tmp_path / f"{EXPECTED_FILE}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(EXPECTED_FILE, "stub")

    KaggleDownloader._extract_archives(tmp_path)

    assert (tmp_path / EXPECTED_FILE).read_text(encoding="utf-8") == "stub"
    assert not archive.exists()


def test_archive_escaping_the_destination_is_refused(tmp_path: Path) -> None:
    destination = tmp_path / "ibm_aml"
    destination.mkdir()
    archive = destination / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.csv", "stub")

    with pytest.raises(DownloadFailedError, match="would extract outside"):
        KaggleDownloader._extract_archives(destination)

    assert not (tmp_path / "escaped.csv").exists()


def test_corrupt_archive_is_reported_as_a_download_failure(tmp_path: Path) -> None:
    (tmp_path / "broken.zip").write_text("not a zip file", encoding="utf-8")
    with pytest.raises(DownloadFailedError, match="could not be extracted"):
        KaggleDownloader._extract_archives(tmp_path)


# --- Error classification -----------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("401 Client Error: Unauthorized", AuthenticationError),
        ("403 Forbidden", AuthenticationError),
        ("could not authenticate", AuthenticationError),
        ("invalid credential supplied", AuthenticationError),
        ("Connection aborted", NetworkError),
        ("Read timed out", NetworkError),
        ("Temporary failure in DNS resolution", NetworkError),
        ("Network is unreachable", NetworkError),
        ("404 Not Found", DownloadFailedError),
        ("something entirely unexpected", DownloadFailedError),
    ],
)
def test_sdk_exceptions_are_classified(
    tmp_path: Path, message: str, expected: type[Exception]
) -> None:
    classified = KaggleDownloader._classify(RuntimeError(message), make_request(tmp_path))
    assert isinstance(classified, expected)


def test_classification_never_leaks_the_original_message_for_auth_errors(
    tmp_path: Path,
) -> None:
    """An auth failure may quote a token back; only the exception type is kept."""
    leaky = RuntimeError("401 Unauthorized for key abcdef0123456789fake")
    classified = KaggleDownloader._classify(leaky, make_request(tmp_path))
    assert "abcdef0123456789fake" not in str(classified)


# --- Authentication failure modes ---------------------------------------------


def install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch, authenticate: Callable[[], None]
) -> None:
    """Replace the SDK module that the adapter imports lazily.

    Injecting the module makes these tests independent of whether the real Kaggle
    package is installed, and lets ``authenticate`` fail in a chosen way without
    any network use.
    """

    class FakeKaggleApi:
        """Stands in for ``kaggle.api.kaggle_api_extended.KaggleApi``."""

        def authenticate(self) -> None:
            """Delegate to the behaviour the test asked for."""
            authenticate()

    module = ModuleType("kaggle.api.kaggle_api_extended")
    module.__dict__["KaggleApi"] = FakeKaggleApi
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)


@pytest.fixture
def placeholder_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the credentials provider with values that are not real."""
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, "example-user")
    monkeypatch.setenv(KAGGLE_KEY_ENV, "not-a-real-token-9f3b2c8a")


@pytest.mark.usefixtures("placeholder_credentials")
def test_the_sdk_calling_exit_becomes_an_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the SDK aborts the process instead of raising.

    ``KaggleApi.authenticate`` prints setup guidance and calls ``exit(1)`` when it
    finds no usable credentials. :class:`SystemExit` derives from
    :class:`BaseException`, so ``except Exception`` does not see it and the host
    process would simply die. It must surface through the port's error contract.
    """

    def abort() -> None:
        raise SystemExit(1)

    install_fake_sdk(monkeypatch, abort)

    with pytest.raises(AuthenticationError, match="no usable credentials"):
        KaggleDownloader()._authenticated_api()


@pytest.mark.usefixtures("placeholder_credentials")
def test_an_opaque_authentication_exception_becomes_an_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise RuntimeError("token rejected: not-a-real-token-9f3b2c8a")

    install_fake_sdk(monkeypatch, fail)

    with pytest.raises(AuthenticationError) as exc_info:
        KaggleDownloader()._authenticated_api()

    assert "not-a-real-token-9f3b2c8a" not in str(exc_info.value)


@pytest.mark.usefixtures("placeholder_credentials")
def test_a_successful_authentication_returns_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sdk(monkeypatch, lambda: None)

    assert KaggleDownloader()._authenticated_api() is not None


@pytest.mark.usefixtures("placeholder_credentials")
def test_authentication_logs_the_username_but_never_the_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install_fake_sdk(monkeypatch, lambda: None)

    with caplog.at_level(logging.INFO, logger=DOWNLOADER_LOGGER):
        KaggleDownloader()._authenticated_api()

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "not-a-real-token-9f3b2c8a" not in text


def test_a_missing_sdk_is_reported_as_a_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete install must explain itself, not raise ImportError."""
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, "example-user")
    monkeypatch.setenv(KAGGLE_KEY_ENV, "not-a-real-token-9f3b2c8a")
    # An empty module makes `from ... import KaggleApi` raise ImportError, which
    # is exactly what an absent or broken install produces at that line.
    monkeypatch.setitem(
        sys.modules,
        "kaggle.api.kaggle_api_extended",
        ModuleType("kaggle.api.kaggle_api_extended"),
    )

    with pytest.raises(DownloadFailedError, match="not installed"):
        KaggleDownloader()._authenticated_api()
