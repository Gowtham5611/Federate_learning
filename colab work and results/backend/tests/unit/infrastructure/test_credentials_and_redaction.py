"""Tests for credential handling and log redaction.

These tests assert the two guarantees that matter most for this module: a
credential is never rendered in a string, and never reaches a log record.
"""

from __future__ import annotations

import logging

import pytest

from domain.exceptions.download import MissingCredentialsError
from infrastructure.data.downloaders.credentials import (
    KAGGLE_KEY_ENV,
    KAGGLE_USERNAME_ENV,
    EnvironmentKaggleCredentialsProvider,
    KaggleCredentials,
)
from infrastructure.utils.logging_config import (
    REDACTION_PLACEHOLDER,
    SecretRedactingFilter,
    clear_secrets,
    redact,
    register_secret,
)

pytestmark = pytest.mark.unit

FAKE_KEY = "abcdef0123456789fake"  # noqa: S105 - a deliberately fake test value
FAKE_USER = "test_user"


@pytest.fixture(autouse=True)
def _isolate_secret_registry() -> object:
    """Keep the process-wide secret registry from leaking between tests."""
    clear_secrets()
    yield None
    clear_secrets()


# --- Credential resolution ---------------------------------------------------


def test_reads_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, FAKE_USER)
    monkeypatch.setenv(KAGGLE_KEY_ENV, FAKE_KEY)
    credentials = EnvironmentKaggleCredentialsProvider().get()
    assert credentials.username == FAKE_USER
    assert credentials.key == FAKE_KEY


@pytest.mark.parametrize(
    ("username", "key", "expected"),
    [
        ("", "", "KAGGLE_USERNAME, KAGGLE_KEY"),
        (FAKE_USER, "", "KAGGLE_KEY"),
        ("", FAKE_KEY, "KAGGLE_USERNAME"),
        ("   ", "   ", "KAGGLE_USERNAME, KAGGLE_KEY"),
    ],
)
def test_missing_credentials_names_the_absent_variables(
    monkeypatch: pytest.MonkeyPatch, username: str, key: str, expected: str
) -> None:
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, username)
    monkeypatch.setenv(KAGGLE_KEY_ENV, key)
    with pytest.raises(MissingCredentialsError, match=expected):
        EnvironmentKaggleCredentialsProvider().get()


def test_missing_credentials_error_never_contains_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, FAKE_USER)
    monkeypatch.delenv(KAGGLE_KEY_ENV, raising=False)
    with pytest.raises(MissingCredentialsError) as exc_info:
        EnvironmentKaggleCredentialsProvider().get()
    assert FAKE_KEY not in str(exc_info.value)


def test_reading_credentials_registers_the_key_for_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KAGGLE_USERNAME_ENV, FAKE_USER)
    monkeypatch.setenv(KAGGLE_KEY_ENV, FAKE_KEY)
    EnvironmentKaggleCredentialsProvider().get()
    assert redact(f"token={FAKE_KEY}") == f"token={REDACTION_PLACEHOLDER}"


# --- The value object cannot render its own secret ---------------------------


def test_repr_hides_the_key() -> None:
    credentials = KaggleCredentials(username=FAKE_USER, key=FAKE_KEY)
    assert FAKE_KEY not in repr(credentials)
    assert FAKE_KEY not in str(credentials)
    assert FAKE_KEY not in f"{credentials}"
    assert FAKE_USER in repr(credentials)


def test_credentials_are_immutable() -> None:
    credentials = KaggleCredentials(username=FAKE_USER, key=FAKE_KEY)
    with pytest.raises(AttributeError):
        credentials.key = "changed"  # type: ignore[misc]


# --- Log redaction -----------------------------------------------------------


def test_registered_secret_is_redacted_from_a_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_secret(FAKE_KEY)
    logger = logging.getLogger("test.redaction")
    logger.addFilter(SecretRedactingFilter())

    with caplog.at_level(logging.INFO, logger="test.redaction"):
        logger.info("authenticating with key %s", FAKE_KEY)

    emitted = " ".join(record.getMessage() for record in caplog.records)
    assert FAKE_KEY not in emitted
    assert REDACTION_PLACEHOLDER in emitted


def test_redaction_leaves_unrelated_text_intact() -> None:
    register_secret(FAKE_KEY)
    assert redact("downloading HI-Small_Trans.csv") == "downloading HI-Small_Trans.csv"


def test_short_values_are_not_registered() -> None:
    register_secret("abc")
    assert redact("abc") == "abc"


def test_empty_and_none_values_are_ignored() -> None:
    register_secret("")
    register_secret(None)
    assert redact("anything") == "anything"


def test_filter_passes_records_through_when_no_secrets_registered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.no_secrets")
    logger.addFilter(SecretRedactingFilter())
    with caplog.at_level(logging.INFO, logger="test.no_secrets"):
        logger.info("plain message")
    assert "plain message" in caplog.text
