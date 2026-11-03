"""Tests for ``.env`` loading.

The loader is the bridge between a developer's git-ignored ``.env`` file and the
environment-reading credential provider. Two properties matter and are pinned
here: an explicitly exported variable always beats the file, and **no value from
the file is ever written to a log**.

Every test uses a file in ``tmp_path``. The project's own ``.env`` is never read,
so the suite behaves identically on a machine that has one and a machine that
does not.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from infrastructure.utils.env import ENV_FILENAME, load_env_file
from infrastructure.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.unit

ENV_LOGGER = "infrastructure.utils.env"
SECRET = "not-a-real-token-9f3b2c8a"


def write_env(path: Path, body: str) -> Path:
    """Write an env file, dedented by the caller."""
    target = path / ENV_FILENAME
    target.write_text(body, encoding="utf-8")
    return target


# --- A missing file is not an error -------------------------------------------


def test_a_missing_file_loads_nothing_and_is_not_an_error(tmp_path: Path) -> None:
    """The project must run with no ``.env`` at all."""
    assert load_env_file(tmp_path / ENV_FILENAME) == 0


def test_a_missing_file_leaves_the_environment_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FEDSECURE_TEST_VARIABLE", raising=False)

    load_env_file(tmp_path / ENV_FILENAME)

    assert "FEDSECURE_TEST_VARIABLE" not in os.environ


def test_a_directory_is_treated_as_absent(tmp_path: Path) -> None:
    (tmp_path / ENV_FILENAME).mkdir()
    assert load_env_file(tmp_path / ENV_FILENAME) == 0


# --- Variables reach the environment ------------------------------------------


def test_a_defined_variable_reaches_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    path = write_env(tmp_path, "KAGGLE_USERNAME=example-user\n")

    load_env_file(path)

    assert os.environ["KAGGLE_USERNAME"] == "example-user"


def test_the_count_reports_how_many_variables_were_defined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ALPHA_ONE", "ALPHA_TWO"):
        monkeypatch.delenv(name, raising=False)
    path = write_env(tmp_path, "ALPHA_ONE=1\nALPHA_TWO=2\n")

    assert load_env_file(path) == 2


def test_blank_and_commented_entries_are_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.env.example`` ships with blank credential slots; those are not values."""
    for name in ("SET_VARIABLE", "BLANK_VARIABLE"):
        monkeypatch.delenv(name, raising=False)
    path = write_env(
        tmp_path,
        "# a comment\n\nSET_VARIABLE=present\nBLANK_VARIABLE=\n",
    )

    assert load_env_file(path) == 1


def test_a_blank_value_does_not_satisfy_the_credential_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied-but-unfilled ``.env`` must fail the same way as no file at all."""
    from domain.exceptions.download import MissingCredentialsError
    from infrastructure.data.downloaders.credentials import (
        EnvironmentKaggleCredentialsProvider,
    )

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    path = write_env(tmp_path, "KAGGLE_USERNAME=\nKAGGLE_KEY=\n")
    load_env_file(path)

    with pytest.raises(MissingCredentialsError):
        EnvironmentKaggleCredentialsProvider().get()


# --- Precedence ---------------------------------------------------------------


def test_an_exported_variable_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate shell export must not be silently replaced by a stale file."""
    monkeypatch.setenv("KAGGLE_USERNAME", "from-the-shell")
    path = write_env(tmp_path, "KAGGLE_USERNAME=from-the-file\n")

    load_env_file(path)

    assert os.environ["KAGGLE_USERNAME"] == "from-the-shell"


def test_override_replaces_an_exported_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "from-the-shell")
    path = write_env(tmp_path, "KAGGLE_USERNAME=from-the-file\n")

    load_env_file(path, override=True)

    assert os.environ["KAGGLE_USERNAME"] == "from-the-file"


# --- The default location -----------------------------------------------------


def test_the_default_path_is_dot_env_beside_the_project_root() -> None:
    """Documented in ``.env.example``; asserted so it cannot drift."""
    assert (PROJECT_ROOT / ENV_FILENAME).name == ".env"


# --- Nothing sensitive is logged ----------------------------------------------


def test_no_value_from_the_file_is_ever_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point of the module docstring's promise, asserted."""
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    path = write_env(tmp_path, f"KAGGLE_KEY={SECRET}\n")

    with caplog.at_level(logging.DEBUG, logger=ENV_LOGGER):
        load_env_file(path)

    text = "\n".join(record.getMessage() for record in caplog.records)

    assert SECRET not in text
    assert "from-the-file" not in text


def test_the_log_confirms_the_file_was_used_without_naming_its_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    path = write_env(tmp_path, f"KAGGLE_KEY={SECRET}\n")

    with caplog.at_level(logging.INFO, logger=ENV_LOGGER):
        load_env_file(path)

    message = caplog.records[-1].getMessage()

    assert "1 variable(s)" in message
    assert str(path) in message
    assert "KAGGLE_KEY" not in message, "even the name is unnecessary here"


# --- Degrading without python-dotenv ------------------------------------------


def test_a_missing_dotenv_package_is_reported_rather_than_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Startup must survive an incomplete install; the file is simply ignored."""
    # An empty stand-in module makes `from dotenv import ...` raise ImportError,
    # which is what a missing or broken install produces at that line.
    monkeypatch.setitem(sys.modules, "dotenv", ModuleType("dotenv"))
    path = write_env(tmp_path, "KAGGLE_USERNAME=example-user\n")

    with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
        loaded = load_env_file(path)

    assert loaded == 0
    assert "python-dotenv is not installed" in caplog.records[-1].getMessage()


def test_the_file_is_not_applied_when_the_package_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No silent half-load: the environment must be exactly as it was."""
    monkeypatch.setitem(sys.modules, "dotenv", ModuleType("dotenv"))
    monkeypatch.delenv("FEDSECURE_TEST_VARIABLE", raising=False)
    path = write_env(tmp_path, "FEDSECURE_TEST_VARIABLE=present\n")

    load_env_file(path)

    assert "FEDSECURE_TEST_VARIABLE" not in os.environ
