"""Kaggle credentials: value object and environment-backed provider.

This module is the *only* place in the project that reads Kaggle credentials.
Nothing above the infrastructure layer references them, and no credential value
is ever written to source, the dataset registry, or a log.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from domain.exceptions.download import MissingCredentialsError
from infrastructure.utils.logging_config import register_secret

KAGGLE_USERNAME_ENV = "KAGGLE_USERNAME"
KAGGLE_KEY_ENV = "KAGGLE_KEY"


@dataclass(frozen=True, slots=True)
class KaggleCredentials:
    """A Kaggle username and API token.

    ``__repr__`` and ``__str__`` are overridden so the token cannot leak through
    an f-string, a log call, an exception traceback, or a debugger's default
    rendering of local variables.
    """

    username: str
    key: str

    def __repr__(self) -> str:
        """Render without exposing the token."""
        return f"KaggleCredentials(username={self.username!r}, key='***')"

    def __str__(self) -> str:
        """Render without exposing the token."""
        return self.__repr__()


class KaggleCredentialsProvider(Protocol):
    """Supplies Kaggle credentials on demand.

    Resolution is deferred to call time, not construction time, so that a
    downloader can be built — and dataset metadata queried — with no credentials
    present at all.
    """

    def get(self) -> KaggleCredentials:
        """Return credentials, or raise :class:`MissingCredentialsError`."""
        ...


class EnvironmentKaggleCredentialsProvider:
    """Reads credentials from environment variables.

    Populated either by the shell or by a ``.env`` file loaded at startup.
    Every token read is registered with the logging redaction filter
    immediately, so it is scrubbed even if some other component tries to log it.
    """

    def get(self) -> KaggleCredentials:
        """Read and validate credentials from the environment.

        Raises:
            MissingCredentialsError: If either variable is unset or blank. The
                error names the variables but never their values.
        """
        username = (os.environ.get(KAGGLE_USERNAME_ENV) or "").strip()
        key = (os.environ.get(KAGGLE_KEY_ENV) or "").strip()

        missing = tuple(
            name
            for name, value in ((KAGGLE_USERNAME_ENV, username), (KAGGLE_KEY_ENV, key))
            if not value
        )
        if missing:
            raise MissingCredentialsError("Kaggle", missing)

        register_secret(key)
        return KaggleCredentials(username=username, key=key)
