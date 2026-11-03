"""Download errors.

These form the error contract of the downloader port: an adapter translates
whatever its client library raises into one of these, so the application layer
can react to *why* a download failed without importing a provider SDK.
"""

from __future__ import annotations

from domain.exceptions.base import FedSecureAIError


class DownloadError(FedSecureAIError):
    """Base class for every download failure."""


class MissingCredentialsError(DownloadError):
    """Required credentials were not supplied.

    Carries only the *names* of the missing variables — never their values.
    """

    def __init__(self, source: str, variable_names: tuple[str, ...]) -> None:
        """Record which named credentials were absent."""
        self.source = source
        self.variable_names = variable_names
        names = ", ".join(variable_names)
        super().__init__(
            f"Missing {source} credentials: {names} are not set. "
            f"Copy .env.example to .env and fill them in, or export them in the "
            f"environment. Credentials are never stored in the repository."
        )


class AuthenticationError(DownloadError):
    """Credentials were supplied but rejected by the provider."""

    def __init__(self, source: str, detail: str | None = None) -> None:
        """Record the provider that rejected the credentials."""
        self.source = source
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"{source} rejected the supplied credentials{suffix}. "
            f"Verify the account has accepted the dataset's terms and that the "
            f"API token is current."
        )


class NetworkError(DownloadError):
    """The provider could not be reached."""

    def __init__(self, source: str, detail: str | None = None) -> None:
        """Record the unreachable provider."""
        self.source = source
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Could not reach {source}{suffix}. Check network connectivity.")


class DownloadFailedError(DownloadError):
    """The transfer started but did not complete successfully."""

    def __init__(self, dataset_id: str, detail: str | None = None) -> None:
        """Record which dataset failed to transfer."""
        self.dataset_id = dataset_id
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Download of '{dataset_id}' failed{suffix}.")


class IncompleteDownloadError(DownloadError):
    """The transfer reported success but the expected files are not all present.

    Guards against a truncated transfer, a partially extracted archive, or a
    downloader that reports success without delivering.
    """

    def __init__(self, dataset_id: str, missing_files: tuple[str, ...], destination: str) -> None:
        """Record what was expected but absent after the transfer."""
        self.dataset_id = dataset_id
        self.missing_files = missing_files
        self.destination = destination
        missing = ", ".join(missing_files)
        super().__init__(
            f"Download of '{dataset_id}' reported success but these expected files "
            f"are missing or empty in '{destination}': {missing}. "
            f"Treat the local copy as incomplete and re-download."
        )
