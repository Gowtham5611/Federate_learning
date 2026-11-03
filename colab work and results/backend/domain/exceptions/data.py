"""Data ingestion errors: reading tabular files and validating them against a schema.

These extend the existing :class:`~domain.exceptions.base.FedSecureAIError`
hierarchy rather than introducing a parallel one, so a caller can still catch a
single project-wide base class. The loader translates whatever the CSV parser
raises into one of these, so nothing above the infrastructure boundary needs to
import pandas' exception types.
"""

from __future__ import annotations

from pathlib import Path

from domain.exceptions.base import FedSecureAIError


class DataError(FedSecureAIError):
    """Base class for problems reading or validating a data file."""


class DataFileNotFoundError(DataError):
    """An expected data file does not exist on this machine."""

    def __init__(self, path: Path | str, *, hint: str = "") -> None:
        """Record the missing path and how to obtain the file."""
        self.path = Path(path)
        suffix = f" {hint}" if hint else ""
        super().__init__(f"Data file not found: '{self.path}'.{suffix}")


class EmptyDataFileError(DataError):
    """A data file exists but contains nothing that can be parsed.

    Distinct from a file whose header parses but which has no data rows: that is
    structurally valid and is reported by the loader as zero chunks, not as an
    error.
    """

    def __init__(self, path: Path | str, reason: str = "no columns could be parsed") -> None:
        """Record which file was unusable and why."""
        self.path = Path(path)
        super().__init__(
            f"Data file '{self.path}' is empty: {reason}. "
            f"A zero-byte file is not a dataset; re-download it."
        )


class MalformedDataError(DataError):
    """A data file could not be parsed as well-formed CSV."""

    def __init__(self, path: Path | str, detail: str) -> None:
        """Record the file and the parser's complaint."""
        self.path = Path(path)
        super().__init__(
            f"Data file '{self.path}' is malformed: {detail}. "
            f"The file may be truncated, wrongly delimited, or not CSV at all."
        )


class UnreadableDataFileError(DataError):
    """A data file exists but the operating system refused to read it."""

    def __init__(self, path: Path | str, detail: str) -> None:
        """Record the file and the OS-level reason."""
        self.path = Path(path)
        super().__init__(
            f"Data file '{self.path}' could not be read: {detail}. "
            f"Check file permissions and that the path is a file, not a directory."
        )


class InvalidChunkSizeError(DataError):
    """A streaming read was requested with an unusable chunk size."""

    def __init__(self, chunk_size: object) -> None:
        """Record the rejected value."""
        self.chunk_size = chunk_size
        super().__init__(
            f"Invalid chunk size {chunk_size!r}: must be a positive integer. "
            f"Chunked reading is what keeps memory bounded, so it cannot be "
            f"disabled by passing zero or a negative size."
        )


class SchemaDefinitionError(DataError):
    """A schema or column specification is internally inconsistent.

    Raised while *building* a schema, not while validating data — this is a
    programming error in the schema definition itself.
    """

    def __init__(self, reason: str, subject: str | None = None) -> None:
        """Record what was wrong with the definition."""
        self.subject = subject
        prefix = f"Schema '{subject}'" if subject else "Schema definition"
        super().__init__(f"{prefix} is invalid: {reason}")


class SchemaNotVerifiedError(DataError):
    """A schema was requested whose fields have not been verified against real data.

    Deliberately a hard failure. The alternative — returning a schema assembled
    from remembered or guessed column names — would let the whole pipeline run
    and produce results that silently describe a dataset that does not exist.
    """

    def __init__(self, dataset_name: str, procedure: str) -> None:
        """Record which schema is unverified and how to verify it."""
        self.dataset_name = dataset_name
        super().__init__(
            f"The {dataset_name} schema has not been verified against the real "
            f"dataset, so it is not defined. Column names, data types and "
            f"permitted values must be read from the actual file rather than "
            f"assumed. To define it: {procedure}"
        )


class SchemaValidationError(DataError):
    """Data did not conform to its schema.

    Raised by the ingestion pipeline in strict mode. The message carries an
    aggregated summary rather than one line per offending row.
    """

    def __init__(self, source: str, chunk_index: int, summary: str) -> None:
        """Record where the invalid data came from and what was wrong with it."""
        self.source = source
        self.chunk_index = chunk_index
        self.summary = summary
        super().__init__(
            f"Schema validation failed for '{source}' at chunk {chunk_index}: {summary}"
        )
