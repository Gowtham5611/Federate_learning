"""Test doubles for the three data ingestion ports.

None of these touch a real file, so the ingestion service can be tested without
any dataset present. They also make the *streaming* contract observable: the
loader double records how many chunks have actually been pulled, which is how a
test can tell a lazy pipeline from one that reads ahead.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from pandas import DataFrame

from application.dto.validation import ChunkValidationResult, ValidationIssue
from application.ports.output.data_loader import DEFAULT_CHUNK_SIZE, ChunkedDataLoaderPort
from application.ports.output.schema_validator import SchemaValidatorPort
from application.ports.output.transaction_data_source import TransactionDataSourcePort
from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.enums.validation_severity import ValidationSeverity
from domain.exceptions.data import DataFileNotFoundError
from domain.value_objects.table_schema import TableSchema
from tests.fixtures.tabular_fixtures import make_synthetic_schema


class StubTransactionSource(TransactionDataSourcePort):
    """Reports a fixed identity and either resolves to a path or raises."""

    def __init__(
        self,
        file_path: Path | None = None,
        *,
        dataset_id: str = "synthetic_test_dataset",
        variant: str = "TEST-Small",
        filename: str = "synthetic_records.csv",
        directory: Path | None = None,
    ) -> None:
        """Configure the identity to report and the file to resolve to."""
        self._file_path = file_path
        self._dataset_id = dataset_id
        self._variant = variant
        self._filename = filename
        self._directory = directory or Path("data/raw/synthetic")
        self.resolve_calls = 0

    @property
    def dataset_id(self) -> str:
        """Invented dataset id."""
        return self._dataset_id

    @property
    def variant(self) -> str:
        """Invented variant label."""
        return self._variant

    @property
    def expected_filename(self) -> str:
        """Invented file name."""
        return self._filename

    @property
    def local_directory(self) -> Path:
        """Invented directory."""
        return self._directory

    def resolve_file(self) -> Path:
        """Return the configured path, or fail as a missing dataset would."""
        self.resolve_calls += 1
        if self._file_path is None:
            raise DataFileNotFoundError(self._directory / self._filename)
        return self._file_path


class RecordingLoader(ChunkedDataLoaderPort):
    """Yields preset frames and records what was asked of it."""

    def __init__(self, frames: Sequence[DataFrame]) -> None:
        """Store the frames to hand out, in order."""
        self._frames = list(frames)
        self.calls: list[tuple[Path, int]] = []
        self.requested_dtypes: list[Mapping[str, str] | None] = []
        self.chunks_pulled = 0

    def stream_chunks(
        self,
        file_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        dtypes: Mapping[str, str] | None = None,
    ) -> Iterator[DataFrame]:
        """Record the request and yield the preset frames lazily."""
        self.calls.append((file_path, chunk_size))
        self.requested_dtypes.append(dtypes)
        return self._iterate()

    def _iterate(self) -> Iterator[DataFrame]:
        """Count each frame as it is actually consumed."""
        for frame in self._frames:
            self.chunks_pulled += 1
            yield frame


class RecordingValidator(SchemaValidatorPort):
    """Returns preset results and records the context it was given."""

    def __init__(
        self,
        results: Sequence[ChunkValidationResult],
        *,
        schema: TableSchema | None = None,
    ) -> None:
        """Store the results to return, one per chunk, in order."""
        self._results = list(results)
        self._schema = schema or make_synthetic_schema()
        self.calls: list[tuple[int, str, int]] = []

    @property
    def schema(self) -> TableSchema:
        """The schema this double claims to enforce."""
        return self._schema

    def validate_chunk(
        self, chunk: DataFrame, *, chunk_index: int, source: str
    ) -> ChunkValidationResult:
        """Record ``(chunk_index, source, row_count)`` and return the next result."""
        self.calls.append((chunk_index, source, len(chunk)))
        return self._results[chunk_index]


def make_result(
    chunk_index: int,
    *,
    rows: int = 2,
    source: str = "synthetic_records.csv",
    issues: tuple[ValidationIssue, ...] = (),
) -> ChunkValidationResult:
    """Build a validation result without running a validator."""
    return ChunkValidationResult(
        source=source,
        chunk_index=chunk_index,
        row_count=rows,
        columns=("a", "b"),
        issues=issues,
    )


def make_error_issue(column: str = "value", occurrences: int = 3) -> ValidationIssue:
    """Build a blocking finding, for testing the strictness policy."""
    return ValidationIssue(
        kind=ValidationIssueKind.WRONG_TYPE,
        severity=ValidationSeverity.ERROR,
        column=column,
        detail="invented failure for testing",
        occurrences=occurrences,
        example_rows=(0,),
    )


def make_warning_issue(column: str = "extra") -> ValidationIssue:
    """Build a non-blocking finding, for testing tolerated deviations."""
    return ValidationIssue(
        kind=ValidationIssueKind.UNEXPECTED_COLUMN,
        severity=ValidationSeverity.WARNING,
        column=column,
        detail="invented tolerated deviation for testing",
    )
