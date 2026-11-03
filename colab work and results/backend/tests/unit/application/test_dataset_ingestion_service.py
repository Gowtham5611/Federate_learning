"""Tests for the dataset ingestion service.

The service composes the three ports and owns one decision: what to do with a
chunk that fails validation. These tests pin that decision, and pin the two
properties that make the composition memory-safe - nothing is read until a chunk
is asked for, and no chunk is retained once it has been yielded.

The doubles do all the work, so no file, no dataset and no credentials are
involved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from application.dto.validation import ChunkValidationResult
from application.ports.output.data_loader import DEFAULT_CHUNK_SIZE
from application.services.dataset_ingestion_service import DatasetIngestionService
from domain.exceptions.data import DataFileNotFoundError, SchemaValidationError
from tests.fixtures.ingestion_doubles import (
    RecordingLoader,
    RecordingValidator,
    StubTransactionSource,
    make_error_issue,
    make_result,
    make_warning_issue,
)

pytestmark = pytest.mark.unit

SERVICE_LOGGER = "application.services.dataset_ingestion_service"
FILE_NAME = "synthetic_records.csv"


def frame(rows: int = 2) -> pd.DataFrame:
    """A small invented frame; its contents are irrelevant to the service."""
    return pd.DataFrame({"a": list(range(rows)), "b": ["x"] * rows})


def build(
    tmp_path: Path,
    *,
    frames: int = 3,
    results: list[ChunkValidationResult] | None = None,
    strict: bool = True,
    chunk_size: int | None = None,
) -> tuple[DatasetIngestionService, RecordingLoader, RecordingValidator]:
    """Wire a service over doubles, returning them for inspection."""
    path = tmp_path / FILE_NAME
    source = StubTransactionSource(path, filename=FILE_NAME)
    loader = RecordingLoader([frame() for _ in range(frames)])
    validator = RecordingValidator(
        results or [make_result(index, source=FILE_NAME) for index in range(frames)]
    )
    service = DatasetIngestionService(
        source,
        loader,
        validator,
        strict=strict,
        **({} if chunk_size is None else {"chunk_size": chunk_size}),
    )
    return service, loader, validator


# --- The file is located eagerly ------------------------------------------------


def test_a_missing_dataset_fails_at_the_call_site() -> None:
    """Not on first iteration: a caller should not have to iterate to find out."""
    loader = RecordingLoader([frame()])
    service = DatasetIngestionService(
        StubTransactionSource(None), loader, RecordingValidator([make_result(0)])
    )

    with pytest.raises(DataFileNotFoundError):
        service.stream_validated_chunks()

    assert loader.calls == [], "the loader must not be reached"


def test_the_file_is_resolved_once_per_run(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path)
    source = service._source  # noqa: SLF001 - the double is the subject here
    assert isinstance(source, StubTransactionSource)

    list(service.stream_validated_chunks())

    assert source.resolve_calls == 1


# --- Reading is lazy and bounded ------------------------------------------------


def test_nothing_is_read_until_a_chunk_is_requested(tmp_path: Path) -> None:
    """Only the file lookup is eager; the parser is not even opened.

    ``stream_validated_chunks`` resolves the path and returns a generator, so a
    caller that never iterates causes no reading at all.
    """
    service, loader, validator = build(tmp_path)

    stream = service.stream_validated_chunks()

    assert loader.calls == [], "the loader must not even be asked for a stream yet"
    assert loader.chunks_pulled == 0
    assert validator.calls == []

    next(stream)

    assert loader.calls, "the first step must open the stream"
    assert loader.chunks_pulled == 1
    assert len(validator.calls) == 1


def test_exactly_one_chunk_is_read_per_step(tmp_path: Path) -> None:
    """The direct statement of memory safety through the whole pipeline."""
    service, loader, _ = build(tmp_path, frames=5)

    stream = service.stream_validated_chunks()

    for expected in (1, 2, 3):
        next(stream)
        assert loader.chunks_pulled == expected


def test_an_abandoned_stream_leaves_the_rest_of_the_file_unread(tmp_path: Path) -> None:
    service, loader, _ = build(tmp_path, frames=100)

    next(service.stream_validated_chunks())

    assert loader.chunks_pulled == 1


def test_the_chunks_are_not_concatenated(tmp_path: Path) -> None:
    """Each pair carries one chunk; combining them is the caller's decision."""
    service, _, _ = build(tmp_path, frames=3)

    pairs = list(service.stream_validated_chunks())

    assert len(pairs) == 3
    assert all(len(chunk) == 2 for chunk, _ in pairs)


def test_the_yielded_chunk_is_the_loaders_own_frame(tmp_path: Path) -> None:
    """No copy, no conversion, no cleaning: the service transforms nothing."""
    path = tmp_path / FILE_NAME
    original = frame(3)
    loader = RecordingLoader([original])
    service = DatasetIngestionService(
        StubTransactionSource(path, filename=FILE_NAME),
        loader,
        RecordingValidator([make_result(0, source=FILE_NAME)]),
    )

    chunk, _ = next(service.stream_validated_chunks())

    assert chunk is original


# --- Each chunk is validated with its context ----------------------------------


def test_every_chunk_is_validated(tmp_path: Path) -> None:
    service, _, validator = build(tmp_path, frames=4)

    list(service.stream_validated_chunks())

    assert len(validator.calls) == 4


def test_chunk_indices_are_sequential_from_zero(tmp_path: Path) -> None:
    service, _, validator = build(tmp_path, frames=4)

    list(service.stream_validated_chunks())

    assert [index for index, _, _ in validator.calls] == [0, 1, 2, 3]


def test_the_validator_is_told_which_file_the_chunk_came_from(tmp_path: Path) -> None:
    service, _, validator = build(tmp_path, frames=2)

    list(service.stream_validated_chunks())

    assert {source for _, source, _ in validator.calls} == {FILE_NAME}


def test_each_result_is_paired_with_its_own_chunk(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path, frames=3)

    pairs = list(service.stream_validated_chunks())

    assert [result.chunk_index for _, result in pairs] == [0, 1, 2]


# --- The chunk size is the caller's ---------------------------------------------


def test_the_configured_chunk_size_reaches_the_loader(tmp_path: Path) -> None:
    service, loader, _ = build(tmp_path, chunk_size=250)

    list(service.stream_validated_chunks())

    assert loader.calls[0][1] == 250


def test_the_default_chunk_size_is_the_shared_one(tmp_path: Path) -> None:
    """One default, defined with the port, rather than a second opinion here."""
    service, loader, _ = build(tmp_path)

    list(service.stream_validated_chunks())

    assert loader.calls[0][1] == DEFAULT_CHUNK_SIZE


def test_the_resolved_path_reaches_the_loader(tmp_path: Path) -> None:
    service, loader, _ = build(tmp_path)

    list(service.stream_validated_chunks())

    assert loader.calls[0][0] == tmp_path / FILE_NAME


# --- Strict mode aborts on the first invalid chunk ------------------------------


def invalid_at(index: int, total: int, *, errors: int = 1) -> list[ChunkValidationResult]:
    """Results where one chunk fails validation and the rest pass."""
    return [
        make_result(
            position,
            source=FILE_NAME,
            issues=(
                tuple(make_error_issue(f"column_{n}") for n in range(errors))
                if position == index
                else ()
            ),
        )
        for position in range(total)
    ]


def test_strict_is_the_default(tmp_path: Path) -> None:
    """Training on data that failed its schema produces uninterpretable results."""
    path = tmp_path / FILE_NAME
    service = DatasetIngestionService(
        StubTransactionSource(path, filename=FILE_NAME),
        RecordingLoader([frame()]),
        RecordingValidator(invalid_at(0, 1)),
    )

    with pytest.raises(SchemaValidationError):
        list(service.stream_validated_chunks())


def test_an_invalid_chunk_aborts_the_run(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path, frames=3, results=invalid_at(1, 3))

    with pytest.raises(SchemaValidationError) as raised:
        list(service.stream_validated_chunks())

    assert raised.value.chunk_index == 1
    assert raised.value.source == FILE_NAME


def test_the_abort_stops_reading_immediately(tmp_path: Path) -> None:
    """The remaining chunks must not be read after the decision to stop."""
    service, loader, _ = build(tmp_path, frames=10, results=invalid_at(1, 10))

    with pytest.raises(SchemaValidationError):
        list(service.stream_validated_chunks())

    assert loader.chunks_pulled == 2


def test_the_valid_chunks_before_the_failure_were_still_yielded(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path, frames=4, results=invalid_at(2, 4))
    stream = service.stream_validated_chunks()

    delivered = [next(stream), next(stream)]

    assert [result.chunk_index for _, result in delivered] == [0, 1]
    with pytest.raises(SchemaValidationError):
        next(stream)


def test_the_abort_message_names_the_offending_column(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path, frames=1, results=invalid_at(0, 1))

    with pytest.raises(SchemaValidationError, match="column_0") as raised:
        list(service.stream_validated_chunks())

    assert "chunk 0" in str(raised.value)


def test_the_abort_message_quotes_a_bounded_number_of_findings(tmp_path: Path) -> None:
    """A wholly broken chunk must not produce a page-long exception message."""
    service, _, _ = build(tmp_path, frames=1, results=invalid_at(0, 1, errors=9))

    with pytest.raises(SchemaValidationError) as raised:
        list(service.stream_validated_chunks())

    message = str(raised.value)

    assert message.count("[error]") == 5
    assert "4 further error(s)" in message
    assert "see the log for the full report" in message


def test_warnings_alone_do_not_abort_a_strict_run(tmp_path: Path) -> None:
    """Strictness is about errors; a tolerated deviation is not one."""
    results = [make_result(0, source=FILE_NAME, issues=(make_warning_issue(),))]
    service, _, _ = build(tmp_path, frames=1, results=results)

    pairs = list(service.stream_validated_chunks())

    assert len(pairs) == 1
    assert pairs[0][1].warnings


# --- Non-strict mode reports and continues --------------------------------------


def test_a_non_strict_run_yields_the_invalid_chunk(tmp_path: Path) -> None:
    service, _, _ = build(tmp_path, frames=3, results=invalid_at(1, 3), strict=False)

    pairs = list(service.stream_validated_chunks())

    assert len(pairs) == 3
    assert [result.is_valid for _, result in pairs] == [True, False, True]


def test_a_non_strict_run_hands_the_findings_to_the_caller(tmp_path: Path) -> None:
    """The chunk is yielded *with* its report, so the decision can still be made."""
    service, _, _ = build(tmp_path, frames=2, results=invalid_at(0, 2), strict=False)

    _, result = next(service.stream_validated_chunks())

    assert not result.is_valid
    assert result.errors[0].column == "column_0"


# --- Logging -------------------------------------------------------------------


def test_an_abort_is_logged_once_with_the_full_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Once, not once per layer - the point of logging here rather than below."""
    service, _, _ = build(tmp_path, frames=1, results=invalid_at(0, 1, errors=9))

    with caplog.at_level(logging.ERROR, logger=SERVICE_LOGGER), pytest.raises(
        SchemaValidationError
    ):
        list(service.stream_validated_chunks())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]

    assert len(errors) == 1
    assert errors[0].getMessage().count("[error]") == 9, "the log keeps every finding"


def test_a_valid_run_logs_no_errors_or_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, _ = build(tmp_path, frames=3)

    with caplog.at_level(logging.DEBUG, logger=SERVICE_LOGGER):
        list(service.stream_validated_chunks())

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_run_logs_once_at_the_start_and_once_at_the_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Per-run, not per-row: a 500-chunk file must not produce 500 progress lines."""
    service, _, _ = build(tmp_path, frames=20)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        list(service.stream_validated_chunks())

    assert len(caplog.records) == 2
    assert "Ingestion starting" in caplog.records[0].getMessage()
    assert "Ingestion finished" in caplog.records[1].getMessage()


def test_the_closing_log_line_counts_what_was_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, _ = build(tmp_path, frames=3, results=invalid_at(1, 3), strict=False)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        list(service.stream_validated_chunks())

    finished = caplog.records[-1].getMessage()

    assert "chunks=3" in finished
    assert "rows=6" in finished
    assert "invalid_chunks=1" in finished


def test_a_non_strict_invalid_chunk_is_logged_as_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, _ = build(tmp_path, frames=2, results=invalid_at(0, 2), strict=False)

    with caplog.at_level(logging.WARNING, logger=SERVICE_LOGGER):
        list(service.stream_validated_chunks())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]

    assert len(warnings) == 1
    assert "non-strict" in warnings[0].getMessage()


def test_the_logs_never_contain_the_data(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Findings and counts are logged; transaction rows are not."""
    service, _, _ = build(tmp_path, frames=2, results=invalid_at(0, 2), strict=False)

    with caplog.at_level(logging.DEBUG, logger=SERVICE_LOGGER):
        list(service.stream_validated_chunks())

    text = "\n".join(record.getMessage() for record in caplog.records)

    for forbidden in ("KAGGLE", "password", "token"):
        assert forbidden.lower() not in text.lower()
    assert "to_string" not in text
    assert text.count("\n") < 20, "the log must not grow with the data"
