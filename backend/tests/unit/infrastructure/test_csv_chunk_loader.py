"""Tests for the memory-safe CSV loader.

The central claim under test is *structural* memory safety: there is no code path
in the loader that reads a whole file. That is checked directly, by recording
every call the loader makes to the CSV parser and asserting each one was chunked
- not by measuring RAM, which this project does not claim figures for.

Fixtures are small synthetic files written to ``tmp_path``. The real dataset is
never used, and no file here is named after it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from application.ports.output.data_loader import DEFAULT_CHUNK_SIZE
from domain.exceptions.data import (
    DataFileNotFoundError,
    EmptyDataFileError,
    InvalidChunkSizeError,
    MalformedDataError,
    UnreadableDataFileError,
)
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from tests.fixtures.tabular_fixtures import (
    SYNTHETIC_COLUMNS,
    SYNTHETIC_HEADER,
    synthetic_row,
    write_csv,
    write_synthetic_csv,
)

pytestmark = pytest.mark.unit


class ReadCsvSpy:
    """Records every call to ``pandas.read_csv`` and delegates to the real one.

    Patched onto ``pandas`` itself rather than onto a name inside the loader, so
    a read from anywhere in the call chain is recorded.
    """

    def __init__(self) -> None:
        """Capture the genuine parser before it is replaced."""
        self.calls: list[dict[str, Any]] = []
        self._real = pd.read_csv

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record the keyword arguments, then parse for real."""
        self.calls.append(dict(kwargs))
        return self._real(*args, **kwargs)


# --- 7. Reads a valid CSV -----------------------------------------------------


def test_reads_a_valid_csv(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=5)

    chunks = list(CsvChunkLoader().stream_chunks(path, chunk_size=10))

    assert len(chunks) == 1
    assert len(chunks[0]) == 5
    assert chunks[0]["record_id"].tolist() == [1000, 1001, 1002, 1003, 1004]


# --- 8. Streams multiple chunks ----------------------------------------------


def test_streams_multiple_chunks(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=6)

    chunks = list(CsvChunkLoader().stream_chunks(path, chunk_size=2))

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [2, 2, 2]


def test_chunks_are_not_concatenated_and_together_cover_every_row(tmp_path: Path) -> None:
    """Each chunk is a distinct slice; the caller decides whether to combine them."""
    path = write_synthetic_csv(tmp_path / "records.csv", rows=7)

    chunks = list(CsvChunkLoader().stream_chunks(path, chunk_size=3))

    assert [len(chunk) for chunk in chunks] == [3, 3, 1], "a short final chunk is expected"
    seen = [value for chunk in chunks for value in chunk["record_id"].tolist()]
    assert seen == list(range(1000, 1007))
    assert len(seen) == len(set(seen)), "no row may appear in two chunks"


# --- 9. Respects the configured chunk size -----------------------------------


@pytest.mark.parametrize(
    ("chunk_size", "expected"),
    [(1, [1] * 6), (4, [4, 2]), (6, [6]), (99, [6])],
)
def test_respects_the_configured_chunk_size(
    tmp_path: Path, chunk_size: int, expected: list[int]
) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=6)

    chunks = list(CsvChunkLoader().stream_chunks(path, chunk_size=chunk_size))

    assert [len(chunk) for chunk in chunks] == expected


def test_the_default_chunk_size_is_used_when_none_is_given(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=3)
    spy = ReadCsvSpy()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("pandas.read_csv", spy)
        list(CsvChunkLoader().stream_chunks(path))

    assert spy.calls[0]["chunksize"] == DEFAULT_CHUNK_SIZE


# --- 10. Does not eagerly load the entire file -------------------------------


def test_nothing_is_parsed_until_the_first_chunk_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=6)
    spy = ReadCsvSpy()
    monkeypatch.setattr("pandas.read_csv", spy)

    stream = CsvChunkLoader().stream_chunks(path, chunk_size=2)
    assert spy.calls == [], "calling stream_chunks must not read the file"

    first = next(stream)

    assert len(first) == 2
    assert len(spy.calls) == 1


def test_every_parser_call_is_chunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-file read path must not exist, not merely be unused."""
    path = write_synthetic_csv(tmp_path / "records.csv", rows=6)
    spy = ReadCsvSpy()
    monkeypatch.setattr("pandas.read_csv", spy)

    list(CsvChunkLoader().stream_chunks(path, chunk_size=2))

    assert spy.calls, "the parser was never called"
    for call in spy.calls:
        assert call.get("chunksize"), f"unchunked read_csv call: {call}"


def test_one_chunk_never_contains_the_whole_file(tmp_path: Path) -> None:
    """The direct statement of memory safety: a caller cannot receive the file."""
    path = write_synthetic_csv(tmp_path / "records.csv", rows=500)

    first = next(CsvChunkLoader().stream_chunks(path, chunk_size=10))

    assert len(first) == 10, "the caller received more than one chunk's worth of rows"


# --- 11. Missing file --------------------------------------------------------


def test_a_missing_file_is_reported_immediately(tmp_path: Path) -> None:
    """The error arrives at the call site, not on first iteration."""
    with pytest.raises(DataFileNotFoundError, match="not found"):
        CsvChunkLoader().stream_chunks(tmp_path / "absent.csv", chunk_size=10)


def test_a_directory_is_not_mistaken_for_a_missing_file(tmp_path: Path) -> None:
    directory = tmp_path / "records.csv"
    directory.mkdir()

    with pytest.raises(UnreadableDataFileError, match="not a regular file"):
        CsvChunkLoader().stream_chunks(directory, chunk_size=10)


# --- 12. Empty file ----------------------------------------------------------


def test_a_zero_byte_file_is_reported_as_empty(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "records.csv", [])

    with pytest.raises(EmptyDataFileError, match="zero bytes"):
        CsvChunkLoader().stream_chunks(path, chunk_size=10)


def test_a_file_with_no_parseable_columns_is_reported_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("\n\n\n", encoding="utf-8")

    with pytest.raises(EmptyDataFileError):
        list(CsvChunkLoader().stream_chunks(path, chunk_size=10))


def test_a_header_only_file_yields_no_chunks(tmp_path: Path) -> None:
    """Structurally valid but rowless: not an error, and not a silent empty frame."""
    path = write_csv(tmp_path / "records.csv", [SYNTHETIC_HEADER])

    assert list(CsvChunkLoader().stream_chunks(path, chunk_size=10)) == []


# --- 13. Malformed CSV -------------------------------------------------------


def test_an_unterminated_quote_is_reported_as_malformed(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "records.csv", ["a,b", '1,"unclosed', "3,4"])

    with pytest.raises(MalformedDataError, match="malformed"):
        list(CsvChunkLoader().stream_chunks(path, chunk_size=2))


def test_a_row_with_too_many_fields_is_reported_as_malformed(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "records.csv", ["a,b", "1,2", "3,4,5"])

    with pytest.raises(MalformedDataError, match="Expected 2 fields"):
        list(CsvChunkLoader().stream_chunks(path, chunk_size=2))


def test_undecodable_bytes_are_reported_as_malformed(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_bytes(b"a,b\n1,\xff\xfe\n")

    with pytest.raises(MalformedDataError):
        list(CsvChunkLoader().stream_chunks(path, chunk_size=2))


def test_a_malformed_file_is_still_detected_when_only_part_is_read(tmp_path: Path) -> None:
    """A defect anywhere in the file must not be swallowed by chunking."""
    lines = [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(4)), '1,"unclosed']
    path = write_csv(tmp_path / "records.csv", lines)

    with pytest.raises(MalformedDataError):
        list(CsvChunkLoader().stream_chunks(path, chunk_size=2))


# --- 14. Preserves column names ----------------------------------------------


def test_column_names_are_preserved_exactly(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=4)

    for chunk in CsvChunkLoader().stream_chunks(path, chunk_size=2):
        assert tuple(chunk.columns) == SYNTHETIC_COLUMNS


def test_awkward_column_names_are_not_normalised(tmp_path: Path) -> None:
    """Renaming headers is preprocessing; the loader must not do it.

    The names below are invented awkward cases - spaces, double spaces, mixed
    case - chosen so that nothing here resembles or implies the real IBM AML
    header, which this repository does not know.
    """
    header = "Some Column,Another  Column,mixed_Case"
    path = write_csv(tmp_path / "records.csv", [header, "1,2,3"])

    chunk = next(CsvChunkLoader().stream_chunks(path, chunk_size=10))

    assert tuple(chunk.columns) == ("Some Column", "Another  Column", "mixed_Case")


# --- 15. Invalid chunk size --------------------------------------------------


@pytest.mark.parametrize("chunk_size", [0, -1, -1000, True, 2.5, "100", None])
def test_an_invalid_chunk_size_is_rejected(tmp_path: Path, chunk_size: Any) -> None:
    """``True`` is included on purpose: it is an ``int`` of value 1 in Python."""
    path = write_synthetic_csv(tmp_path / "records.csv", rows=2)

    with pytest.raises(InvalidChunkSizeError):
        CsvChunkLoader().stream_chunks(path, chunk_size=chunk_size)


def test_the_chunk_size_is_checked_before_the_file(tmp_path: Path) -> None:
    """A bad call is a bad call whether or not the data happens to be present."""
    with pytest.raises(InvalidChunkSizeError):
        CsvChunkLoader().stream_chunks(tmp_path / "absent.csv", chunk_size=0)


# --- Determinism -------------------------------------------------------------


def test_iteration_is_deterministic_across_passes(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / "records.csv", rows=7)
    loader = CsvChunkLoader()

    first = list(loader.stream_chunks(path, chunk_size=3))
    second = list(loader.stream_chunks(path, chunk_size=3))

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.equals(right)
