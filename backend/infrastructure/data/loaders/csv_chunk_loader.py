"""Memory-safe CSV reader: streams a file in chunks, never in full.

The only place in the project that calls pandas' CSV parser. Everything above it
receives frames one chunk at a time and translates parser failures into the
project's error contract, so no other module imports ``pandas.errors``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from application.ports.output.data_loader import DEFAULT_CHUNK_SIZE, ChunkedDataLoaderPort
from domain.exceptions.data import (
    DataFileNotFoundError,
    EmptyDataFileError,
    InvalidChunkSizeError,
    MalformedDataError,
    UnreadableDataFileError,
)

logger = logging.getLogger(__name__)


class CsvChunkLoader(ChunkedDataLoaderPort):
    """Streams a CSV file in bounded chunks.

    Memory safety is structural, not advisory. ``pandas.read_csv`` is only ever
    called with ``chunksize`` set, which returns a reader that parses on demand
    instead of a frame containing the file; the loader yields those pieces
    straight through and holds no reference to any it has already emitted. There
    is no code path here that reads a whole file, so the selected IBM AML variant
    cannot be loaded entirely into memory by accident.

    No RAM figure is claimed for this. Peak usage depends on chunk size, column
    count and inferred dtypes, and this project records measured numbers only.

    Known pandas behaviour that callers must expect: dtypes are inferred **per
    chunk**, so one column can arrive as ``int64`` in one chunk and as text in
    the next if a later chunk contains a non-numeric value. Validation therefore
    checks values by coercion rather than trusting the parser's dtype, and any
    dtype normalisation belongs to preprocessing. A caller that needs a column's
    type to be identical in every chunk — preprocessing does, for identifier
    columns — passes ``dtypes`` explicitly.
    """

    def __init__(self, *, encoding: str = "utf-8") -> None:
        """Configure the reader.

        Args:
            encoding: Text encoding of the files to read. The project does not
                assume the IBM AML file's encoding; UTF-8 is the default because
                it is the safe first attempt, and a decode failure is reported as
                a malformed file rather than silently replaced.
        """
        self._encoding = encoding

    def stream_chunks(
        self,
        file_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        dtypes: Mapping[str, str] | None = None,
    ) -> Iterator[DataFrame]:
        """Yield the file's rows in successive chunks.

        Preconditions are checked immediately so a bad call fails at the call
        site; parsing is deferred to the returned iterator so nothing is read
        until a chunk is actually requested.

        Args:
            file_path: CSV file to read.
            chunk_size: Maximum rows per chunk.
            dtypes: Optional explicit column types. Omitted by default, which
                leaves the per-chunk inference described above in place;
                supplying it makes a column's type identical in every chunk,
                which preprocessing requires for identifier columns.

        Returns:
            A lazy iterator over chunks.

        Raises:
            InvalidChunkSizeError: ``chunk_size`` is not a positive integer.
            DataFileNotFoundError: The path does not exist.
            EmptyDataFileError: The file is zero bytes.
            UnreadableDataFileError: The path is not a regular file.
        """
        self._validate_chunk_size(chunk_size)
        self._validate_file(file_path)
        return self._iterate(file_path, chunk_size, dtypes)

    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> None:
        """Reject a chunk size that would defeat chunking.

        ``bool`` is excluded explicitly: ``True`` is a valid ``int`` of value 1
        in Python, and accepting it would silently produce a one-row-per-chunk
        stream from what is obviously a mistaken call.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise InvalidChunkSizeError(chunk_size)
        if chunk_size <= 0:
            raise InvalidChunkSizeError(chunk_size)

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        """Check the path is a readable, non-empty regular file."""
        if not file_path.exists():
            raise DataFileNotFoundError(file_path)
        if not file_path.is_file():
            raise UnreadableDataFileError(file_path, "the path is not a regular file")
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            raise UnreadableDataFileError(file_path, str(exc)) from exc
        if size == 0:
            raise EmptyDataFileError(file_path, "the file is zero bytes")

    def _iterate(
        self, file_path: Path, chunk_size: int, dtypes: Mapping[str, str] | None = None
    ) -> Iterator[DataFrame]:
        """Parse and yield chunks, translating parser failures.

        ``EmptyDataError`` is caught before ``ParserError`` because pandas
        derives the former from the latter.
        """
        logger.info(
            "Streaming '%s' in chunks of %d row(s)", file_path.name, chunk_size
        )
        chunks = 0
        rows = 0
        try:
            # chunksize is what makes this bounded; on_bad_lines states the
            # strictness explicitly rather than relying on the pandas default.
            # Note: with chunk_size == 1 the C parser tolerates a row with too
            # many fields instead of failing, so a malformed file may only be
            # detected at larger chunk sizes.
            with pd.read_csv(
                file_path,
                chunksize=chunk_size,
                encoding=self._encoding,
                on_bad_lines="error",
                dtype=dict(dtypes) if dtypes else None,
            ) as reader:
                for frame in reader:
                    if frame.empty:
                        # A trailing or header-only chunk carries no rows.
                        continue
                    chunks += 1
                    rows += len(frame)
                    logger.debug(
                        "Chunk %d of '%s': %d row(s), %d column(s)",
                        chunks - 1,
                        file_path.name,
                        len(frame),
                        len(frame.columns),
                    )
                    yield frame
        except pd.errors.EmptyDataError as exc:
            raise EmptyDataFileError(file_path, str(exc)) from exc
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise MalformedDataError(file_path, str(exc)) from exc
        except OSError as exc:
            raise UnreadableDataFileError(file_path, str(exc)) from exc

        if chunks == 0:
            logger.warning(
                "File '%s' parsed successfully but contains no data rows.",
                file_path.name,
            )
        else:
            logger.info(
                "Finished streaming '%s': %d chunk(s), %d row(s)",
                file_path.name,
                chunks,
                rows,
            )
