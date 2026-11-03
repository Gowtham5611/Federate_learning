"""Outbound port for reading a tabular data file without loading it whole."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path

from pandas import DataFrame

DEFAULT_CHUNK_SIZE = 100_000
"""Rows per chunk when the caller does not choose.

Sized to keep one chunk small relative to a typical development machine's RAM
while still amortising per-chunk overhead. It is a default, not a measured
optimum: no memory figure is claimed for it anywhere in this project.
"""


class ChunkedDataLoaderPort(ABC):
    """The single tabular-loading abstraction the application depends on.

    The contract is *streaming*: an implementation yields the file piece by
    piece and never materialises it in full. The selected IBM AML variant is a
    single multi-hundred-megabyte CSV, and the same code will later run inside
    several simulated federated clients in one process, so whole-file reads are
    excluded by design rather than by discipline.

    ``DataFrame`` appears in this signature knowingly. The alternative — a
    project-defined table type wrapping pandas — would add an abstraction that
    every consumer immediately unwraps, and the project's rules forbid
    unnecessary layers. This mirrors the existing precedent of ``pathlib.Path``
    in ``DownloadRequest``: a stable, ubiquitous type is treated as vocabulary
    rather than as a leaked dependency.
    """

    @abstractmethod
    def stream_chunks(
        self,
        file_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        dtypes: Mapping[str, str] | None = None,
    ) -> Iterator[DataFrame]:
        """Yield the file's rows in successive chunks of at most ``chunk_size``.

        Implementations must be lazy: opening the file and parsing the first
        chunk happen when the first chunk is requested, not when this method is
        called. They must not accumulate or concatenate chunks — a caller that
        wants the whole file must ask for it explicitly and accept the cost.

        Column names must be preserved exactly as they appear in the file.
        Renaming, type conversion and cleaning are preprocessing concerns and do
        not belong here.

        Args:
            file_path: File to read.
            chunk_size: Maximum rows per chunk. Must be a positive integer.
            dtypes: Optional column-to-dtype mapping passed to the parser. The
                default of ``None`` keeps the historical behaviour — the parser
                infers types **per chunk**, which is safe for validation because
                the validator coerces rather than trusting dtypes. Preprocessing
                needs the stronger guarantee: an identifier column that happens
                to hold only digits in one chunk must not become an integer
                there and text elsewhere, because the two forms compare
                unequal. Passing explicit dtypes is how a caller asks for that
                guarantee; it is not a performance option.

        Yields:
            One frame per chunk, in file order. A file whose header parses but
            which contains no data rows yields nothing.

        Raises:
            InvalidChunkSizeError: ``chunk_size`` is not a positive integer.
            DataFileNotFoundError: The file does not exist.
            EmptyDataFileError: The file contains nothing parseable.
            MalformedDataError: The file is not well-formed CSV.
            UnreadableDataFileError: The file could not be read.
        """
