"""CSV-backed interim storage.

Composes :class:`~infrastructure.data.writers.interim_partition_store.InterimDayPartitionStore`
for the day partitions and a single append-only file for the staged training
rows. Both are plain CSV for the same reason the processed output is: no
columnar writer is installed in this environment, and this phase does not add a
dependency it can do without.

All reads go through the project's :class:`ChunkedDataLoaderPort`, so the
interim files are subject to the same bounded-reading contract as the raw
dataset — including :meth:`read_partition`, which is bounded not by the loader
but by the choice of a *day* as the partition key.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TextIO

import pandas as pd
from pandas import DataFrame

from application.ports.output.data_loader import ChunkedDataLoaderPort
from application.ports.output.interim_storage import InterimStoragePort
from domain.exceptions.data import EmptyDataFileError
from infrastructure.data.writers.interim_partition_store import (
    InterimDayPartitionStore,
    remove_tree,
)

logger = logging.getLogger(__name__)

PARTITIONS_SUBDIRECTORY = "partitions"
STAGED_FILENAME = "staged_train.csv"


class CsvInterimStorage(InterimStoragePort):
    """Day-partitioned CSV working storage with an append-only staging file."""

    def __init__(
        self,
        directory: Path,
        loader: ChunkedDataLoaderPort,
        *,
        partition_dtypes: Mapping[str, str] | None = None,
    ) -> None:
        """Prepare the working directory.

        Args:
            directory: Root for the working files. Created if absent.
            loader: Used for every read, so nothing here bypasses the project's
                chunked-reading contract.
            partition_dtypes: Explicit types for reading partitions back.
                Supplied by the caller because the columns being partitioned are
                the caller's concern; identifier columns in particular must be
                read as text or they will not compare equal across files.
        """
        self._directory = directory
        self._loader = loader
        self._partition_dtypes = dict(partition_dtypes) if partition_dtypes else None
        self._partitions: InterimDayPartitionStore | None = None
        self._staged_handle: TextIO | None = None
        self._staged_columns: tuple[str, ...] = ()
        self._staged_rows = 0

    @property
    def directory(self) -> Path:
        """Root directory the working files live in."""
        return self._directory

    @property
    def staged_path(self) -> Path:
        """Where staged training rows are written."""
        return self._directory / STAGED_FILENAME

    # --- Day partitions ------------------------------------------------------

    def begin_partitions(self, columns: tuple[str, ...]) -> None:
        """Start a partitioning pass, fixing the partition column order."""
        self._partitions = InterimDayPartitionStore(
            self._directory / PARTITIONS_SUBDIRECTORY, columns
        )

    def add_to_partition(self, key: str, frame: DataFrame) -> int:
        """Append rows to the partition named ``key``."""
        return self._require_partitions().append(key, frame)

    def end_partitions(self) -> None:
        """Close every partition file so they can be read back."""
        self._require_partitions().flush()

    def partition_keys(self) -> list[str]:
        """Partition keys in ascending order."""
        return self._require_partitions().days()

    def read_partition(self, key: str) -> DataFrame:
        """Read one whole day partition.

        Uses a chunk size larger than any single day can be and concatenates
        what comes back — normally one chunk. The concatenation is the
        pipeline's one deliberate materialisation, and it is bounded by the size
        of a day rather than the size of the dataset.
        """
        path = self._require_partitions().path_for(key)
        chunks = list(self._loader.stream_chunks(path, 2_000_000, dtypes=self._partition_dtypes))
        if not chunks:
            return DataFrame(columns=list(self._require_partitions().columns))
        return chunks[0] if len(chunks) == 1 else pd.concat(chunks, ignore_index=True)

    def drop_partition(self, key: str) -> None:
        """Delete a consumed partition."""
        self._require_partitions().discard(key)

    # --- Staged rows ---------------------------------------------------------

    def stage(self, frame: DataFrame) -> int:
        """Append engineered training rows to the staging file."""
        if frame.empty:
            return 0

        columns = tuple(frame.columns)
        if self._staged_handle is None:
            self.staged_path.parent.mkdir(parents=True, exist_ok=True)
            self._staged_handle = self.staged_path.open("w", encoding="utf-8", newline="")
            self._staged_handle.write(",".join(columns) + "\n")
            self._staged_columns = columns
        elif columns != self._staged_columns:
            raise ValueError(
                f"staged column order changed: opened with {self._staged_columns}, "
                f"received {columns}"
            )

        frame.to_csv(self._staged_handle, header=False, index=False, lineterminator="\n")
        self._staged_rows += len(frame)
        return len(frame)

    def end_staging(self) -> None:
        """Close the staging file so it can be streamed back."""
        if self._staged_handle is not None:
            self._staged_handle.close()
            self._staged_handle = None

    def read_staged(self, chunk_size: int) -> Iterator[DataFrame]:
        """Stream the staged rows back in bounded chunks, in the order written.

        Yields nothing if no rows were ever staged, rather than failing: a run
        over data with no training rows is a data problem the caller reports,
        not a storage error.
        """
        if not self.staged_path.is_file():
            return
        try:
            yield from self._loader.stream_chunks(
                self.staged_path, chunk_size, dtypes=self._partition_dtypes
            )
        except EmptyDataFileError:
            return

    def staged_row_count(self) -> int:
        """Rows staged so far."""
        return self._staged_rows

    # --- Disposal ------------------------------------------------------------

    def cleanup(self) -> None:
        """Delete every working file this instance created. Idempotent.

        Never raises. This runs after the processed splits have been written and
        the raw file re-verified, so the run has already succeeded; a scratch
        file that some other process is holding open is worth a warning, not a
        failed run. Whatever survives is named in the log so it can be deleted
        by hand, and it is inert either way — nothing downstream reads it.
        """
        self.end_staging()
        leftovers: list[Path] = []

        if self._partitions is not None:
            if not self._partitions.remove_all():
                leftovers.append(self._partitions.directory)
            self._partitions = None

        try:
            self.staged_path.unlink(missing_ok=True)
        except OSError:
            leftovers.append(self.staged_path)

        if not leftovers and self._directory.is_dir() and not any(self._directory.iterdir()):
            remove_tree(self._directory)

        if leftovers:
            logger.warning(
                "Interim working files could not be removed and can be deleted "
                "by hand: %s",
                ", ".join(str(path) for path in leftovers),
            )
        else:
            logger.info("Interim working files removed from %s", self._directory)

    def _require_partitions(self) -> InterimDayPartitionStore:
        """Return the partition store, or explain that the pass never started."""
        if self._partitions is None:
            raise RuntimeError(
                "begin_partitions() must be called before partitions are used"
            )
        return self._partitions
