"""Outbound port for the bounded working storage a multi-pass pipeline needs.

Preprocessing this dataset cannot be done in one pass. The raw file is not in
time order, so chronological features and a chronological split both need the
rows re-ordered first; and scaling parameters cannot be applied to the training
rows until every training row has contributed to them. Both problems are solved
by writing intermediate results to disk and reading them back, which is what
this port abstracts.

Two distinct kinds of working file are involved and the interface names them
separately rather than pretending they are one thing:

*Partitions* are the raw rows bucketed by calendar day. They exist to turn an
unordered file into an ordered stream, and are consumed day by day.

*Staged* rows are engineered training rows waiting for the scaler to finish
being fitted. They exist because a statistic cannot be applied before it is
complete.

Everything written here is disposable. The port offers no way to read a working
file after :meth:`cleanup`, because nothing outside the run should ever depend
on one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from pandas import DataFrame


class InterimStoragePort(ABC):
    """Disposable, bounded storage for intermediate preprocessing results."""

    @property
    @abstractmethod
    def directory(self) -> Path:
        """Root directory the working files live in."""

    # --- Day partitions ------------------------------------------------------

    @abstractmethod
    def begin_partitions(self, columns: tuple[str, ...]) -> None:
        """Start a partitioning pass, fixing the column order of every partition."""

    @abstractmethod
    def add_to_partition(self, key: str, frame: DataFrame) -> int:
        """Append rows to the partition named ``key``.

        Returns:
            Rows appended.
        """

    @abstractmethod
    def end_partitions(self) -> None:
        """Finish the partitioning pass so the partitions can be read back."""

    @abstractmethod
    def partition_keys(self) -> list[str]:
        """Partition keys in ascending order."""

    @abstractmethod
    def read_partition(self, key: str) -> DataFrame:
        """Read one whole partition into memory.

        This is the pipeline's single deliberate whole-object read, and the
        reason partitions are days: one day of the verified dataset is a few
        hundred thousand rows, which is a bounded and modest amount of memory,
        whereas the dataset is five million.
        """

    @abstractmethod
    def drop_partition(self, key: str) -> None:
        """Delete a consumed partition, keeping peak disk usage down."""

    # --- Staged rows ---------------------------------------------------------

    @abstractmethod
    def stage(self, frame: DataFrame) -> int:
        """Append engineered training rows awaiting a fitted transformation.

        Returns:
            Rows appended.
        """

    @abstractmethod
    def end_staging(self) -> None:
        """Finish staging so the rows can be streamed back."""

    @abstractmethod
    def read_staged(self, chunk_size: int) -> Iterator[DataFrame]:
        """Stream the staged rows back in bounded chunks, in the order written."""

    @abstractmethod
    def staged_row_count(self) -> int:
        """Rows staged so far."""

    # --- Disposal ------------------------------------------------------------

    @abstractmethod
    def cleanup(self) -> None:
        """Delete every working file this instance created. Idempotent."""
