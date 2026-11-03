"""Outbound port for writing processed splits and their metadata.

Kept behind a port for one concrete reason rather than on principle: the output
format is the part of this phase most likely to change. Parquet is the natural
target for a dataset of this size, but neither ``pyarrow`` nor ``fastparquet``
is installed in this environment, so the shipped adapter writes CSV. When a
columnar writer becomes available, it replaces the adapter and nothing in the
application layer moves.

The interface is **append-only and streaming**. There is no ``write(frame)``
taking a whole split, because a whole split does not fit in memory and offering
the method would invite exactly that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from pandas import DataFrame

from domain.enums.split_name import SplitName


class ProcessedDataWriterPort(ABC):
    """Accumulates processed rows into per-split outputs, one chunk at a time."""

    @property
    @abstractmethod
    def output_root(self) -> Path:
        """Directory the splits and metadata are written under."""

    @abstractmethod
    def append(self, split: SplitName, frame: DataFrame) -> int:
        """Append rows to one split's output.

        The first call for a split creates its output and writes the header;
        later calls must append without repeating it. Column order must be
        identical across every call for a split — a caller relying on the
        header to describe the file cannot tolerate a reordered chunk.

        Args:
            split: Which split these rows belong to.
            frame: Rows to append. May be empty, in which case nothing is
                written and zero is returned.

        Returns:
            Number of rows appended.
        """

    @abstractmethod
    def write_metadata(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Write one metadata artefact.

        Args:
            name: Artefact name without extension, e.g. ``"feature_metadata"``.
            payload: JSON-serialisable content. Implementations must refuse to
                write credentials; see the adapter for the enforced key list.

        Returns:
            Path the artefact was written to.
        """

    @abstractmethod
    def rows_written(self, split: SplitName) -> int:
        """Rows appended to ``split`` so far."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release every open output. Idempotent."""

    def __enter__(self) -> ProcessedDataWriterPort:
        """Support ``with`` so outputs close even when a run fails."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the outputs on the way out, successfully or not."""
        self.close()
