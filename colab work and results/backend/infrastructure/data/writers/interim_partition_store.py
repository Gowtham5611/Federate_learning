"""Day-partitioned interim storage: how an unsorted file is read in time order.

The pipeline needs the transactions in chronological order, and the raw file is
not in chronological order — that was measured, not assumed. The two obvious
fixes both fail: sorting five million rows in memory defeats the point of
chunked reading, and processing the file as it lies produces account-history
features that are simply wrong.

The way out is an external bucket sort with the calendar as the bucket key.
Streaming the raw file once and appending each row to a file named after its
date leaves one file per day. The verified data spans eighteen days, so each
file holds on the order of a few hundred thousand rows — small enough to load
and sort exactly, in memory, one at a time. Reading the day files in date order
and sorting within each one yields a total chronological order over the whole
dataset, from a peak memory cost of one day rather than one dataset.

Partitions are written under ``data/interim`` and deleted once consumed unless
the caller asks to keep them. They are working files, not results: nothing
downstream should ever read them.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import TextIO

from pandas import DataFrame

logger = logging.getLogger(__name__)

PARTITION_PREFIX = "day="
PARTITION_SUFFIX = ".csv"

REMOVAL_ATTEMPTS = 3
"""How many times a directory removal is retried before giving up.

On Windows a directory whose contents were just deleted can still be held open
briefly by an indexer or a file-sync client, which surfaces as ``WinError 5``
on the final ``rmdir``. The handle is released a moment later, so retrying
converts a spurious failure into a success. This was observed on the real run,
inside a OneDrive-synced working tree.
"""

REMOVAL_RETRY_SECONDS = 0.5


def remove_tree(directory: Path) -> bool:
    """Delete ``directory`` and its contents, retrying a transient lock.

    Args:
        directory: Tree to remove. A path that does not exist counts as removed.

    Returns:
        Whether the tree is gone. ``False`` means something else is holding it;
        the caller decides whether that is fatal.
    """
    for attempt in range(1, REMOVAL_ATTEMPTS + 1):
        if not directory.exists():
            return True
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            if attempt == REMOVAL_ATTEMPTS:
                logger.warning(
                    "Could not remove '%s' after %d attempt(s): %s",
                    directory,
                    attempt,
                    exc,
                )
                return False
            time.sleep(REMOVAL_RETRY_SECONDS)
        else:
            return True
    return not directory.exists()


MAX_OPEN_PARTITIONS = 64
"""Cap on simultaneously open partition files.

Eighteen days need eighteen handles, so the cap never binds on the real
dataset. It exists because the number of partitions is a property of the *data*,
not of the code, and a dataset spanning years would otherwise exhaust the
process's file descriptors halfway through a run.
"""


class InterimDayPartitionStore:
    """Appends rows to one CSV per calendar day, then serves them in date order."""

    def __init__(self, directory: Path, columns: tuple[str, ...]) -> None:
        """Prepare the partition directory.

        Args:
            directory: Where partition files are written. Created if absent.
            columns: Column order written to every partition's header.
        """
        self._directory = directory
        self._columns = tuple(columns)
        self._handles: dict[str, TextIO] = {}
        self._started: set[str] = set()
        self._counts: dict[str, int] = {}
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Where the partitions live."""
        return self._directory

    @property
    def columns(self) -> tuple[str, ...]:
        """Column order every partition carries."""
        return self._columns

    def append(self, day: str, frame: DataFrame) -> int:
        """Append one day's rows to its partition.

        Args:
            day: Partition key, an ISO date such as ``"2022-09-01"``.
            frame: Rows for that day, carrying exactly :attr:`columns`.

        Returns:
            Rows appended.
        """
        if frame.empty:
            return 0
        handle = self._handle_for(day)
        frame.to_csv(handle, header=False, index=False, lineterminator="\n")
        self._counts[day] = self._counts.get(day, 0) + len(frame)
        return len(frame)

    def flush(self) -> None:
        """Close every open partition so they can be read back."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def days(self) -> list[str]:
        """Partition keys in chronological order.

        ISO dates sort lexicographically into date order, so listing the
        directory and sorting is enough — no parsing, and no reliance on the
        order the days happened to be created in.
        """
        return sorted(
            path.name[len(PARTITION_PREFIX) : -len(PARTITION_SUFFIX)]
            for path in self._directory.glob(f"{PARTITION_PREFIX}*{PARTITION_SUFFIX}")
        )

    def path_for(self, day: str) -> Path:
        """Path of one day's partition."""
        return self._directory / f"{PARTITION_PREFIX}{day}{PARTITION_SUFFIX}"

    def rows_in(self, day: str) -> int:
        """Rows this store wrote for ``day``."""
        return self._counts.get(day, 0)

    def discard(self, day: str) -> None:
        """Delete one consumed partition.

        Called as soon as a day has been processed, so peak interim disk usage
        stays near one copy of the dataset rather than growing to two.
        """
        self.path_for(day).unlink(missing_ok=True)

    def remove_all(self) -> bool:
        """Delete the whole partition directory.

        Only ever removes the directory this store created; it never touches a
        path it was not given.

        Returns:
            Whether the directory is gone. A working directory that cannot be
            removed is reported, not raised: by the time this runs the outputs
            are already written and verified, and failing the run over a locked
            scratch file would throw away a correct result.
        """
        self.flush()
        removed = remove_tree(self._directory)
        if removed:
            logger.info("Removed interim partitions at %s", self._directory)
        return removed

    def _handle_for(self, day: str) -> TextIO:
        """Return an open handle for ``day``, respecting the open-file cap."""
        existing = self._handles.get(day)
        if existing is not None:
            return existing

        if len(self._handles) >= MAX_OPEN_PARTITIONS:
            evicted, handle = next(iter(self._handles.items()))
            handle.close()
            del self._handles[evicted]

        path = self.path_for(day)
        first_write = day not in self._started
        handle = path.open("w" if first_write else "a", encoding="utf-8", newline="")
        if first_write:
            handle.write(",".join(self._columns) + "\n")
            self._started.add(day)
        self._handles[day] = handle
        return handle
