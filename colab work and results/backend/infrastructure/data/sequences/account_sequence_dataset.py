"""Per-account temporal sequences over one split's rows.

The dataset holds the split's features **once** and assembles windows on demand.
The alternative — materialising every window — stores each transaction's features
once per window it appears in, roughly ``sequence_length`` copies of the split. At
3,047,216 training rows, 57 features and length 10 that is about 7 GB, on a
machine with under 5 GB free. So the windows are described by an index instead of
being stored.

The index is one array: ``previous_row[i]`` is the row of the same account's
immediately preceding transaction, or ``-1`` if there is none. Walking it back
``sequence_length - 1`` steps from a target reproduces that account's recent
history exactly, and costs one 4-byte integer per row rather than a tensor.

Three properties follow from the construction rather than from a later check:

* No element of a sequence comes from after its target, because the walk only
  ever moves backwards.
* No sequence crosses a split boundary, because the index is built from one
  split's rows.
* Every element of a sequence belongs to the same account, because the index
  links same-account rows only.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from application.ports.output.sequence_data import SequenceBatch, SequenceDataset
from domain.enums.split_name import SplitName

NO_PREVIOUS_ROW = -1
"""Sentinel in ``previous_row`` for an account's first transaction in the split."""


class AccountSequenceDataset(SequenceDataset):
    """One split's sequences, one window per classified transaction."""

    def __init__(
        self,
        *,
        split: SplitName,
        features: NDArray[np.float32],
        targets: NDArray[np.float32],
        previous_row: NDArray[np.int64],
        sequence_length: int,
        stride: int,
    ) -> None:
        """Hold a split's features and the index that describes its windows.

        Args:
            split: Which split these rows are.
            features: ``(rows, feature_dimension)`` float32 model inputs. Holds no
                identifier columns and no target column.
            targets: ``(rows,)`` float32 labels.
            previous_row: ``(rows,)`` row of the same account's previous
                transaction, or :data:`NO_PREVIOUS_ROW`.
            sequence_length: Maximum window length.
            stride: Emit a target every ``stride`` rows.

        Raises:
            ValueError: The arrays disagree on the number of rows.
        """
        rows = int(features.shape[0])
        if targets.shape[0] != rows or previous_row.shape[0] != rows:
            raise ValueError(
                f"features has {rows} row(s) but targets has {targets.shape[0]} and "
                f"previous_row has {previous_row.shape[0]}; all three describe the "
                f"same rows and must be the same length"
            )

        self._split = split
        self._features = features
        self._targets = targets
        self._previous = previous_row
        self._length = sequence_length
        self._stride = stride
        self._row_count = rows
        # Stride 1 - the default and what every reported baseline uses - selects
        # every row, so no labelled transaction is dropped.
        self._target_rows: NDArray[np.int64] = np.arange(
            0, rows, stride, dtype=np.int64
        )
        self._labels: NDArray[np.float32] = targets[self._target_rows]

    # --- Description ---------------------------------------------------------

    @property
    def split(self) -> SplitName:
        """Which split these sequences come from."""
        return self._split

    @property
    def sequence_count(self) -> int:
        """Sequences available."""
        return int(self._target_rows.shape[0])

    @property
    def row_count(self) -> int:
        """Rows read from the split's file."""
        return self._row_count

    @property
    def feature_dimension(self) -> int:
        """Width of the feature axis."""
        return int(self._features.shape[1])

    @property
    def sequence_length(self) -> int:
        """Maximum window length."""
        return self._length

    @property
    def labels(self) -> NDArray[np.float32]:
        """Every target, in unshuffled iteration order."""
        return self._labels

    @property
    def target_rows(self) -> NDArray[np.int64]:
        """Split-relative row of each classified transaction, in label order."""
        return self._target_rows

    def history_lengths(self) -> NDArray[np.int64]:
        """True length of every sequence, for reporting how much history exists.

        Computed by walking the whole index once, so it is a measurement of the
        data rather than an assumption that accounts have long histories.
        """
        return self._walk(self._target_rows)[1]

    # --- Iteration -----------------------------------------------------------

    def batches(
        self, batch_size: int, *, shuffle: bool = False, seed: int | None = None
    ) -> Iterator[SequenceBatch]:
        """Yield mini-batches, assembling each window as it is needed.

        Args:
            batch_size: Maximum sequences per batch. The last batch is smaller
                rather than dropped.
            shuffle: Permute target order. Training does; evaluation must not.
            seed: Seed for the permutation.

        Yields:
            Batches covering every sequence exactly once.

        Raises:
            ValueError: ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size!r}")

        order = self._target_rows
        if shuffle:
            generator = np.random.default_rng(seed)
            order = generator.permutation(order)

        total = int(order.shape[0])
        for start in range(0, total, batch_size):
            yield self._assemble(order[start : start + batch_size])

    def _assemble(self, target_rows: NDArray[np.int64]) -> SequenceBatch:
        """Build one batch's tensor from the index."""
        walk, lengths = self._walk(target_rows)

        # walk[:, k] is k steps back from the target. The batch wants position j
        # counted forwards from the oldest real element, so position j of a
        # sequence of length L is (L - 1 - j) steps back. Where that is negative
        # the position is padding.
        positions = np.arange(self._length, dtype=np.int64)
        steps = lengths[:, None] - 1 - positions[None, :]
        real = steps >= 0
        rows = np.take_along_axis(walk, np.maximum(steps, 0), axis=1)

        features = self._features[np.maximum(rows, 0)]
        # Padding is written as zeros as well as being masked by `lengths`. Both,
        # not either: `lengths` is what a packed sequence uses, and the zeros mean
        # that an implementation which ignored `lengths` would read neutral values
        # rather than another account's transactions.
        features[~real] = 0.0

        return SequenceBatch(
            features=features,
            lengths=lengths,
            labels=self._targets[target_rows],
            row_indices=target_rows,
        )

    def _walk(
        self, target_rows: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Follow the previous-row index back from each target.

        Returns:
            ``walk`` of shape ``(n, sequence_length)`` where column ``k`` is the
            row ``k`` steps before the target (:data:`NO_PREVIOUS_ROW` once the
            account's history runs out), and ``lengths`` counting the real rows in
            each row of ``walk``.
        """
        count = int(target_rows.shape[0])
        walk = np.full((count, self._length), NO_PREVIOUS_ROW, dtype=np.int64)
        walk[:, 0] = target_rows

        current = target_rows
        for step in range(1, self._length):
            # np.maximum keeps the gather in bounds for rows that have already run
            # out of history; np.where then discards whatever it read.
            current = np.where(
                current >= 0, self._previous[np.maximum(current, 0)], NO_PREVIOUS_ROW
            )
            walk[:, step] = current

        lengths = (walk >= 0).sum(axis=1).astype(np.int64)
        return walk, lengths


def build_previous_row_index(account_codes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Link each row to the same account's previous row.

    Args:
        account_codes: ``(rows,)`` integer account identity per row, in the file's
            chronological order.

    Returns:
        ``(rows,)`` index where entry ``i`` is the row of the previous transaction
        by the same account, or :data:`NO_PREVIOUS_ROW`.

    A stable sort groups each account's rows together while preserving their
    relative order, so within a group the predecessor is simply the entry before.
    Vectorised because the alternative — a Python dictionary walk — takes seconds
    per split at three million rows and is executed on every run.
    """
    rows = int(account_codes.shape[0])
    previous = np.full(rows, NO_PREVIOUS_ROW, dtype=np.int64)
    if rows == 0:
        return previous

    grouped = np.argsort(account_codes, kind="stable")
    same_account = account_codes[grouped[1:]] == account_codes[grouped[:-1]]
    previous[grouped[1:]] = np.where(same_account, grouped[:-1], NO_PREVIOUS_ROW)
    return previous
