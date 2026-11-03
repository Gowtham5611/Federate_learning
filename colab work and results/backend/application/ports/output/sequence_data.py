"""Outbound port for turning processed rows into fixed-shape temporal sequences.

A recurrent model needs ``(batch, sequence_length, feature_dimension)``. The
processed splits are flat transaction rows, so something has to decide *which*
rows form a sequence — and that decision is where a temporal model quietly
becomes either meaningful or a shape-satisfying fiction. The contract is stated
here so the decision is visible from the application layer rather than buried in
a data adapter:

1. A sequence is a window over **one entity's** history, in chronological order,
   oldest first. Windows over the global transaction stream are excluded: two
   adjacent rows there usually belong to unrelated accounts, and a model reading
   them as a sequence is reading noise.
2. A sequence ends at the transaction being classified. No element may come from
   after it. This is the same causality rule the Phase 3 history features obey.
3. A sequence never crosses a split boundary, because an implementation builds
   each split's windows from that split's rows alone. Cross-boundary leakage is
   made impossible by construction rather than checked for afterwards.
4. Every architecture receives sequences built the same way, which is what makes
   the GRU, LSTM and BiLSTM baselines comparable.

``numpy`` appears in these signatures. It joins ``pathlib.Path`` and
``pandas.DataFrame`` as the project's third deliberate purity exception: a batch
of numbers has to be *some* type, and a wrapper around ``ndarray`` would be a
layer every consumer immediately unwraps. Keeping numpy rather than torch here is
what lets the sequence machinery be tested, and the application service be
orchestrated, without importing a deep-learning framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from application.dto.experiment import DataProvenance, SequenceStrategy
from application.dto.training import TrainingConfig
from domain.enums.split_name import SplitName


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """One mini-batch of sequences.

    Attributes:
        features: ``(batch, sequence_length, feature_dimension)``, float32.
            Real transactions occupy positions ``0 .. length-1`` in chronological
            order, oldest first, and the remainder is padding.
        lengths: ``(batch,)`` true sequence lengths, each at least 1. Supplied so
            a consumer can mask the padding instead of letting the recurrent
            encoder read it as data — which matters most for the bidirectional
            model, whose backward pass would otherwise start inside the padding.
        labels: ``(batch,)`` float32 targets in ``{0.0, 1.0}``, the label of the
            **last real** transaction in each sequence.
        row_indices: ``(batch,)`` position of that classified transaction within
            its split. Carries traceability back to a specific row — and to its
            timestamp and account identifiers — without those identifiers ever
            entering the feature tensor.
    """

    features: NDArray[np.float32]
    lengths: NDArray[np.int64]
    labels: NDArray[np.float32]
    row_indices: NDArray[np.int64]

    def __len__(self) -> int:
        """Sequences in this batch."""
        return int(self.features.shape[0])


class SequenceDataset(ABC):
    """A split's worth of sequences, iterable in bounded batches.

    Deliberately not indexable. Materialising every sequence would store each
    transaction's features once per window it appears in — for the training split
    that is roughly ``sequence_length`` copies of the split. Implementations are
    expected to keep the rows once and assemble windows per batch.
    """

    @property
    @abstractmethod
    def split(self) -> SplitName:
        """Which split these sequences come from."""

    @property
    @abstractmethod
    def sequence_count(self) -> int:
        """Sequences available, which is the number of classified transactions."""

    @property
    @abstractmethod
    def row_count(self) -> int:
        """Rows read from the split's file.

        Equal to :attr:`sequence_count` at stride 1. A difference between the two
        is the count of labelled transactions a larger stride discarded, and is
        reported rather than left to be inferred.
        """

    @property
    @abstractmethod
    def feature_dimension(self) -> int:
        """Width of the feature axis: what the model's input layer must accept."""

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        """Maximum window length; the size of the time axis."""

    @property
    @abstractmethod
    def labels(self) -> NDArray[np.float32]:
        """Every target, in iteration order at ``shuffle=False``.

        Held separately from the batches so class balance can be reported, and
        metrics reconciled, without a forward pass.
        """

    @property
    def positive_count(self) -> int:
        """Laundering-labelled targets in this split."""
        return int(self.labels.sum())

    @abstractmethod
    def batches(
        self, batch_size: int, *, shuffle: bool = False, seed: int | None = None
    ) -> Iterator[SequenceBatch]:
        """Yield mini-batches.

        Args:
            batch_size: Maximum sequences per batch. The final batch may be
                smaller and is never dropped — discarding it would silently drop
                labelled transactions, and at 0.1% positives possibly some
                positives.
            shuffle: Whether to permute the order of targets. Training shuffles;
                evaluation must not, so that ``labels`` lines up with the
                predictions positionally.
            seed: Seed for the permutation, so a shuffled epoch is reproducible.

        Yields:
            Batches covering every sequence exactly once.
        """


class SequenceDataProviderPort(ABC):
    """Builds sequence datasets from the processed splits.

    An implementation also owns reading the Phase 3 metadata: which columns are
    features, which are identifiers to keep out of the tensor, and which is the
    target. That keeps the answer in one place, read from the artefacts Phase 3
    actually wrote rather than restated as a constant here.
    """

    @abstractmethod
    def load(self, split: SplitName, config: TrainingConfig) -> SequenceDataset:
        """Read one split and build its sequences.

        Args:
            split: Which split to read.
            config: Supplies sequence length, stride, the nominal-code expansion
                choice and any row limit.

        Returns:
            The split's sequences.

        Raises:
            EmptySequenceDatasetError: The split produced no sequences.
            UnsortedSplitError: The split's rows are not in chronological order,
                which would make window order meaningless.
        """

    @abstractmethod
    def describe_features(self, config: TrainingConfig) -> DataProvenance:
        """Report where the data comes from and what the tensor will contain.

        Callable before any split is read, so a run can record its provenance —
        and a caller can check the feature dimension against a checkpoint —
        without paying for the data.
        """

    @abstractmethod
    def describe_strategy(self, config: TrainingConfig) -> SequenceStrategy:
        """Report how rows become sequences, for the experiment record."""
