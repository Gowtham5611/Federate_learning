"""Outbound port for fitting a sequence classifier and scoring a split.

One engine trains all three architectures. The alternative — a loop per
architecture — would let the GRU and the LSTM baselines drift apart in ways that
have nothing to do with the architectures, which is exactly the comparison this
project is built to make.

The port is also shaped for the later phases without implementing them. A
federated client's local training round is *this* ``fit`` call, on that client's
own dataset, starting from the parameters the server sent. So ``fit`` accepts
optional starting parameters and returns the fitted ones, which is equally what
warm-starting and resuming need. Nothing here aggregates, averages or
communicates: choosing what to do with several clients' returned parameters is a
later phase's decision, and there is no code for it yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from application.dto.training import EpochRecord, TrainingConfig
from application.ports.output.sequence_data import SequenceDataset

ParameterState = Mapping[str, Any]
"""Opaque handle on a model's parameters.

Deliberately untyped beyond "a mapping of names to values". The application layer
passes these through — from a checkpoint into ``fit``, from ``fit`` into a
checkpoint — and never inspects them, so the framework's tensor type stays inside
the adapter.
"""

EpochCallback = Callable[[EpochRecord], None]
"""Invoked once per completed epoch, for logging and progress reporting."""


@dataclass(frozen=True, slots=True)
class SplitPredictions:
    """Scores for one split, aligned positionally with the dataset's labels.

    Attributes:
        labels: ``(n,)`` true targets, in dataset order.
        probabilities: ``(n,)`` predicted probability of the positive class.
            Sigmoid has already been applied here and nowhere else: the model
            emits logits, and this is the one place they become probabilities.
        row_indices: ``(n,)`` row of the classified transaction within its split,
            so any prediction can be traced back to a timestamp and an account.
        mean_loss: Mean binary cross-entropy over the split, computed from the
            logits rather than from these probabilities.
        seconds: Wall-clock scoring time.
    """

    labels: NDArray[np.float32]
    probabilities: NDArray[np.float32]
    row_indices: NDArray[np.int64]
    mean_loss: float
    seconds: float

    def __len__(self) -> int:
        """Predictions made."""
        return int(self.labels.shape[0])


@dataclass(frozen=True, slots=True)
class FitOutcome:
    """What a completed fit produced.

    Attributes:
        epochs: One record per completed epoch, in order.
        best_epoch: Epoch whose parameters were kept, one-based.
        best_state: Parameters from that epoch. What a later federated round
            would send back to a server, and what evaluation scores.
        final_state: Parameters after the last epoch. Kept distinct from
            ``best_state`` so the two checkpoints can never be confused: with
            early stopping they are usually different models.
        parameter_count: Trainable parameters.
        device: Device the fit ran on.
        early_stopped: Whether the run ended before its epoch budget.
        training_seconds: Fitting time, data preparation excluded.
        deterministic: Whether bitwise-reproducible kernels were successfully
            requested. Reported honestly; ``False`` is a normal outcome.
    """

    epochs: tuple[EpochRecord, ...]
    best_epoch: int
    best_state: ParameterState
    final_state: ParameterState
    parameter_count: int
    device: str
    early_stopped: bool
    training_seconds: float
    deterministic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingEnginePort(ABC):
    """Fits a sequence classifier and scores a dataset with it."""

    @abstractmethod
    def fit(
        self,
        config: TrainingConfig,
        train: SequenceDataset,
        validation: SequenceDataset,
        *,
        initial_parameters: ParameterState | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> FitOutcome:
        """Train on ``train``, measuring on ``validation`` after each epoch.

        The validation split is used for two things and no others: choosing which
        epoch's parameters to keep, and deciding when to stop early. The test
        split is not an argument to this method, which is the structural half of
        "never tune on test".

        Args:
            config: Hyperparameters. Also fixes the seed.
            train: Sequences to fit on.
            validation: Sequences to measure on. Never trained on.
            initial_parameters: Parameters to start from instead of a fresh
                initialisation. Used for resuming; also what a federated round
                would supply.
            on_epoch: Called with each epoch's record as it completes.

        Returns:
            The fitted parameters and the training curve.

        Raises:
            FeatureDimensionMismatchError: ``train`` and ``validation`` disagree
                on feature width, or supplied parameters describe another width.
            NoCompletedEpochError: No epoch finished.
        """

    @abstractmethod
    def predict(
        self,
        config: TrainingConfig,
        dataset: SequenceDataset,
        parameters: ParameterState,
    ) -> SplitPredictions:
        """Score every sequence in ``dataset`` with fixed parameters.

        No gradients, no parameter updates, no shuffling: the returned arrays line
        up positionally with ``dataset.labels``.

        Args:
            config: Supplies architecture, batch size and device.
            dataset: Sequences to score.
            parameters: Frozen parameters to score with.

        Returns:
            Labels, probabilities and the mean loss.
        """

    @abstractmethod
    def count_parameters(self, config: TrainingConfig, feature_dimension: int) -> int:
        """Trainable parameters a model built from ``config`` would have.

        Answerable without training, so a report can state model size even for a
        run that fails later.
        """
