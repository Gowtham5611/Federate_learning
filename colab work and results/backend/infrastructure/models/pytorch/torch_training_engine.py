"""One PyTorch training engine, shared by all three architectures.

There is a single training loop in this project. Three loops — one per
architecture — would be the obvious way to write this and the wrong one: the
comparison this project exists to make is between GRU, LSTM and BiLSTM on
identical treatment, and three loops drift. A learning-rate change applied to two
of them, a shuffle seeded differently in the third, and the measured difference
between architectures is partly a difference between code paths.

The same reasoning shapes the loop for what comes later. A federated client's
local training round is this ``fit`` call on that client's own data, starting from
parameters the server sent, returning the parameters it reached. So ``fit`` takes
optional starting parameters, returns state rather than mutating a hidden model,
and holds no reference to a "global" dataset. Nothing here aggregates or
communicates, and nothing here knows that clients will exist.

Loss is unweighted ``BCEWithLogitsLoss``. The 0.1% positive rate is the phenomenon
these baselines are meant to measure, not a defect to correct before measuring it;
cost-sensitive weighting belongs to the later phase that studies it, and applying
it here would leave nothing to compare that phase against.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from application.dto.experiment import ClassificationMetrics
from application.dto.training import EpochRecord, TrainingConfig
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    EpochCallback,
    FitOutcome,
    ParameterState,
    SplitPredictions,
    TrainingEnginePort,
)
from application.ports.output.sequence_data import SequenceBatch, SequenceDataset
from domain.enums.training_mode import SelectionMetric
from domain.exceptions.training import (
    FeatureDimensionMismatchError,
    NoCompletedEpochError,
)
from infrastructure.models.pytorch.reproducibility import (
    describe_runtime,
    request_deterministic_kernels,
    resolve_device,
    seed_everything,
)
from infrastructure.models.pytorch.sequence_classifier import (
    RecurrentSequenceClassifier,
    build_classifier,
)

logger = logging.getLogger(__name__)

INPUT_WEIGHT_KEY = "encoder.weight_ih_l0"
"""State-dict entry whose second axis is the model's input width.

Used to check supplied parameters against the data before loading them, so a
stale checkpoint produces a named error instead of a shape failure from inside
PyTorch.
"""

PROGRESS_FRACTIONS = 10
"""Progress lines per epoch.

A full training epoch here is thousands of batches and minutes long. Ten lines is
enough to see that a run is alive and moving without turning the log into a
transcript.
"""

_OptimizerFactory = Callable[
    [Iterable[nn.Parameter], float], torch.optim.Optimizer
]
"""Signature the optimiser table is used through: parameters, then learning rate.

Typed as a callable rather than as ``type[Optimizer]`` because the base class does
not accept a learning rate — only its subclasses do, each as their second argument.
"""

_OPTIMIZERS: dict[str, _OptimizerFactory] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


class TorchTrainingEngine(TrainingEnginePort):
    """Fits and scores recurrent classifiers with PyTorch."""

    def __init__(self, metrics: MetricsCalculatorPort) -> None:
        """Wire the metrics calculator used for per-epoch validation.

        Args:
            metrics: Computes the validation metrics that choose the best epoch.
                Injected rather than computed here so that epoch selection and the
                final report use the same definitions of recall and PR-AUC.
        """
        self._metrics = metrics

    # --- Fitting -------------------------------------------------------------

    def fit(
        self,
        config: TrainingConfig,
        train: SequenceDataset,
        validation: SequenceDataset,
        *,
        initial_parameters: ParameterState | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> FitOutcome:
        """Train on ``train``, selecting the best epoch by validation score.

        Raises:
            FeatureDimensionMismatchError: The splits disagree on feature width, or
                ``initial_parameters`` was trained on a different width.
            NoCompletedEpochError: The epoch budget produced no completed epoch.
        """
        if train.feature_dimension != validation.feature_dimension:
            raise FeatureDimensionMismatchError(
                train.feature_dimension,
                validation.feature_dimension,
                f"the '{validation.split.value}' split",
            )

        seed_everything(config.random_seed)
        deterministic = request_deterministic_kernels()
        device = resolve_device(config.device)
        runtime = describe_runtime(device, deterministic=deterministic)

        model = build_classifier(config, train.feature_dimension)
        if initial_parameters is not None:
            self._load_into(model, initial_parameters, train.feature_dimension)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = self._build_optimizer(config, model)
        logger.info(
            "Training %s on %s: %d parameter(s), %d sequence(s), batch size %d, "
            "%s, lr=%g, torch %s",
            type(model).__name__,
            device,
            model.parameter_count(),
            train.sequence_count,
            config.batch_size,
            config.optimizer,
            config.learning_rate,
            runtime["framework_version"],
        )

        records: list[EpochRecord] = []
        best_state: ParameterState = self._snapshot(model)
        final_state: ParameterState = best_state
        best_score: float | None = None
        best_epoch = 0
        stale_epochs = 0
        early_stopped = False
        started = time.perf_counter()

        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = self._train_one_epoch(
                config, model, train, criterion, optimizer, epoch
            )
            scored = self._score(config, model, validation, device)
            measured = self._metrics.evaluate(
                scored.labels,
                scored.probabilities,
                threshold=config.threshold,
                loss=scored.mean_loss,
            )
            score = _selection_value(config.selection_metric, measured, scored.mean_loss)
            improved = best_score is None or _is_better(
                config.selection_metric, score, best_score
            )

            final_state = self._snapshot(model)
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = final_state
                stale_epochs = 0
            else:
                stale_epochs += 1

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=scored.mean_loss,
                validation_metrics=_epoch_metrics(measured, config.selection_metric, score),
                seconds=time.perf_counter() - epoch_started,
                # "Best so far" while the run is in progress. Rewritten below once
                # the run has ended, because only one epoch becomes the checkpoint.
                selected=improved,
            )
            records.append(record)
            if on_epoch is not None:
                on_epoch(record)

            if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
                # Only a stop that skipped epochs is an early stop. Patience
                # reached on the final epoch of the budget ended nothing.
                early_stopped = epoch < config.epochs
                logger.info(
                    "Stopping early after epoch %d: %d epoch(s) without an "
                    "improvement in validation %s (patience %d). Epoch %d stays "
                    "selected.",
                    epoch,
                    stale_epochs,
                    config.selection_metric.value,
                    config.early_stopping_patience,
                    best_epoch,
                )
                break

        if not records:
            raise NoCompletedEpochError(config.architecture.value)

        return FitOutcome(
            epochs=_mark_selected(records, best_epoch),
            best_epoch=best_epoch,
            best_state=best_state,
            final_state=final_state,
            parameter_count=model.parameter_count(),
            device=str(device),
            early_stopped=early_stopped,
            training_seconds=time.perf_counter() - started,
            deterministic=deterministic,
            extra=dict(runtime),
        )

    def _train_one_epoch(
        self,
        config: TrainingConfig,
        model: RecurrentSequenceClassifier,
        train: SequenceDataset,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """Run one pass over the training split and return its mean loss."""
        device = next(model.parameters()).device
        model.train()

        total_batches = max(1, -(-train.sequence_count // config.batch_size))
        report_every = max(1, total_batches // PROGRESS_FRACTIONS)
        loss_sum = 0.0
        seen = 0

        # The shuffle seed advances with the epoch, so each epoch sees a different
        # order while the whole run stays reproducible from one seed.
        batches = train.batches(
            config.batch_size, shuffle=True, seed=config.random_seed + epoch
        )
        for index, batch in enumerate(batches, start=1):
            features, lengths, labels = _to_torch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            count = len(batch)
            # Weighted by batch size: the final batch is usually smaller, and a
            # plain mean of batch means would quietly overweight it.
            loss_sum += float(loss.detach()) * count
            seen += count
            if index % report_every == 0 or index == total_batches:
                logger.info(
                    "  epoch %d: batch %d/%d, running train loss %.6f",
                    epoch,
                    index,
                    total_batches,
                    loss_sum / seen,
                )

        return loss_sum / seen if seen else 0.0

    # --- Scoring -------------------------------------------------------------

    def predict(
        self,
        config: TrainingConfig,
        dataset: SequenceDataset,
        parameters: ParameterState,
    ) -> SplitPredictions:
        """Score a split with frozen parameters.

        Raises:
            FeatureDimensionMismatchError: ``parameters`` expects a different
                feature width than ``dataset`` provides.
        """
        device = resolve_device(config.device)
        model = build_classifier(config, dataset.feature_dimension)
        self._load_into(model, parameters, dataset.feature_dimension)
        model.to(device)
        return self._score(config, model, dataset, device)

    def _score(
        self,
        config: TrainingConfig,
        model: RecurrentSequenceClassifier,
        dataset: SequenceDataset,
        device: torch.device,
    ) -> SplitPredictions:
        """Run the model over a split without gradients or shuffling."""
        model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        started = time.perf_counter()

        probabilities: list[NDArray[np.float32]] = []
        labels: list[NDArray[np.float32]] = []
        row_indices: list[NDArray[np.int64]] = []
        loss_sum = 0.0
        seen = 0

        with torch.no_grad():
            # shuffle=False: the returned arrays must line up positionally with
            # dataset.labels, which is what lets a prediction be traced back to a
            # row and what makes the metrics reproducible.
            for batch in dataset.batches(config.batch_size, shuffle=False):
                features, lengths, batch_labels = _to_torch(batch, device)
                logits = model(features, lengths)
                loss_sum += float(criterion(logits, batch_labels))
                # The only sigmoid in the project. The model emits logits; the loss
                # consumes logits; probabilities exist for evaluation alone.
                probabilities.append(
                    torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                )
                labels.append(batch.labels)
                row_indices.append(batch.row_indices)
                seen += len(batch)

        return SplitPredictions(
            labels=np.concatenate(labels) if labels else np.empty(0, dtype=np.float32),
            probabilities=(
                np.concatenate(probabilities)
                if probabilities
                else np.empty(0, dtype=np.float32)
            ),
            row_indices=(
                np.concatenate(row_indices)
                if row_indices
                else np.empty(0, dtype=np.int64)
            ),
            mean_loss=loss_sum / seen if seen else 0.0,
            seconds=time.perf_counter() - started,
        )

    def count_parameters(self, config: TrainingConfig, feature_dimension: int) -> int:
        """Trainable parameters a model from this configuration would have."""
        return build_classifier(config, feature_dimension).parameter_count()

    # --- Helpers -------------------------------------------------------------

    @staticmethod
    def _build_optimizer(
        config: TrainingConfig, model: RecurrentSequenceClassifier
    ) -> torch.optim.Optimizer:
        """Construct the configured optimiser.

        Raises:
            ValueError: The name is not one of the supported optimisers.
        """
        name = config.optimizer.strip().lower()
        factory = _OPTIMIZERS.get(name)
        if factory is None:
            supported = ", ".join(sorted(_OPTIMIZERS))
            raise ValueError(
                f"unknown optimizer {config.optimizer!r}; supported: {supported}"
            )
        return factory(model.parameters(), config.learning_rate)

    @staticmethod
    def _snapshot(model: RecurrentSequenceClassifier) -> ParameterState:
        """Detached CPU copy of the model's parameters.

        Cloned rather than referenced: ``state_dict`` hands back live tensors, so a
        stored "best" state that was not copied would keep changing as training
        continued and would no longer be the parameters whose score was reported.
        """
        return {
            name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in model.state_dict().items()
        }

    @staticmethod
    def _load_into(
        model: RecurrentSequenceClassifier,
        parameters: ParameterState,
        feature_dimension: int,
    ) -> None:
        """Load parameters after checking they describe this data's width.

        Raises:
            FeatureDimensionMismatchError: The stored input width differs.
        """
        stored = parameters.get(INPUT_WEIGHT_KEY)
        if isinstance(stored, Tensor) and stored.dim() == 2:
            expected = int(stored.shape[1])
            if expected != feature_dimension:
                raise FeatureDimensionMismatchError(
                    expected, feature_dimension, "the supplied parameters"
                )
        model.load_state_dict(dict(parameters))


def _to_torch(
    batch: SequenceBatch, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Move one batch onto the device.

    ``from_numpy`` shares memory with the numpy array rather than copying it, which
    matters at this batch rate; ``to(device)`` copies only when the device differs.
    """
    features = torch.from_numpy(batch.features).to(device)
    lengths = torch.from_numpy(batch.lengths)
    labels = torch.from_numpy(batch.labels).to(device)
    return features, lengths, labels


def _mark_selected(
    records: list[EpochRecord], best_epoch: int
) -> tuple[EpochRecord, ...]:
    """Flag exactly the one epoch that became the checkpoint.

    During the run ``selected`` means "best so far", because that is all a
    callback watching the curve can know. Once the run has ended only one epoch is
    the checkpoint, so the earlier bests are cleared: a curve with three selected
    rows would not say which parameters were saved.
    """
    return tuple(
        dataclasses.replace(record, selected=record.epoch == best_epoch)
        for record in records
    )


def _selection_value(
    metric: SelectionMetric, measured: ClassificationMetrics, loss: float
) -> float:
    """Extract the quantity that decides which epoch is best."""
    if metric is SelectionMetric.LOSS:
        return loss
    if metric is SelectionMetric.RECALL:
        return measured.recall
    if metric is SelectionMetric.F1:
        return measured.f1
    return measured.pr_auc


def _is_better(metric: SelectionMetric, candidate: float, incumbent: float) -> bool:
    """Whether ``candidate`` beats ``incumbent`` under this metric's direction."""
    if metric.higher_is_better:
        return candidate > incumbent
    return candidate < incumbent


def _epoch_metrics(
    measured: ClassificationMetrics, metric: SelectionMetric, score: float
) -> dict[str, float]:
    """Validation metrics recorded for one epoch of the training curve.

    Includes the selection score under its own key as well as the metric it came
    from, so the curve shows what the choice of best epoch was actually based on.
    """
    values = {
        name: float(value)
        for name, value in measured.to_dict().items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    values[f"selection_{metric.value}"] = score
    return values
