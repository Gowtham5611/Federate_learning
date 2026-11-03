"""Training configuration and per-epoch history.

Two immutable records. :class:`TrainingConfig` is the complete description of an
experiment's hyperparameters — complete in the strong sense that two runs sharing
a config and a seed are intended to produce the same model, so every knob that
could change the outcome is a field here rather than a default buried in an
adapter. :class:`EpochRecord` is one row of the training curve.

Both are frozen and slotted for the same reason the preprocessing DTOs are: a
configuration that a training loop can mutate halfway through is a configuration
that no longer describes the run it produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from domain.enums.model_architecture import ModelArchitecture
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy

DEFAULT_SEQUENCE_LENGTH = 10
"""Transactions per sequence, including the one being classified.

Ten is a choice, not a measured optimum, and nothing in this project claims
otherwise. It is long enough for a short behavioural pattern in one account's
history to be visible and short enough that every one of the 3,047,216 training
targets keeps its own sequence on a CPU-only machine.
"""

DEFAULT_SEQUENCE_STRIDE = 1
"""Emit a sequence for every transaction.

Any larger value discards labelled transactions. With 0.1019% positives that
would also change the effective class balance, which
:mod:`application.dto.experiment` would then be reporting for a dataset nobody
described. Configurable because the spec asks for it; left at 1 for every
reported baseline.
"""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Every hyperparameter that defines one centralized experiment.

    Attributes:
        architecture: Which recurrent encoder to train.
        sequence_length: Maximum transactions per sequence, newest last.
        sequence_stride: Emit a target every ``stride`` transactions.
        hidden_size: Recurrent hidden width per direction.
        num_layers: Stacked recurrent layers.
        dropout: Dropout between recurrent layers. Ignored by PyTorch when
            ``num_layers == 1``, which is why :meth:`effective_dropout` reports
            what actually applies rather than what was requested.
        batch_size: Sequences per optimisation step.
        learning_rate: Optimiser step size.
        optimizer: Optimiser name, resolved by the training engine.
        epochs: Maximum passes over the training split.
        random_seed: Seed for Python, NumPy and PyTorch.
        early_stopping_patience: Epochs without improvement before stopping.
            Zero disables early stopping.
        selection_metric: Validation quantity that picks the best checkpoint.
        threshold: Decision threshold, and the value used as-is under
            :attr:`ThresholdPolicy.FIXED`.
        threshold_policy: How the threshold applied to test is arrived at.
        one_hot_nominal_codes: Expand the three nominal code columns into
            indicator columns. Phase 3 recorded this expansion as deferred to the
            modelling phase precisely because the codes carry no ordering.
        gradient_clip_norm: Global gradient-norm cap, or ``None`` for no clipping.
        row_limit: Read at most this many rows per split. Present so that a
            deliberately reduced smoke run is recorded as reduced and can never
            be mistaken for a full one. ``None`` means the whole split.
        num_workers: Data-loading worker processes. Zero loads in-process.
        device: Torch device string, or ``None`` to select automatically.
    """

    architecture: ModelArchitecture
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    sequence_stride: int = DEFAULT_SEQUENCE_STRIDE
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    epochs: int = 5
    random_seed: int = 42
    early_stopping_patience: int = 2
    selection_metric: SelectionMetric = SelectionMetric.PR_AUC
    threshold: float = 0.5
    threshold_policy: ThresholdPolicy = ThresholdPolicy.VALIDATION_F1
    one_hot_nominal_codes: bool = True
    gradient_clip_norm: float | None = 5.0
    row_limit: int | None = None
    num_workers: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        """Reject values that cannot describe a runnable experiment.

        Checked here rather than at first use so that a bad configuration fails
        before any data is read, in the same spirit as the processed-data writer
        refusing a bad output directory at construction.
        """
        self._require_positive("sequence_length", self.sequence_length)
        self._require_positive("sequence_stride", self.sequence_stride)
        self._require_positive("hidden_size", self.hidden_size)
        self._require_positive("num_layers", self.num_layers)
        self._require_positive("batch_size", self.batch_size)
        self._require_positive("epochs", self.epochs)

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate!r}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1); got {self.threshold!r}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience cannot be negative; "
                f"got {self.early_stopping_patience!r}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative; got {self.num_workers!r}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError(
                f"gradient_clip_norm must be positive or None; "
                f"got {self.gradient_clip_norm!r}"
            )
        if self.row_limit is not None and self.row_limit <= 0:
            raise ValueError(f"row_limit must be positive or None; got {self.row_limit!r}")

    @staticmethod
    def _require_positive(name: str, value: int) -> None:
        """Reject a non-positive integer field."""
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")

    @property
    def effective_dropout(self) -> float:
        """Dropout that actually applies.

        PyTorch ignores ``dropout`` on a single-layer RNN and warns about it. The
        reported figure is what the model does, not what was asked for, so a
        result record cannot claim regularisation that never happened.
        """
        return self.dropout if self.num_layers > 1 else 0.0

    @property
    def is_reduced_run(self) -> bool:
        """Whether this configuration deliberately reads less than the whole split."""
        return self.row_limit is not None

    def with_architecture(self, architecture: ModelArchitecture) -> TrainingConfig:
        """Return a copy targeting a different architecture.

        How ``--model all`` guarantees the three baselines are scientifically
        comparable: one configuration object, three architectures, nothing else
        varied. Any per-architecture tuning would have to go through this method
        and would therefore be visible.
        """
        return TrainingConfig(
            architecture=architecture,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            epochs=self.epochs,
            random_seed=self.random_seed,
            early_stopping_patience=self.early_stopping_patience,
            selection_metric=self.selection_metric,
            threshold=self.threshold,
            threshold_policy=self.threshold_policy,
            one_hot_nominal_codes=self.one_hot_nominal_codes,
            gradient_clip_norm=self.gradient_clip_norm,
            row_limit=self.row_limit,
            num_workers=self.num_workers,
            device=self.device,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingConfig:
        """Rebuild a configuration from :meth:`to_dict`.

        Needed to read a checkpoint back: a stored model is only reusable if the
        configuration that produced it can be reconstructed exactly. Keys that
        :meth:`to_dict` derives rather than stores — ``effective_dropout`` — are
        ignored, and anything absent falls back to the field default, so a
        checkpoint written before a field existed still loads.

        Raises:
            KeyError: ``architecture`` is absent. It has no sensible default: a
                guess would produce a model of the wrong kind.
            ValueError: A stored enum value is not a member, or a value fails the
                same validation a fresh configuration would face.
        """
        return cls(
            architecture=ModelArchitecture(payload["architecture"]),
            sequence_length=int(payload.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)),
            sequence_stride=int(payload.get("sequence_stride", DEFAULT_SEQUENCE_STRIDE)),
            hidden_size=int(payload.get("hidden_size", 64)),
            num_layers=int(payload.get("num_layers", 1)),
            dropout=float(payload.get("dropout", 0.0)),
            batch_size=int(payload.get("batch_size", 1024)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            optimizer=str(payload.get("optimizer", "adam")),
            epochs=int(payload.get("epochs", 5)),
            random_seed=int(payload.get("random_seed", 42)),
            early_stopping_patience=int(payload.get("early_stopping_patience", 2)),
            selection_metric=SelectionMetric(
                payload.get("selection_metric", SelectionMetric.PR_AUC.value)
            ),
            threshold=float(payload.get("threshold", 0.5)),
            threshold_policy=ThresholdPolicy(
                payload.get("threshold_policy", ThresholdPolicy.VALIDATION_F1.value)
            ),
            one_hot_nominal_codes=bool(payload.get("one_hot_nominal_codes", True)),
            gradient_clip_norm=_optional_float(payload.get("gradient_clip_norm", 5.0)),
            row_limit=_optional_int(payload.get("row_limit")),
            num_workers=int(payload.get("num_workers", 0)),
            device=_optional_str(payload.get("device")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the experiment record and the checkpoint."""
        return {
            "architecture": self.architecture.value,
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "effective_dropout": self.effective_dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "random_seed": self.random_seed,
            "early_stopping_patience": self.early_stopping_patience,
            "selection_metric": self.selection_metric.value,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy.value,
            "one_hot_nominal_codes": self.one_hot_nominal_codes,
            "gradient_clip_norm": self.gradient_clip_norm,
            "row_limit": self.row_limit,
            "num_workers": self.num_workers,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One epoch's measurements: the training curve, one row at a time.

    Attributes:
        epoch: One-based epoch number.
        train_loss: Mean training loss over the epoch.
        validation_loss: Mean validation loss.
        validation_metrics: Threshold-free and threshold-based validation
            metrics, keyed by metric name.
        seconds: Wall-clock time for the epoch, training and validation together.
        selected: Whether this epoch became the best checkpoint.
    """

    epoch: int
    train_loss: float
    validation_loss: float
    validation_metrics: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the experiment record."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_metrics": dict(self.validation_metrics),
            "seconds": self.seconds,
            "selected": self.selected,
        }


def _optional_float(value: Any) -> float | None:
    """Coerce a stored value to a float, preserving ``None``.

    ``None`` is meaningful in these fields — no gradient clipping, no row limit, no
    pinned device — so it must survive a round trip rather than becoming ``0.0``.
    """
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    """Coerce a stored value to an int, preserving ``None``."""
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    """Coerce a stored value to a non-empty string, preserving ``None``."""
    if value is None:
        return None
    text = str(value)
    return text or None

