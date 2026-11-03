"""Preprocessing errors: transforming validated raw data into model-ready data.

These extend :class:`~domain.exceptions.base.FedSecureAIError` alongside the
ingestion errors in :mod:`domain.exceptions.data`, so a caller can still catch
one project-wide base class. They exist as a separate module because they
describe a different failure surface: ingestion fails on *files*, preprocessing
fails on *invariants* — a statistic used before it was fitted, a split boundary
that does not separate time, an output directory that would overwrite the input.
"""

from __future__ import annotations

from pathlib import Path

from domain.exceptions.base import FedSecureAIError


class PreprocessingError(FedSecureAIError):
    """Base class for problems transforming a dataset."""


class TransformerNotFittedError(PreprocessingError):
    """A transformation was applied before its parameters were learned.

    The guard that makes "fit on train only" enforceable rather than merely
    intended: applying an unfitted encoder or scaler is a programming error, and
    silently fitting on whatever data arrived first is exactly the leak this
    project must not have.
    """

    def __init__(self, transformer: str, operation: str = "transform") -> None:
        """Record which transformer was used too early."""
        self.transformer = transformer
        super().__init__(
            f"'{transformer}' cannot {operation}: it has not been fitted. "
            f"Preprocessing statistics are learned from the training split "
            f"only, so fit it on training data before applying it to any split."
        )


class TransformerAlreadyFittedError(PreprocessingError):
    """A fitted transformer was asked to learn from more data.

    Refitting after the training split has been consumed would fold validation
    or test statistics into the parameters, which is a leak whether or not it
    was intended.
    """

    def __init__(self, transformer: str) -> None:
        """Record which transformer was re-fitted."""
        self.transformer = transformer
        super().__init__(
            f"'{transformer}' is already fitted and cannot learn from further "
            f"data. Fitting again would mix statistics from outside the "
            f"training split into the learned parameters."
        )


class InvalidSplitFractionsError(PreprocessingError):
    """Requested split proportions cannot produce three non-empty splits."""

    def __init__(self, train_fraction: float, validation_fraction: float) -> None:
        """Record the rejected proportions."""
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        super().__init__(
            f"Invalid split fractions: train={train_fraction!r}, "
            f"validation={validation_fraction!r}. Both must be greater than 0 "
            f"and their sum must be less than 1 so that a non-empty test split "
            f"remains."
        )


class ChronologyViolationError(PreprocessingError):
    """A split boundary does not separate the splits in time.

    Raised when the produced splits overlap chronologically — the failure that
    would make every downstream metric optimistic and unexplainable.
    """

    def __init__(self, earlier: str, later: str, detail: str) -> None:
        """Record which pair of splits overlapped."""
        self.earlier = earlier
        self.later = later
        super().__init__(
            f"Chronological ordering violated between the '{earlier}' and "
            f"'{later}' splits: {detail}. Every '{earlier}' transaction must "
            f"occur no later than every '{later}' transaction."
        )


class NotEnoughDataToSplitError(PreprocessingError):
    """The dataset cannot be divided into three non-empty chronological splits.

    Distinct from bad fractions: the proportions are usable but the data is too
    small or too concentrated in time — every row sharing a single timestamp
    cannot be split chronologically at all.
    """

    def __init__(self, row_count: int, distinct_timestamps: int) -> None:
        """Record how little there was to work with."""
        self.row_count = row_count
        self.distinct_timestamps = distinct_timestamps
        super().__init__(
            f"Cannot form three non-empty chronological splits from "
            f"{row_count} row(s) spanning {distinct_timestamps} distinct "
            f"timestamp(s). A chronological split cuts between timestamps, so "
            f"at least 3 distinct timestamps are required and each split must "
            f"receive at least one row."
        )


class RawDatasetModifiedError(PreprocessingError):
    """The read-only source dataset changed while it was being read.

    The pipeline never writes to its input; this detects the case where
    something else did, which would make the produced splits describe a file
    that no longer exists.
    """

    def __init__(self, path: Path | str, before: str, after: str) -> None:
        """Record the file and the two digests that disagreed."""
        self.path = Path(path)
        self.before = before
        self.after = after
        super().__init__(
            f"The raw dataset '{self.path}' changed during preprocessing: its "
            f"SHA-256 was {before[:16]}... before the run and {after[:16]}... "
            f"after. The raw dataset is read-only; the outputs of this run "
            f"describe neither version reliably and should be discarded."
        )


class OutputWouldOverwriteInputError(PreprocessingError):
    """A configured output path resolves inside the raw dataset directory.

    A configuration mistake rather than a data problem, caught before anything
    is written: the raw dataset is the one artefact that must survive every run
    untouched.
    """

    def __init__(self, output_path: Path | str, raw_path: Path | str) -> None:
        """Record the offending pair of paths."""
        self.output_path = Path(output_path)
        self.raw_path = Path(raw_path)
        super().__init__(
            f"Refusing to write to '{self.output_path}': it is inside the raw "
            f"dataset location '{self.raw_path}'. Processed data must be "
            f"written elsewhere so the downloaded dataset is never overwritten."
        )
