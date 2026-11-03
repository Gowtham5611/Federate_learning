"""Training and modelling errors: turning processed data into a fitted model.

A third failure surface alongside :mod:`domain.exceptions.data` (files) and
:mod:`domain.exceptions.preprocessing` (invariants of a transformation). These
describe invariants of an *experiment*: a tensor whose width does not match the
model that must consume it, a threshold frozen from the wrong split, a checkpoint
that cannot be reloaded into the architecture that wrote it.

Every one of them is raised instead of continuing with a plausible-looking
substitute, because a silently degraded experiment produces numbers that look
like results.
"""

from __future__ import annotations

from pathlib import Path

from domain.exceptions.base import FedSecureAIError


class TrainingError(FedSecureAIError):
    """Base class for problems training or evaluating a model."""


class FeatureDimensionMismatchError(TrainingError):
    """The data's feature width differs from the width the model expects.

    Almost always a stale checkpoint or a changed feature set rather than a bug
    in the tensors. PyTorch would raise its own shape error deeper in the call
    stack; this one names the two numbers and where each came from.
    """

    def __init__(self, expected: int, received: int, source: str) -> None:
        """Record the disagreement and what produced the unexpected width."""
        self.expected = expected
        self.received = received
        self.source = source
        super().__init__(
            f"Feature dimension mismatch: the model expects {expected} input "
            f"feature(s) but {source} supplies {received}. The processed feature "
            f"set and the model configuration must describe the same data; a "
            f"checkpoint trained on a different feature set cannot be reused."
        )


class EmptySequenceDatasetError(TrainingError):
    """A split produced no training sequences at all.

    Distinct from an empty file: the split may hold rows and still yield nothing
    if every candidate target was filtered out, which is a configuration mistake
    worth stopping for rather than training on nothing.
    """

    def __init__(self, split: str, rows: int) -> None:
        """Record which split came back empty and how many rows it held."""
        self.split = split
        self.rows = rows
        super().__init__(
            f"The '{split}' split produced no sequences from {rows} row(s). "
            f"Check the configured sequence length and stride: a stride larger "
            f"than the split, or a length filter that excludes every row, leaves "
            f"nothing to train on or evaluate against."
        )


class UnsortedSplitError(TrainingError):
    """A split's rows are not in chronological order.

    Sequence construction reads order from row position, so an unsorted split
    would silently build sequences out of time order — the leakage this project
    exists to prevent, arriving through the back door.
    """

    def __init__(self, split: str, first_offending_row: int) -> None:
        """Record which split broke order and where."""
        self.split = split
        self.first_offending_row = first_offending_row
        super().__init__(
            f"The '{split}' split is not in chronological order: row "
            f"{first_offending_row} has an earlier timestamp than the row before "
            f"it. Sequences are built from row order, so an unsorted split would "
            f"place a later transaction before an earlier one in the same "
            f"sequence."
        )


class ThresholdFittedOnWrongSplitError(TrainingError):
    """A decision threshold was selected using something other than validation.

    The structural guard behind "never tune on test". Selecting a threshold is
    fitting, and fitting is only ever allowed on validation once training is
    done, so the attempt is refused rather than logged.
    """

    def __init__(self, split: str) -> None:
        """Record the split that was wrongly offered for threshold selection."""
        self.split = split
        super().__init__(
            f"Refusing to select a decision threshold from the '{split}' split. "
            f"A threshold is a fitted parameter and may only be chosen on the "
            f"validation split; the test split must stay untouched until the "
            f"threshold is frozen."
        )


class CheckpointNotFoundError(TrainingError):
    """A checkpoint expected on disk is not there."""

    def __init__(self, path: Path | str) -> None:
        """Record the missing path."""
        self.path = Path(path)
        super().__init__(
            f"No checkpoint at '{self.path}'. Train the model first, or point "
            f"the configuration at the run whose checkpoint you meant."
        )


class CheckpointArchitectureMismatchError(TrainingError):
    """A checkpoint was written by a different architecture than the one loading it.

    Loading GRU weights into an LSTM would either fail on key names or, worse,
    partially succeed. Both are caught by comparing the architecture recorded in
    the checkpoint against the model asking for it.
    """

    def __init__(self, expected: str, found: str, path: Path | str) -> None:
        """Record both architectures and the checkpoint they disagree about."""
        self.expected = expected
        self.found = found
        self.path = Path(path)
        super().__init__(
            f"Checkpoint '{self.path}' was written by a '{found}' model but is "
            f"being loaded into a '{expected}' model. Parameters from different "
            f"architectures are never interchangeable in this project."
        )


class ProcessedMetadataError(TrainingError):
    """The preprocessing artefacts needed to build a tensor are missing or unusable.

    Phase 4 reads which columns are features, which are identifiers and which is
    the target from the artefacts Phase 3 wrote, rather than restating them as
    constants. That is the safer arrangement — the two phases cannot drift apart —
    but it means a missing artefact or a missing key has to stop the run. The
    alternative, falling back to a hard-coded column list, is how a model quietly
    ends up trained on the wrong features.
    """

    def __init__(self, detail: str, path: Path | str) -> None:
        """Record what was wrong and which artefact it was wrong in."""
        self.detail = detail
        self.path = Path(path)
        super().__init__(
            f"Cannot read the processed feature metadata at '{self.path}': "
            f"{detail}. Run the 'preprocess' command first, or point the "
            f"configuration at the directory that preprocessing actually wrote."
        )


class NoCompletedEpochError(TrainingError):
    """Training finished without completing a single epoch.

    There is no checkpoint to select and no validation measurement to report, so
    there is nothing to persist. Reported rather than writing an empty result
    that would later read as a real experiment.
    """

    def __init__(self, experiment_id: str) -> None:
        """Record the experiment that produced nothing."""
        self.experiment_id = experiment_id
        super().__init__(
            f"Experiment '{experiment_id}' completed no epochs, so no checkpoint "
            f"was selected and no results exist to report."
        )


class PredictionAlignmentError(TrainingError):
    """Component predictions are misaligned in length, ordering, or truth labels."""

