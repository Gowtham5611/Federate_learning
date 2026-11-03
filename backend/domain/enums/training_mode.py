"""How a model was trained, and how its classification threshold was chosen.

Both enums exist so that a stored result says what produced it. A dashboard
comparing centralized against federated runs, or a fixed threshold against a
validation-selected one, reads these fields rather than inferring the setup from
a filename.
"""

from __future__ import annotations

from enum import StrEnum


class TrainingMode(StrEnum):
    """Which training regime produced a result.

    Only ``CENTRALIZED`` exists yet, because only centralized training is
    implemented. The enum is still worth having now: it puts the field in the
    result schema with a real type from the start, so the later federated phases
    add a member instead of changing every stored record's shape.
    """

    CENTRALIZED = "centralized"
    FEDERATED = "federated"


class ThresholdPolicy(StrEnum):
    """How the decision threshold on the predicted probability was chosen.

    The distinction matters more than usual here. At 0.1% positives a fixed 0.5
    cut can leave a well-ranked model reporting zero recall, which says something
    about the threshold rather than about the model. Recording the policy
    alongside the threshold keeps those two readings apart.
    """

    FIXED = "fixed"
    """Use the configured constant. Nothing is learned from any split."""

    VALIDATION_F1 = "validation_f1"
    """Sweep candidate thresholds on validation and keep the best F1.

    Fitted on validation only, then frozen before test is touched. Test never
    influences the choice.
    """

    @property
    def is_learned(self) -> bool:
        """Whether the threshold is selected from data rather than configured."""
        return self is ThresholdPolicy.VALIDATION_F1


class SelectionMetric(StrEnum):
    """The validation quantity used to pick the best epoch's checkpoint.

    ``PR_AUC`` is the default elsewhere in this project rather than ``LOSS``.
    Under this dataset's imbalance the mean binary cross-entropy is dominated by
    the 99.9% of rows that are negative, so it can improve while the ranking of
    the rare positives gets worse. Average precision measures that ranking
    directly and needs no threshold.
    """

    PR_AUC = "pr_auc"
    RECALL = "recall"
    F1 = "f1"
    LOSS = "loss"

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger value is an improvement."""
        return self is not SelectionMetric.LOSS
