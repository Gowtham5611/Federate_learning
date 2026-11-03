"""Metrics adapter: scikit-learn for the curves, numpy for the confusion matrix.

Split deliberately. ``average_precision_score`` and ``roc_auc_score`` implement
ranking summaries whose details (interpolation, tie handling) are worth taking from
a reference implementation. The confusion matrix, precision, recall and F1 are four
comparisons and three divisions, and computing them here means the degenerate cases
are handled explicitly rather than depending on a library's choice of default.

Those degenerate cases are not hypothetical at a 0.1% positive rate. A model that
flags nothing has an undefined precision; a threshold high enough to flag nothing
has an undefined F1. Both are reported as ``0.0`` with the confusion counts beside
them, because the counts show what happened and a ``nan`` in a results table
propagates into every average computed from it.

Threshold selection lives here and refuses any split but validation.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from application.dto.experiment import ClassificationMetrics, ConfusionCounts
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from domain.enums.split_name import SplitName
from domain.exceptions.training import ThresholdFittedOnWrongSplitError

logger = logging.getLogger(__name__)

UNDEFINED_METRIC = 0.0
"""Value reported when a metric's denominator is zero.

Chosen over ``nan`` so that a results table stays numeric and averages over it stay
meaningful. The confusion counts published alongside make the difference between
"scored zero" and "was undefined" recoverable.
"""

FALLBACK_THRESHOLD = 0.5
"""Threshold returned when the validation split contains no positives.

Selecting a threshold needs at least one positive to score against. With none,
every candidate ties at F1 zero, so there is nothing to choose; the neutral value
is returned and the situation is logged rather than being presented as a fitted
result.
"""


class SklearnMetricsCalculator(MetricsCalculatorPort):
    """Computes classification metrics and selects F1-optimal thresholds."""

    def evaluate(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        threshold: float,
        loss: float | None = None,
    ) -> ClassificationMetrics:
        """Score predictions at one threshold.

        Raises:
            ValueError: ``labels`` and ``probabilities`` have different lengths.
        """
        if labels.shape[0] != probabilities.shape[0]:
            raise ValueError(
                f"labels has {labels.shape[0]} entry/entries and probabilities has "
                f"{probabilities.shape[0]}; they describe the same predictions"
            )

        truth = labels.astype(bool)
        # >= rather than >: the threshold is the smallest score that counts as an
        # alert, which is what makes a selected threshold behave as measured.
        flagged = probabilities >= threshold

        true_positives = int(np.count_nonzero(truth & flagged))
        false_positives = int(np.count_nonzero(~truth & flagged))
        false_negatives = int(np.count_nonzero(truth & ~flagged))
        true_negatives = int(np.count_nonzero(~truth & ~flagged))
        confusion = ConfusionCounts(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

        precision = _ratio(true_positives, true_positives + false_positives)
        recall = _ratio(true_positives, true_positives + false_negatives)
        total = confusion.total

        return ClassificationMetrics(
            precision=precision,
            recall=recall,
            f1=_ratio(2.0 * precision * recall, precision + recall),
            pr_auc=self._pr_auc(labels, probabilities),
            roc_auc=self._roc_auc(labels, probabilities),
            accuracy=_ratio(true_positives + true_negatives, total),
            confusion=confusion,
            threshold=threshold,
            positive_rate=_ratio(confusion.actual_positives, total),
            loss=loss,
        )

    def select_threshold(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        split: SplitName,
    ) -> float:
        """Choose the F1-maximising threshold on validation predictions.

        Raises:
            ThresholdFittedOnWrongSplitError: ``split`` is not validation.
        """
        if split is not SplitName.VALIDATION:
            raise ThresholdFittedOnWrongSplitError(split.value)

        positives = int(np.count_nonzero(labels.astype(bool)))
        if positives == 0 or probabilities.shape[0] == 0:
            logger.warning(
                "The validation split holds %d positive(s) in %d prediction(s), so "
                "no threshold can be fitted. Falling back to %.2f, which is a "
                "default rather than a selected value.",
                positives,
                probabilities.shape[0],
                FALLBACK_THRESHOLD,
            )
            return FALLBACK_THRESHOLD

        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        # precision_recall_curve returns one more precision/recall point than
        # thresholds: the trailing (1, 0) point corresponds to no threshold at all.
        precision = precision[:-1]
        recall = recall[:-1]
        denominator = precision + recall
        f1 = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

        best = int(np.argmax(f1))
        chosen = float(thresholds[best])
        logger.info(
            "Selected threshold %.6f on the validation split: F1 %.6f, precision "
            "%.6f, recall %.6f, from %d candidate(s).",
            chosen,
            float(f1[best]),
            float(precision[best]),
            float(recall[best]),
            int(thresholds.shape[0]),
        )
        return chosen

    # --- Ranking summaries ---------------------------------------------------

    @staticmethod
    def _pr_auc(
        labels: NDArray[np.float32], probabilities: NDArray[np.float32]
    ) -> float:
        """Average precision, or the undefined value when one class is absent."""
        if not _has_both_classes(labels):
            return UNDEFINED_METRIC
        return float(average_precision_score(labels, probabilities))

    @staticmethod
    def _roc_auc(
        labels: NDArray[np.float32], probabilities: NDArray[np.float32]
    ) -> float:
        """ROC-AUC, or the undefined value when one class is absent."""
        if not _has_both_classes(labels):
            return UNDEFINED_METRIC
        return float(roc_auc_score(labels, probabilities))


def _has_both_classes(labels: NDArray[np.float32]) -> bool:
    """Whether the labels contain at least one positive and one negative.

    Both ranking metrics are undefined with a single class present, and both
    scikit-learn functions raise rather than returning a value. Checked first so
    that a run does not abort over a split that simply held no positives, which is
    a real possibility for a small reduced run.
    """
    if labels.shape[0] == 0:
        return False
    positives = int(np.count_nonzero(labels.astype(bool)))
    return bool(0 < positives < labels.shape[0])


def _ratio(numerator: float, denominator: float) -> float:
    """Divide, reporting :data:`UNDEFINED_METRIC` for a zero denominator."""
    return numerator / denominator if denominator else UNDEFINED_METRIC
