"""Outbound port for scoring predictions and choosing a decision threshold.

Behind a port for a reason that is about evidence rather than swappability: the
numbers this produces are the project's results, so it has to be replaceable by a
second implementation and checkable against one. The shipped adapter delegates to
scikit-learn.

Threshold selection lives here too, next to the metrics it optimises, and the
port makes the split an explicit argument so that an implementation can refuse to
fit on anything but validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from application.dto.experiment import ClassificationMetrics
from domain.enums.split_name import SplitName


class MetricsCalculatorPort(ABC):
    """Computes classification metrics and selects thresholds."""

    @abstractmethod
    def evaluate(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        threshold: float,
        loss: float | None = None,
    ) -> ClassificationMetrics:
        """Score predictions at one threshold.

        ``pr_auc`` and ``roc_auc`` describe the ranking and do not depend on
        ``threshold``; everything else does.

        Args:
            labels: ``(n,)`` true targets in ``{0.0, 1.0}``.
            probabilities: ``(n,)`` predicted positive-class probabilities.
            threshold: Probabilities at or above this count as positive.
            loss: Mean loss to record alongside, if it was measured.

        Returns:
            Metrics, including the four confusion-matrix counts.
        """

    @abstractmethod
    def select_threshold(
        self,
        labels: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        *,
        split: SplitName,
    ) -> float:
        """Choose the threshold maximising F1 on the given predictions.

        Args:
            labels: ``(n,)`` true targets.
            probabilities: ``(n,)`` predicted probabilities.
            split: The split these predictions came from. Implementations must
                refuse anything but :attr:`SplitName.VALIDATION`: a threshold is
                a fitted parameter, and fitting it on test would make every test
                metric a best case rather than a measurement.

        Returns:
            The selected threshold.

        Raises:
            ThresholdFittedOnWrongSplitError: ``split`` is not validation.
        """
