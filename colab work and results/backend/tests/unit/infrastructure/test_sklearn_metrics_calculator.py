"""Tests for :mod:`infrastructure.evaluation.sklearn_metrics_calculator`.

Two things are checked, and the second is the reason the first matters.

The arithmetic is checked against confusion matrices written out by hand. At a
0.1% positive rate a mistake here does not look like a bug; it looks like a result.
False negatives get their own tests because a missed laundering transaction is the
error this project is about, and because ``fn`` is the one count that a model
flagging nothing drives to its maximum while accuracy stays at 99.9%.

The degenerate cases are checked as behaviour rather than as edge cases: a split
with no positives, a threshold nothing clears, a single-class split. Each has an
undefined metric somewhere, and each is reported as zero with the counts intact
rather than as ``nan`` — which is what keeps averages over the results table
meaningful.

Threshold selection is checked for the one rule it must never break: it refuses
any split but validation.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from numpy.typing import NDArray

from domain.enums.split_name import SplitName
from domain.exceptions.training import ThresholdFittedOnWrongSplitError
from infrastructure.evaluation.sklearn_metrics_calculator import (
    FALLBACK_THRESHOLD,
    UNDEFINED_METRIC,
    SklearnMetricsCalculator,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def calculator() -> SklearnMetricsCalculator:
    """The adapter under test."""
    return SklearnMetricsCalculator()


def labels_of(*values: float) -> NDArray[np.float32]:
    """Labels as the engine supplies them."""
    return np.array(values, dtype=np.float32)


def scores_of(*values: float) -> NDArray[np.float32]:
    """Predicted probabilities as the engine supplies them."""
    return np.array(values, dtype=np.float32)


# --- The confusion matrix ----------------------------------------------------


def test_counts_are_read_off_the_predictions(
    calculator: SklearnMetricsCalculator,
) -> None:
    # truth:  1    1    0    0
    # score: .9   .2   .8   .1   at threshold .5
    #        TP   FN   FP   TN
    metrics = calculator.evaluate(
        labels_of(1, 1, 0, 0), scores_of(0.9, 0.2, 0.8, 0.1), threshold=0.5
    )

    assert metrics.confusion.true_positives == 1
    assert metrics.confusion.false_negatives == 1
    assert metrics.confusion.false_positives == 1
    assert metrics.confusion.true_negatives == 1


def test_the_threshold_is_the_lowest_score_that_alerts(
    calculator: SklearnMetricsCalculator,
) -> None:
    # A score exactly at the threshold counts as an alert. Otherwise a selected
    # threshold would score differently from the sweep that selected it.
    metrics = calculator.evaluate(labels_of(1), scores_of(0.5), threshold=0.5)

    assert metrics.confusion.true_positives == 1
    assert metrics.confusion.false_negatives == 0


def test_a_higher_threshold_trades_recall_for_precision(
    calculator: SklearnMetricsCalculator,
) -> None:
    labels = labels_of(1, 1, 0, 0, 0)
    scores = scores_of(0.9, 0.4, 0.6, 0.2, 0.1)

    lenient = calculator.evaluate(labels, scores, threshold=0.3)
    strict = calculator.evaluate(labels, scores, threshold=0.8)

    assert lenient.recall > strict.recall
    assert lenient.precision < strict.precision
    assert strict.false_negatives > lenient.false_negatives


def test_the_counts_sum_to_the_predictions_made(
    calculator: SklearnMetricsCalculator,
) -> None:
    labels = labels_of(1, 0, 0, 1, 0, 0, 0)
    metrics = calculator.evaluate(
        labels, scores_of(0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.4), threshold=0.5
    )

    assert metrics.confusion.total == 7
    assert metrics.confusion.actual_positives == 2


# --- Precision, recall, F1 ---------------------------------------------------


def test_precision_and_recall_match_their_definitions(
    calculator: SklearnMetricsCalculator,
) -> None:
    # 2 true positives, 1 false positive, 2 false negatives.
    metrics = calculator.evaluate(
        labels_of(1, 1, 1, 1, 0, 0),
        scores_of(0.9, 0.8, 0.3, 0.2, 0.7, 0.1),
        threshold=0.5,
    )

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 4)


def test_f1_is_the_harmonic_mean_of_the_two(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 1, 1, 1, 0, 0),
        scores_of(0.9, 0.8, 0.3, 0.2, 0.7, 0.1),
        threshold=0.5,
    )

    expected = (
        2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
    )
    assert metrics.f1 == pytest.approx(expected)


def test_perfect_separation_scores_one_everywhere(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 1, 0, 0), scores_of(0.99, 0.95, 0.02, 0.01), threshold=0.5
    )

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.false_negatives == 0


# --- False negatives ---------------------------------------------------------


def test_false_negatives_are_reported_as_their_own_number(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 1, 1, 0), scores_of(0.9, 0.1, 0.1, 0.1), threshold=0.5
    )

    assert metrics.false_negatives == 2
    assert metrics.confusion.false_negatives == 2
    assert metrics.to_dict()["fn"] == 2


def test_a_model_that_flags_nothing_is_visibly_useless(
    calculator: SklearnMetricsCalculator,
) -> None:
    # This is the failure mode that a fixed 0.5 threshold produces under extreme
    # imbalance, and the reason accuracy is not the primary metric here: 99.9%
    # accurate, every positive missed.
    labels = labels_of(*([0.0] * 999 + [1.0]))
    scores = scores_of(*([0.01] * 1000))

    metrics = calculator.evaluate(labels, scores, threshold=0.5)

    assert metrics.accuracy == pytest.approx(0.999)
    assert metrics.recall == UNDEFINED_METRIC
    assert metrics.false_negatives == 1
    assert metrics.confusion.missed_positive_rate == pytest.approx(1.0)


def test_the_missed_positive_rate_complements_recall(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 1, 1, 1, 0), scores_of(0.9, 0.8, 0.1, 0.1, 0.1), threshold=0.5
    )

    assert metrics.recall == pytest.approx(0.5)
    assert metrics.confusion.missed_positive_rate == pytest.approx(0.5)


# --- Degenerate cases --------------------------------------------------------


def test_flagging_nothing_gives_an_undefined_precision_not_a_crash(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 0, 0), scores_of(0.1, 0.1, 0.1), threshold=0.9
    )

    assert metrics.precision == UNDEFINED_METRIC
    assert metrics.f1 == UNDEFINED_METRIC
    # The counts stay intact, so "undefined" and "scored zero" remain distinguishable.
    assert metrics.confusion.predicted_positives == 0
    assert metrics.confusion.false_negatives == 1


def test_a_split_with_no_positives_reports_zero_rather_than_nan(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(0, 0, 0, 0), scores_of(0.9, 0.4, 0.2, 0.1), threshold=0.5
    )

    for value in (metrics.recall, metrics.f1, metrics.pr_auc, metrics.roc_auc):
        assert value == UNDEFINED_METRIC
        assert not np.isnan(value)


def test_a_split_with_no_negatives_also_reports_zero(
    calculator: SklearnMetricsCalculator,
) -> None:
    # Both ranking metrics are undefined with one class present, and scikit-learn
    # raises rather than returning a value. A small reduced run can genuinely land
    # here, and it must not abort the run.
    metrics = calculator.evaluate(
        labels_of(1, 1, 1), scores_of(0.9, 0.8, 0.7), threshold=0.5
    )

    assert metrics.pr_auc == UNDEFINED_METRIC
    assert metrics.roc_auc == UNDEFINED_METRIC
    assert metrics.recall == pytest.approx(1.0)


def test_no_predictions_at_all_is_reported_rather_than_divided_by_zero(
    calculator: SklearnMetricsCalculator,
) -> None:
    empty = np.empty(0, dtype=np.float32)

    metrics = calculator.evaluate(empty, empty, threshold=0.5)

    assert metrics.confusion.total == 0
    assert metrics.accuracy == UNDEFINED_METRIC
    assert metrics.positive_rate == UNDEFINED_METRIC


def test_mismatched_lengths_are_refused(
    calculator: SklearnMetricsCalculator,
) -> None:
    with pytest.raises(ValueError, match="same predictions"):
        calculator.evaluate(labels_of(1, 0), scores_of(0.9), threshold=0.5)


# --- Ranking summaries -------------------------------------------------------


def test_pr_auc_ranks_rather_than_thresholds(
    calculator: SklearnMetricsCalculator,
) -> None:
    # Same ranking, scores shifted below the threshold: the threshold-based metrics
    # collapse and the ranking metrics do not. This is why PR-AUC is the default
    # epoch-selection metric in this project.
    labels = labels_of(1, 1, 0, 0)
    ranked = calculator.evaluate(labels, scores_of(0.4, 0.3, 0.2, 0.1), threshold=0.5)

    assert ranked.recall == UNDEFINED_METRIC
    assert ranked.pr_auc == pytest.approx(1.0)
    assert ranked.roc_auc == pytest.approx(1.0)


def test_the_threshold_that_was_used_is_recorded_with_the_metrics(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 0), scores_of(0.9, 0.1), threshold=0.0123
    )

    assert metrics.threshold == pytest.approx(0.0123)
    assert metrics.to_dict()["threshold"] == pytest.approx(0.0123)


def test_the_loss_is_carried_through_when_given(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 0), scores_of(0.9, 0.1), threshold=0.5, loss=0.25
    )

    assert metrics.loss == pytest.approx(0.25)


def test_the_loss_stays_absent_when_not_given(
    calculator: SklearnMetricsCalculator,
) -> None:
    # Threshold sweeps have no loss to report, and inventing 0.0 would read as a
    # perfect one.
    metrics = calculator.evaluate(labels_of(1, 0), scores_of(0.9, 0.1), threshold=0.5)

    assert metrics.loss is None


def test_the_positive_rate_describes_the_split_not_the_predictions(
    calculator: SklearnMetricsCalculator,
) -> None:
    metrics = calculator.evaluate(
        labels_of(1, 0, 0, 0), scores_of(0.9, 0.9, 0.9, 0.9), threshold=0.5
    )

    assert metrics.positive_rate == pytest.approx(0.25)


# --- Threshold selection -----------------------------------------------------


def test_the_selected_threshold_maximises_f1_on_the_predictions(
    calculator: SklearnMetricsCalculator,
) -> None:
    labels = labels_of(1, 1, 0, 0, 0, 0)
    scores = scores_of(0.8, 0.6, 0.5, 0.3, 0.2, 0.1)

    chosen = calculator.select_threshold(labels, scores, split=SplitName.VALIDATION)

    at_chosen = calculator.evaluate(labels, scores, threshold=chosen).f1
    for candidate in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        assert at_chosen >= calculator.evaluate(labels, scores, threshold=candidate).f1


def test_the_selected_threshold_is_one_of_the_observed_scores(
    calculator: SklearnMetricsCalculator,
) -> None:
    # A threshold between two observed scores would classify identically to the
    # lower one while being harder to justify.
    scores = scores_of(0.81, 0.62, 0.53, 0.31)

    chosen = calculator.select_threshold(
        labels_of(1, 1, 0, 0), scores, split=SplitName.VALIDATION
    )

    assert chosen in scores.astype(float).tolist()


def test_selection_finds_a_threshold_below_the_default_when_scores_are_low(
    calculator: SklearnMetricsCalculator,
) -> None:
    # The situation this project expects: a well-ranked model whose probabilities
    # all sit far below 0.5 because positives are rare.
    labels = labels_of(1, 1, 0, 0, 0, 0, 0, 0)
    scores = scores_of(0.09, 0.07, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01)

    chosen = calculator.select_threshold(labels, scores, split=SplitName.VALIDATION)

    assert chosen < 0.5
    assert calculator.evaluate(labels, scores, threshold=chosen).recall > 0.0


@pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.TEST])
def test_selection_refuses_every_split_but_validation(
    calculator: SklearnMetricsCalculator, split: SplitName
) -> None:
    # A threshold is a fitted parameter. Fitting it on test would be tuning on
    # test, so the attempt is refused rather than logged.
    with pytest.raises(ThresholdFittedOnWrongSplitError) as caught:
        calculator.select_threshold(
            labels_of(1, 0), scores_of(0.9, 0.1), split=split
        )

    assert split.value in str(caught.value)


def test_selection_on_a_split_with_no_positives_falls_back_and_says_so(
    calculator: SklearnMetricsCalculator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        chosen = calculator.select_threshold(
            labels_of(0, 0, 0), scores_of(0.9, 0.2, 0.1), split=SplitName.VALIDATION
        )

    assert chosen == FALLBACK_THRESHOLD
    assert "default rather than a selected value" in caplog.text


def test_selection_on_no_predictions_falls_back(
    calculator: SklearnMetricsCalculator,
) -> None:
    empty = np.empty(0, dtype=np.float32)

    chosen = calculator.select_threshold(empty, empty, split=SplitName.VALIDATION)

    assert chosen == FALLBACK_THRESHOLD


def test_selection_is_reproducible_for_the_same_predictions(
    calculator: SklearnMetricsCalculator,
) -> None:
    labels = labels_of(1, 0, 1, 0, 0, 0)
    scores = scores_of(0.7, 0.6, 0.5, 0.4, 0.3, 0.2)

    first = calculator.select_threshold(labels, scores, split=SplitName.VALIDATION)
    second = calculator.select_threshold(labels, scores, split=SplitName.VALIDATION)

    assert first == second


def test_selection_reads_nothing_but_the_arrays_it_is_given(
    calculator: SklearnMetricsCalculator,
) -> None:
    # No state carried between calls: a second, different validation set must be
    # able to produce a different answer.
    low = calculator.select_threshold(
        labels_of(1, 0, 0), scores_of(0.05, 0.02, 0.01), split=SplitName.VALIDATION
    )
    high = calculator.select_threshold(
        labels_of(1, 0, 0), scores_of(0.95, 0.20, 0.10), split=SplitName.VALIDATION
    )

    assert low != high
