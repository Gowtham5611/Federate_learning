"""Tests for :mod:`application.services.centralized_training_service`.

The service is checked for the properties that make its output trustworthy, and
most of those are properties of *order* rather than of a return value:

- test is not read until the threshold is frozen;
- the threshold is only ever fitted on validation;
- best and final checkpoints go to separate paths;
- every result row carries the Phase 4 values, answers the two federated fields a
  centralized run can answer, and leaves the other three null rather than
  inventing them.

All five collaborators are doubles sharing one event log, so "before" and "after"
are assertions about list positions rather than about intent.
"""

from __future__ import annotations

import numpy as np
import pytest

from application.dto.training import TrainingConfig
from application.services.centralized_training_service import (
    LOSS_TYPE,
    CentralizedTrainingService,
)
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import (
    SelectionMetric,
    ThresholdPolicy,
    TrainingMode,
)
from domain.exceptions.training import FeatureDimensionMismatchError
from tests.fixtures.training_doubles import (
    EventLog,
    RecordingCheckpointStore,
    RecordingMetricsCalculator,
    RecordingResultStore,
    RecordingSequenceProvider,
    StubTrainingEngine,
    make_dataset,
)

pytestmark = pytest.mark.unit


class Harness:
    """One service plus the doubles behind it and the log they share."""

    def __init__(
        self,
        *,
        provider: RecordingSequenceProvider | None = None,
        engine: StubTrainingEngine | None = None,
        metrics: RecordingMetricsCalculator | None = None,
    ) -> None:
        """Assemble a service over recording doubles."""
        self.events: EventLog = []
        self.provider = provider or RecordingSequenceProvider(self.events)
        self.engine = engine or StubTrainingEngine(self.events)
        self.metrics = metrics or RecordingMetricsCalculator(self.events)
        self.checkpoints = RecordingCheckpointStore(self.events)
        self.results = RecordingResultStore(self.events)
        self.service = CentralizedTrainingService(
            sequences=self.provider,
            engine=self.engine,
            metrics=self.metrics,
            checkpoints=self.checkpoints,
            results=self.results,
        )

    def position(self, event: str) -> int:
        """Index of an event in the shared log.

        Raises:
            AssertionError: The event never happened.
        """
        assert event in self.events, f"{event!r} not in {self.events}"
        return self.events.index(event)


def make_config(**overrides: object) -> TrainingConfig:
    """A small valid configuration, overridable per test."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "sequence_length": 3,
        "batch_size": 4,
        "epochs": 2,
        "hidden_size": 4,
        "early_stopping_patience": 0,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


# --- The central ordering rule -----------------------------------------------


def test_test_split_is_not_read_until_the_threshold_is_frozen() -> None:
    harness = Harness()
    harness.service.run(make_config())

    frozen = harness.position("select_threshold:validation")
    test_read = harness.position("load:test")
    assert test_read > frozen, harness.events


def test_test_split_is_not_read_until_training_has_finished() -> None:
    harness = Harness()
    harness.service.run(make_config())

    assert harness.position("load:test") > harness.position("fit")


def test_train_and_validation_are_read_before_fitting() -> None:
    harness = Harness()
    harness.service.run(make_config())

    fit = harness.position("fit")
    assert harness.position("load:train") < fit
    assert harness.position("load:validation") < fit


def test_each_split_is_read_exactly_once() -> None:
    harness = Harness()
    harness.service.run(make_config())

    assert harness.provider.loaded == [
        SplitName.TRAIN,
        SplitName.VALIDATION,
        SplitName.TEST,
    ]


def test_fit_is_given_train_and_validation_only() -> None:
    harness = Harness()
    harness.service.run(make_config())

    assert harness.engine.fit_calls == [(SplitName.TRAIN, SplitName.VALIDATION)]


# --- Threshold discipline ----------------------------------------------------


def test_threshold_is_only_ever_selected_on_validation() -> None:
    harness = Harness()
    harness.service.run(make_config(threshold_policy=ThresholdPolicy.VALIDATION_F1))

    assert harness.metrics.selection_splits == [SplitName.VALIDATION]
    assert SplitName.TEST not in harness.metrics.selection_splits


def test_learned_threshold_is_recorded_with_the_split_it_came_from() -> None:
    harness = Harness()
    report = harness.service.run(
        make_config(threshold_policy=ThresholdPolicy.VALIDATION_F1)
    )

    assert report.frozen_threshold == pytest.approx(harness.metrics.selected)
    assert report.threshold_selected_on == SplitName.VALIDATION.value


def test_fixed_policy_selects_no_threshold_at_all() -> None:
    harness = Harness()
    report = harness.service.run(
        make_config(threshold_policy=ThresholdPolicy.FIXED, threshold=0.3)
    )

    assert harness.metrics.selection_splits == []
    assert report.frozen_threshold == pytest.approx(0.3)
    # None, not "validation": a configured constant is a different claim from a
    # fitted threshold and must not be reported as one.
    assert report.threshold_selected_on is None


def test_frozen_threshold_is_applied_unchanged_to_test() -> None:
    harness = Harness(metrics=RecordingMetricsCalculator([], selected=0.37))
    harness.metrics.events = harness.events
    report = harness.service.run(
        make_config(threshold_policy=ThresholdPolicy.VALIDATION_F1)
    )

    learned_rows = [
        result
        for result in report.results
        if result.threshold_policy is ThresholdPolicy.VALIDATION_F1
    ]
    assert {result.split for result in learned_rows} == {
        SplitName.VALIDATION,
        SplitName.TEST,
    }
    for result in learned_rows:
        assert result.metrics.threshold == pytest.approx(0.37)


def test_both_threshold_policies_are_reported_for_each_split() -> None:
    harness = Harness()
    report = harness.service.run(
        make_config(threshold_policy=ThresholdPolicy.VALIDATION_F1)
    )

    pairs = {(result.split, result.threshold_policy) for result in report.results}
    assert pairs == {
        (SplitName.VALIDATION, ThresholdPolicy.FIXED),
        (SplitName.VALIDATION, ThresholdPolicy.VALIDATION_F1),
        (SplitName.TEST, ThresholdPolicy.FIXED),
        (SplitName.TEST, ThresholdPolicy.VALIDATION_F1),
    }


def test_fixed_policy_reports_one_row_per_split() -> None:
    harness = Harness()
    report = harness.service.run(make_config(threshold_policy=ThresholdPolicy.FIXED))

    assert len(report.results) == 2
    assert all(
        result.threshold_policy is ThresholdPolicy.FIXED for result in report.results
    )


# --- Checkpoints -------------------------------------------------------------


def test_both_checkpoints_are_written() -> None:
    harness = Harness()
    harness.service.run(make_config())

    assert set(harness.checkpoints.saved) == {
        CheckpointKind.BEST,
        CheckpointKind.FINAL,
    }


def test_best_and_final_checkpoints_go_to_different_paths() -> None:
    harness = Harness()
    harness.service.run(make_config())

    best = harness.checkpoints.path_for("run", CheckpointKind.BEST)
    final = harness.checkpoints.path_for("run", CheckpointKind.FINAL)
    assert best != final


def test_final_checkpoint_does_not_overwrite_the_best_parameters() -> None:
    # Best epoch 1, two epochs run: the two states are genuinely different, which
    # is exactly the case where an overwrite would go unnoticed.
    harness = Harness(engine=StubTrainingEngine([], epochs=2, best_epoch=1))
    harness.engine.events = harness.events
    harness.service.run(make_config(epochs=2))

    best = harness.checkpoints.saved[CheckpointKind.BEST]
    final = harness.checkpoints.saved[CheckpointKind.FINAL]
    assert best.parameters == harness.engine.best_state
    assert final.parameters == harness.engine.final_state
    assert best.epoch == 1
    assert final.epoch == 2


def test_checkpoints_carry_the_frozen_threshold_and_the_architecture() -> None:
    harness = Harness()
    harness.service.run(
        make_config(
            architecture=ModelArchitecture.BILSTM,
            threshold_policy=ThresholdPolicy.VALIDATION_F1,
        )
    )

    for checkpoint in harness.checkpoints.saved.values():
        assert checkpoint.architecture is ModelArchitecture.BILSTM
        assert checkpoint.threshold == pytest.approx(harness.metrics.selected)
        assert checkpoint.feature_dimension == harness.provider.feature_dimension
        assert checkpoint.random_seed == 42


def test_checkpoints_are_written_after_test_has_been_scored() -> None:
    harness = Harness()
    harness.service.run(make_config())

    assert harness.position("save_checkpoint:best") > harness.position("predict:test")


# --- Result rows -------------------------------------------------------------


def test_every_row_carries_the_phase_four_values() -> None:
    harness = Harness()
    report = harness.service.run(make_config())

    for result in report.results:
        assert result.training_mode is TrainingMode.CENTRALIZED
        assert result.loss_type == LOSS_TYPE
        row = result.to_dict()
        assert row["training_mode"] == "centralized"
        assert row["data_distribution"] == "centralized"
        assert row["federated_algorithm"] == "none"


def test_client_round_and_mu_are_null_rather_than_invented() -> None:
    harness = Harness()
    report = harness.service.run(make_config())

    for result in report.results:
        assert result.client_id is None
        assert result.federated_round is None
        assert result.mu is None
        # Null, not zero: round 0 and mu 0.0 are both real federated settings, so
        # writing either here would put a fabricated value in the table.
        row = result.to_dict()
        assert row["client_id"] is None
        assert row["round"] is None
        assert row["mu"] is None


def test_rows_report_false_negatives_explicitly() -> None:
    harness = Harness()
    report = harness.service.run(make_config())

    for result in report.results:
        row = result.to_dict()
        assert "fn" in row
        assert isinstance(row["fn"], int)


def test_rows_record_the_hyperparameters_that_produced_them() -> None:
    harness = Harness()
    report = harness.service.run(make_config(hidden_size=8, sequence_length=3))

    for result in report.results:
        stored = result.to_dict()["hyperparameters"]
        assert stored["hidden_size"] == 8
        assert stored["sequence_length"] == 3
        assert stored["architecture"] == "gru"


def test_no_training_split_row_is_emitted() -> None:
    # Training-split metrics are not reported: they measure memorisation, and a
    # table that mixes them with held-out numbers invites the wrong comparison.
    harness = Harness()
    report = harness.service.run(make_config())

    assert SplitName.TRAIN not in {result.split for result in report.results}


def test_report_is_persisted_and_rows_appended() -> None:
    harness = Harness()
    report = harness.service.run(make_config())

    assert harness.results.reports == [report]
    assert len(harness.results.rows) == len(report.results)
    assert harness.position("save_report") < harness.position("append_rows")


# --- Provenance and history --------------------------------------------------


def test_epoch_history_is_carried_into_the_report() -> None:
    harness = Harness(engine=StubTrainingEngine([], epochs=3, best_epoch=2))
    harness.engine.events = harness.events
    report = harness.service.run(make_config(epochs=3))

    assert [record.epoch for record in report.epochs] == [1, 2, 3]
    assert report.selected_epoch == 2


def test_provenance_is_re_read_after_every_split_has_been_loaded() -> None:
    harness = Harness()
    report = harness.service.run(make_config())

    # The first describe_features call happens before any split is read, so only
    # the second one can report what the run consumed.
    assert harness.events.count("describe_features") == 2
    assert set(report.provenance.rows_per_split) == {
        SplitName.TRAIN.value,
        SplitName.VALIDATION.value,
        SplitName.TEST.value,
    }


def test_experiment_id_names_the_architecture_the_mode_and_the_seed() -> None:
    harness = Harness()
    report = harness.service.run(
        make_config(architecture=ModelArchitecture.LSTM, random_seed=7)
    )

    assert report.experiment_id.startswith("centralized_lstm_seed7_")


def test_reduced_runs_are_recorded_as_reduced() -> None:
    harness = Harness()
    report = harness.service.run(make_config(row_limit=50))

    assert report.config.is_reduced_run
    assert report.config.to_dict()["row_limit"] == 50


# --- Guards ------------------------------------------------------------------


def test_mismatched_feature_widths_are_refused_before_fitting() -> None:
    events: EventLog = []
    provider = RecordingSequenceProvider(
        events,
        datasets={
            SplitName.TRAIN: make_dataset(SplitName.TRAIN, feature_dimension=4),
            SplitName.VALIDATION: make_dataset(
                SplitName.VALIDATION, feature_dimension=5
            ),
            SplitName.TEST: make_dataset(SplitName.TEST, feature_dimension=4),
        },
    )
    harness = Harness(provider=provider)
    provider.events = harness.events

    with pytest.raises(FeatureDimensionMismatchError) as caught:
        harness.service.run(make_config())

    assert "fit" not in harness.events
    assert "load:test" not in harness.events
    assert "4" in str(caught.value)
    assert "5" in str(caught.value)


def test_selection_metric_choice_is_passed_through_to_the_report() -> None:
    harness = Harness()
    report = harness.service.run(make_config(selection_metric=SelectionMetric.RECALL))

    assert report.config.selection_metric is SelectionMetric.RECALL


def test_predictions_come_from_the_best_state_not_the_final_one() -> None:
    harness = Harness(engine=StubTrainingEngine([], epochs=3, best_epoch=1))
    harness.engine.events = harness.events

    seen: list[object] = []
    original = harness.engine.predict

    def spy(config_arg: TrainingConfig, dataset: object, parameters: object) -> object:
        seen.append(parameters)
        return original(config_arg, dataset, parameters)  # type: ignore[arg-type]

    harness.engine.predict = spy  # type: ignore[method-assign, assignment]
    harness.service.run(make_config(epochs=3))

    assert seen == [harness.engine.best_state, harness.engine.best_state]


def test_metrics_are_computed_from_the_predicted_probabilities() -> None:
    # A perfectly separable stub must produce a perfect confusion matrix; if the
    # service passed labels or probabilities to the wrong parameter this breaks.
    labels_seen: dict[SplitName, int] = {}
    events: EventLog = []
    provider = RecordingSequenceProvider(events)
    for split, dataset in provider.datasets.items():
        labels_seen[split] = int(np.count_nonzero(dataset.labels > 0.5))

    harness = Harness(provider=provider)
    provider.events = harness.events
    report = harness.service.run(make_config(threshold_policy=ThresholdPolicy.FIXED))

    test_row = report.result_for(SplitName.TEST, ThresholdPolicy.FIXED)
    assert test_row is not None
    assert test_row.metrics.confusion.false_negatives == 0
    assert test_row.metrics.confusion.true_positives == labels_seen[SplitName.TEST]
