"""End-to-end centralized training over a synthetic processed tree.

=============================================================================
SYNTHETIC TEST DATA - NOT IBM AML DATA
=============================================================================
Every row read here is written by :mod:`tests.fixtures.sequence_fixtures` into
pytest's ``tmp_path``. No count, metric, timing or checkpoint produced by this
module describes the real IBM AML dataset, and nothing here reads
``data/processed/``. The real tree is never touched by a test.

What this file checks is the *wiring*, with no doubles anywhere: the real metadata
reader, the real sequence provider, the real PyTorch engine, the real scikit-learn
metrics calculator, and both real stores, composed the way ``composition_root``
composes them. The unit tests establish each component's behaviour in isolation;
these tests establish that the composition still holds the properties that make a
result trustworthy:

1. **The test split is not read until the threshold is frozen.** Enforced by
   watching file access, not by reading the service and believing it.
2. **The frozen threshold is fitted on validation and applied unchanged to test.**
3. **Both checkpoints land on disk, best and final, at separate paths.**
4. **The results table and the JSON report agree with the returned report.**
5. **One configuration, three architectures, byte-identical inputs.**
6. **A reduced run is marked as reduced**, so its numbers can never be mistaken
   for a full run's.

Two epochs on a few hundred synthetic rows: seconds, not minutes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from application.dto.experiment import ExperimentReport
from application.dto.training import TrainingConfig
from application.services.centralized_training_service import (
    LOSS_TYPE,
    CentralizedTrainingService,
)
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ORDERED_ARCHITECTURES, ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy, TrainingMode
from domain.exceptions.training import ProcessedMetadataError
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine
from infrastructure.persistence.csv_experiment_result_store import (
    CsvExperimentResultStore,
)
from infrastructure.persistence.torch_checkpoint_store import TorchCheckpointStore
from tests.fixtures.sequence_fixtures import (
    EXPANDED_DIMENSION,
    write_processed_tree,
)

pytestmark = pytest.mark.integration


def make_config(**overrides: object) -> TrainingConfig:
    """A configuration that trains in seconds on the synthetic tree."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "sequence_length": 4,
        "sequence_stride": 1,
        "hidden_size": 6,
        "batch_size": 32,
        "epochs": 2,
        "learning_rate": 0.02,
        "early_stopping_patience": 0,
        "random_seed": 17,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


class _Harness:
    """The five real adapters, composed, plus the paths they write to."""

    def __init__(self, root: Path) -> None:
        """Build the processed tree and wire the service over it."""
        self.processed = write_processed_tree(root / "processed")
        self.checkpoint_root = root / "checkpoints"
        self.results_root = root / "results"
        self.provider = ProcessedSequenceProvider(self.processed)
        self.metrics = SklearnMetricsCalculator()
        self.checkpoints = TorchCheckpointStore(self.checkpoint_root)
        self.results = CsvExperimentResultStore(self.results_root)
        self.service = CentralizedTrainingService(
            sequences=self.provider,
            engine=TorchTrainingEngine(self.metrics),
            metrics=self.metrics,
            checkpoints=self.checkpoints,
            results=self.results,
        )

    def run(self, config: TrainingConfig | None = None) -> ExperimentReport:
        """Run one experiment and return its report."""
        return self.service.run(config or make_config())


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    """A freshly composed pipeline over a synthetic tree."""
    return _Harness(tmp_path)


@pytest.fixture
def report(harness: _Harness) -> ExperimentReport:
    """One completed run, shared by the tests that only read its outcome."""
    return harness.run()


# --- The run completes -------------------------------------------------------


def test_a_run_produces_a_report_naming_its_own_configuration(
    report: ExperimentReport,
) -> None:
    assert report.config.architecture is ModelArchitecture.GRU
    assert report.config.random_seed == 17
    assert "gru" in report.experiment_id
    assert "centralized" in report.experiment_id


def test_the_run_trains_for_the_configured_number_of_epochs(
    report: ExperimentReport,
) -> None:
    assert len(report.epochs) == 2
    assert [record.epoch for record in report.epochs] == [1, 2]


def test_exactly_one_epoch_is_marked_as_selected(report: ExperimentReport) -> None:
    selected = [record for record in report.epochs if record.selected]

    assert len(selected) == 1
    assert selected[0].epoch == report.selected_epoch


def test_the_measured_input_width_matches_the_recorded_metadata(
    report: ExperimentReport,
) -> None:
    # The expansion happens once, in the provider, and the report states the width
    # the model actually saw rather than the source column count.
    assert report.provenance.feature_dimension == EXPANDED_DIMENSION
    assert len(report.provenance.feature_columns) < EXPANDED_DIMENSION


def test_the_identifier_columns_are_recorded_as_excluded(
    report: ExperimentReport,
) -> None:
    for column in report.provenance.identifier_columns:
        assert column not in report.provenance.feature_columns


def test_the_timings_are_measured_rather_than_left_at_zero(
    report: ExperimentReport,
) -> None:
    assert report.training_seconds > 0.0
    assert report.data_preparation_seconds > 0.0
    assert report.total_seconds >= report.training_seconds


def test_the_runtime_facts_come_from_the_running_process(
    report: ExperimentReport,
) -> None:
    # Not a configured string: a report whose torch version was assumed could not
    # be used to explain a number that differs between machines.
    assert report.torch_version == torch.__version__
    assert isinstance(report.deterministic, bool)
    assert report.device == "cpu"


def test_every_split_is_counted_as_it_was_read(report: ExperimentReport) -> None:
    for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        assert report.provenance.rows_per_split[split.value] > 0
        assert split.value in report.provenance.positives_per_split


def test_the_recorded_strategy_describes_what_was_built(
    report: ExperimentReport,
) -> None:
    strategy = report.sequences

    assert strategy.sequence_length == 4
    assert strategy.stride == 1
    assert "newest" in strategy.target_assignment
    assert "no window spans" in strategy.boundary_policy


# --- The test split is read last ---------------------------------------------


def test_the_test_split_is_not_read_until_the_threshold_is_frozen(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule this phase exists to keep, watched rather than trusted.

    The order of provider loads and threshold selections is recorded, so "test is
    not read until the threshold has been frozen" is a check on observed calls.
    """
    events: list[str] = []
    load = ProcessedSequenceProvider.load
    select = SklearnMetricsCalculator.select_threshold

    def watched_load(
        self: ProcessedSequenceProvider, split: SplitName, config: TrainingConfig
    ) -> object:
        events.append(f"load:{split.value}")
        return load(self, split, config)

    def watched_select(
        self: SklearnMetricsCalculator, *args: object, **kwargs: object
    ) -> float:
        events.append("freeze")
        return select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ProcessedSequenceProvider, "load", watched_load)
    monkeypatch.setattr(SklearnMetricsCalculator, "select_threshold", watched_select)

    harness.run()

    assert events.index("freeze") < events.index("load:test")
    assert events[: events.index("freeze")] == ["load:train", "load:validation"]


def test_the_threshold_is_only_ever_fitted_on_validation(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The calculator refuses any other split, so recording the argument shows the
    # refusal is never reached rather than merely available.
    splits: list[SplitName] = []
    select = SklearnMetricsCalculator.select_threshold

    def watched(
        self: SklearnMetricsCalculator, labels: object, scores: object, *, split: SplitName
    ) -> float:
        splits.append(split)
        return select(self, labels, scores, split=split)  # type: ignore[arg-type]

    monkeypatch.setattr(SklearnMetricsCalculator, "select_threshold", watched)

    harness.run()

    assert splits == [SplitName.VALIDATION]


def test_a_fixed_policy_fits_no_threshold_at_all(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("no threshold may be fitted under a fixed policy")

    monkeypatch.setattr(SklearnMetricsCalculator, "select_threshold", refuse)

    report = harness.run(make_config(threshold_policy=ThresholdPolicy.FIXED))

    assert report.threshold_selected_on is None
    assert report.frozen_threshold == pytest.approx(report.config.threshold)


def test_the_frozen_threshold_reaches_the_test_row_unchanged(
    report: ExperimentReport,
) -> None:
    learned = report.result_for(SplitName.TEST, report.config.threshold_policy)

    assert report.threshold_selected_on == SplitName.VALIDATION.value
    assert learned is not None
    assert learned.metrics.threshold == pytest.approx(report.frozen_threshold)


def test_both_thresholds_are_reported_for_both_scored_splits(
    report: ExperimentReport,
) -> None:
    # At 0.1% positives the fixed and fitted thresholds can tell different stories,
    # and reporting only one would be a choice disguised as a measurement.
    for split in (SplitName.VALIDATION, SplitName.TEST):
        assert report.result_for(split, ThresholdPolicy.FIXED) is not None
        assert report.result_for(split, ThresholdPolicy.VALIDATION_F1) is not None


def test_the_train_split_is_never_scored_into_a_result_row(
    report: ExperimentReport,
) -> None:
    assert {result.split for result in report.results} == {
        SplitName.VALIDATION,
        SplitName.TEST,
    }


# --- Checkpoints -------------------------------------------------------------


def test_both_checkpoints_land_on_disk_at_separate_paths(
    harness: _Harness, report: ExperimentReport
) -> None:
    best = harness.checkpoints.path_for(report.experiment_id, CheckpointKind.BEST)
    final = harness.checkpoints.path_for(report.experiment_id, CheckpointKind.FINAL)

    assert best.is_file()
    assert final.is_file()
    assert best != final


def test_the_best_checkpoint_reloads_into_the_architecture_that_wrote_it(
    harness: _Harness, report: ExperimentReport
) -> None:
    reloaded = harness.checkpoints.load(
        report.experiment_id,
        CheckpointKind.BEST,
        expected_architecture=report.config.architecture,
    )

    assert reloaded.epoch == report.selected_epoch
    assert reloaded.config == report.config
    assert reloaded.feature_dimension == report.provenance.feature_dimension


def test_the_checkpoint_carries_the_frozen_threshold(
    harness: _Harness, report: ExperimentReport
) -> None:
    # The threshold is part of the decision rule; a checkpoint without it would
    # score differently when reloaded.
    reloaded = harness.checkpoints.load(report.experiment_id, CheckpointKind.BEST)

    assert reloaded.threshold == pytest.approx(report.frozen_threshold)


def test_the_final_checkpoint_records_the_last_epoch(
    harness: _Harness, report: ExperimentReport
) -> None:
    reloaded = harness.checkpoints.load(report.experiment_id, CheckpointKind.FINAL)

    assert reloaded.epoch == report.epochs[-1].epoch


def test_the_stored_parameters_reproduce_the_reported_scores(
    harness: _Harness, report: ExperimentReport
) -> None:
    # What makes the checkpoint a result rather than a file: reloading it and
    # scoring the test split again gives the numbers the report already published.
    reloaded = harness.checkpoints.load(report.experiment_id, CheckpointKind.BEST)
    test = harness.provider.load(SplitName.TEST, report.config)
    engine = TorchTrainingEngine(harness.metrics)

    predictions = engine.predict(report.config, test, reloaded.parameters)
    rescored = harness.metrics.evaluate(
        predictions.labels,
        predictions.probabilities,
        threshold=report.frozen_threshold,
    )
    published = report.result_for(SplitName.TEST, report.config.threshold_policy)

    assert published is not None
    assert rescored.recall == pytest.approx(published.metrics.recall)
    assert rescored.false_negatives == published.metrics.false_negatives
    assert rescored.pr_auc == pytest.approx(published.metrics.pr_auc, abs=1e-6)


# --- What was written --------------------------------------------------------


def test_the_report_file_matches_the_returned_report(
    harness: _Harness, report: ExperimentReport
) -> None:
    payload = json.loads(
        harness.results.report_path(report.experiment_id).read_text(encoding="utf-8")
    )

    assert payload["experiment_id"] == report.experiment_id
    assert payload["selected_epoch"] == report.selected_epoch
    assert payload["frozen_threshold"] == pytest.approx(report.frozen_threshold)
    assert len(payload["results"]) == len(report.results)
    assert len(payload["training_curve"]) == len(report.epochs)


def test_the_table_gains_one_row_per_result(
    harness: _Harness, report: ExperimentReport
) -> None:
    rows = harness.results.load_rows()

    assert len(rows) == len(report.results)
    assert {row["experiment_id"] for row in rows} == {report.experiment_id}


def test_every_written_row_carries_the_phase_four_column_values(
    harness: _Harness, report: ExperimentReport
) -> None:
    # The three questions a centralized run can answer are answered; the three that
    # belong to a federated run are blank rather than zero.
    for row in harness.results.load_rows():
        assert row["training_mode"] == TrainingMode.CENTRALIZED.value
        assert row["data_distribution"] == "centralized"
        assert row["federated_algorithm"] == "none"
        assert row["loss_type"] == LOSS_TYPE
        assert row["client_id"] == ""
        assert row["round"] == ""
        assert row["mu"] == ""
        assert row["model"] == report.config.architecture.value


def test_false_negatives_are_written_as_their_own_column(
    harness: _Harness,
) -> None:
    for row in harness.results.load_rows():
        assert row["fn"] != ""
        assert row["false_negatives"] == row["fn"]


def test_the_written_hyperparameters_describe_the_run(
    harness: _Harness, report: ExperimentReport
) -> None:
    stored = harness.results.load_rows()[0]["hyperparameters"]

    assert isinstance(stored, dict)
    assert stored["hidden_size"] == report.config.hidden_size
    assert stored["random_seed"] == report.config.random_seed


def test_a_second_run_appends_rather_than_replacing(harness: _Harness) -> None:
    first = harness.run()
    second = harness.run(make_config(random_seed=18))

    identifiers = {row["experiment_id"] for row in harness.results.load_rows()}

    assert first.experiment_id != second.experiment_id
    assert identifiers == {first.experiment_id, second.experiment_id}


def test_two_runs_write_to_separate_checkpoint_directories(
    harness: _Harness,
) -> None:
    # A second run must not silently overwrite the first run's best parameters.
    first = harness.run()
    second = harness.run(make_config(random_seed=18))

    assert harness.checkpoints.path_for(
        first.experiment_id, CheckpointKind.BEST
    ).is_file()
    assert harness.checkpoints.path_for(
        second.experiment_id, CheckpointKind.BEST
    ).is_file()


# --- One configuration, three architectures ----------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_runs_through_the_same_pipeline(
    harness: _Harness, architecture: ModelArchitecture
) -> None:
    report = harness.run(make_config(architecture=architecture))

    assert report.config.architecture is architecture
    assert report.parameter_count > 0
    assert report.result_for(SplitName.TEST, ThresholdPolicy.FIXED) is not None


def test_the_three_baselines_see_identical_data(harness: _Harness) -> None:
    """No architecture gets information the others do not.

    The provenance and the sequence strategy are compared across all three runs,
    which is where a per-architecture data difference would show up.
    """
    reports = [
        harness.run(make_config(architecture=architecture))
        for architecture in ORDERED_ARCHITECTURES
    ]

    first = reports[0]
    for report in reports[1:]:
        assert report.provenance.to_dict() == first.provenance.to_dict()
        assert report.sequences == first.sequences
        assert report.results[0].sequence_count == first.results[0].sequence_count


def test_the_three_baselines_differ_only_in_architecture(harness: _Harness) -> None:
    base = make_config()

    reports = [
        harness.run(base.with_architecture(architecture))
        for architecture in ORDERED_ARCHITECTURES
    ]

    baseline = reports[0].config.to_dict()
    for report in reports[1:]:
        assert report.config.to_dict() | {
            "architecture": baseline["architecture"]
        } == baseline


def test_the_bidirectional_model_is_larger_than_the_one_directional_one(
    harness: _Harness,
) -> None:
    # The doubled hidden representation is accounted for rather than dropped.
    lstm = harness.run(make_config(architecture=ModelArchitecture.LSTM))
    bilstm = harness.run(make_config(architecture=ModelArchitecture.BILSTM))

    assert bilstm.parameter_count > lstm.parameter_count


# --- Reproducibility ---------------------------------------------------------


def test_the_same_seed_reproduces_the_same_measurements(tmp_path: Path) -> None:
    # Two independently composed pipelines over identically generated trees.
    first = _Harness(tmp_path / "a").run(make_config())
    second = _Harness(tmp_path / "b").run(make_config())

    left = first.result_for(SplitName.TEST, ThresholdPolicy.FIXED)
    right = second.result_for(SplitName.TEST, ThresholdPolicy.FIXED)

    assert left is not None
    assert right is not None
    assert left.metrics.to_dict() == right.metrics.to_dict()
    assert first.frozen_threshold == pytest.approx(second.frozen_threshold)


def test_a_different_seed_gives_a_different_model(tmp_path: Path) -> None:
    first = _Harness(tmp_path / "a").run(make_config(random_seed=1))
    second = _Harness(tmp_path / "b").run(make_config(random_seed=2))

    assert first.epochs[0].train_loss != second.epochs[0].train_loss


# --- Reduced runs are labelled ------------------------------------------------


def test_a_row_limited_run_is_marked_as_reduced_in_its_own_record(
    harness: _Harness,
) -> None:
    # So a smoke run's numbers cannot be read as a full run's.
    report = harness.run(make_config(row_limit=60))

    assert report.config.is_reduced_run
    stored = harness.results.load_rows()[-1]["hyperparameters"]
    assert isinstance(stored, dict)
    assert stored["row_limit"] == 60


def test_a_reduced_run_reads_fewer_rows(harness: _Harness) -> None:
    full = harness.run(make_config())
    reduced = _Harness(harness.processed.parent.parent / "reduced")
    limited = reduced.run(make_config(row_limit=40))

    assert (
        limited.provenance.rows_per_split[SplitName.TRAIN.value]
        < full.provenance.rows_per_split[SplitName.TRAIN.value]
    )


def test_a_reduced_run_says_so_in_the_log(
    harness: _Harness, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        harness.run(make_config(row_limit=60))

    assert "REDUCED run" in caplog.text


# --- Failure surfaces --------------------------------------------------------


def test_a_missing_processed_tree_is_refused_at_construction(tmp_path: Path) -> None:
    # Running `train` before `preprocess` fails with an explanation, not a
    # traceback from a CSV reader.
    with pytest.raises(ProcessedMetadataError) as caught:
        ProcessedSequenceProvider(tmp_path / "never_preprocessed")

    assert "does not exist" in str(caught.value)


def test_a_missing_split_file_names_the_split(harness: _Harness) -> None:
    (harness.processed / SplitName.TEST.value / "transactions.csv").unlink()

    with pytest.raises(ProcessedMetadataError) as caught:
        harness.run()

    assert "test" in str(caught.value)


def test_a_split_out_of_chronological_order_is_refused(tmp_path: Path) -> None:
    """Sequences take their order from row position, so the order is verified."""
    harness = _Harness(tmp_path)
    path = harness.processed / SplitName.TRAIN.value / "transactions.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8"
    )

    with pytest.raises(Exception, match="chronological|order"):
        harness.run()


def test_a_run_that_fails_writes_no_result_row(harness: _Harness) -> None:
    # A partially written table would be worse than none: the row would look like
    # a completed experiment.
    (harness.processed / SplitName.TEST.value / "transactions.csv").unlink()

    with pytest.raises(ProcessedMetadataError):
        harness.run()

    assert harness.results.load_rows() == []


# --- Selection metrics -------------------------------------------------------


@pytest.mark.parametrize("metric", list(SelectionMetric))
def test_every_selection_metric_picks_an_epoch(
    harness: _Harness, metric: SelectionMetric
) -> None:
    report = harness.run(make_config(selection_metric=metric))

    assert 1 <= report.selected_epoch <= len(report.epochs)
    assert report.epochs[report.selected_epoch - 1].selected


def test_the_selection_score_is_recorded_beside_the_metric_it_came_from(
    harness: _Harness,
) -> None:
    report = harness.run(make_config(selection_metric=SelectionMetric.RECALL))

    for record in report.epochs:
        assert f"selection_{SelectionMetric.RECALL.value}" in record.validation_metrics


def test_early_stopping_can_end_a_run_before_its_budget(harness: _Harness) -> None:
    # Not asserted to trigger — that depends on the synthetic data — but if it did,
    # the report must say so and the curve must be shorter than the budget.
    report = harness.run(make_config(epochs=6, early_stopping_patience=1))

    if report.early_stopped:
        assert len(report.epochs) < 6
    else:
        assert len(report.epochs) == 6


# --- The non-negotiables ------------------------------------------------------


def test_no_written_artefact_mentions_a_credential(
    harness: _Harness, report: ExperimentReport
) -> None:
    text = harness.results.report_path(report.experiment_id).read_text(
        encoding="utf-8"
    ).lower()
    text += harness.results.table_path.read_text(encoding="utf-8").lower()

    for word in ("kaggle", "password", "token", "api_key", "secret"):
        assert word not in text


def test_nothing_federated_is_claimed_anywhere_in_the_report(
    harness: _Harness, report: ExperimentReport
) -> None:
    # Phase 4 does not implement federated learning, and its output must not read
    # as though it had.
    payload = json.loads(
        harness.results.report_path(report.experiment_id).read_text(encoding="utf-8")
    )
    text = json.dumps(payload).lower()

    for word in ("fedavg", "fedprox", "aggregation", "non_iid", "non-iid"):
        assert word not in text
    for result in payload["results"]:
        assert result["client_id"] is None
        assert result["round"] is None
        assert result["mu"] is None


def test_the_loss_is_recorded_as_the_unweighted_objective(
    report: ExperimentReport,
) -> None:
    for result in report.results:
        assert result.loss_type == LOSS_TYPE


def test_the_synthetic_tree_is_the_only_data_this_test_reads(
    harness: _Harness, tmp_path: Path
) -> None:
    """Nothing here can reach ``data/processed/``."""
    assert tmp_path in harness.processed.parents
    assert "data" not in harness.processed.relative_to(tmp_path).parts


def _split_files(root: Path) -> Iterator[Path]:
    """Every split CSV under a processed tree."""
    return root.glob("*/transactions.csv")


def test_the_processed_tree_is_left_unmodified_by_a_run(harness: _Harness) -> None:
    # Training reads; it never writes back into the data it was given.
    before = {
        path.name: path.read_bytes() for path in _split_files(harness.processed)
    }

    harness.run()

    after = {path.name: path.read_bytes() for path in _split_files(harness.processed)}
    assert after == before
