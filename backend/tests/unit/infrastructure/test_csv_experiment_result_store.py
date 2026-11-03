"""Tests for :mod:`infrastructure.persistence.csv_experiment_result_store`.

The store writes the table the final comparison will read, so the tests are about
whether that table stays trustworthy as it grows:

- **the header is fixed at the first append**, and a later run whose rows have a
  different shape is refused rather than written misaligned;
- **a not-applicable field stays empty** on the way out and on the way back, because
  a blank cell coerced to ``0.0`` would turn "centralized run, no federated round"
  into a measurement;
- **the required Phase 4 column values are present** — ``training_mode``,
  ``data_distribution`` and ``federated_algorithm`` are answered, the three
  genuinely inapplicable fields are null;
- **rows accumulate in append order**, so the table is a log rather than a snapshot.

Every fixture here is a hand-built report. Nothing is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.dto.experiment import (
    CENTRALIZED_DISTRIBUTION,
    NO_FEDERATED_ALGORITHM,
    ClassificationMetrics,
    ConfusionCounts,
    DataProvenance,
    ExperimentReport,
    ExperimentResult,
    SequenceStrategy,
)
from application.dto.training import EpochRecord, TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy, TrainingMode
from infrastructure.persistence.csv_experiment_result_store import (
    CENTRALIZED_RESULTS_TABLE,
    JSON_COLUMNS,
    RESULTS_TABLE,
    CsvExperimentResultStore,
    HeaderMismatchError,
)

pytestmark = pytest.mark.unit

EXPERIMENT = "gru_centralized_seed42"
LOSS_TYPE = "bce_with_logits"


def make_metrics(*, false_negatives: int = 3, threshold: float = 0.04) -> ClassificationMetrics:
    """Metrics with recognisable counts."""
    return ClassificationMetrics(
        precision=0.25,
        recall=0.5,
        f1=1 / 3,
        pr_auc=0.2,
        roc_auc=0.8,
        accuracy=0.999,
        confusion=ConfusionCounts(
            true_positives=3,
            true_negatives=900,
            false_positives=9,
            false_negatives=false_negatives,
        ),
        threshold=threshold,
        positive_rate=0.006,
        loss=0.012,
    )


def make_result(
    split: SplitName = SplitName.TEST,
    *,
    experiment_id: str = EXPERIMENT,
    policy: ThresholdPolicy = ThresholdPolicy.VALIDATION_F1,
    architecture: ModelArchitecture = ModelArchitecture.GRU,
    **overrides: object,
) -> ExperimentResult:
    """One result row with the Phase 4 defaults left in place."""
    values: dict[str, object] = {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "training_mode": TrainingMode.CENTRALIZED,
        "split": split,
        "metrics": make_metrics(),
        "threshold_policy": policy,
        "seed": 42,
        "loss_type": LOSS_TYPE,
        "training_time_seconds": 12.5,
        "evaluation_time_seconds": 1.25,
        "sequence_count": 915,
        "hyperparameters": {"hidden_size": 64, "optimizer": "adam"},
    }
    values.update(overrides)
    return ExperimentResult(**values)  # type: ignore[arg-type]


def make_report(
    *results: ExperimentResult, experiment_id: str = EXPERIMENT
) -> ExperimentReport:
    """A report around the given rows."""
    return ExperimentReport(
        experiment_id=experiment_id,
        config=TrainingConfig(architecture=ModelArchitecture.GRU, hidden_size=64),
        provenance=DataProvenance(
            processed_directory="data/processed/ibm_aml/HI-Small",
            preprocessing_version="test-0.0.0",
            feature_columns=("amount_log", "same_bank"),
            identifier_columns=("timestamp", "from_account", "to_account"),
            target_column="is_laundering",
            feature_dimension=5,
            rows_per_split={"train": 120},
            positives_per_split={"train": 18},
        ),
        sequences=SequenceStrategy(
            grouping_column="from_account",
            sequence_length=10,
            stride=1,
            target_assignment="newest transaction in the window",
            padding="right-padded with zeros",
            boundary_policy="no sequence crosses train/validation or validation/test",
            ordering="row order, verified non-decreasing by timestamp",
        ),
        epochs=(
            EpochRecord(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                validation_metrics={"recall": 0.5},
                seconds=1.0,
                selected=True,
            ),
        ),
        results=results,
        selected_epoch=1,
        frozen_threshold=0.04,
        threshold_selected_on=SplitName.VALIDATION.value,
        device="cpu",
        parameter_count=1234,
        training_seconds=12.5,
        data_preparation_seconds=2.0,
        total_seconds=16.0,
        early_stopped=False,
        torch_version="0.0.0-test",
        deterministic=False,
    )


@pytest.fixture
def store(tmp_path: Path) -> CsvExperimentResultStore:
    """A store over a temporary root."""
    return CsvExperimentResultStore(tmp_path / "results")


# --- The report file ---------------------------------------------------------


def test_the_report_is_written_under_its_own_experiment_name(
    store: CsvExperimentResultStore,
) -> None:
    path = store.save_report(make_report(make_result()))

    assert path == store.report_path(EXPERIMENT)
    assert path.name == f"{EXPERIMENT}.json"
    assert path.is_file()


def test_the_report_holds_the_whole_run(
    store: CsvExperimentResultStore,
) -> None:
    path = store.save_report(make_report(make_result()))

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == EXPERIMENT
    assert payload["configuration"]["hidden_size"] == 64
    assert payload["data_provenance"]["model_input_dimension"] == 5
    assert payload["sequence_strategy"]["sequence_length"] == 10
    assert len(payload["training_curve"]) == 1
    assert len(payload["results"]) == 1


def test_the_report_records_the_provenance_needed_to_trace_a_number(
    store: CsvExperimentResultStore,
) -> None:
    path = store.save_report(make_report(make_result()))
    provenance = json.loads(path.read_text(encoding="utf-8"))["data_provenance"]

    assert provenance["processed_directory"] == "data/processed/ibm_aml/HI-Small"
    assert provenance["target_column"] == "is_laundering"
    # The traceability columns are named as excluded, not merely absent.
    assert provenance["identifier_columns_excluded_from_model"] == [
        "timestamp",
        "from_account",
        "to_account",
    ]


def test_the_report_states_the_determinism_answer_rather_than_omitting_it(
    store: CsvExperimentResultStore,
) -> None:
    path = store.save_report(make_report(make_result()))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["deterministic_kernels_requested"] is False
    assert payload["torch_version"] == "0.0.0-test"


def test_two_reports_of_the_same_configuration_differ_only_in_measurements(
    store: CsvExperimentResultStore,
) -> None:
    # sort_keys makes the files diffable, which is how two runs get compared.
    first = store.save_report(make_report(make_result(), experiment_id="run_a"))
    second = store.save_report(make_report(make_result(), experiment_id="run_b"))

    left = first.read_text(encoding="utf-8").replace("run_a", "RUN")
    right = second.read_text(encoding="utf-8").replace("run_b", "RUN")
    assert left == right


def test_resaving_a_report_replaces_it(store: CsvExperimentResultStore) -> None:
    store.save_report(make_report(make_result(SplitName.VALIDATION)))
    store.save_report(make_report(make_result(SplitName.TEST)))

    payload = json.loads(store.report_path(EXPERIMENT).read_text(encoding="utf-8"))
    assert [row["split"] for row in payload["results"]] == ["test"]


# --- The results table -------------------------------------------------------


def test_the_table_is_created_with_a_header_on_the_first_append(
    store: CsvExperimentResultStore,
) -> None:
    path = store.append_rows(make_report(make_result()))

    assert path == store.table_path
    assert path.name in {RESULTS_TABLE, CENTRALIZED_RESULTS_TABLE.name}
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("experiment_id,model,training_mode")


def test_the_header_covers_the_whole_result_schema(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result()))

    header = store.table_path.read_text(encoding="utf-8").splitlines()[0].split(",")

    for name in (
        "experiment_id",
        "model",
        "training_mode",
        "data_distribution",
        "federated_algorithm",
        "client_id",
        "round",
        "loss_type",
        "mu",
        "split",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "threshold",
        "training_time",
        "evaluation_time",
        "seed",
        "hyperparameters",
    ):
        assert name in header, name


def test_a_second_run_appends_rather_than_replacing(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result(experiment_id="first")))
    store.append_rows(make_report(make_result(experiment_id="second")))

    rows = store.load_rows()

    assert [row["experiment_id"] for row in rows] == ["first", "second"]


def test_rows_keep_the_order_they_were_appended_in(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(
        make_report(
            make_result(SplitName.VALIDATION),
            make_result(SplitName.TEST),
        )
    )
    store.append_rows(make_report(make_result(SplitName.TEST, experiment_id="later")))

    rows = store.load_rows()

    assert [(row["experiment_id"], row["split"]) for row in rows] == [
        (EXPERIMENT, "validation"),
        (EXPERIMENT, "test"),
        ("later", "test"),
    ]


def test_the_header_is_written_once_no_matter_how_many_appends(
    store: CsvExperimentResultStore,
) -> None:
    for index in range(3):
        store.append_rows(make_report(make_result(experiment_id=f"run_{index}")))

    text = store.table_path.read_text(encoding="utf-8")
    assert text.count("experiment_id") == 1


def test_an_empty_report_writes_nothing_but_still_names_the_table(
    store: CsvExperimentResultStore,
) -> None:
    # A run that produced no rows must not create a headerless file that the next
    # append would then treat as the header.
    path = store.append_rows(make_report())

    assert path == store.table_path
    assert not path.exists()


def test_appending_after_an_empty_report_still_writes_the_header(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report())
    store.append_rows(make_report(make_result()))

    assert store.table_path.read_text(encoding="utf-8").count("experiment_id") == 1
    assert len(store.load_rows()) == 1


# --- The Phase 4 column values -----------------------------------------------


def test_every_row_answers_the_three_questions_a_centralized_run_can_answer(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(
        make_report(make_result(SplitName.VALIDATION), make_result(SplitName.TEST))
    )

    for row in store.load_rows():
        assert row["training_mode"] == TrainingMode.CENTRALIZED.value
        assert row["data_distribution"] == CENTRALIZED_DISTRIBUTION
        assert row["federated_algorithm"] == NO_FEDERATED_ALGORITHM
        assert row["loss_type"] == LOSS_TYPE


def test_the_inapplicable_federated_fields_are_blank_not_zero(
    store: CsvExperimentResultStore,
) -> None:
    # A blank cell says "not applicable". A 0 would say round zero, or mu zero,
    # both of which are meaningful federated settings this run never had.
    store.append_rows(make_report(make_result()))

    line = store.table_path.read_text(encoding="utf-8").splitlines()[1]
    row = store.load_rows()[0]

    assert row["client_id"] == ""
    assert row["round"] == ""
    assert row["mu"] == ""
    assert ",0," not in line.replace(",0.0,", ",X,")


def test_false_negatives_are_a_column_of_their_own(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result()))

    row = store.load_rows()[0]

    assert row["fn"] == "3"
    assert row["false_negatives"] == "3"


def test_both_threshold_policies_can_sit_in_the_table_together(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(
        make_report(
            make_result(SplitName.TEST, policy=ThresholdPolicy.FIXED),
            make_result(SplitName.TEST, policy=ThresholdPolicy.VALIDATION_F1),
        )
    )

    policies = [row["threshold_policy"] for row in store.load_rows()]
    assert policies == ["fixed", "validation_f1"]


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_each_architecture_is_recorded_under_the_model_column(
    store: CsvExperimentResultStore, architecture: ModelArchitecture
) -> None:
    store.append_rows(make_report(make_result(architecture=architecture)))

    assert store.load_rows()[0]["model"] == architecture.value


# --- The JSON-encoded column -------------------------------------------------


def test_the_hyperparameters_travel_as_one_json_cell(
    store: CsvExperimentResultStore,
) -> None:
    # One cell rather than one column per field, so adding a hyperparameter later
    # does not change the header and trip the mismatch check.
    assert "hyperparameters" in JSON_COLUMNS
    store.append_rows(make_report(make_result()))

    header = store.table_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header.count("hyperparameters") == 1


def test_the_hyperparameters_come_back_as_a_dictionary(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(
        make_report(make_result(hyperparameters={"hidden_size": 64, "epochs": 5}))
    )

    row = store.load_rows()[0]

    assert row["hyperparameters"] == {"hidden_size": 64, "epochs": 5}


def test_an_empty_hyperparameter_cell_reads_back_as_an_empty_mapping(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result(hyperparameters={})))

    assert store.load_rows()[0]["hyperparameters"] == {}


def test_a_hyperparameter_cell_that_is_not_json_is_returned_as_text(
    store: CsvExperimentResultStore,
) -> None:
    # Better than raising: the rest of the row is still a readable result.
    store.append_rows(make_report(make_result()))
    text = store.table_path.read_text(encoding="utf-8")
    store.table_path.write_text(
        text.replace('"{""epochs"": 5, ', "{broken, ").replace(
            '"{""hidden_size"": 64, ""optimizer"": ""adam""}"', "{broken"
        ),
        encoding="utf-8",
    )

    row = store.load_rows()[0]

    assert row["hyperparameters"] == "{broken"
    assert row["experiment_id"] == EXPERIMENT


def test_a_comma_inside_the_json_cell_does_not_split_the_row(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(
        make_report(make_result(hyperparameters={"a": 1, "b": 2, "c": 3}))
    )

    rows = store.load_rows()

    assert len(rows) == 1
    assert rows[0]["hyperparameters"] == {"a": 1, "b": 2, "c": 3}


# --- Header stability --------------------------------------------------------


def test_a_row_shape_that_does_not_match_the_table_is_refused(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result()))
    # Simulate a later phase whose rows carry an extra column.
    text = store.table_path.read_text(encoding="utf-8").splitlines()
    store.table_path.write_text(
        "\n".join([text[0] + ",new_column", *text[1:]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(HeaderMismatchError) as caught:
        store.append_rows(make_report(make_result()))

    assert "new_column" in str(caught.value)


def test_the_refusal_names_both_what_is_new_and_what_is_missing(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result()))
    text = store.table_path.read_text(encoding="utf-8").splitlines()
    store.table_path.write_text(
        "\n".join([text[0].replace("mu", "proximal_mu"), *text[1:]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HeaderMismatchError) as caught:
        store.append_rows(make_report(make_result()))

    message = str(caught.value)
    assert "'mu'" in message
    assert "'proximal_mu'" in message


def test_a_refused_append_leaves_the_table_as_it_was(
    store: CsvExperimentResultStore,
) -> None:
    store.append_rows(make_report(make_result()))
    text = store.table_path.read_text(encoding="utf-8").splitlines()
    store.table_path.write_text(
        "\n".join([text[0] + ",extra", *text[1:]]) + "\n", encoding="utf-8"
    )
    before = store.table_path.read_text(encoding="utf-8")

    with pytest.raises(HeaderMismatchError):
        store.append_rows(make_report(make_result()))

    assert store.table_path.read_text(encoding="utf-8") == before


def test_the_same_schema_appends_without_complaint(
    store: CsvExperimentResultStore,
) -> None:
    for index in range(4):
        store.append_rows(make_report(make_result(experiment_id=f"run_{index}")))

    assert len(store.load_rows()) == 4


# --- Reading back ------------------------------------------------------------


def test_reading_a_table_that_does_not_exist_yet_returns_nothing(
    store: CsvExperimentResultStore,
) -> None:
    assert store.load_rows() == []


def test_values_are_returned_as_written_rather_than_re_typed(
    store: CsvExperimentResultStore,
) -> None:
    # Deliberate: coercing a blank cell to 0.0 would turn "not applicable" into a
    # measurement, and that decision belongs to whoever reads the table.
    store.append_rows(make_report(make_result()))

    row = store.load_rows()[0]

    assert row["recall"] == "0.5"
    assert row["seed"] == "42"


def test_the_root_is_resolved_to_an_absolute_path(tmp_path: Path) -> None:
    store = CsvExperimentResultStore(tmp_path / "a" / ".." / "results")

    assert store.root.is_absolute()
    assert ".." not in store.root.parts


def test_the_reports_directory_and_the_table_are_siblings(
    store: CsvExperimentResultStore,
) -> None:
    store.save_report(make_report(make_result()))
    store.append_rows(make_report(make_result()))

    assert store.report_path(EXPERIMENT).parent.parent == store.root
    assert store.root in store.table_path.parents


def test_explicit_table_path_can_be_sibling_of_reports(tmp_path: Path) -> None:
    store = CsvExperimentResultStore(tmp_path / "results", table_path=RESULTS_TABLE)
    store.save_report(make_report(make_result()))
    store.append_rows(make_report(make_result()))

    assert store.table_path.parent == store.root
