"""Tests for :mod:`application.dto.training`.

:class:`TrainingConfig` is the object that decides whether two runs are comparable,
so the tests are about the ways it could quietly stop describing its own run:

- **a configuration that cannot describe a runnable experiment is rejected at
  construction**, before any data is read, and the message says which field;
- **a frozen record stays frozen**, because a loop that could retune halfway
  through would leave a config that no longer matches the model it produced;
- **``effective_dropout`` reports what the model does**, not what was asked for —
  PyTorch ignores dropout on a single-layer RNN, and a record claiming
  regularisation that never happened is a fabricated experimental detail;
- **``with_architecture`` varies the architecture and nothing else**, which is the
  mechanism behind the ``--model all`` comparability claim;
- **``from_dict`` ∘ ``to_dict`` is the identity**, because a checkpoint is only
  reusable if its configuration comes back exactly.

No data, no model, no training here — these are pure records.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from application.dto.training import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEQUENCE_STRIDE,
    EpochRecord,
    TrainingConfig,
)
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy

pytestmark = pytest.mark.unit


def make_config(**overrides: object) -> TrainingConfig:
    """A configuration with nothing left at its default.

    Every field differs from the default, so a round trip that dropped one would
    show up as a changed value rather than as a coincidentally equal one.
    """
    values: dict[str, object] = {
        "architecture": ModelArchitecture.BILSTM,
        "sequence_length": 7,
        "sequence_stride": 3,
        "hidden_size": 32,
        "num_layers": 2,
        "dropout": 0.15,
        "batch_size": 256,
        "learning_rate": 0.005,
        "optimizer": "adamw",
        "epochs": 9,
        "random_seed": 7,
        "early_stopping_patience": 4,
        "selection_metric": SelectionMetric.RECALL,
        "threshold": 0.03,
        "threshold_policy": ThresholdPolicy.FIXED,
        "one_hot_nominal_codes": False,
        "gradient_clip_norm": 1.5,
        "row_limit": 5000,
        "num_workers": 2,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


# --- Defaults ----------------------------------------------------------------


def test_only_the_architecture_is_required() -> None:
    # Everything else has a defensible default, so a smoke run needs one argument.
    config = TrainingConfig(architecture=ModelArchitecture.GRU)

    assert config.architecture is ModelArchitecture.GRU
    assert config.sequence_length == DEFAULT_SEQUENCE_LENGTH
    assert config.sequence_stride == DEFAULT_SEQUENCE_STRIDE


def test_the_default_stride_keeps_every_labelled_transaction() -> None:
    # A larger stride discards targets, which would also shift the class balance
    # away from the rate the provenance record reports.
    assert DEFAULT_SEQUENCE_STRIDE == 1


def test_the_defaults_select_on_a_ranking_metric_not_accuracy() -> None:
    config = TrainingConfig(architecture=ModelArchitecture.GRU)

    assert config.selection_metric is SelectionMetric.PR_AUC
    assert config.selection_metric.higher_is_better


def test_the_default_threshold_policy_learns_from_validation() -> None:
    config = TrainingConfig(architecture=ModelArchitecture.GRU)

    assert config.threshold_policy is ThresholdPolicy.VALIDATION_F1
    assert config.threshold_policy.is_learned


def test_the_default_run_reads_whole_splits() -> None:
    config = TrainingConfig(architecture=ModelArchitecture.GRU)

    assert config.row_limit is None
    assert config.is_reduced_run is False


def test_the_default_device_is_chosen_at_run_time() -> None:
    # None means "resolve it now, on this machine", so a config written on a GPU
    # box still runs here.
    assert TrainingConfig(architecture=ModelArchitecture.GRU).device is None


# --- Immutability ------------------------------------------------------------


def test_a_configuration_cannot_be_retuned_after_construction() -> None:
    config = make_config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.learning_rate = 0.9  # type: ignore[misc]


def test_a_configuration_has_no_instance_dictionary_to_smuggle_fields_into() -> None:
    # slots=True: an adapter cannot attach an undeclared hyperparameter that the
    # record would then fail to report.
    with pytest.raises(AttributeError):
        make_config().undeclared = 1  # type: ignore[attr-defined]


def test_two_configurations_with_the_same_values_are_equal() -> None:
    # Equality is what lets a checkpoint round trip be asserted at all.
    assert make_config() == make_config()


# --- Rejected values ---------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["sequence_length", "sequence_stride", "hidden_size", "num_layers", "batch_size", "epochs"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_count_is_refused_by_name(field: str, value: int) -> None:
    # Named, because "must be a positive integer" alone would not say which knob.
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        make_config(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_a_dropout_outside_the_unit_interval_is_refused(value: float) -> None:
    # 1.0 would zero every activation, which is not regularisation.
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        make_config(dropout=value)


def test_zero_dropout_is_allowed() -> None:
    assert make_config(dropout=0.0).dropout == 0.0


@pytest.mark.parametrize("value", [0.0, -0.001])
def test_a_non_positive_learning_rate_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        make_config(learning_rate=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.2, 1.2])
def test_a_threshold_outside_the_open_unit_interval_is_refused(value: float) -> None:
    # 0.0 alerts on everything and 1.0 on nothing; neither is a decision rule.
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\)"):
        make_config(threshold=value)


def test_a_negative_patience_is_refused() -> None:
    with pytest.raises(ValueError, match="early_stopping_patience cannot be negative"):
        make_config(early_stopping_patience=-1)


def test_zero_patience_is_allowed_and_means_no_early_stopping() -> None:
    assert make_config(early_stopping_patience=0).early_stopping_patience == 0


def test_a_negative_worker_count_is_refused() -> None:
    with pytest.raises(ValueError, match="num_workers cannot be negative"):
        make_config(num_workers=-1)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_non_positive_gradient_clip_is_refused(value: float) -> None:
    # None disables clipping; 0.0 would zero every gradient and train nothing.
    with pytest.raises(ValueError, match="gradient_clip_norm must be positive or None"):
        make_config(gradient_clip_norm=value)


def test_no_gradient_clipping_is_expressed_as_none() -> None:
    assert make_config(gradient_clip_norm=None).gradient_clip_norm is None


@pytest.mark.parametrize("value", [0, -10])
def test_a_non_positive_row_limit_is_refused(value: int) -> None:
    with pytest.raises(ValueError, match="row_limit must be positive or None"):
        make_config(row_limit=value)


def test_the_rejection_reports_the_value_it_saw() -> None:
    # So a bad CLI argument is traceable to what was typed.
    with pytest.raises(ValueError, match="-3"):
        make_config(hidden_size=-3)


# --- Derived answers ---------------------------------------------------------


def test_dropout_on_a_single_layer_reports_as_zero() -> None:
    # PyTorch ignores it and warns. Reporting 0.3 would be a fabricated detail.
    config = make_config(num_layers=1, dropout=0.3)

    assert config.dropout == pytest.approx(0.3)
    assert config.effective_dropout == 0.0


def test_dropout_on_stacked_layers_reports_as_requested() -> None:
    config = make_config(num_layers=3, dropout=0.3)

    assert config.effective_dropout == pytest.approx(0.3)


def test_a_row_limited_run_is_marked_as_reduced() -> None:
    # The flag exists so a smoke run's numbers can never be read as a full run's.
    assert make_config(row_limit=1000).is_reduced_run is True
    assert make_config(row_limit=None).is_reduced_run is False


# --- Varying one thing -------------------------------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_with_architecture_changes_the_architecture(
    architecture: ModelArchitecture,
) -> None:
    assert make_config().with_architecture(architecture).architecture is architecture


def test_with_architecture_changes_nothing_else() -> None:
    # The mechanism behind the comparability claim: `--model all` runs one
    # configuration three times, so no architecture gets a tuned advantage.
    original = make_config()

    derived = original.with_architecture(ModelArchitecture.LSTM)

    assert derived.to_dict() | {"architecture": original.architecture.value} == (
        original.to_dict()
    )


def test_with_architecture_leaves_the_original_alone() -> None:
    original = make_config(architecture=ModelArchitecture.GRU)

    original.with_architecture(ModelArchitecture.BILSTM)

    assert original.architecture is ModelArchitecture.GRU


def test_with_architecture_carries_every_field() -> None:
    # A field added later and forgotten here would silently revert to its default
    # for two of the three baselines.
    original = make_config()

    derived = original.with_architecture(ModelArchitecture.GRU)

    for field in dataclasses.fields(TrainingConfig):
        if field.name != "architecture":
            assert getattr(derived, field.name) == getattr(original, field.name), field.name


# --- Serialisation -----------------------------------------------------------


def test_every_field_appears_in_the_serialised_form() -> None:
    payload = make_config().to_dict()

    for field in dataclasses.fields(TrainingConfig):
        assert field.name in payload, field.name


def test_the_serialised_form_holds_enum_values_not_enum_objects() -> None:
    # It goes into JSON and a torch payload loaded under weights_only=True.
    payload = make_config().to_dict()

    assert payload["architecture"] == ModelArchitecture.BILSTM.value
    assert payload["selection_metric"] == SelectionMetric.RECALL.value
    assert payload["threshold_policy"] == ThresholdPolicy.FIXED.value
    assert isinstance(payload["architecture"], str)


def test_the_serialised_form_reports_the_dropout_that_applies_as_well() -> None:
    payload = make_config(num_layers=1, dropout=0.4).to_dict()

    assert payload["dropout"] == pytest.approx(0.4)
    assert payload["effective_dropout"] == 0.0


def test_a_configuration_survives_a_round_trip() -> None:
    original = make_config()

    assert TrainingConfig.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_survives_a_round_trip(
    architecture: ModelArchitecture,
) -> None:
    original = make_config(architecture=architecture)

    assert TrainingConfig.from_dict(original.to_dict()) == original


def test_the_derived_dropout_is_ignored_on_the_way_back() -> None:
    # to_dict emits effective_dropout for readers; from_dict must not treat it as
    # a field, or a single-layer round trip would lose the requested value.
    original = make_config(num_layers=1, dropout=0.4)

    rebuilt = TrainingConfig.from_dict(original.to_dict())

    assert rebuilt.dropout == pytest.approx(0.4)


def test_an_absent_architecture_is_refused_rather_than_guessed() -> None:
    # Any default would build a model of the wrong kind under the right name.
    payload: dict[str, Any] = make_config().to_dict()
    del payload["architecture"]

    with pytest.raises(KeyError):
        TrainingConfig.from_dict(payload)


def test_an_unknown_architecture_is_refused() -> None:
    with pytest.raises(ValueError, match="transformer"):
        TrainingConfig.from_dict({"architecture": "transformer"})


def test_a_payload_written_before_a_field_existed_still_loads() -> None:
    # Forward compatibility for old checkpoints: absent keys take field defaults.
    config = TrainingConfig.from_dict({"architecture": ModelArchitecture.GRU.value})

    assert config == TrainingConfig(architecture=ModelArchitecture.GRU)


def test_the_optional_fields_come_back_as_none_rather_than_zero() -> None:
    # 0.0 clipping, a 0-row limit and a "" device are all different claims from
    # "not set", and two of the three would fail validation.
    original = make_config(gradient_clip_norm=None, row_limit=None, device=None)

    rebuilt = TrainingConfig.from_dict(original.to_dict())

    assert rebuilt.gradient_clip_norm is None
    assert rebuilt.row_limit is None
    assert rebuilt.device is None


def test_an_empty_device_string_reads_back_as_unset() -> None:
    assert TrainingConfig.from_dict(
        {"architecture": ModelArchitecture.GRU.value, "device": ""}
    ).device is None


def test_stored_numbers_are_coerced_rather_than_carried_as_text() -> None:
    # A CSV or JSON round trip through a loose writer can hand back strings.
    config = TrainingConfig.from_dict(
        {
            "architecture": ModelArchitecture.GRU.value,
            "hidden_size": "48",
            "learning_rate": "0.002",
            "one_hot_nominal_codes": True,
        }
    )

    assert config.hidden_size == 48
    assert config.learning_rate == pytest.approx(0.002)


def test_a_stored_value_that_is_out_of_range_is_still_refused() -> None:
    # from_dict runs the same validation, so a hand-edited checkpoint cannot
    # smuggle in a configuration a fresh run would have rejected.
    payload = make_config().to_dict() | {"epochs": 0}

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        TrainingConfig.from_dict(payload)


# --- The training curve ------------------------------------------------------


def test_an_epoch_record_defaults_to_no_metrics_and_not_selected() -> None:
    record = EpochRecord(epoch=1, train_loss=0.5, validation_loss=0.4)

    assert record.validation_metrics == {}
    assert record.seconds == 0.0
    assert record.selected is False


def test_an_epoch_record_is_frozen() -> None:
    record = EpochRecord(epoch=1, train_loss=0.5, validation_loss=0.4)

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.selected = True  # type: ignore[misc]


def test_an_epoch_record_serialises_its_whole_row() -> None:
    payload = EpochRecord(
        epoch=3,
        train_loss=0.25,
        validation_loss=0.3,
        validation_metrics={"recall": 0.5, "pr_auc": 0.2},
        seconds=61.5,
        selected=True,
    ).to_dict()

    assert payload == {
        "epoch": 3,
        "train_loss": 0.25,
        "validation_loss": 0.3,
        "validation_metrics": {"recall": 0.5, "pr_auc": 0.2},
        "seconds": 61.5,
        "selected": True,
    }


def test_the_serialised_metrics_are_a_copy() -> None:
    # The record is frozen, but a shared inner dict would let a reader edit the
    # curve after the fact.
    record = EpochRecord(
        epoch=1, train_loss=0.5, validation_loss=0.4, validation_metrics={"recall": 0.5}
    )

    record.to_dict()["validation_metrics"]["recall"] = 0.9

    assert record.validation_metrics == {"recall": 0.5}


def test_each_epoch_record_gets_its_own_metrics_mapping() -> None:
    first = EpochRecord(epoch=1, train_loss=0.5, validation_loss=0.4)
    second = EpochRecord(epoch=2, train_loss=0.4, validation_loss=0.3)

    first.validation_metrics["recall"] = 0.5

    assert second.validation_metrics == {}
