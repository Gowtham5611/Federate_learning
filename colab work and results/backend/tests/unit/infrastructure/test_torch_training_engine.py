"""Tests for :mod:`infrastructure.models.pytorch.torch_training_engine`.

The engine is one loop shared by three architectures, and these tests check the
things that a shared loop makes possible and a bug would quietly undo:

- **the loss is unweighted binary cross-entropy over logits** — no class
  weighting, no focal loss, no resampling, because the natural imbalance is what
  these baselines exist to measure;
- **best and final states are separate snapshots**, and the best one is a detached
  copy rather than a live view of a model that keeps training;
- **the same seed reproduces the run**, and a different seed does not;
- **``fit`` is reusable for a later local training round**: it accepts starting
  parameters, leaves them unmodified, and returns state instead of mutating
  hidden global state;
- **scoring does not shuffle**, so predictions stay aligned with labels and rows.

Models here are hidden size 4 over a dozen synthetic sequences, so a full run
takes milliseconds. The real processed dataset is not involved.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from application.dto.training import EpochRecord, TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import SelectionMetric
from domain.exceptions.training import FeatureDimensionMismatchError
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
)
from infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from infrastructure.models.pytorch.reproducibility import seed_everything
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import (
    INPUT_WEIGHT_KEY,
    TorchTrainingEngine,
)
from tests.fixtures.training_doubles import make_dataset

pytestmark = pytest.mark.unit

FEATURES = 4


def make_config(**overrides: object) -> TrainingConfig:
    """A configuration that trains in milliseconds."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "sequence_length": 3,
        "hidden_size": 4,
        "batch_size": 4,
        "epochs": 2,
        "learning_rate": 0.05,
        "early_stopping_patience": 0,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


@pytest.fixture
def engine() -> TorchTrainingEngine:
    """An engine over the real metrics calculator."""
    return TorchTrainingEngine(SklearnMetricsCalculator())


def train_split(**overrides: int) -> AccountSequenceDataset:
    """A training dataset of the fixture width."""
    return make_dataset(SplitName.TRAIN, feature_dimension=FEATURES, **overrides)


def validation_split(**overrides: int) -> AccountSequenceDataset:
    """A validation dataset of the fixture width."""
    return make_dataset(SplitName.VALIDATION, feature_dimension=FEATURES, **overrides)


# --- The loss ----------------------------------------------------------------


def test_reported_loss_is_the_unweighted_mean_over_every_sequence(
    engine: TorchTrainingEngine,
) -> None:
    # Recompute binary cross-entropy from the engine's own probabilities, giving
    # every sequence equal weight. A positive class weight, a focal term or any
    # resampling would move the reported figure away from this one.
    config = make_config(epochs=1)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)
    scored = engine.predict(config, validation, outcome.best_state)

    probabilities = np.clip(scored.probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    labels = scored.labels.astype(np.float64)
    per_sequence = -(
        labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    )

    assert scored.mean_loss == pytest.approx(float(per_sequence.mean()), rel=1e-3)


def test_positives_are_not_upweighted(engine: TorchTrainingEngine) -> None:
    # The unweighted mean and a positive-weighted mean differ whenever the classes
    # are unbalanced and the losses are not identical, which is the case here. The
    # reported loss must be the unweighted one.
    config = make_config(epochs=1)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)
    scored = engine.predict(config, validation, outcome.best_state)

    probabilities = np.clip(scored.probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    labels = scored.labels.astype(np.float64)
    per_sequence = -(
        labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    weights = np.where(labels > 0.5, 10.0, 1.0)
    weighted = float((weights * per_sequence).sum() / weights.sum())

    assert scored.mean_loss != pytest.approx(weighted, rel=1e-3)


def test_training_loss_is_a_finite_positive_number(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(make_config(), train_split(), validation_split())

    for record in outcome.epochs:
        assert np.isfinite(record.train_loss)
        assert record.train_loss > 0.0


def test_training_moves_the_weights(engine: TorchTrainingEngine) -> None:
    # A check that gradients actually reach the encoder. `fit` seeds before
    # building, so the untrained parameters are reconstructible exactly.
    config = make_config(epochs=1)
    outcome = engine.fit(config, train_split(), validation_split())

    seed_everything(config.random_seed)
    untrained = build_classifier(config, FEATURES).state_dict()

    assert not torch.equal(
        untrained[INPUT_WEIGHT_KEY], outcome.final_state[INPUT_WEIGHT_KEY]
    )


# --- Best and final states ---------------------------------------------------


def test_best_and_final_states_differ_when_the_last_epoch_was_not_the_best(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(make_config(epochs=3), train_split(), validation_split())

    if outcome.best_epoch != len(outcome.epochs):
        assert not torch.equal(
            outcome.best_state[INPUT_WEIGHT_KEY],
            outcome.final_state[INPUT_WEIGHT_KEY],
        )


def test_stored_states_are_detached_cpu_copies(engine: TorchTrainingEngine) -> None:
    # A state that still shares storage with the live model would keep changing as
    # training continued, and would stop being the parameters whose score was
    # reported.
    outcome = engine.fit(make_config(epochs=3), train_split(), validation_split())

    for state in (outcome.best_state, outcome.final_state):
        for name, tensor in state.items():
            assert isinstance(tensor, torch.Tensor), name
            assert not tensor.requires_grad, name
            assert tensor.device.type == "cpu", name


def test_the_best_state_survives_further_training(
    engine: TorchTrainingEngine,
) -> None:
    # Same run, more epochs: whatever epoch 1 produced must be reproducible from
    # the longer run's history rather than overwritten by later epochs.
    short = engine.fit(make_config(epochs=1), train_split(), validation_split())
    long = engine.fit(make_config(epochs=3), train_split(), validation_split())

    if long.best_epoch == 1:
        assert torch.equal(
            short.best_state[INPUT_WEIGHT_KEY], long.best_state[INPUT_WEIGHT_KEY]
        )


def test_the_selected_epoch_is_the_one_the_metric_prefers(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(
        make_config(epochs=4, selection_metric=SelectionMetric.LOSS),
        train_split(),
        validation_split(),
    )

    best = outcome.epochs[outcome.best_epoch - 1]
    assert best.selected
    # Loss is minimised, so no completed epoch may have scored lower.
    assert best.validation_loss == min(
        record.validation_loss for record in outcome.epochs
    )


def test_pr_auc_selection_maximises_rather_than_minimises(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(
        make_config(epochs=3, selection_metric=SelectionMetric.PR_AUC),
        train_split(),
        validation_split(),
    )

    best = outcome.epochs[outcome.best_epoch - 1]
    scores = [
        record.validation_metrics["selection_pr_auc"] for record in outcome.epochs
    ]
    assert best.validation_metrics["selection_pr_auc"] == max(scores)


def test_the_selection_score_is_recorded_next_to_the_metric_it_came_from(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(
        make_config(epochs=1, selection_metric=SelectionMetric.RECALL),
        train_split(),
        validation_split(),
    )

    metrics = outcome.epochs[0].validation_metrics
    assert metrics["selection_recall"] == pytest.approx(metrics["recall"])


def test_every_epoch_records_the_primary_metrics(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(make_config(epochs=1), train_split(), validation_split())

    metrics = outcome.epochs[0].validation_metrics
    for name in ("recall", "f1", "pr_auc", "precision", "fn"):
        assert name in metrics, name


# --- Early stopping ----------------------------------------------------------


def test_early_stopping_keeps_the_best_epoch_selected(
    engine: TorchTrainingEngine,
) -> None:
    # A learning rate this small barely moves the model, so improvements stop
    # quickly and patience is exercised.
    outcome = engine.fit(
        make_config(epochs=12, early_stopping_patience=1, learning_rate=1e-6),
        train_split(),
        validation_split(),
    )

    assert 1 <= outcome.best_epoch <= len(outcome.epochs)
    assert outcome.epochs[outcome.best_epoch - 1].selected


def test_early_stopping_is_reported_rather_than_inferred(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(
        make_config(epochs=12, early_stopping_patience=1, learning_rate=1e-6),
        train_split(),
        validation_split(),
    )

    assert outcome.early_stopped == (len(outcome.epochs) < 12)


def test_patience_zero_runs_the_full_budget(engine: TorchTrainingEngine) -> None:
    outcome = engine.fit(
        make_config(epochs=3, early_stopping_patience=0),
        train_split(),
        validation_split(),
    )

    assert len(outcome.epochs) == 3
    assert not outcome.early_stopped


def test_epoch_records_are_numbered_from_one_without_gaps(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(make_config(epochs=4), train_split(), validation_split())

    assert [record.epoch for record in outcome.epochs] == [1, 2, 3, 4]


def test_the_epoch_callback_sees_every_epoch_as_it_completes(
    engine: TorchTrainingEngine,
) -> None:
    seen: list[EpochRecord] = []
    outcome = engine.fit(
        make_config(epochs=3),
        train_split(),
        validation_split(),
        on_epoch=seen.append,
    )

    assert [record.epoch for record in seen] == [1, 2, 3]
    assert seen == list(outcome.epochs)


# --- The test split is not an argument ----------------------------------------


def test_fit_takes_only_train_and_validation(engine: TorchTrainingEngine) -> None:
    # The structural half of "never tune on test": there is no parameter through
    # which the test split could reach epoch selection or early stopping.
    from inspect import signature

    parameters = set(signature(engine.fit).parameters)

    assert parameters == {
        "config",
        "train",
        "validation",
        "initial_parameters",
        "on_epoch",
    }


# --- Reproducibility ---------------------------------------------------------


def test_the_same_seed_reproduces_the_run(engine: TorchTrainingEngine) -> None:
    config = make_config(epochs=2, random_seed=99)
    first = engine.fit(config, train_split(), validation_split())
    second = engine.fit(config, train_split(), validation_split())

    assert [record.train_loss for record in first.epochs] == [
        record.train_loss for record in second.epochs
    ]
    for name, tensor in first.best_state.items():
        assert torch.equal(tensor, second.best_state[name]), name


def test_a_different_seed_produces_a_different_run(
    engine: TorchTrainingEngine,
) -> None:
    first = engine.fit(make_config(random_seed=1), train_split(), validation_split())
    second = engine.fit(make_config(random_seed=2), train_split(), validation_split())

    assert not torch.equal(
        first.best_state[INPUT_WEIGHT_KEY], second.best_state[INPUT_WEIGHT_KEY]
    )


def test_each_epoch_shuffles_differently_within_one_seeded_run(
    engine: TorchTrainingEngine,
) -> None:
    # The shuffle seed advances with the epoch, so two epochs are not identical
    # passes; the run as a whole still reproduces from one seed, which the test
    # above checks.
    outcome = engine.fit(make_config(epochs=3), train_split(rows=24), validation_split())

    losses = [record.train_loss for record in outcome.epochs]
    assert len(set(losses)) == len(losses)


def test_the_runtime_is_recorded_as_measured(engine: TorchTrainingEngine) -> None:
    outcome = engine.fit(make_config(epochs=1), train_split(), validation_split())

    assert outcome.device == "cpu"
    assert outcome.extra["framework_version"] == torch.__version__
    assert outcome.extra["device"] == "cpu"
    # Reported, not claimed: False is a legitimate value on this backend.
    assert isinstance(outcome.deterministic, bool)
    assert outcome.extra["deterministic_kernels_requested"] == str(
        outcome.deterministic
    )


def test_training_time_is_measured_not_estimated(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(make_config(epochs=2), train_split(), validation_split())

    assert outcome.training_seconds > 0.0
    assert outcome.training_seconds >= sum(
        record.seconds for record in outcome.epochs
    ) * 0.5


# --- Scoring -----------------------------------------------------------------


def test_predictions_line_up_positionally_with_the_dataset(
    engine: TorchTrainingEngine,
) -> None:
    config = make_config(epochs=1)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)

    scored = engine.predict(config, validation, outcome.best_state)

    assert scored.labels.tolist() == validation.labels.tolist()
    assert scored.row_indices.tolist() == validation.target_rows.tolist()


def test_probabilities_are_in_the_unit_interval(engine: TorchTrainingEngine) -> None:
    config = make_config(epochs=1)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)

    scored = engine.predict(config, validation, outcome.best_state)

    assert float(scored.probabilities.min()) >= 0.0
    assert float(scored.probabilities.max()) <= 1.0


def test_scoring_is_deterministic_for_fixed_parameters(
    engine: TorchTrainingEngine,
) -> None:
    config = make_config(epochs=1)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)

    first = engine.predict(config, validation, outcome.best_state)
    second = engine.predict(config, validation, outcome.best_state)

    assert first.probabilities.tolist() == second.probabilities.tolist()
    assert first.row_indices.tolist() == second.row_indices.tolist()


def test_every_sequence_is_scored_exactly_once(engine: TorchTrainingEngine) -> None:
    config = make_config(epochs=1, batch_size=5)
    validation = validation_split(rows=17)
    outcome = engine.fit(config, train_split(), validation)

    scored = engine.predict(config, validation, outcome.best_state)

    assert len(scored) == validation.sequence_count
    assert sorted(scored.row_indices.tolist()) == sorted(
        validation.target_rows.tolist()
    )


def test_batch_size_does_not_change_the_scores(engine: TorchTrainingEngine) -> None:
    # Different batch sizes mean different padding widths per batch. If padding
    # were read rather than masked, the scores would move.
    validation = validation_split(rows=15)
    outcome = engine.fit(make_config(epochs=1), train_split(), validation)

    small = engine.predict(make_config(batch_size=2), validation, outcome.best_state)
    large = engine.predict(make_config(batch_size=64), validation, outcome.best_state)

    assert small.probabilities == pytest.approx(large.probabilities, abs=1e-6)


def test_the_best_state_and_the_final_state_score_the_same_split_differently(
    engine: TorchTrainingEngine,
) -> None:
    validation = validation_split()
    outcome = engine.fit(make_config(epochs=3), train_split(), validation)

    if outcome.best_epoch != len(outcome.epochs):
        best = engine.predict(make_config(), validation, outcome.best_state)
        final = engine.predict(make_config(), validation, outcome.final_state)
        assert best.probabilities.tolist() != final.probabilities.tolist()


# --- Reusable for a later local training round --------------------------------


def test_fit_accepts_starting_parameters(engine: TorchTrainingEngine) -> None:
    # The property a later federated round needs: start from parameters that came
    # from somewhere else. Nothing federated is implemented here; the seam is only
    # kept open.
    config = make_config(epochs=1)
    first = engine.fit(config, train_split(), validation_split())

    resumed = engine.fit(
        config,
        train_split(),
        validation_split(),
        initial_parameters=first.final_state,
    )

    assert len(resumed.epochs) == 1
    assert not torch.equal(
        first.final_state[INPUT_WEIGHT_KEY], resumed.final_state[INPUT_WEIGHT_KEY]
    )


def test_fit_does_not_mutate_the_parameters_it_was_given(
    engine: TorchTrainingEngine,
) -> None:
    supplied = engine.fit(
        make_config(epochs=1), train_split(), validation_split()
    ).final_state
    before = {name: tensor.clone() for name, tensor in supplied.items()}

    engine.fit(
        make_config(epochs=2),
        train_split(),
        validation_split(),
        initial_parameters=supplied,
    )

    for name, tensor in before.items():
        assert torch.equal(supplied[name], tensor), name


def test_one_engine_carries_nothing_between_runs(
    engine: TorchTrainingEngine,
) -> None:
    # Two identical fits through the same engine instance: the second must start
    # afresh rather than continue the first.
    config = make_config(epochs=1, random_seed=5)
    first = engine.fit(config, train_split(), validation_split())
    second = engine.fit(config, train_split(), validation_split())

    assert torch.equal(
        first.final_state[INPUT_WEIGHT_KEY], second.final_state[INPUT_WEIGHT_KEY]
    )


def test_fit_holds_no_reference_to_a_global_dataset(
    engine: TorchTrainingEngine,
) -> None:
    # A later phase trains on a client's own subset. Passing different data to the
    # same engine must produce different parameters, which it cannot if the engine
    # captured a dataset.
    config = make_config(epochs=1)
    small = engine.fit(config, train_split(rows=12), validation_split())
    large = engine.fit(config, train_split(rows=36), validation_split())

    assert not torch.equal(
        small.final_state[INPUT_WEIGHT_KEY], large.final_state[INPUT_WEIGHT_KEY]
    )


# --- Guards ------------------------------------------------------------------


def test_splits_that_disagree_on_width_are_refused(
    engine: TorchTrainingEngine,
) -> None:
    with pytest.raises(FeatureDimensionMismatchError) as caught:
        engine.fit(
            make_config(),
            make_dataset(SplitName.TRAIN, feature_dimension=4),
            make_dataset(SplitName.VALIDATION, feature_dimension=6),
        )

    assert "validation" in str(caught.value)


def test_stale_parameters_are_named_rather_than_left_to_torch(
    engine: TorchTrainingEngine,
) -> None:
    narrow = engine.fit(
        make_config(epochs=1),
        make_dataset(SplitName.TRAIN, feature_dimension=4),
        make_dataset(SplitName.VALIDATION, feature_dimension=4),
    )

    with pytest.raises(FeatureDimensionMismatchError, match="supplied parameters"):
        engine.predict(
            make_config(),
            make_dataset(SplitName.TEST, feature_dimension=6),
            narrow.best_state,
        )


def test_an_unknown_optimizer_lists_the_supported_ones(
    engine: TorchTrainingEngine,
) -> None:
    with pytest.raises(ValueError, match="unknown optimizer") as caught:
        engine.fit(make_config(optimizer="lbfgs"), train_split(), validation_split())

    message = str(caught.value)
    for name in ("adam", "adamw", "rmsprop", "sgd"):
        assert name in message, name


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop"])
def test_each_supported_optimizer_runs(
    engine: TorchTrainingEngine, name: str
) -> None:
    outcome = engine.fit(
        make_config(epochs=1, optimizer=name), train_split(), validation_split()
    )

    assert len(outcome.epochs) == 1


def test_the_optimizer_name_is_matched_case_insensitively(
    engine: TorchTrainingEngine,
) -> None:
    outcome = engine.fit(
        make_config(epochs=1, optimizer="  AdamW "), train_split(), validation_split()
    )

    assert len(outcome.epochs) == 1


def test_gradient_clipping_can_be_switched_off(engine: TorchTrainingEngine) -> None:
    outcome = engine.fit(
        make_config(epochs=1, gradient_clip_norm=None),
        train_split(),
        validation_split(),
    )

    assert len(outcome.epochs) == 1


# --- One loop, three architectures -------------------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_goes_through_the_same_loop(
    engine: TorchTrainingEngine, architecture: ModelArchitecture
) -> None:
    outcome = engine.fit(
        make_config(epochs=2, architecture=architecture),
        train_split(),
        validation_split(),
    )

    assert len(outcome.epochs) == 2
    assert outcome.parameter_count > 0
    assert outcome.device == "cpu"


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_is_scored_the_same_way(
    engine: TorchTrainingEngine, architecture: ModelArchitecture
) -> None:
    config = make_config(epochs=1, architecture=architecture)
    validation = validation_split()
    outcome = engine.fit(config, train_split(), validation)

    scored = engine.predict(config, validation, outcome.best_state)

    assert scored.labels.tolist() == validation.labels.tolist()
    assert len(scored) == validation.sequence_count


def test_count_parameters_matches_the_fitted_model(
    engine: TorchTrainingEngine,
) -> None:
    config = make_config(epochs=1)
    outcome = engine.fit(config, train_split(), validation_split())

    assert engine.count_parameters(config, FEATURES) == outcome.parameter_count


def test_count_parameters_needs_no_training(engine: TorchTrainingEngine) -> None:
    # Answerable before any data is read, so a report can state model size even
    # for a run that fails later.
    assert engine.count_parameters(make_config(), 57) > 0


def test_bilstm_counts_more_parameters_than_lstm(
    engine: TorchTrainingEngine,
) -> None:
    lstm = engine.count_parameters(
        make_config(architecture=ModelArchitecture.LSTM), FEATURES
    )
    bilstm = engine.count_parameters(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )

    assert bilstm > lstm
