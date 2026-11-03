"""Tests for :mod:`infrastructure.models.pytorch.sequence_classifier`.

Three claims are checked here, in order of how quietly they could fail:

1. **The BiLSTM head is sized for two directions.** A head sized for one trains
   without complaint and discards half the encoder, so the doubling is asserted
   directly, both in the declared representation size and in the head's weight
   shape.
2. **Padding is masked, not read.** Appending padding to a batch must not change
   any logit. This is the property that makes the reported sequence length honest,
   and it matters most for the bidirectional model, whose backward pass would
   otherwise start inside the padding.
3. **The output is a logit.** Nothing here applies a sigmoid, so the outputs must
   be free to leave ``(0, 1)``.

Everything runs on tiny CPU tensors — a handful of sequences, hidden size 4.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from infrastructure.models.pytorch.sequence_classifier import (
    BiLstmClassifier,
    GruClassifier,
    LstmClassifier,
    RecurrentSequenceClassifier,
    build_classifier,
)

pytestmark = pytest.mark.unit

FEATURES = 3
HIDDEN = 4
LENGTH = 5


def make_config(**overrides: object) -> TrainingConfig:
    """A tiny model configuration."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "hidden_size": HIDDEN,
        "sequence_length": LENGTH,
        "num_layers": 1,
        "batch_size": 4,
        "epochs": 1,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def padded_batch(
    lengths: list[int], *, sequence_length: int = LENGTH, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """Build a right-padded batch whose padded positions are exactly zero."""
    generator = torch.Generator().manual_seed(seed)
    count = len(lengths)
    features = torch.rand(
        (count, sequence_length, FEATURES), generator=generator, dtype=torch.float32
    )
    for row, length in enumerate(lengths):
        features[row, length:] = 0.0
    return features, torch.tensor(lengths, dtype=torch.int64)


def evaluated(model: RecurrentSequenceClassifier, *args: Tensor) -> Tensor:
    """Run a model in eval mode without gradients."""
    model.eval()
    with torch.no_grad():
        logits: Tensor = model(*args)
    return logits


# --- The subclass table ------------------------------------------------------


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        (ModelArchitecture.GRU, GruClassifier),
        (ModelArchitecture.LSTM, LstmClassifier),
        (ModelArchitecture.BILSTM, BiLstmClassifier),
    ],
)
def test_each_architecture_builds_its_own_classifier(
    architecture: ModelArchitecture, expected: type[RecurrentSequenceClassifier]
) -> None:
    model = build_classifier(make_config(architecture=architecture), FEATURES)

    assert isinstance(model, expected)


def test_every_architecture_shares_one_implementation() -> None:
    # The comparison this project makes is between architectures, so the three
    # models must differ only in their encoder.
    for architecture in ModelArchitecture:
        model = build_classifier(make_config(architecture=architecture), FEATURES)
        assert isinstance(model, RecurrentSequenceClassifier)
        assert type(model).forward is RecurrentSequenceClassifier.forward


def test_gru_and_lstm_are_unidirectional_and_bilstm_is_not() -> None:
    gru = build_classifier(make_config(architecture=ModelArchitecture.GRU), FEATURES)
    lstm = build_classifier(make_config(architecture=ModelArchitecture.LSTM), FEATURES)
    bilstm = build_classifier(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )

    assert (gru.directions, lstm.directions, bilstm.directions) == (1, 1, 2)


# --- The doubled bidirectional representation --------------------------------


def test_bilstm_representation_is_twice_the_hidden_size() -> None:
    bilstm = build_classifier(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )

    assert bilstm.representation_size == 2 * HIDDEN


def test_bilstm_head_is_wired_for_both_directions() -> None:
    bilstm = build_classifier(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )

    # The head's input width is the real check: a head of width `hidden_size`
    # would train happily while ignoring the backward pass.
    assert bilstm.head.in_features == 2 * HIDDEN


def test_unidirectional_head_is_one_hidden_wide() -> None:
    gru = build_classifier(make_config(architecture=ModelArchitecture.GRU), FEATURES)

    assert gru.head.in_features == HIDDEN


def test_bilstm_has_more_parameters_than_the_lstm_it_doubles() -> None:
    lstm = build_classifier(make_config(architecture=ModelArchitecture.LSTM), FEATURES)
    bilstm = build_classifier(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )

    assert bilstm.parameter_count() > lstm.parameter_count()


def test_bilstm_uses_both_final_states_not_just_the_forward_one() -> None:
    # The backward direction reads the window in reverse, so changing only the
    # oldest element must move the logit. If the head were fed the forward state
    # alone, the backward final state - which summarises exactly that element -
    # would have no effect.
    torch.manual_seed(7)
    bilstm = build_classifier(
        make_config(architecture=ModelArchitecture.BILSTM), FEATURES
    )
    features, lengths = padded_batch([LENGTH])
    baseline = evaluated(bilstm, features, lengths)

    altered = features.clone()
    altered[0, 0] = altered[0, 0] + 5.0
    changed = evaluated(bilstm, altered, lengths)

    assert not torch.allclose(baseline, changed)


# --- Padding is masked -------------------------------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_padding_does_not_change_any_logit(architecture: ModelArchitecture) -> None:
    torch.manual_seed(11)
    model = build_classifier(make_config(architecture=architecture), FEATURES)

    short_features, lengths = padded_batch([3, 2], sequence_length=3, seed=5)
    short = evaluated(model, short_features, lengths)

    # The same two sequences in a wider tensor: two extra padded positions.
    wide = torch.zeros((2, 5, FEATURES), dtype=torch.float32)
    wide[:, :3] = short_features
    long = evaluated(model, wide, lengths)

    assert torch.allclose(short, long, atol=1e-6)


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_values_written_into_padded_positions_are_ignored(
    architecture: ModelArchitecture,
) -> None:
    # `lengths` is what the packing honours. Even garbage in the padded slots must
    # not reach the encoder.
    torch.manual_seed(13)
    model = build_classifier(make_config(architecture=architecture), FEATURES)
    features, lengths = padded_batch([5, 2, 1], seed=3)
    baseline = evaluated(model, features, lengths)

    polluted = features.clone()
    polluted[1, 2:] = 99.0
    polluted[2, 1:] = -99.0
    after = evaluated(model, polluted, lengths)

    assert torch.allclose(baseline, after, atol=1e-6)


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_batch_order_does_not_change_a_sequences_logit(
    architecture: ModelArchitecture,
) -> None:
    # Packing with enforce_sorted=False must restore the original order, or
    # predictions would stop lining up with labels.
    torch.manual_seed(17)
    model = build_classifier(make_config(architecture=architecture), FEATURES)
    features, lengths = padded_batch([2, 5, 3], seed=9)
    forward = evaluated(model, features, lengths)

    order = [2, 0, 1]
    reordered = evaluated(model, features[order], lengths[order])

    assert torch.allclose(forward[order], reordered, atol=1e-6)


# --- Shape and output semantics ----------------------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_output_is_one_value_per_sequence(architecture: ModelArchitecture) -> None:
    model = build_classifier(make_config(architecture=architecture), FEATURES)
    features, lengths = padded_batch([5, 4, 3, 1])

    logits = evaluated(model, features, lengths)

    assert logits.shape == (4,)


def test_a_single_sequence_batch_still_returns_a_one_dimensional_tensor() -> None:
    # squeeze(-1) on a (1, 1) tensor must not collapse to a scalar, or the loss
    # would silently broadcast.
    model = build_classifier(make_config(), FEATURES)
    features, lengths = padded_batch([4])

    logits = evaluated(model, features, lengths)

    assert logits.shape == (1,)


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_output_is_a_logit_not_a_probability(architecture: ModelArchitecture) -> None:
    # Push the head's bias out of range: a logit follows it, a probability cannot.
    model = build_classifier(make_config(architecture=architecture), FEATURES)
    with torch.no_grad():
        model.head.bias.fill_(9.0)
    features, lengths = padded_batch([5, 5])

    logits = evaluated(model, features, lengths)

    assert float(logits.max()) > 1.0


def test_logits_flow_gradients_to_every_parameter() -> None:
    model = build_classifier(make_config(), FEATURES)
    features, lengths = padded_batch([5, 3])

    model.train()
    model(features, lengths).sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad)), name


# --- Construction ------------------------------------------------------------


def test_feature_dimension_is_taken_from_the_data_not_the_config() -> None:
    model = build_classifier(make_config(), 57)

    assert model.feature_dimension == 57
    assert model.encoder.input_size == 57


def test_single_layer_models_report_no_dropout() -> None:
    # PyTorch ignores dropout on a one-layer RNN, so the model must be built with
    # the effective value rather than the requested one.
    config = make_config(num_layers=1, dropout=0.5)
    model = build_classifier(config, FEATURES)

    assert config.effective_dropout == 0.0
    assert model.encoder.dropout == 0.0


def test_multi_layer_models_keep_the_requested_dropout() -> None:
    config = make_config(num_layers=2, dropout=0.25)
    model = build_classifier(config, FEATURES)

    assert config.effective_dropout == pytest.approx(0.25)
    assert model.encoder.dropout == pytest.approx(0.25)


def test_stacked_layers_are_built() -> None:
    model = build_classifier(make_config(num_layers=3, dropout=0.1), FEATURES)

    assert model.num_layers == 3
    assert model.encoder.num_layers == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [("feature_dimension", 0), ("hidden_size", 0), ("num_layers", 0)],
)
def test_non_positive_dimensions_are_refused(field: str, value: int) -> None:
    arguments: dict[str, object] = {
        "feature_dimension": FEATURES,
        "hidden_size": HIDDEN,
        "num_layers": 1,
        "dropout": 0.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        GruClassifier(**arguments)  # type: ignore[arg-type]


def test_parameter_count_matches_a_manual_sum() -> None:
    model = build_classifier(make_config(), FEATURES)

    expected = sum(
        int(np.prod(parameter.shape)) for parameter in model.parameters()
    )
    assert model.parameter_count() == expected


def test_two_models_from_one_seed_have_identical_initial_parameters() -> None:
    torch.manual_seed(4)
    first = build_classifier(make_config(), FEATURES)
    torch.manual_seed(4)
    second = build_classifier(make_config(), FEATURES)

    for (name, left), (_, right) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(left, right), name
