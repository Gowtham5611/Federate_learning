"""Tests for the two fitted transformers: the standardiser and the encoder.

The property both classes exist to guarantee is that a statistic learned from
training rows is never influenced by validation or test rows. These tests attack
that from both directions — they check that fitting after freezing is refused,
and that transforming before freezing is refused — because a transformer that
allowed either would let leakage in through call order alone.

The numeric assertions use small hand-checkable values so a wrong answer is
visibly wrong rather than merely different.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pandas import DataFrame

from domain.exceptions.preprocessing import (
    TransformerAlreadyFittedError,
    TransformerNotFittedError,
)
from infrastructure.data.preprocessors.categorical_encoder import (
    HIGH_CARDINALITY_REFUSALS,
    UNKNOWN_CODE,
    OrdinalCategoricalEncoder,
    encode_binary_target,
)
from infrastructure.data.preprocessors.running_stats import (
    MINIMUM_STANDARD_DEVIATION,
    RunningStandardizer,
    describe_log_transform,
)

pytestmark = pytest.mark.unit


def _numbers(values: list[float]) -> DataFrame:
    """One-column frame named ``x``."""
    return DataFrame({"x": values})


# --- RunningStandardizer: streaming statistics --------------------------------


def test_chunked_fitting_matches_a_single_pass_over_the_same_rows() -> None:
    """The whole point: chunking must not change the answer.

    If it did, the memory-bounded pipeline would silently produce different
    features from an in-memory one, and no amount of testing the pipeline's
    plumbing would reveal it.
    """
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]

    whole = RunningStandardizer(("x",))
    whole.update(_numbers(values))
    whole.freeze()

    chunked = RunningStandardizer(("x",))
    for start in range(0, len(values), 3):
        chunked.update(_numbers(values[start : start + 3]))
    chunked.freeze()

    assert whole.parameters()["x"]["mean"] == pytest.approx(
        chunked.parameters()["x"]["mean"]
    )
    assert whole.parameters()["x"]["std"] == pytest.approx(
        chunked.parameters()["x"]["std"]
    )


def test_the_fitted_mean_and_deviation_are_the_population_values() -> None:
    """Population, not sample: pinned down so a reproduction can match it."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
    standardizer.freeze()

    parameters = standardizer.parameters()["x"]

    assert parameters["mean"] == pytest.approx(5.0)
    assert parameters["std"] == pytest.approx(2.0)
    assert parameters["count"] == pytest.approx(8.0)


def test_transforming_centres_and_scales_by_the_fitted_values() -> None:
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
    standardizer.freeze()

    transformed = standardizer.transform(_numbers([5.0, 7.0, 3.0]))

    assert list(transformed["x"]) == pytest.approx([0.0, 1.0, -1.0])


def test_a_constant_column_is_centred_rather_than_amplified() -> None:
    """Dividing by a near-zero deviation turns noise into huge feature values."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([7.0] * 5))
    standardizer.freeze()

    assert standardizer.parameters()["x"]["std"] < MINIMUM_STANDARD_DEVIATION
    assert list(standardizer.transform(_numbers([7.0, 9.0]))["x"]) == pytest.approx(
        [0.0, 2.0]
    )


def test_fitting_is_refused_once_the_statistics_are_frozen() -> None:
    """This is the structural half of 'fitted on train only'."""
    standardizer = RunningStandardizer(("x",))
    standardizer.freeze()

    with pytest.raises(TransformerAlreadyFittedError):
        standardizer.update(_numbers([1.0]))


def test_transforming_is_refused_before_the_statistics_are_frozen() -> None:
    """The other half: no state in which the transformer both learns and applies."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([1.0, 2.0]))

    with pytest.raises(TransformerNotFittedError):
        standardizer.transform(_numbers([1.0]))


def test_reporting_parameters_is_refused_before_freezing() -> None:
    with pytest.raises(TransformerNotFittedError):
        RunningStandardizer(("x",)).parameters()


def test_freezing_twice_is_harmless() -> None:
    """The service freezes lazily and again at the end; both paths must be safe."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([1.0, 3.0]))
    standardizer.freeze()
    standardizer.freeze()

    assert standardizer.parameters()["x"]["mean"] == pytest.approx(2.0)


def test_a_standardizer_restored_from_parameters_transforms_identically() -> None:
    """Reproducing a run must not require re-reading the training rows."""
    original = RunningStandardizer(("x",))
    original.update(_numbers([1.0, 2.0, 3.0, 10.0]))
    original.freeze()

    restored = RunningStandardizer.from_parameters(original.parameters())
    sample = _numbers([4.0, 7.5])

    assert list(restored.transform(sample)["x"]) == pytest.approx(
        list(original.transform(sample)["x"])
    )


def test_missing_values_are_skipped_rather_than_treated_as_zero() -> None:
    """A NaN counted as zero would drag the mean toward it."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(DataFrame({"x": [4.0, None, 6.0]}))
    standardizer.freeze()

    assert standardizer.parameters()["x"]["mean"] == pytest.approx(5.0)
    assert standardizer.parameters()["x"]["count"] == pytest.approx(2.0)


def test_an_empty_chunk_leaves_the_statistics_untouched() -> None:
    """Chunked reading produces empty frames at boundaries; they must be inert."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([2.0, 4.0]))
    standardizer.update(DataFrame({"x": pd.Series(dtype="float64")}))
    standardizer.freeze()

    assert standardizer.parameters()["x"]["mean"] == pytest.approx(3.0)
    assert standardizer.parameters()["x"]["count"] == pytest.approx(2.0)


def test_the_repr_never_shows_the_fitted_values() -> None:
    """Statistics of financial data do not belong in a log line."""
    standardizer = RunningStandardizer(("x",))
    standardizer.update(_numbers([123456.75]))

    assert "123456" not in repr(standardizer)


def test_the_log_transform_is_documented_per_column() -> None:
    described = describe_log_transform(("amount_paid_log",))

    assert described["amount_paid_log"]["transform"] == "log1p"
    assert described["amount_paid_log"]["applied_before"] == "standardization"


# --- OrdinalCategoricalEncoder: deterministic vocabularies ---------------------


def _categories(values: list[str]) -> DataFrame:
    """One-column frame named ``c``."""
    return DataFrame({"c": values})


def test_codes_follow_the_sorted_vocabulary_not_order_of_appearance() -> None:
    """Order-of-appearance codes would differ between runs that chunk differently."""
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["Wire", "ACH", "Cheque"]))
    encoder.freeze()

    assert encoder.mapping()["c"] == {"ACH": 0, "Cheque": 1, "Wire": 2}


def test_chunk_order_does_not_change_the_codes() -> None:
    first = OrdinalCategoricalEncoder(("c",))
    first.update(_categories(["Wire"]))
    first.update(_categories(["ACH", "Cheque"]))
    first.freeze()

    second = OrdinalCategoricalEncoder(("c",))
    second.update(_categories(["Cheque", "ACH"]))
    second.update(_categories(["Wire"]))
    second.freeze()

    assert first.mapping() == second.mapping()


def test_a_value_unseen_in_training_maps_to_the_unknown_code() -> None:
    """Not dropped: an unfamiliar payment format is exactly what AML should see."""
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH", "Wire"]))
    encoder.freeze()

    transformed = encoder.transform(_categories(["ACH", "Bitcoin"]))

    assert list(transformed["c_code"]) == [0, UNKNOWN_CODE]


def test_the_unknown_code_cannot_collide_with_a_real_code() -> None:
    assert UNKNOWN_CODE < 0


def test_unknown_values_are_counted_so_they_can_be_reported() -> None:
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH"]))
    encoder.freeze()

    assert encoder.count_unknown(_categories(["ACH", "Wire", "Cash"])) == {"c": 2}


def test_the_source_column_survives_encoding() -> None:
    """Keeping both makes the code auditable against the value it came from."""
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH"]))
    encoder.freeze()

    transformed = encoder.transform(_categories(["ACH"]))

    assert list(transformed.columns) == ["c", "c_code"]


def test_the_vocabulary_is_closed_once_frozen() -> None:
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.freeze()

    with pytest.raises(TransformerAlreadyFittedError):
        encoder.update(_categories(["ACH"]))


def test_encoding_is_refused_before_the_vocabulary_is_closed() -> None:
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH"]))

    with pytest.raises(TransformerNotFittedError):
        encoder.transform(_categories(["ACH"]))


def test_the_mapping_is_refused_before_the_vocabulary_is_closed() -> None:
    with pytest.raises(TransformerNotFittedError):
        OrdinalCategoricalEncoder(("c",)).mapping()


def test_an_encoder_restored_from_a_mapping_encodes_identically() -> None:
    original = OrdinalCategoricalEncoder(("c",))
    original.update(_categories(["Wire", "ACH"]))
    original.freeze()

    restored = OrdinalCategoricalEncoder.from_mapping(original.mapping())
    sample = _categories(["ACH", "Wire", "Cash"])

    assert list(restored.transform(sample)["c_code"]) == list(
        original.transform(sample)["c_code"]
    )


def test_missing_values_do_not_enter_the_vocabulary() -> None:
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(DataFrame({"c": ["ACH", None]}))
    encoder.freeze()

    assert encoder.vocabulary_size("c") == 1


def test_an_empty_chunk_leaves_the_vocabulary_untouched() -> None:
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH"]))
    encoder.update(DataFrame({"c": pd.Series(dtype="object")}))
    encoder.freeze()

    assert encoder.vocabulary_size("c") == 1


# --- High-cardinality safety ---------------------------------------------------


def test_the_identifier_columns_are_refused_with_a_measured_reason() -> None:
    """Silently not encoding them would be indistinguishable from forgetting to."""
    assert set(HIGH_CARDINALITY_REFUSALS) == {
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
    }
    assert "515,080" in HIGH_CARDINALITY_REFUSALS["Account"]
    assert "356,303" in HIGH_CARDINALITY_REFUSALS["From Bank"]


def test_the_refusals_are_carried_into_the_encoding_metadata() -> None:
    """They belong in the artefact, where a later reader will look."""
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories(["ACH"]))
    encoder.freeze()

    described = encoder.describe()

    assert described["scheme"] == "ordinal"
    assert described["codes_are_nominal"] is True
    assert described["not_encoded"] == HIGH_CARDINALITY_REFUSALS


def test_a_high_cardinality_column_does_not_explode_the_encoder() -> None:
    """Guard on cost: the encoder is linear in distinct values, not quadratic.

    Ten thousand synthetic identifiers stand in for the real account column's
    measured 515,080 without needing them. The assertion is that the mapping
    stays one entry per value — one-hot expansion here is what the design
    refuses.
    """
    encoder = OrdinalCategoricalEncoder(("c",))
    encoder.update(_categories([f"ACCT{index:06d}" for index in range(10_000)]))
    encoder.freeze()

    assert encoder.vocabulary_size("c") == 10_000
    assert len(encoder.mapping()["c"]) == 10_000


# --- Target encoding -----------------------------------------------------------


def test_the_target_maps_through_its_fixed_mapping() -> None:
    encoded = encode_binary_target(pd.Series(["0", "1", "0"]), {"0": 0, "1": 1})

    assert list(encoded) == [0, 1, 0]


def test_a_target_value_outside_the_mapping_becomes_the_unknown_code() -> None:
    """Surfaced, not guessed at: the caller decides what to do with it."""
    encoded = encode_binary_target(pd.Series(["1", "2"]), {"0": 0, "1": 1})

    assert list(encoded) == [1, UNKNOWN_CODE]


def test_numeric_and_text_labels_encode_the_same_way() -> None:
    """CSV round-trips turn 1 into '1'; both must reach the same code."""
    assert list(encode_binary_target(pd.Series([1, 0]), {"0": 0, "1": 1})) == [1, 0]


def test_log1p_keeps_a_zero_count_defined() -> None:
    """Why log1p and not log: derived counts and elapsed times reach zero."""
    assert math.log1p(0.0) == 0.0
