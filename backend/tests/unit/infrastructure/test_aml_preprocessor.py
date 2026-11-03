"""Tests for :class:`AmlPreprocessor`: cleaning, the target, and fit/transform.

This is where the per-chunk contract lives, so it is where the rules that matter
most are checkable on a handful of rows: nothing is dropped without being
counted, the label mapping is the one the file was measured to contain, and
there is no call order that fits a statistic on anything but training rows.

All rows here are synthetic. Where a real measurement is asserted — the two
target values, the eleven columns — it is asserted as a *constant of the module*,
not as a fact rediscovered from data at test time.
"""

from __future__ import annotations

import pytest
from pandas import DataFrame

from domain.exceptions.preprocessing import (
    TransformerAlreadyFittedError,
    TransformerNotFittedError,
)
from infrastructure.data.features.temporal_features import (
    EPOCH_SECONDS_COLUMN,
    PARSED_TIMESTAMP_COLUMN,
)
from infrastructure.data.preprocessors.aml_preprocessor import (
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    OUTPUT_TIMESTAMP_FORMAT,
    POSITIVE_LABEL,
    PREPROCESSING_VERSION,
    RAW_DTYPES,
    SCALED_COLUMNS,
    TARGET_MAPPING,
    TARGET_OUTPUT_COLUMN,
    AmlPreprocessor,
    count_duplicate_rows,
    removal_descriptions,
    summarise_class_counts,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_IS_LAUNDERING,
    COL_TIMESTAMP,
    COL_TO_ACCOUNT,
    RAW_COLUMN_ORDER,
)
from tests.fixtures.preprocessing_fixtures import make_frame, make_transactions

pytestmark = pytest.mark.unit


def _raw(**overrides: object) -> DataFrame:
    """A single synthetic raw row, with any verified column overridable."""
    row = dict(make_transactions(rows=1, days=1, positive_every=0)[0])
    row.update(overrides)
    return DataFrame([row], columns=list(RAW_COLUMN_ORDER))


def _engineered(rows: int = 40) -> DataFrame:
    """Synthetic rows carried through cleaning, ordering and engineering."""
    preprocessor = AmlPreprocessor()
    cleaned, _ = preprocessor.clean(make_frame(rows=rows, days=1))
    return preprocessor.engineer(preprocessor.sort_chronologically(cleaned))


# --- Cleaning: nothing disappears without being counted ------------------------


def test_valid_rows_survive_cleaning_untouched_in_number() -> None:
    cleaned, removed = AmlPreprocessor().clean(make_frame(rows=30))

    assert len(cleaned) == 30
    assert sum(removed.values()) == 0


def test_every_removal_reason_is_reported_even_when_it_removed_nothing() -> None:
    """A silent zero and a missing rule look identical in a report otherwise."""
    _, removed = AmlPreprocessor().clean(make_frame(rows=5))

    assert set(removed) == set(removal_descriptions())


def test_a_missing_required_value_is_removed_and_counted() -> None:
    cleaned, removed = AmlPreprocessor().clean(_raw(**{COL_AMOUNT_PAID: None}))

    assert len(cleaned) == 0
    assert removed["missing_required_value"] == 1


def test_an_unparseable_timestamp_is_removed_and_counted() -> None:
    """A row with no usable time cannot be placed in a chronological split."""
    cleaned, removed = AmlPreprocessor().clean(_raw(**{COL_TIMESTAMP: "not a date"}))

    assert len(cleaned) == 0
    assert removed["unparseable_timestamp"] == 1


def test_a_negative_amount_is_removed_and_counted() -> None:
    cleaned, removed = AmlPreprocessor().clean(_raw(**{COL_AMOUNT_PAID: -5.0}))

    assert len(cleaned) == 0
    assert removed["negative_amount"] == 1


def test_a_label_outside_the_two_verified_values_is_removed_and_counted() -> None:
    cleaned, removed = AmlPreprocessor().clean(_raw(**{COL_IS_LAUNDERING: 7}))

    assert len(cleaned) == 0
    assert removed["unknown_target_value"] == 1


def test_a_very_large_amount_is_kept_because_it_is_not_an_error() -> None:
    """Rule 5, explicitly: in AML an unusual value may be the whole point."""
    cleaned, removed = AmlPreprocessor().clean(_raw(**{COL_AMOUNT_PAID: 1.0463e12}))

    assert len(cleaned) == 1
    assert sum(removed.values()) == 0


def test_a_zero_amount_is_kept() -> None:
    """Zero is not negative; only a negative magnitude is nonsensical."""
    cleaned, _ = AmlPreprocessor().clean(
        _raw(**{COL_AMOUNT_PAID: 0.0, COL_AMOUNT_RECEIVED: 0.0})
    )

    assert len(cleaned) == 1


def test_the_removals_and_the_survivors_account_for_every_input_row() -> None:
    """The ledger has to reconcile, which is the check the final report makes."""
    frame = DataFrame(
        [
            _raw().iloc[0],
            _raw(**{COL_TIMESTAMP: "nonsense"}).iloc[0],
            _raw(**{COL_AMOUNT_RECEIVED: -1.0}).iloc[0],
            _raw(**{COL_IS_LAUNDERING: 9}).iloc[0],
        ]
    )

    cleaned, removed = AmlPreprocessor().clean(frame)

    assert len(cleaned) + sum(removed.values()) == len(frame)


def test_every_removal_reason_carries_a_description_a_reader_can_act_on() -> None:
    descriptions = removal_descriptions()

    assert set(descriptions) == {
        "missing_required_value",
        "unparseable_timestamp",
        "negative_amount",
        "unknown_target_value",
    }
    assert all(len(text) > 40 for text in descriptions.values())
    assert "never treated as outliers" in descriptions["negative_amount"]


def test_cleaning_an_empty_chunk_yields_an_empty_result_not_an_error() -> None:
    """Chunked reading can hand over an empty frame; it must pass straight through."""
    empty = DataFrame(columns=list(RAW_COLUMN_ORDER))

    cleaned, removed = AmlPreprocessor().clean(empty)

    assert len(cleaned) == 0
    assert sum(removed.values()) == 0


def test_cleaning_rewrites_the_timestamp_into_the_canonical_format() -> None:
    """One format everywhere: histogram, boundaries, reports and output agree."""
    cleaned, _ = AmlPreprocessor().clean(_raw(**{COL_TIMESTAMP: "2020/03/07 14:05"}))

    assert cleaned[COL_TIMESTAMP].iloc[0] == "2020-03-07 14:05:00"
    assert OUTPUT_TIMESTAMP_FORMAT == "%Y-%m-%d %H:%M:%S"


def test_the_canonical_timestamp_sorts_lexicographically_as_it_does_in_time() -> None:
    """The property the whole text-comparison design rests on."""
    cleaned, _ = AmlPreprocessor().clean(
        make_frame(rows=40, days=4).sample(frac=1.0, random_state=0)
    )
    stamps = list(cleaned[COL_TIMESTAMP])

    assert sorted(stamps) == list(
        cleaned.sort_values(PARSED_TIMESTAMP_COLUMN)[COL_TIMESTAMP]
    )


def test_cleaning_leaves_the_caller_s_chunk_alone() -> None:
    frame = make_frame(rows=5)

    AmlPreprocessor().clean(frame)

    assert PARSED_TIMESTAMP_COLUMN not in frame.columns


# --- Types forced on read ------------------------------------------------------


def test_identifier_columns_are_read_as_text_so_chunks_agree() -> None:
    """An id inferred as int64 in one chunk and text in another is two ids."""
    assert RAW_DTYPES[COL_FROM_ACCOUNT] == "string"
    assert RAW_DTYPES[COL_TO_ACCOUNT] == "string"
    assert set(RAW_DTYPES) <= set(RAW_COLUMN_ORDER)


def test_the_amount_and_target_columns_are_left_to_explicit_coercion() -> None:
    """They are numbers, and cleaning coerces them where the failures are counted."""
    for column in (COL_AMOUNT_PAID, COL_AMOUNT_RECEIVED, COL_IS_LAUNDERING):
        assert column not in RAW_DTYPES


# --- Day partitioning ----------------------------------------------------------


def test_day_keys_come_from_the_canonical_timestamp_text() -> None:
    """So a partition's name can never disagree with the rows inside it."""
    cleaned, _ = AmlPreprocessor().clean(_raw(**{COL_TIMESTAMP: "2020/03/07 14:05"}))

    assert list(AmlPreprocessor.day_keys(cleaned)) == ["2020-03-07"]


def test_a_partition_read_back_recovers_the_columns_engineering_needs() -> None:
    """Partitions store only the eleven columns; the working ones are re-derived."""
    cleaned, _ = AmlPreprocessor().clean(make_frame(rows=6, days=1))
    round_tripped = cleaned[list(RAW_COLUMN_ORDER)]

    prepared = AmlPreprocessor.prepare_partition(round_tripped)

    assert PARSED_TIMESTAMP_COLUMN in prepared.columns
    assert EPOCH_SECONDS_COLUMN in prepared.columns
    assert list(prepared[EPOCH_SECONDS_COLUMN]) == list(cleaned[EPOCH_SECONDS_COLUMN])


# --- Canonical ordering --------------------------------------------------------


def test_sorting_puts_rows_in_timestamp_order() -> None:
    cleaned, _ = AmlPreprocessor().clean(make_frame(rows=50, days=5))

    ordered = AmlPreprocessor.sort_chronologically(cleaned)

    assert list(ordered[COL_TIMESTAMP]) == sorted(ordered[COL_TIMESTAMP])


def test_the_sort_is_stable_so_two_runs_order_ties_identically() -> None:
    """Minute resolution means ties are everywhere.

    History features depend on the order chosen, so it must be the same order
    every time.
    """
    cleaned, _ = AmlPreprocessor().clean(make_frame(rows=60, days=3))

    first = AmlPreprocessor.sort_chronologically(cleaned)
    second = AmlPreprocessor.sort_chronologically(cleaned)

    assert list(first[COL_FROM_ACCOUNT]) == list(second[COL_FROM_ACCOUNT])


# --- Fit only on training ------------------------------------------------------


def test_transforming_before_fitting_is_refused() -> None:
    """No ordering of calls exists that transforms with unfitted statistics."""
    preprocessor = AmlPreprocessor()

    with pytest.raises(TransformerNotFittedError):
        preprocessor.transform(_engineered())


def test_fitting_after_the_freeze_is_refused() -> None:
    """The freeze is what closes the training split; nothing may reopen it."""
    preprocessor = AmlPreprocessor()
    preprocessor.freeze_fit()

    with pytest.raises(TransformerAlreadyFittedError):
        preprocessor.fit_on_training_chunk(_engineered())


def test_describing_the_preprocessor_before_fitting_is_refused() -> None:
    """The artefacts would otherwise record parameters that do not exist."""
    with pytest.raises(TransformerNotFittedError):
        AmlPreprocessor().describe()


def test_the_scaling_parameters_come_only_from_the_rows_fitted_on() -> None:
    """The leakage check on the statistic itself.

    Two preprocessors are fitted on the same training rows; one then transforms
    a wildly different set of later rows. Its parameters must not move.
    """
    train = _engineered(rows=30)
    later = _engineered(rows=30)
    later[SCALED_COLUMNS[0]] = 10_000.0

    fitted = AmlPreprocessor()
    fitted.fit_on_training_chunk(train)
    fitted.freeze_fit()
    before = fitted.describe()["scaling_parameters"]

    fitted.transform(later)

    assert fitted.describe()["scaling_parameters"] == before


def test_the_preprocessor_reports_whether_it_is_fitted() -> None:
    preprocessor = AmlPreprocessor()
    assert preprocessor.is_fitted is False

    preprocessor.freeze_fit()

    assert preprocessor.is_fitted is True


def test_freezing_twice_is_harmless() -> None:
    preprocessor = AmlPreprocessor()
    preprocessor.fit_on_training_chunk(_engineered())
    preprocessor.freeze_fit()
    preprocessor.freeze_fit()

    assert preprocessor.is_fitted is True


# --- The output shape ----------------------------------------------------------


def _fitted_on(frame: DataFrame) -> AmlPreprocessor:
    preprocessor = AmlPreprocessor()
    preprocessor.fit_on_training_chunk(frame)
    preprocessor.freeze_fit()
    return preprocessor


def test_transform_produces_exactly_the_declared_columns_in_order() -> None:
    """The column order is an artefact; the output has to match it exactly."""
    engineered = _engineered()
    preprocessor = _fitted_on(engineered)

    transformed = preprocessor.transform(engineered)

    assert list(transformed.columns) == list(preprocessor.output_columns)


def test_the_output_is_identifiers_then_features_then_the_target() -> None:
    preprocessor = AmlPreprocessor()

    assert preprocessor.output_columns == (
        *IDENTIFIER_COLUMNS,
        *preprocessor.feature_columns,
        TARGET_OUTPUT_COLUMN,
    )


def test_the_duplicated_account_header_finally_becomes_two_clear_names() -> None:
    """``Account`` / ``Account.1`` are ambiguous; the output names are not."""
    engineered = _engineered()

    transformed = _fitted_on(engineered).transform(engineered)

    assert "from_account" in transformed.columns
    assert "to_account" in transformed.columns
    assert "Account.1" not in transformed.columns


def test_identifiers_are_carried_but_are_not_features() -> None:
    """They exist for sequencing in the next phase, not as model inputs."""
    preprocessor = AmlPreprocessor()

    assert set(IDENTIFIER_COLUMNS).isdisjoint(preprocessor.feature_columns)
    assert TARGET_OUTPUT_COLUMN not in preprocessor.feature_columns


def test_transform_does_not_change_the_number_of_rows() -> None:
    """Preprocessing shapes rows; it never quietly filters them."""
    engineered = _engineered(rows=37)

    transformed = _fitted_on(engineered).transform(engineered)

    assert len(transformed) == len(engineered)


def test_no_row_leaves_transform_with_a_missing_feature_value() -> None:
    engineered = _engineered()

    transformed = _fitted_on(engineered).transform(engineered)

    assert not transformed.isna().to_numpy().any()


def test_transforming_twice_gives_the_same_answer() -> None:
    """Reproducibility, and confirmation that transform holds no hidden state."""
    engineered = _engineered()
    preprocessor = _fitted_on(engineered)

    first = preprocessor.transform(engineered)
    second = preprocessor.transform(engineered)

    assert first.equals(second)


def test_the_engineered_columns_are_what_the_staging_file_must_carry() -> None:
    """Narrower than everything engineered: the interim file pays per column."""
    preprocessor = AmlPreprocessor()
    engineered = _engineered()

    carried = set(preprocessor.engineered_columns)

    assert carried <= set(engineered.columns)
    assert PARSED_TIMESTAMP_COLUMN not in carried
    assert EPOCH_SECONDS_COLUMN not in carried
    # Everything transform reads must still be there.
    assert set(CATEGORICAL_COLUMNS) <= carried
    assert set(SCALED_COLUMNS) <= carried


def test_transform_works_from_the_carried_columns_alone() -> None:
    """The staging round-trip in one assertion: nothing dropped was needed."""
    preprocessor = AmlPreprocessor()
    engineered = _engineered()
    carried = engineered[list(preprocessor.engineered_columns)]
    preprocessor.fit_on_training_chunk(carried)
    preprocessor.freeze_fit()

    transformed = preprocessor.transform(carried)

    assert list(transformed.columns) == list(preprocessor.output_columns)


# --- The target ----------------------------------------------------------------


def test_the_target_mapping_is_the_one_measured_in_the_file() -> None:
    """Two values, counted over all rows.

    Not an assumption about what "should" be in a laundering column.
    """
    assert TARGET_MAPPING == {"0": 0, "1": 1}
    assert POSITIVE_LABEL == 1
    assert AmlPreprocessor().target_mapping() == TARGET_MAPPING


def test_the_target_reaches_the_output_under_its_documented_name() -> None:
    engineered = _engineered()

    transformed = _fitted_on(engineered).transform(engineered)

    assert TARGET_OUTPUT_COLUMN == "is_laundering"
    assert TARGET_OUTPUT_COLUMN in transformed.columns


def test_the_target_values_survive_preprocessing_unchanged() -> None:
    """Rule 9: the class distribution is reported, never rebalanced."""
    engineered = _engineered(rows=42)
    expected = summarise_class_counts(engineered[COL_IS_LAUNDERING])

    transformed = _fitted_on(engineered).transform(engineered)

    assert summarise_class_counts(transformed[TARGET_OUTPUT_COLUMN]) == expected


def test_no_resampling_happens_anywhere_in_the_preprocessor() -> None:
    """Stated as a row-count identity, which resampling would break immediately."""
    engineered = _engineered(rows=50)
    before = summarise_class_counts(engineered[COL_IS_LAUNDERING])

    transformed = _fitted_on(engineered).transform(engineered)
    after = summarise_class_counts(transformed[TARGET_OUTPUT_COLUMN])

    assert after == before
    assert sum(after.values()) == len(engineered)


def test_class_counts_are_returned_as_plain_integers() -> None:
    """They are serialised into JSON artefacts, where numpy scalars fail."""
    counts = summarise_class_counts(_engineered(rows=20)[COL_IS_LAUNDERING])

    assert all(type(key) is int for key in counts)
    assert all(type(value) is int for value in counts.values())


# --- Duplicates: counted and kept ----------------------------------------------


def test_exact_duplicate_rows_are_counted() -> None:
    row = _raw()
    frame = DataFrame([row.iloc[0], row.iloc[0], row.iloc[0]])

    assert count_duplicate_rows(frame) == 2


def test_rows_differing_in_any_verified_column_are_not_duplicates() -> None:
    frame = DataFrame([_raw().iloc[0], _raw(**{COL_AMOUNT_PAID: 101.0}).iloc[0]])

    assert count_duplicate_rows(frame) == 0


def test_a_frame_with_no_duplicates_counts_zero() -> None:
    assert count_duplicate_rows(make_frame(rows=20, days=4)) == 0


def test_duplicates_are_not_removed_by_cleaning() -> None:
    """Reported and kept: two identical transfers in one minute are plausible.

    Deleting real transactions from an AML dataset is the worse error.
    """
    row = _raw()
    frame = DataFrame([row.iloc[0], row.iloc[0]])

    cleaned, removed = AmlPreprocessor().clean(frame)

    assert len(cleaned) == 2
    assert sum(removed.values()) == 0


# --- Metadata ------------------------------------------------------------------


def test_the_description_records_that_fitting_used_training_rows_only() -> None:
    described = _fitted_on(_engineered()).describe()

    assert described["fitted_on"] == "train split only"
    assert described["preprocessing_version"] == PREPROCESSING_VERSION


def test_every_feature_column_is_documented_as_transformed_or_not() -> None:
    """"Not transformed" with a reason is a decision; an absent entry is a gap."""
    preprocessor = _fitted_on(_engineered())
    described = preprocessor.describe()

    documented = set(described["numerical_transformations"])

    assert set(preprocessor.feature_columns) <= documented


def test_the_scaled_columns_are_the_skewed_magnitudes_only() -> None:
    """Rule 8: not every numeric column is blindly transformed."""
    preprocessor = _fitted_on(_engineered())
    described = preprocessor.describe()

    assert all(column.endswith("_log") for column in SCALED_COLUMNS)
    for column in described["unscaled_columns"]:
        assert described["numerical_transformations"][column]["transform"] == "none"
        assert described["numerical_transformations"][column]["reason"]


def test_the_rejected_features_from_every_builder_are_collected() -> None:
    rejected = AmlPreprocessor().rejected_features()

    assert "time_to_next_transaction" in rejected  # temporal
    assert "amount_difference" in rejected  # transaction
    assert "is_fraud_ring_member" in rejected  # history


def test_the_description_names_the_columns_deliberately_left_unencoded() -> None:
    described = _fitted_on(_engineered()).describe()

    assert set(described["columns_not_encoded"]) == {
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
    }


def test_unknown_categories_are_counted_as_transformation_proceeds() -> None:
    """A validation-only payment format is reportable, not silently coded away."""
    train = _engineered(rows=20)
    preprocessor = _fitted_on(train)

    later = _engineered(rows=20)
    later[CATEGORICAL_COLUMNS[0]] = "Format That Never Appeared"
    preprocessor.transform(later)

    assert preprocessor.unknown_category_counts[CATEGORICAL_COLUMNS[0]] == 20
