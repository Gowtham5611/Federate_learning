"""Tests for :mod:`infrastructure.data.sequences.processed_feature_metadata`.

Phase 4 does not restate the schema; it reads Phase 3's artefacts. These tests
check both halves of that: that the recorded description is read faithfully, and
that a description which would put a label or an account identifier on the model's
feature axis is refused rather than believed.

The nominal expansion is checked by its arithmetic, because that is what fixes the
model's input width. In the fixture, two continuous columns plus one nominal column
with a three-value vocabulary give five model inputs; the real artefacts give 57
from 23, by the same rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from domain.exceptions.training import ProcessedMetadataError
from infrastructure.data.sequences.processed_feature_metadata import (
    FEATURE_METADATA_FILE,
    METADATA_DIRECTORY,
    SPLIT_FILENAME,
    TRANSFORMERS_FILE,
    NominalColumn,
    ProcessedFeatureMetadata,
)
from tests.fixtures.sequence_fixtures import (
    CONTINUOUS_COLUMNS,
    EXPANDED_DIMENSION,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    NOMINAL_COLUMN,
    NOMINAL_RAW_NAME,
    NOMINAL_VOCABULARY,
    PREPROCESSING_VERSION,
    RECEIVER_COLUMN,
    SENDER_COLUMN,
    SOURCE_DIMENSION,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    UNKNOWN_CODE,
    feature_metadata_payload,
    transformers_payload,
    write_metadata,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A processed root with valid synthetic metadata."""
    processed = tmp_path / "processed"
    write_metadata(processed)
    return processed


def read_with(
    tmp_path: Path,
    *,
    features: dict[str, Any] | None = None,
    transformers: dict[str, Any] | None = None,
) -> ProcessedFeatureMetadata:
    """Write metadata into a fresh root and read it back."""
    processed = tmp_path / "variant"
    write_metadata(processed, features=features, transformers=transformers)
    return ProcessedFeatureMetadata.read(processed)


# --- Faithful reading --------------------------------------------------------


def test_target_and_columns_are_read_from_the_artefact(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.target_column == TARGET_COLUMN
    assert metadata.feature_columns == FEATURE_COLUMNS
    assert metadata.identifier_columns == IDENTIFIER_COLUMNS
    assert metadata.preprocessing_version == PREPROCESSING_VERSION


def test_processed_column_names_are_translated_not_guessed(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    # Both come from raw_to_output_names: 'Timestamp' and 'Account'.
    assert metadata.timestamp_column == TIMESTAMP_COLUMN
    assert metadata.sender_column == SENDER_COLUMN


def test_split_path_points_at_the_recorded_filename(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.split_path("train") == root / "train" / SPLIT_FILENAME


def test_nominal_columns_come_from_the_recorded_vocabulary(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.nominal_columns == (
        NominalColumn(
            column=NOMINAL_COLUMN,
            vocabulary_size=NOMINAL_VOCABULARY,
            unknown_code=UNKNOWN_CODE,
        ),
    )


def test_continuous_columns_exclude_the_code_columns(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.continuous_columns() == CONTINUOUS_COLUMNS
    assert NOMINAL_COLUMN not in metadata.continuous_columns()


# --- Model columns and the feature width -------------------------------------


def test_expansion_off_feeds_the_source_columns_unchanged(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.model_columns(one_hot_nominal_codes=False) == FEATURE_COLUMNS
    assert metadata.feature_dimension(one_hot_nominal_codes=False) == SOURCE_DIMENSION


def test_expansion_on_replaces_each_code_column_with_its_indicator_block(
    root: Path,
) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.model_columns(one_hot_nominal_codes=True) == (
        *CONTINUOUS_COLUMNS,
        "payment_format_is_0",
        "payment_format_is_1",
        "payment_format_is_2",
    )


def test_expansion_arithmetic_is_continuous_plus_vocabularies(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    expanded = metadata.feature_dimension(one_hot_nominal_codes=True)
    assert expanded == EXPANDED_DIMENSION
    assert expanded == len(CONTINUOUS_COLUMNS) + NOMINAL_VOCABULARY
    # Same rule as the real artefacts: sources, minus the code columns, plus their
    # vocabularies.
    assert expanded == SOURCE_DIMENSION - 1 + NOMINAL_VOCABULARY


def test_indicator_block_is_positioned_where_the_code_column_was(
    tmp_path: Path,
) -> None:
    # Nominal column in the middle rather than at the end, so a naive "append the
    # indicators" implementation would produce a different order.
    features = feature_metadata_payload(
        feature_columns=["amount_log", NOMINAL_COLUMN, "same_bank"],
        column_order=[
            *IDENTIFIER_COLUMNS,
            "amount_log",
            NOMINAL_COLUMN,
            "same_bank",
            TARGET_COLUMN,
        ],
    )
    metadata = read_with(tmp_path, features=features)

    assert metadata.model_columns(one_hot_nominal_codes=True) == (
        "amount_log",
        "payment_format_is_0",
        "payment_format_is_1",
        "payment_format_is_2",
        "same_bank",
    )


def test_indicator_names_are_derived_from_the_column_stem() -> None:
    nominal = NominalColumn(
        column="receiving_currency_code", vocabulary_size=2, unknown_code=-1
    )

    assert nominal.indicator_names() == (
        "receiving_currency_is_0",
        "receiving_currency_is_1",
    )


def test_a_non_ordinal_scheme_expands_nothing(tmp_path: Path) -> None:
    # If the artefact says the codes were not ordinal-encoded, inventing an
    # expansion would contradict the record of what was actually done.
    transformers = transformers_payload(
        categorical_encoding={
            "scheme": "one_hot",
            "unknown_code": UNKNOWN_CODE,
            "columns": {},
        }
    )
    metadata = read_with(tmp_path, transformers=transformers)

    assert metadata.nominal_columns == ()
    assert metadata.feature_dimension(one_hot_nominal_codes=True) == SOURCE_DIMENSION


def test_a_coded_column_that_is_not_a_feature_is_ignored(tmp_path: Path) -> None:
    transformers = transformers_payload(
        categorical_encoding={
            "scheme": "ordinal",
            "unknown_code": UNKNOWN_CODE,
            "columns": {
                NOMINAL_RAW_NAME: {"vocabulary_size": NOMINAL_VOCABULARY},
                "Unused Column": {"vocabulary_size": 99},
            },
        }
    )
    features = feature_metadata_payload(
        raw_to_output_names={
            "Timestamp": TIMESTAMP_COLUMN,
            "Account": SENDER_COLUMN,
            "Account.1": RECEIVER_COLUMN,
            f"{NOMINAL_RAW_NAME}_code": NOMINAL_COLUMN,
            "Unused Column_code": "unused_column_code",
            "Is Laundering": TARGET_COLUMN,
        }
    )
    metadata = read_with(tmp_path, features=features, transformers=transformers)

    assert metadata.nominal_column_names() == (NOMINAL_COLUMN,)
    assert metadata.feature_dimension(one_hot_nominal_codes=True) == EXPANDED_DIMENSION


# --- What is read from the split files ---------------------------------------


def test_columns_to_read_covers_ordering_grouping_features_and_target(
    root: Path,
) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert metadata.columns_to_read() == (
        TIMESTAMP_COLUMN,
        SENDER_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    )


def test_the_receiving_account_is_never_read(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    assert RECEIVER_COLUMN not in metadata.columns_to_read()


def test_identifiers_are_not_on_the_model_feature_axis(root: Path) -> None:
    metadata = ProcessedFeatureMetadata.read(root)

    for one_hot in (True, False):
        columns = metadata.model_columns(one_hot_nominal_codes=one_hot)
        assert TIMESTAMP_COLUMN not in columns
        assert SENDER_COLUMN not in columns
        assert RECEIVER_COLUMN not in columns
        assert TARGET_COLUMN not in columns


def test_columns_to_read_has_no_duplicates(tmp_path: Path) -> None:
    # A schema in which the timestamp is also a feature must not be read twice.
    features = feature_metadata_payload(
        feature_columns=[TIMESTAMP_COLUMN, *FEATURE_COLUMNS],
        identifier_columns=[SENDER_COLUMN, RECEIVER_COLUMN],
    )
    metadata = read_with(tmp_path, features=features)

    columns = metadata.columns_to_read()
    assert len(columns) == len(set(columns))


# --- Refusals ----------------------------------------------------------------


def test_missing_feature_metadata_is_named(tmp_path: Path) -> None:
    with pytest.raises(ProcessedMetadataError) as caught:
        ProcessedFeatureMetadata.read(tmp_path / "absent")

    assert FEATURE_METADATA_FILE in str(caught.value)


def test_missing_transformers_artefact_is_named(tmp_path: Path) -> None:
    processed = tmp_path / "partial"
    directory = processed / METADATA_DIRECTORY
    directory.mkdir(parents=True)
    (directory / FEATURE_METADATA_FILE).write_text(
        json.dumps(feature_metadata_payload()), encoding="utf-8"
    )

    with pytest.raises(ProcessedMetadataError) as caught:
        ProcessedFeatureMetadata.read(processed)

    assert TRANSFORMERS_FILE in str(caught.value)


def test_unparseable_artefact_is_reported_with_its_path(tmp_path: Path) -> None:
    processed = tmp_path / "broken"
    write_metadata(processed)
    (processed / METADATA_DIRECTORY / FEATURE_METADATA_FILE).write_text(
        "{not json", encoding="utf-8"
    )

    with pytest.raises(ProcessedMetadataError, match="could not be parsed"):
        ProcessedFeatureMetadata.read(processed)


def test_target_listed_as_a_feature_is_refused(tmp_path: Path) -> None:
    features = feature_metadata_payload(
        feature_columns=[*FEATURE_COLUMNS, TARGET_COLUMN]
    )

    with pytest.raises(ProcessedMetadataError, match="leak the label"):
        read_with(tmp_path, features=features)


def test_a_column_that_is_both_identifier_and_feature_is_refused(
    tmp_path: Path,
) -> None:
    features = feature_metadata_payload(
        feature_columns=[*FEATURE_COLUMNS, SENDER_COLUMN]
    )

    with pytest.raises(ProcessedMetadataError, match="both identifiers and"):
        read_with(tmp_path, features=features)


def test_a_described_column_absent_from_the_file_is_refused(tmp_path: Path) -> None:
    features = feature_metadata_payload(
        feature_columns=[*FEATURE_COLUMNS, "column_that_was_never_written"]
    )

    with pytest.raises(ProcessedMetadataError, match="absent from column_order"):
        read_with(tmp_path, features=features)


def test_a_missing_required_key_is_named(tmp_path: Path) -> None:
    features = feature_metadata_payload()
    del features["target_column"]

    with pytest.raises(ProcessedMetadataError, match="'target_column'"):
        read_with(tmp_path, features=features)


def test_an_untranslatable_raw_name_is_refused(tmp_path: Path) -> None:
    features = feature_metadata_payload(
        raw_to_output_names={"Account": SENDER_COLUMN, "Is Laundering": TARGET_COLUMN}
    )

    with pytest.raises(ProcessedMetadataError, match="Timestamp"):
        read_with(tmp_path, features=features)


def test_a_vocabulary_size_that_is_not_positive_is_refused(tmp_path: Path) -> None:
    transformers = transformers_payload(
        categorical_encoding={
            "scheme": "ordinal",
            "unknown_code": UNKNOWN_CODE,
            "columns": {NOMINAL_RAW_NAME: {"vocabulary_size": 0}},
        }
    )

    with pytest.raises(ProcessedMetadataError, match="vocabulary_size"):
        read_with(tmp_path, transformers=transformers)


def test_a_missing_unknown_code_is_refused(tmp_path: Path) -> None:
    transformers = transformers_payload(
        categorical_encoding={
            "scheme": "ordinal",
            "columns": {NOMINAL_RAW_NAME: {"vocabulary_size": NOMINAL_VOCABULARY}},
        }
    )

    with pytest.raises(ProcessedMetadataError, match="unknown_code"):
        read_with(tmp_path, transformers=transformers)


def test_an_empty_feature_list_is_refused(tmp_path: Path) -> None:
    features = feature_metadata_payload(feature_columns=[])

    with pytest.raises(ProcessedMetadataError, match="non-empty list"):
        read_with(tmp_path, features=features)
