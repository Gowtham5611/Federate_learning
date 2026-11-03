"""Tests for the schema validator.

Two properties dominate: the validator must not touch the caller's data, and it
must report one finding per ``(column, kind)`` rather than one per bad row. Both
are asserted directly rather than assumed.

Every schema and value here is synthetic. Nothing in this file describes the IBM
AML dataset, whose schema this repository does not know.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from application.dto.validation import ChunkValidationResult
from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.enums.validation_severity import ValidationSeverity
from domain.value_objects.column_spec import ColumnSpec
from domain.value_objects.table_schema import TableSchema
from infrastructure.data.validators.dataframe_schema_validator import DataFrameSchemaValidator
from tests.fixtures.tabular_fixtures import (
    SYNTHETIC_COLUMNS,
    make_synthetic_schema,
    synthetic_row,
)

pytestmark = pytest.mark.unit

SOURCE = "synthetic_records.csv"


def valid_frame(rows: int = 3, *, start: int = 0) -> pd.DataFrame:
    """A frame of text values that conforms to the synthetic schema.

    Built as strings on purpose: a chunked CSV read produces whatever the parser
    inferred for that chunk, so the validator must judge values by coercion
    rather than by dtype.
    """
    data = [synthetic_row(index).split(",") for index in range(start, start + rows)]
    return pd.DataFrame(data, columns=list(SYNTHETIC_COLUMNS))


def validate(frame: pd.DataFrame, *, chunk_index: int = 0, **kwargs: Any) -> ChunkValidationResult:
    """Validate a frame against the synthetic schema."""
    validator = DataFrameSchemaValidator(make_synthetic_schema(**kwargs))
    return validator.validate_chunk(frame, chunk_index=chunk_index, source=SOURCE)


def validate_column(spec: ColumnSpec, values: list[Any]) -> ChunkValidationResult:
    """Validate a single column against a single-column schema."""
    schema = TableSchema(
        name="single_column",
        columns=(spec,),
        unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
    )
    frame = pd.DataFrame({spec.name: values})
    return DataFrameSchemaValidator(schema).validate_chunk(
        frame, chunk_index=0, source=SOURCE
    )


def kinds(result: ChunkValidationResult) -> list[ValidationIssueKind]:
    """The kinds of every finding, in reported order."""
    return [issue.kind for issue in result.issues]


# --- 16. Conforming data is accepted -----------------------------------------


def test_conforming_data_produces_no_findings() -> None:
    result = validate(valid_frame(5))

    assert result.is_valid
    assert result.issues == ()
    assert result.row_count == 5
    assert result.columns == SYNTHETIC_COLUMNS


def test_an_empty_but_correctly_shaped_chunk_is_valid() -> None:
    """Zero rows cannot violate a value constraint."""
    result = validate(valid_frame(0))

    assert result.is_valid
    assert result.row_count == 0


# --- 17. Missing required column ---------------------------------------------


def test_a_missing_required_column_is_an_error() -> None:
    frame = valid_frame(3).drop(columns=["value"])

    result = validate(frame)

    assert not result.is_valid
    assert kinds(result) == [ValidationIssueKind.MISSING_REQUIRED_COLUMN]
    assert result.issues[0].column == "value"


def test_a_missing_optional_column_is_not_an_error() -> None:
    frame = valid_frame(3).drop(columns=["receiver_ref"])

    result = validate(frame)

    assert result.is_valid
    assert result.issues == ()


def test_each_missing_required_column_is_reported_once() -> None:
    frame = valid_frame(3).drop(columns=["value", "unit"])

    result = validate(frame)

    assert kinds(result) == [ValidationIssueKind.MISSING_REQUIRED_COLUMN] * 2
    assert sorted(issue.column or "" for issue in result.issues) == ["unit", "value"]


def test_column_order_does_not_affect_validity() -> None:
    """Headers in a different order still describe the same table."""
    frame = valid_frame(3)[list(reversed(SYNTHETIC_COLUMNS))]

    assert validate(frame).is_valid


# --- 18. Invalid data types --------------------------------------------------


@pytest.mark.parametrize(
    ("data_type", "values"),
    [
        (ColumnDataType.INTEGER, ["1", "not-a-number"]),
        (ColumnDataType.INTEGER, ["1", "3.5"]),
        (ColumnDataType.FLOAT, ["1.5", "not-a-number"]),
        (ColumnDataType.DATETIME, ["2024-01-01T00:00:00", "not-a-date"]),
        (ColumnDataType.BOOLEAN, ["true", "maybe"]),
    ],
)
def test_a_value_that_is_not_the_declared_type_is_an_error(
    data_type: ColumnDataType, values: list[Any]
) -> None:
    result = validate_column(ColumnSpec(name="c", data_type=data_type), values)

    assert not result.is_valid
    assert kinds(result) == [ValidationIssueKind.WRONG_TYPE]
    assert result.issues[0].occurrences == 1


@pytest.mark.parametrize(
    ("data_type", "values"),
    [
        (ColumnDataType.INTEGER, ["1", "-2", "3"]),
        (ColumnDataType.FLOAT, ["1.5", "-0.25", "3"]),
        (ColumnDataType.DATETIME, ["2024-01-01T00:00:00", "2024-06-30T23:59:59"]),
        (ColumnDataType.BOOLEAN, ["true", "FALSE", "1", "0", "yes", "n"]),
        (ColumnDataType.STRING, ["anything", "at", "all"]),
    ],
)
def test_values_matching_the_declared_type_are_accepted(
    data_type: ColumnDataType, values: list[Any]
) -> None:
    assert validate_column(ColumnSpec(name="c", data_type=data_type), values).is_valid


def test_a_string_column_never_produces_a_type_error() -> None:
    """Every CSV field is text, so a string column has no type to fail."""
    result = validate_column(
        ColumnSpec(name="c", data_type=ColumnDataType.STRING), ["1", "x", "2024-01-01"]
    )

    assert result.is_valid


def test_the_declared_type_is_checked_regardless_of_the_parsers_dtype() -> None:
    """The same bad column must be caught whether it arrived as text or numbers."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER)

    as_text = validate_column(spec, ["1", "2.5"])
    as_numbers = validate_column(spec, [1.0, 2.5])

    assert kinds(as_text) == kinds(as_numbers) == [ValidationIssueKind.WRONG_TYPE]


# --- 19. Invalid values where a constraint is defined ------------------------


def test_a_value_outside_a_closed_vocabulary_is_an_error() -> None:
    frame = valid_frame(3)
    frame.loc[1, "unit"] = "CCC"

    result = validate(frame)

    assert kinds(result) == [ValidationIssueKind.VALUE_NOT_ALLOWED]
    assert result.issues[0].column == "unit"
    assert result.issues[0].occurrences == 1


def test_the_permitted_vocabulary_is_named_in_the_finding() -> None:
    """An error a reader cannot act on is not much better than no error."""
    frame = valid_frame(2)
    frame.loc[0, "unit"] = "ZZZ"

    detail = validate(frame).issues[0].detail

    assert "'AAA'" in detail
    assert "'BBB'" in detail


def test_a_value_below_the_minimum_is_an_error() -> None:
    frame = valid_frame(3)
    frame.loc[0, "value"] = "-1.00"

    result = validate(frame)

    assert kinds(result) == [ValidationIssueKind.VALUE_OUT_OF_RANGE]
    assert result.issues[0].column == "value"


def test_a_value_above_the_maximum_is_an_error() -> None:
    frame = valid_frame(3)
    frame.loc[2, "flag"] = "7"

    result = validate(frame)

    assert kinds(result) == [ValidationIssueKind.VALUE_OUT_OF_RANGE]
    assert result.issues[0].column == "flag"


def test_both_bounds_of_one_column_produce_a_single_finding() -> None:
    """Splitting 'too low' from 'too high' would be two messages about one column."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER, minimum=0, maximum=1)

    result = validate_column(spec, ["-5", "0", "1", "9"])

    assert kinds(result) == [ValidationIssueKind.VALUE_OUT_OF_RANGE]
    assert result.issues[0].occurrences == 2


def test_an_unconstrained_column_accepts_any_in_type_value() -> None:
    spec = ColumnSpec(name="c", data_type=ColumnDataType.FLOAT)

    assert validate_column(spec, ["-1e9", "0", "1e9"]).is_valid


# --- 20. Unexpected-column policy --------------------------------------------


def test_an_undeclared_column_is_an_error_under_the_reject_policy() -> None:
    frame = valid_frame(2)
    frame["surprise"] = "x"

    result = validate(frame, policy=UnexpectedColumnPolicy.REJECT)

    assert not result.is_valid
    assert kinds(result) == [ValidationIssueKind.UNEXPECTED_COLUMN]
    assert result.issues[0].severity is ValidationSeverity.ERROR


def test_an_undeclared_column_is_a_warning_under_the_allow_policy() -> None:
    frame = valid_frame(2)
    frame["surprise"] = "x"

    result = validate(frame, policy=UnexpectedColumnPolicy.ALLOW)

    assert result.is_valid, "an allowed extension must not invalidate the chunk"
    assert len(result.warnings) == 1
    assert result.warnings[0].kind is ValidationIssueKind.UNEXPECTED_COLUMN


def test_an_undeclared_column_is_never_dropped_silently() -> None:
    """Under either policy the column is reported; it is never ignored."""
    frame = valid_frame(2)
    frame["surprise"] = "x"

    for policy in (UnexpectedColumnPolicy.REJECT, UnexpectedColumnPolicy.ALLOW):
        result = validate(frame, policy=policy)
        assert "surprise" in [issue.column for issue in result.issues]
        assert "surprise" in result.columns


def test_the_policy_is_named_in_the_finding() -> None:
    frame = valid_frame(2)
    frame["surprise"] = "x"

    detail = validate(frame, policy=UnexpectedColumnPolicy.ALLOW).issues[0].detail

    assert "allow" in detail


# --- 21. Nullable fields -----------------------------------------------------


def test_nulls_in_a_nullable_column_are_accepted() -> None:
    frame = valid_frame(3)
    frame.loc[1, "receiver_ref"] = None

    assert validate(frame).is_valid


def test_nulls_in_a_non_nullable_column_are_an_error() -> None:
    frame = valid_frame(3)
    frame.loc[1, "sender_ref"] = None

    result = validate(frame)

    assert kinds(result) == [ValidationIssueKind.NULL_IN_NON_NULLABLE]
    assert result.issues[0].column == "sender_ref"
    assert result.issues[0].occurrences == 1


def test_nulls_are_not_also_reported_as_type_failures() -> None:
    """An empty cell is missing, not malformed; reporting both would double-count."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER, nullable=True)

    result = validate_column(spec, ["1", None, "3"])

    assert result.is_valid


def test_an_all_null_nullable_column_is_valid() -> None:
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER, nullable=True)

    assert validate_column(spec, [None, None]).is_valid


# --- 22. Validation does not mutate the caller's data ------------------------


def test_validation_does_not_mutate_a_conforming_frame() -> None:
    frame = valid_frame(4)
    before = frame.copy(deep=True)

    validate(frame)

    assert frame.equals(before)
    assert list(frame.dtypes.astype(str)) == list(before.dtypes.astype(str))
    assert list(frame.columns) == list(before.columns)


def test_validation_does_not_mutate_a_non_conforming_frame() -> None:
    """The tempting bug: coercing a column 'while checking it' and writing it back."""
    frame = valid_frame(4)
    frame.loc[0, "value"] = "not-a-number"
    frame.loc[1, "unit"] = "CCC"
    frame.loc[2, "sender_ref"] = None
    before = frame.copy(deep=True)

    result = validate(frame)

    assert not result.is_valid, "the frame really was invalid"
    assert frame.equals(before)
    assert list(frame.dtypes.astype(str)) == list(before.dtypes.astype(str))


def test_validation_leaves_text_values_exactly_as_they_arrived() -> None:
    """No trimming, no case folding: normalising text is preprocessing."""
    spec = ColumnSpec(
        name="c", data_type=ColumnDataType.STRING, allowed_values=("AAA",)
    )
    frame = pd.DataFrame({"c": [" AAA ", "aaa"]})
    before = frame.copy(deep=True)

    result = DataFrameSchemaValidator(
        TableSchema(
            name="s",
            columns=(spec,),
            unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
        )
    ).validate_chunk(frame, chunk_index=0, source=SOURCE)

    assert result.issues[0].occurrences == 2, "neither variant is the declared value"
    assert frame.equals(before)


# --- 23. Multiple chunks -----------------------------------------------------


def test_successive_chunks_are_validated_independently() -> None:
    validator = DataFrameSchemaValidator(make_synthetic_schema())
    good = valid_frame(3, start=0)
    bad = valid_frame(3, start=3)
    bad.loc[bad.index[0], "unit"] = "CCC"

    first = validator.validate_chunk(good, chunk_index=0, source=SOURCE)
    second = validator.validate_chunk(bad, chunk_index=1, source=SOURCE)
    third = validator.validate_chunk(good, chunk_index=2, source=SOURCE)

    assert first.is_valid
    assert not second.is_valid
    assert third.is_valid, "a bad chunk must not leave the validator in a bad state"


def test_each_chunk_reports_its_own_index() -> None:
    validator = DataFrameSchemaValidator(make_synthetic_schema())

    indices = [
        validator.validate_chunk(valid_frame(2), chunk_index=index, source=SOURCE).chunk_index
        for index in range(3)
    ]

    assert indices == [0, 1, 2]


# --- 24. Repeated violations are aggregated, not repeated --------------------


def test_many_bad_rows_in_one_column_produce_exactly_one_finding() -> None:
    """The duplicate-message problem: 50 bad rows must not mean 50 messages."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER)

    result = validate_column(spec, ["bad"] * 50)

    assert len(result.issues) == 1
    assert result.issues[0].occurrences == 50


def test_an_aggregated_finding_keeps_a_few_example_rows() -> None:
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER)

    issue = validate_column(spec, ["1", "bad", "3", "bad", "bad", "bad"]).issues[0]

    assert issue.occurrences == 4
    assert issue.example_rows == (1, 3, 4), "capped at the default of three examples"


def test_example_rows_point_at_positions_in_the_file_not_the_chunk() -> None:
    """A chunked read keeps a continuing index, which is what makes rows locatable."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER)
    frame = pd.DataFrame({"c": ["1", "bad"]}, index=pd.RangeIndex(start=500, stop=502))
    schema = TableSchema(
        name="s", columns=(spec,), unexpected_column_policy=UnexpectedColumnPolicy.REJECT
    )

    issue = DataFrameSchemaValidator(schema).validate_chunk(
        frame, chunk_index=5, source=SOURCE
    ).issues[0]

    assert issue.example_rows == (501,)


def test_different_columns_still_produce_separate_findings() -> None:
    """Aggregation is per column: two broken columns are two problems."""
    frame = valid_frame(3)
    frame.loc[0, "value"] = "bad"
    frame.loc[0, "unit"] = "CCC"

    result = validate(frame)

    assert len(result.issues) == 2
    assert {issue.column for issue in result.issues} == {"value", "unit"}


def test_different_problems_in_one_column_are_reported_separately() -> None:
    """A null and a bad value are different defects needing different fixes."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER, minimum=0, maximum=1)

    result = validate_column(spec, [None, "5"])

    assert set(kinds(result)) == {
        ValidationIssueKind.NULL_IN_NON_NULLABLE,
        ValidationIssueKind.VALUE_OUT_OF_RANGE,
    }


def test_the_example_count_is_configurable() -> None:
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER)
    schema = TableSchema(
        name="s", columns=(spec,), unexpected_column_policy=UnexpectedColumnPolicy.REJECT
    )
    frame = pd.DataFrame({"c": ["bad"] * 10})

    issue = DataFrameSchemaValidator(schema, max_examples=0).validate_chunk(
        frame, chunk_index=0, source=SOURCE
    ).issues[0]

    assert issue.example_rows == ()
    assert issue.occurrences == 10, "suppressing examples must not lose the count"


# --- 25. Chunk context is preserved -----------------------------------------


def test_the_result_carries_the_context_needed_to_act_on_it() -> None:
    frame = valid_frame(3)
    frame.loc[0, "unit"] = "CCC"

    result = validate(frame, chunk_index=7)

    assert result.source == SOURCE
    assert result.chunk_index == 7
    assert result.row_count == 3
    assert SOURCE in result.summary()
    assert "chunk 7" in result.summary()


def test_the_report_names_the_offending_column_and_row() -> None:
    frame = valid_frame(3)
    frame.loc[1, "unit"] = "CCC"

    report = validate(frame, chunk_index=2).report()

    assert "chunk 2" in report
    assert "unit" in report
    assert "row 1" in report


# --- 26. Determinism ---------------------------------------------------------


def test_validating_the_same_chunk_twice_gives_equal_results() -> None:
    frame = valid_frame(4)
    frame.loc[0, "value"] = "bad"
    frame.loc[1, "unit"] = "CCC"
    frame.loc[2, "sender_ref"] = None
    validator = DataFrameSchemaValidator(make_synthetic_schema())

    first = validator.validate_chunk(frame, chunk_index=0, source=SOURCE)
    second = validator.validate_chunk(frame, chunk_index=0, source=SOURCE)

    assert first == second


def test_finding_order_does_not_depend_on_column_order() -> None:
    """Reordering the frame's columns must not reorder the report."""
    frame = valid_frame(3)
    frame.loc[0, "value"] = "bad"
    frame.loc[0, "unit"] = "CCC"
    reordered = frame[list(reversed(SYNTHETIC_COLUMNS))]

    assert kinds(validate(frame)) == kinds(validate(reordered))
    assert [i.column for i in validate(frame).issues] == [
        i.column for i in validate(reordered).issues
    ]


def test_structural_findings_are_reported_before_value_findings() -> None:
    """A missing column explains the errors that follow it, so it is read first."""
    frame = valid_frame(3).drop(columns=["value"])
    frame.loc[0, "unit"] = "CCC"

    assert kinds(validate(frame)) == [
        ValidationIssueKind.MISSING_REQUIRED_COLUMN,
        ValidationIssueKind.VALUE_NOT_ALLOWED,
    ]
