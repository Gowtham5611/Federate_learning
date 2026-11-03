"""Tests for the validation result DTOs.

These objects are what a reader of a failed ingestion run actually sees, so the
tests are about legibility and reproducibility: one finding per class of
violation, a deterministic order, and messages that name the column, the count
and where to look.
"""

from __future__ import annotations

import pytest

from application.dto.validation import ChunkValidationResult, ValidationIssue
from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.enums.validation_severity import ValidationSeverity

pytestmark = pytest.mark.unit


def issue(
    *,
    kind: ValidationIssueKind = ValidationIssueKind.WRONG_TYPE,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
    column: str | None = "value",
    detail: str = "invented detail",
    occurrences: int = 0,
    example_rows: tuple[int, ...] = (),
) -> ValidationIssue:
    """Build a finding with every field defaulted."""
    return ValidationIssue(
        kind=kind,
        severity=severity,
        column=column,
        detail=detail,
        occurrences=occurrences,
        example_rows=example_rows,
    )


def result(*issues: ValidationIssue, chunk_index: int = 0) -> ChunkValidationResult:
    """Build a result carrying the given findings."""
    return ChunkValidationResult(
        source="synthetic_records.csv",
        chunk_index=chunk_index,
        row_count=10,
        columns=("a", "b"),
        issues=issues,
    )


# --- A finding describes itself ------------------------------------------------


def test_a_structural_finding_names_the_column_and_the_severity() -> None:
    text = issue(
        kind=ValidationIssueKind.MISSING_REQUIRED_COLUMN,
        column="value",
        detail="required column is absent",
    ).describe()

    assert "[error]" in text
    assert "column 'value'" in text
    assert "required column is absent" in text


def test_a_structural_finding_reports_no_row_count() -> None:
    """A missing column is not row-scoped, so 'affected rows' would be noise."""
    text = issue(kind=ValidationIssueKind.MISSING_REQUIRED_COLUMN, occurrences=0).describe()

    assert "affected row" not in text


def test_a_row_scoped_finding_reports_how_many_rows_are_affected() -> None:
    text = issue(occurrences=4000).describe()

    assert "4000 affected row(s)" in text


def test_a_row_scoped_finding_points_at_example_rows() -> None:
    """The count says how bad it is; the examples say where to look."""
    text = issue(occurrences=4000, example_rows=(7, 19, 23)).describe()

    assert "e.g. row 7, 19, 23" in text


def test_a_finding_without_a_column_is_described_as_chunk_wide() -> None:
    text = issue(column=None, detail="row length varies").describe()

    assert "chunk: row length varies" in text


def test_a_warning_is_labelled_as_a_warning() -> None:
    text = issue(severity=ValidationSeverity.WARNING).describe()

    assert "[warning]" in text


def test_string_conversion_matches_the_description() -> None:
    """So a finding can be interpolated into a message without ceremony."""
    finding = issue(occurrences=2, example_rows=(1,))

    assert str(finding) == finding.describe()
    assert f"{finding}" == finding.describe()


def test_a_finding_is_immutable() -> None:
    finding = issue()

    with pytest.raises((AttributeError, TypeError)):
        finding.detail = "rewritten"  # type: ignore[misc]


def test_findings_with_equal_fields_compare_equal() -> None:
    assert issue(occurrences=3) == issue(occurrences=3)
    assert issue(occurrences=3) != issue(occurrences=4)


# --- Findings are ordered reproducibly ----------------------------------------


def test_findings_are_sorted_on_construction() -> None:
    """Order must not depend on the order the validator happened to check in."""
    unordered = result(
        issue(kind=ValidationIssueKind.VALUE_OUT_OF_RANGE, column="b"),
        issue(kind=ValidationIssueKind.MISSING_REQUIRED_COLUMN, column="a"),
    )

    assert [finding.kind for finding in unordered.issues] == [
        ValidationIssueKind.MISSING_REQUIRED_COLUMN,
        ValidationIssueKind.VALUE_OUT_OF_RANGE,
    ]


def test_structural_kinds_sort_before_value_kinds() -> None:
    """The order in which a reader can act on them: shape first, then contents."""
    ordering = [
        ValidationIssueKind.MISSING_REQUIRED_COLUMN,
        ValidationIssueKind.UNEXPECTED_COLUMN,
        ValidationIssueKind.WRONG_TYPE,
        ValidationIssueKind.NULL_IN_NON_NULLABLE,
        ValidationIssueKind.VALUE_NOT_ALLOWED,
        ValidationIssueKind.VALUE_OUT_OF_RANGE,
    ]

    keys = [kind.sort_key for kind in ordering]

    assert keys == sorted(keys)
    assert len(set(keys)) == len(ordering), "every kind needs a distinct rank"


def test_findings_of_one_kind_are_ordered_by_column() -> None:
    ordered = result(
        issue(column="zebra"), issue(column="alpha"), issue(column="mango")
    )

    assert [finding.column for finding in ordered.issues] == ["alpha", "mango", "zebra"]


def test_the_same_findings_in_any_order_produce_equal_results() -> None:
    first = issue(column="a")
    second = issue(column="b", kind=ValidationIssueKind.VALUE_NOT_ALLOWED)

    assert result(first, second) == result(second, first)


def test_a_finding_without_a_column_sorts_deterministically() -> None:
    """``None`` must not blow up the comparison against a named column."""
    ordered = result(issue(column="a"), issue(column=None))

    assert [finding.column for finding in ordered.issues] == [None, "a"]


# --- A result classifies its findings ------------------------------------------


def test_a_result_with_no_findings_is_valid() -> None:
    empty = result()

    assert empty.is_valid
    assert empty.errors == ()
    assert empty.warnings == ()


def test_errors_and_warnings_are_separated() -> None:
    mixed = result(
        issue(severity=ValidationSeverity.ERROR, column="a"),
        issue(severity=ValidationSeverity.WARNING, column="b"),
        issue(severity=ValidationSeverity.WARNING, column="c"),
    )

    assert len(mixed.errors) == 1
    assert len(mixed.warnings) == 2
    assert len(mixed.issues) == 3


def test_an_error_invalidates_the_chunk() -> None:
    assert not result(issue(severity=ValidationSeverity.ERROR)).is_valid


def test_warnings_alone_do_not_invalidate_the_chunk() -> None:
    """A tolerated deviation must not stop a run that was configured to allow it."""
    warned = result(issue(severity=ValidationSeverity.WARNING))

    assert warned.is_valid
    assert warned.warnings


def test_only_errors_are_blocking() -> None:
    assert ValidationSeverity.ERROR.is_blocking
    assert not ValidationSeverity.WARNING.is_blocking


# --- A result carries its own context ------------------------------------------


def test_the_summary_names_the_source_chunk_and_counts() -> None:
    text = result(
        issue(severity=ValidationSeverity.ERROR, column="a"),
        issue(severity=ValidationSeverity.WARNING, column="b"),
        chunk_index=3,
    ).summary()

    assert "chunk 3" in text
    assert "synthetic_records.csv" in text
    assert "10 row(s)" in text
    assert "1 error(s)" in text
    assert "1 warning(s)" in text


def test_the_summary_omits_the_individual_findings() -> None:
    """So it can be logged per chunk without flooding the log."""
    text = result(issue(detail="a very specific complaint")).summary()

    assert "a very specific complaint" not in text


def test_the_report_is_the_summary_plus_every_finding() -> None:
    full = result(
        issue(column="a", detail="first complaint"),
        issue(column="b", detail="second complaint"),
    ).report()

    lines = full.splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("chunk 0")
    assert "first complaint" in lines[1]
    assert "second complaint" in lines[2]


def test_a_clean_report_is_a_single_line() -> None:
    assert result().report().splitlines() == [result().summary()]


def test_a_result_is_immutable() -> None:
    outcome = result()

    with pytest.raises((AttributeError, TypeError)):
        outcome.chunk_index = 9  # type: ignore[misc]


def test_the_row_count_is_reported_whether_or_not_rows_were_valid() -> None:
    """Knowing 10 of 10 rows failed is different from knowing 10 failed."""
    assert result(issue(occurrences=10)).row_count == 10
