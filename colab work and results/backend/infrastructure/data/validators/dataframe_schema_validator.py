"""Schema validation for pandas chunks: reports problems, changes nothing.

The only module that compares data against a :class:`TableSchema`. It is a pure
function of ``(schema, chunk)``: no logging, no state, no mutation — so the same
chunk always produces the same report, and running validation cannot change what
the rest of the pipeline sees.
"""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame, Series

from application.dto.validation import ChunkValidationResult, ValidationIssue
from application.ports.output.schema_validator import SchemaValidatorPort
from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.enums.validation_severity import ValidationSeverity
from domain.value_objects.column_spec import ColumnSpec
from domain.value_objects.table_schema import TableSchema

_BOOLEAN_TOKENS = frozenset({"true", "false", "t", "f", "yes", "no", "y", "n"})
"""Textual forms accepted for a boolean column, in addition to 0 and 1."""

_DEFAULT_MAX_EXAMPLES = 3


class DataFrameSchemaValidator(SchemaValidatorPort):
    """Checks a chunk against a schema without modifying it.

    Two properties are enforced by construction rather than by care:

    * **Nothing is written back.** Type conformance is established by *coercing a
      copy* — ``pd.to_numeric`` and friends return new objects — and the result is
      used only to count failures. The caller's frame is never assigned to, so
      validation cannot double as an accidental preprocessing step.
    * **Findings are aggregated.** One finding per ``(column, kind)`` per chunk,
      carrying an occurrence count and a few example row labels. A column with
      50 000 bad values produces one line, not 50 000.

    Values are checked by coercion rather than by inspecting the parser's dtype,
    because chunked CSV reads infer dtypes per chunk: the same column can arrive
    as ``int64`` in one chunk and as text in the next. Trusting the dtype would
    make validation depend on chunk boundaries.

    Row labels in ``example_rows`` are the frame's index labels. For a chunked
    read these continue across chunks, so an example label points at the row's
    position in the original file rather than within the chunk.
    """

    def __init__(self, schema: TableSchema, *, max_examples: int = _DEFAULT_MAX_EXAMPLES) -> None:
        """Bind the validator to one schema.

        Args:
            schema: Structure to enforce.
            max_examples: How many example row labels to keep per finding. Small
                on purpose: examples exist to locate a problem, not to enumerate
                it.
        """
        self._schema = schema
        self._max_examples = max(0, max_examples)

    @property
    def schema(self) -> TableSchema:
        """The schema being enforced."""
        return self._schema

    def validate_chunk(
        self, chunk: DataFrame, *, chunk_index: int, source: str
    ) -> ChunkValidationResult:
        """Check one chunk and return an aggregated, deterministically ordered report.

        Args:
            chunk: Rows to check. Returned untouched.
            chunk_index: Zero-based position in the stream.
            source: Where the data came from, normally a file name.

        Returns:
            The findings for this chunk.
        """
        present = tuple(str(name) for name in chunk.columns)
        issues: list[ValidationIssue] = []

        issues.extend(self._check_structure(present))
        declared = set(self._schema.column_names)
        for spec in self._schema.columns:
            if spec.name in present and spec.name in declared:
                issues.extend(self._check_column(spec, chunk[spec.name]))

        return ChunkValidationResult(
            source=source,
            chunk_index=chunk_index,
            row_count=len(chunk),
            columns=present,
            issues=tuple(issues),
        )

    # -- Structure ---------------------------------------------------------

    def _check_structure(self, present: tuple[str, ...]) -> list[ValidationIssue]:
        """Report missing required columns and columns the schema does not declare."""
        issues: list[ValidationIssue] = []
        seen = set(present)

        for spec in self._schema.required_columns:
            if spec.name not in seen:
                issues.append(
                    ValidationIssue(
                        kind=ValidationIssueKind.MISSING_REQUIRED_COLUMN,
                        severity=ValidationSeverity.ERROR,
                        column=spec.name,
                        detail=(
                            f"required column is absent; the file provides "
                            f"{len(present)} column(s) and none of them is this one"
                        ),
                    )
                )

        declared = set(self._schema.column_names)
        policy = self._schema.unexpected_column_policy
        for name in present:
            if name in declared:
                continue
            reject = policy is UnexpectedColumnPolicy.REJECT
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.UNEXPECTED_COLUMN,
                    severity=ValidationSeverity.ERROR if reject else ValidationSeverity.WARNING,
                    column=name,
                    detail=(
                        "column is not declared in the schema and the schema's "
                        f"policy is '{policy.value}'"
                        + (
                            "; the file's format has changed, or the wrong file "
                            "is being read"
                            if reject
                            else "; it is carried through unvalidated"
                        )
                    ),
                )
            )
        return issues

    # -- Values ------------------------------------------------------------

    def _check_column(self, spec: ColumnSpec, column: Series) -> list[ValidationIssue]:
        """Check one present column's values against its specification."""
        issues: list[ValidationIssue] = []

        null_mask = column.isna()
        if not spec.nullable and bool(null_mask.any()):
            issues.append(
                self._value_issue(
                    ValidationIssueKind.NULL_IN_NON_NULLABLE,
                    spec.name,
                    "column is declared non-nullable but contains empty values",
                    null_mask,
                )
            )

        non_null = column[~null_mask]
        if non_null.empty:
            return issues

        numeric = self._coerced_numeric(spec, non_null)
        issues.extend(self._check_type(spec, non_null, numeric))
        issues.extend(self._check_allowed_values(spec, non_null))
        if numeric is not None:
            issues.extend(self._check_bounds(spec, numeric))
        return issues

    @staticmethod
    def _coerced_numeric(spec: ColumnSpec, non_null: Series) -> Series | None:
        """Numeric view of a numeric column, as a new Series, or ``None``.

        The coercion result is reused for both the type check and the bounds
        check so the column is only converted once.
        """
        if not spec.data_type.is_numeric:
            return None
        return pd.to_numeric(non_null, errors="coerce")

    def _check_type(
        self,
        spec: ColumnSpec,
        non_null: Series,
        numeric: Series | None,
    ) -> list[ValidationIssue]:
        """Report values that cannot be read as the declared type.

        ``STRING`` is absent by design: every CSV field is text, so a string
        column has no type failure to report. Its constraint is
        ``allowed_values``.
        """
        bad: Series | None = None
        expectation = ""

        if spec.data_type is ColumnDataType.INTEGER and numeric is not None:
            bad = numeric.isna() | (numeric % 1 != 0)
            expectation = "a whole number"
        elif spec.data_type is ColumnDataType.FLOAT and numeric is not None:
            bad = numeric.isna()
            expectation = "a number"
        elif spec.data_type is ColumnDataType.DATETIME:
            bad = pd.to_datetime(non_null, errors="coerce").isna()
            expectation = "a date/time"
        elif spec.data_type is ColumnDataType.BOOLEAN:
            text = non_null.astype(str).str.strip().str.lower()
            bad = ~(pd.to_numeric(text, errors="coerce").isin([0, 1]) | text.isin(_BOOLEAN_TOKENS))
            expectation = "a boolean (true/false, yes/no, 1/0)"

        if bad is None or not bool(bad.any()):
            return []
        return [
            self._value_issue(
                ValidationIssueKind.WRONG_TYPE,
                spec.name,
                f"values are not {expectation} as declared "
                f"('{spec.data_type.value}')",
                bad,
            )
        ]

    def _check_allowed_values(
        self,
        spec: ColumnSpec,
        non_null: Series,
    ) -> list[ValidationIssue]:
        """Report values outside a declared closed vocabulary.

        Comparison is exact. Trimming or case-folding first would make the
        validator quietly accept data it was told to reject, and normalising
        text is preprocessing.
        """
        if not spec.allowed_values:
            return []
        bad = ~non_null.astype(str).isin(spec.allowed_values)
        if not bool(bad.any()):
            return []
        vocabulary = ", ".join(f"'{value}'" for value in spec.allowed_values)
        return [
            self._value_issue(
                ValidationIssueKind.VALUE_NOT_ALLOWED,
                spec.name,
                f"values are outside the permitted set ({vocabulary})",
                bad,
            )
        ]

    def _check_bounds(
        self,
        spec: ColumnSpec,
        numeric: Series,
    ) -> list[ValidationIssue]:
        """Report values outside the declared inclusive bounds.

        Both bounds produce a single finding, because ``(column, kind)`` is the
        aggregation key and splitting "too low" from "too high" would reintroduce
        two messages about one broken column.
        """
        if spec.minimum is None and spec.maximum is None:
            return []
        below = numeric < spec.minimum if spec.minimum is not None else False
        above = numeric > spec.maximum if spec.maximum is not None else False
        bad = below | above
        if isinstance(bad, bool) or not bool(bad.any()):
            return []
        low = "-inf" if spec.minimum is None else str(spec.minimum)
        high = "+inf" if spec.maximum is None else str(spec.maximum)
        return [
            self._value_issue(
                ValidationIssueKind.VALUE_OUT_OF_RANGE,
                spec.name,
                f"values fall outside the permitted range [{low}, {high}]",
                bad,
            )
        ]

    # -- Finding construction ----------------------------------------------

    def _value_issue(
        self,
        kind: ValidationIssueKind,
        column: str,
        detail: str,
        mask: Series,
    ) -> ValidationIssue:
        """Aggregate a row-level failure mask into exactly one finding."""
        return ValidationIssue(
            kind=kind,
            severity=ValidationSeverity.ERROR,
            column=column,
            detail=detail,
            occurrences=int(mask.sum()),
            example_rows=self._examples(mask),
        )

    def _examples(self, mask: Series) -> tuple[int, ...]:
        """First few index labels where ``mask`` is true, as integers.

        Non-integer labels are skipped rather than coerced: a wrong row number is
        worse than no row number.
        """
        if self._max_examples == 0:
            return ()
        labels = mask.index[mask.to_numpy(dtype=bool)][: self._max_examples]
        examples: list[int] = []
        for label in labels:
            try:
                examples.append(int(label))
            except (TypeError, ValueError):
                continue
        return tuple(examples)
