"""Schema validation findings, aggregated per column rather than per row."""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.enums.validation_severity import ValidationSeverity


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One aggregated schema violation.

    An issue describes a *class* of violation within one chunk, not a single bad
    cell. A chunk of 100 000 transactions with 4 000 malformed amounts yields one
    issue with ``occurrences=4000``, not 4 000 issues — the aggregation key is
    ``(column, kind)``. This is deliberate: per-row findings make a real problem
    unreadable and are what produced repeated identical messages previously.

    Attributes:
        kind: Category of violation. Half of the aggregation key.
        severity: Whether this invalidates the chunk.
        column: Column the finding concerns, or ``None`` for a whole-chunk
            finding. The other half of the aggregation key.
        detail: What was wrong, phrased so it is actionable on its own.
        occurrences: Number of affected rows. ``0`` marks a structural finding
            (a missing or undeclared column), which is not row-scoped.
        example_rows: A few affected row labels, to make the problem locatable
            without listing every row. Ascending.
    """

    kind: ValidationIssueKind
    severity: ValidationSeverity
    column: str | None
    detail: str
    occurrences: int = 0
    example_rows: tuple[int, ...] = ()

    @property
    def sort_key(self) -> tuple[int, str, str]:
        """Ordering that makes a result reproducible across runs."""
        return (self.kind.sort_key, self.column or "", self.detail)

    def describe(self) -> str:
        """One-line human-readable form, used in logs and error messages."""
        location = f"column '{self.column}'" if self.column else "chunk"
        text = f"[{self.severity.value}] {location}: {self.detail}"
        if self.occurrences:
            text += f" ({self.occurrences} affected row(s)"
            if self.example_rows:
                examples = ", ".join(str(row) for row in self.example_rows)
                text += f", e.g. row {examples}"
            text += ")"
        return text

    def __str__(self) -> str:
        """Same as :meth:`describe`, so interpolation stays readable."""
        return self.describe()


@dataclass(frozen=True, slots=True)
class ChunkValidationResult:
    """The outcome of validating one chunk against a schema.

    Carries its own context (``source``, ``chunk_index``) so a finding remains
    interpretable after it has been separated from the loop that produced it —
    "wrong type in column 'amount'" is not actionable without knowing which
    chunk of which file.

    Findings are sorted on construction, so validating identical input twice
    produces equal results.

    Attributes:
        source: Label for the data being validated, normally the file name.
        chunk_index: Zero-based position of this chunk in the stream.
        row_count: Rows in the chunk, whether or not they were valid.
        columns: Column names as they arrived, in file order.
        issues: Aggregated findings, deterministically ordered.
    """

    source: str
    chunk_index: int
    row_count: int
    columns: tuple[str, ...]
    issues: tuple[ValidationIssue, ...]

    def __post_init__(self) -> None:
        """Normalise finding order so equal input yields equal results."""
        object.__setattr__(
            self, "issues", tuple(sorted(self.issues, key=lambda issue: issue.sort_key))
        )

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Findings that invalidate the chunk."""
        return tuple(issue for issue in self.issues if issue.severity.is_blocking)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Findings worth reporting that do not invalidate the chunk."""
        return tuple(issue for issue in self.issues if not issue.severity.is_blocking)

    @property
    def is_valid(self) -> bool:
        """Whether the chunk conforms to the schema."""
        return not self.errors

    def summary(self) -> str:
        """One-line count summary, without the individual findings."""
        return (
            f"chunk {self.chunk_index} of '{self.source}': {self.row_count} row(s), "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )

    def report(self) -> str:
        """Full multi-line report: the summary followed by every finding."""
        lines = [self.summary()]
        lines.extend(f"  - {issue.describe()}" for issue in self.issues)
        return "\n".join(lines)
