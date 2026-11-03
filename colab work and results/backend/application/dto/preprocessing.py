"""Preprocessing results: what the pipeline did, in a form that can be audited.

Every dataclass here is a *record of fact*, not a knob. The pipeline produces
them, the CLI prints them and the metadata writer serialises them; nothing reads
one back to change its behaviour. That separation is what lets the final report
be assembled from measurements rather than from assumptions about what the code
was supposed to do.

Counts are deliberately redundant — ``rows_in``, ``rows_removed`` and
``rows_out`` are all stored rather than derived — because reconciling them is
the check that the pipeline lost nothing silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.enums.split_name import SplitName


@dataclass(frozen=True, slots=True)
class RemovalRecord:
    """One reason rows were dropped, and how many were dropped for it.

    Rule of the project: no row disappears without a line in the ledger. A
    reason that removed nothing is still recorded, because "0 rows failed the
    timestamp parse" is itself a finding worth reporting.

    Attributes:
        reason: Stable machine-readable key, e.g. ``"unparseable_timestamp"``.
        description: What the check tested, phrased for a human reader.
        rows_removed: How many rows the check removed across the whole dataset.
    """

    reason: str
    description: str
    rows_removed: int


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """The complete accounting of rows in, rows dropped and rows kept.

    Attributes:
        rows_read: Data rows parsed from the raw file.
        removals: One record per cleaning rule, in the order applied.
        rows_written: Rows that survived to the processed output.
        duplicate_rows_detected: Exact duplicate rows found.
        duplicate_rows_removed: How many of those were dropped. Kept separate
            from ``duplicate_rows_detected`` because this project *keeps*
            duplicates by default and reports them.
    """

    rows_read: int
    removals: tuple[RemovalRecord, ...]
    rows_written: int
    duplicate_rows_detected: int
    duplicate_rows_removed: int

    @property
    def rows_removed(self) -> int:
        """Total rows dropped by all cleaning rules."""
        return sum(record.rows_removed for record in self.removals)

    @property
    def reconciles(self) -> bool:
        """Whether ``rows_read - rows_removed == rows_written``.

        The single check that proves no row vanished unaccounted for.
        """
        return self.rows_read - self.rows_removed == self.rows_written


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """What one split contains, measured after it was written.

    Attributes:
        name: Which split.
        row_count: Rows written for this split.
        first_timestamp: Earliest transaction timestamp, ISO-8601.
        last_timestamp: Latest transaction timestamp, ISO-8601.
        positive_count: Rows whose target is the positive class.
        negative_count: Rows whose target is the negative class.
    """

    name: SplitName
    row_count: int
    first_timestamp: str
    last_timestamp: str
    positive_count: int
    negative_count: int

    @property
    def positive_rate(self) -> float:
        """Share of positive-class rows. ``0.0`` for an empty split."""
        return self.positive_count / self.row_count if self.row_count else 0.0


@dataclass(frozen=True, slots=True)
class SplitReport:
    """The chronological split as a whole: boundaries, sizes and class balance.

    Attributes:
        train_last_timestamp: The last timestamp assigned to training.
        validation_last_timestamp: The last timestamp assigned to validation.
        tie_policy: How rows sharing a boundary timestamp were assigned.
        summaries: One summary per split, in chronological order.
    """

    train_last_timestamp: str
    validation_last_timestamp: str
    tie_policy: str
    summaries: tuple[SplitSummary, ...]

    @property
    def total_rows(self) -> int:
        """Rows across all splits."""
        return sum(summary.row_count for summary in self.summaries)

    def summary_for(self, split: SplitName) -> SplitSummary:
        """Return the summary for one split.

        Raises:
            KeyError: The split was not produced.
        """
        for summary in self.summaries:
            if summary.name is split:
                return summary
        raise KeyError(f"no summary for split '{split.value}'")

    def share(self, split: SplitName) -> float:
        """Fraction of all rows that landed in ``split``."""
        total = self.total_rows
        return self.summary_for(split).row_count / total if total else 0.0

    def chronology_holds(self) -> bool:
        """Whether each split ends no later than the next one begins.

        Timestamps are ISO-8601 with fixed-width fields, so lexicographic
        comparison is chronological comparison; no parsing is needed to check
        the property the split exists to guarantee.
        """
        ordered = sorted(self.summaries, key=lambda summary: summary.name.position)
        populated = [summary for summary in ordered if summary.row_count]
        return all(
            earlier.last_timestamp <= later.first_timestamp
            for earlier, later in zip(populated, populated[1:], strict=False)
        )


@dataclass(frozen=True, slots=True)
class TargetReport:
    """The label column as it was actually found, not as it was expected.

    Attributes:
        source_column: The raw column the label came from.
        output_column: The column name in the processed output.
        observed_values: Every distinct raw value seen, sorted.
        mapping: Raw value (as text) to encoded value.
        positive_label: Which encoded value means "laundering".
        class_counts: Encoded value to row count, over the whole dataset.
    """

    source_column: str
    output_column: str
    observed_values: tuple[str, ...]
    mapping: dict[str, int]
    positive_label: int
    class_counts: dict[int, int]

    @property
    def total(self) -> int:
        """Rows counted across all classes."""
        return sum(self.class_counts.values())

    def percentage(self, label: int) -> float:
        """Share of rows carrying ``label``, as a percentage."""
        total = self.total
        return 100.0 * self.class_counts.get(label, 0) / total if total else 0.0


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    """Identity of the exact input bytes a run consumed.

    Attributes:
        file_name: Name of the raw file.
        size_bytes: Its size.
        sha256: Digest of its full contents.
        verified_unchanged: Whether the digest still matched after the run.
    """

    file_name: str
    size_bytes: int
    sha256: str
    verified_unchanged: bool


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Measured cost of a run. Every field is timed, none is estimated.

    Attributes:
        seconds_per_stage: Wall-clock seconds per named stage, in order.
        peak_chunk_rows: Largest number of rows held at once by design.
        distinct_accounts_tracked: Size of the behavioural state, which is what
            actually bounds memory — it grows with distinct accounts, not with
            rows.
    """

    seconds_per_stage: dict[str, float]
    peak_chunk_rows: int
    distinct_accounts_tracked: int

    @property
    def total_seconds(self) -> float:
        """Wall-clock seconds across all stages."""
        return sum(self.seconds_per_stage.values())

    def rows_per_second(self, rows: int) -> float:
        """Throughput over the whole run."""
        total = self.total_seconds
        return rows / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class PreprocessingReport:
    """Everything one preprocessing run established, in one object.

    Attributes:
        version: Preprocessing version stamped into the outputs.
        source_file: Absolute path of the raw file, as text.
        output_directory: Where the processed splits were written, as text.
        fingerprint: Identity of the consumed bytes.
        cleaning: The row ledger.
        splits: Boundaries, sizes and class balance.
        target: The label as found.
        feature_columns: Model-input column names, in output order.
        identifier_columns: Non-feature columns carried for traceability.
        excluded_features: Feature name to the reason it was not implemented.
        performance: Measured timings.
    """

    version: str
    source_file: str
    output_directory: str
    fingerprint: DatasetFingerprint
    cleaning: CleaningReport
    splits: SplitReport
    target: TargetReport
    feature_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    excluded_features: dict[str, str] = field(default_factory=dict)
    performance: PerformanceReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, for the metadata artefacts."""
        return {
            "preprocessing_version": self.version,
            "source_file": self.source_file,
            "output_directory": self.output_directory,
            "dataset_fingerprint": {
                "file_name": self.fingerprint.file_name,
                "size_bytes": self.fingerprint.size_bytes,
                "sha256": self.fingerprint.sha256,
                "verified_unchanged_after_run": self.fingerprint.verified_unchanged,
            },
            "cleaning": {
                "rows_read": self.cleaning.rows_read,
                "rows_removed": self.cleaning.rows_removed,
                "rows_written": self.cleaning.rows_written,
                "reconciles": self.cleaning.reconciles,
                "duplicate_rows_detected": self.cleaning.duplicate_rows_detected,
                "duplicate_rows_removed": self.cleaning.duplicate_rows_removed,
                "removals": [
                    {
                        "reason": record.reason,
                        "description": record.description,
                        "rows_removed": record.rows_removed,
                    }
                    for record in self.cleaning.removals
                ],
            },
            "splits": {
                "tie_policy": self.splits.tie_policy,
                "train_last_timestamp": self.splits.train_last_timestamp,
                "validation_last_timestamp": self.splits.validation_last_timestamp,
                "chronology_holds": self.splits.chronology_holds(),
                "per_split": [
                    {
                        "name": summary.name.value,
                        "rows": summary.row_count,
                        "share": self.splits.share(summary.name),
                        "first_timestamp": summary.first_timestamp,
                        "last_timestamp": summary.last_timestamp,
                        "positive": summary.positive_count,
                        "negative": summary.negative_count,
                        "positive_rate": summary.positive_rate,
                    }
                    for summary in self.splits.summaries
                ],
            },
            "target": {
                "source_column": self.target.source_column,
                "output_column": self.target.output_column,
                "observed_values": list(self.target.observed_values),
                "mapping": self.target.mapping,
                "positive_label": self.target.positive_label,
                "class_counts": {str(k): v for k, v in self.target.class_counts.items()},
                "class_percentages": {
                    str(k): self.target.percentage(k) for k in sorted(self.target.class_counts)
                },
            },
            "columns": {
                "features": list(self.feature_columns),
                "identifiers": list(self.identifier_columns),
            },
            "excluded_features": dict(self.excluded_features),
            "performance": (
                {
                    "seconds_per_stage": dict(self.performance.seconds_per_stage),
                    "total_seconds": self.performance.total_seconds,
                    "rows_per_second": self.performance.rows_per_second(
                        self.cleaning.rows_read
                    ),
                    "peak_chunk_rows": self.performance.peak_chunk_rows,
                    "distinct_accounts_tracked": (
                        self.performance.distinct_accounts_tracked
                    ),
                }
                if self.performance is not None
                else None
            ),
        }
