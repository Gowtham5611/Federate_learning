"""Preprocessing CLI command.

Deliberately thin. It builds nothing and decides nothing: the service is
injected as a factory so that constructing it — which creates output
directories — happens only when the command actually runs, not when the parser
is built.

This command never downloads anything. It operates on the dataset already
present on disk and fails with acquisition instructions if it is not.
"""

from __future__ import annotations

from typing import Protocol

from application.dto.preprocessing import PreprocessingReport
from application.services.preprocessing_service import PreprocessingService
from domain.enums.split_name import ORDERED_SPLITS

_WIDTH = 40


class PreprocessingServiceFactory(Protocol):
    """Builds a preprocessing service once the command has decided to run one."""

    def __call__(
        self, *, chunk_size: int | None, keep_interim: bool
    ) -> PreprocessingService:
        """Construct the service with any command-line overrides applied."""
        ...


def preprocess_command(
    factory: PreprocessingServiceFactory,
    *,
    chunk_size: int | None = None,
    keep_interim: bool = False,
) -> int:
    """Run the preprocessing pipeline over the local dataset.

    Args:
        factory: Builds the configured service.
        chunk_size: Override for the configured rows-per-read.
        keep_interim: Leave the working files in place for inspection.

    Returns:
        Process exit code. Always 0 — every failure raises a
        :class:`~domain.exceptions.base.FedSecureAIError`, which the entry point
        reports without a traceback.
    """
    service = factory(chunk_size=chunk_size, keep_interim=keep_interim)
    rule = "=" * _WIDTH
    print(
        "\n".join(
            [
                rule,
                "FedSecureAI Dataset Preprocessing",
                rule,
                "",
                "Reading the local dataset in bounded chunks. Nothing is",
                "downloaded, and the raw file is opened read-only and verified",
                "byte-for-byte unchanged when the run finishes.",
                "",
            ]
        )
    )

    report = service.run()
    print(_render(report))
    return 0


def _render(report: PreprocessingReport) -> str:
    """Format the measured report as the CLI report block."""
    rule = "=" * _WIDTH
    cleaning = report.cleaning
    lines = [
        rule,
        "FedSecureAI Preprocessing Report",
        rule,
        "",
        f"Version:        {report.version}",
        f"Source:         {report.source_file}",
        f"Output:         {report.output_directory}",
        f"SHA-256:        {report.fingerprint.sha256}",
        f"Raw unchanged:  {'yes' if report.fingerprint.verified_unchanged else 'NO'}",
        "",
        "Rows",
        f"  read:         {cleaning.rows_read:,}",
        f"  removed:      {cleaning.rows_removed:,}",
        f"  written:      {cleaning.rows_written:,}",
        f"  reconciles:   {'yes' if cleaning.reconciles else 'NO'}",
        "",
    ]

    if cleaning.rows_removed:
        lines.append("Removals")
        lines.extend(
            f"  {record.rows_removed:>10,}  {record.reason}"
            for record in cleaning.removals
            if record.rows_removed
        )
        lines.append("")

    lines.extend(
        [
            f"Duplicate rows detected: {cleaning.duplicate_rows_detected:,} "
            f"(kept, not removed)",
            "",
            "Splits (chronological, 60/20/20 by timestamp)",
        ]
    )
    for summary in report.splits.summaries:
        share = report.splits.share(summary.name)
        lines.extend(
            [
                f"  {summary.name.value}",
                f"    rows:       {summary.row_count:,} ({share:.4%})",
                f"    window:     {summary.first_timestamp} .. {summary.last_timestamp}",
                f"    positives:  {summary.positive_count:,} "
                f"({summary.positive_rate:.6%})",
            ]
        )
    lines.extend(
        [
            "",
            f"Boundaries:     train <= {report.splits.train_last_timestamp}, "
            f"validation <= {report.splits.validation_last_timestamp}",
            f"Tie policy:     {report.splits.tie_policy}",
            f"Chronology:     {'holds' if report.splits.chronology_holds() else 'VIOLATED'}",
            "",
            "Target",
            f"  column:       {report.target.source_column} -> "
            f"{report.target.output_column}",
            f"  mapping:      {report.target.mapping}",
        ]
    )
    lines.extend(
        f"  class {label}:      {count:,} ({report.target.percentage(label):.4f}%)"
        for label, count in sorted(report.target.class_counts.items())
    )

    lines.extend(
        [
            "",
            "Features",
            f"  model inputs: {len(report.feature_columns)}",
            f"  identifiers:  {len(report.identifier_columns)} (not model inputs)",
            f"  documented exclusions: {len(report.excluded_features)}",
            "",
        ]
    )

    performance = report.performance
    if performance is not None:
        lines.extend(
            [
                "Performance",
                f"  total:        {performance.total_seconds:.1f}s",
                f"  throughput:   "
                f"{performance.rows_per_second(cleaning.rows_read):,.0f} rows/s",
                f"  chunk size:   {performance.peak_chunk_rows:,} rows",
                f"  accounts:     {performance.distinct_accounts_tracked:,} tracked",
                *(
                    f"  {stage:<13} {seconds:.1f}s"
                    for stage, seconds in performance.seconds_per_stage.items()
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Written:",
            *(f"  {split.value}/transactions.csv" for split in ORDERED_SPLITS),
            "  metadata/*.json",
            "",
            "This phase produces data only. No model was trained.",
            rule,
        ]
    )
    return "\n".join(lines)
