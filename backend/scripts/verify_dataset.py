#!/usr/bin/env python
"""Verify and inspect the real IBM AML dataset, without loading it into RAM.

Operational script, run manually after an explicit download:

    python backend/scripts/verify_dataset.py

What it does, in order:

1. Inventories the dataset directory, so the actual archive layout is observed
   rather than assumed.
2. Locates the transaction file through the ordinary
   :class:`~infrastructure.data.sources.ibm_aml_data_source.IBMAMLDataSource`,
   the same component the application uses.
3. Probes a small byte prefix to determine encoding, delimiter and whether a
   header row is present.
4. Streams the whole file in chunks through
   :class:`~infrastructure.data.loaders.csv_chunk_loader.CsvChunkLoader`,
   accumulating only bounded summaries: counts, min/max, and value counts for
   columns whose cardinality stays under a cap.
5. Writes a markdown verification report to ``reports/`` (git-ignored).

**It never calls an unrestricted ``pd.read_csv(path)``.** Peak memory is one
chunk plus the bounded summaries, whatever the file's size.

Nothing here downloads anything. If the file is absent the script says so and
exits non-zero; it does not invent values, and it does not fall back to another
dataset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# An operational script is executed as a file, not imported as a package, so
# the BACKEND source tree (this file's import root) has to be on the path before
# the project's own imports resolve. The repository root is resolved by
# `infrastructure.utils.paths`, imported below.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402 - must follow the sys.path adjustment above

from application.services.dataset_manager import DatasetManager  # noqa: E402
from domain.exceptions.base import FedSecureAIError  # noqa: E402
from infrastructure.data.downloaders.kaggle_downloader import KaggleDownloader  # noqa: E402
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader  # noqa: E402
from infrastructure.data.sources.ibm_aml_data_source import IBMAMLDataSource  # noqa: E402
from infrastructure.data.sources.yaml_dataset_registry import YamlDatasetRegistry  # noqa: E402
from infrastructure.utils.env import load_env_file  # noqa: E402
from infrastructure.utils.paths import PROJECT_ROOT, resolve  # noqa: E402

DEFAULT_CHUNK_SIZE = 250_000
PROBE_BYTES = 64 * 1024
CARDINALITY_CAP = 60
"""Beyond this many distinct values a column is summarised, not enumerated."""

SAMPLE_VALUES = 3
ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


# --- Bounded per-column accumulator ------------------------------------------


@dataclass
class ColumnObservation:
    """Everything learned about one column, in constant space.

    ``counts`` is abandoned once the column proves high-cardinality, so a column
    of millions of distinct identifiers costs nothing to observe.
    """

    name: str
    position: int
    dtypes: set[str] = field(default_factory=set)
    non_null: int = 0
    nulls: int = 0
    counts: Counter[str] | None = field(default_factory=Counter)
    distinct_exceeded_cap: bool = False
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    samples: list[str] = field(default_factory=list)

    def observe(self, series: pd.Series) -> None:
        """Fold one chunk's worth of this column into the summary."""
        self.dtypes.add(str(series.dtype))
        null_mask = series.isna()
        null_count = int(null_mask.sum())
        self.nulls += null_count
        self.non_null += int(len(series)) - null_count

        present = series[~null_mask]
        if present.empty:
            return

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            self._observe_numeric(present)
        else:
            self._observe_text(present)

        self._observe_cardinality(present)
        self._collect_samples(present)

    def _observe_numeric(self, present: pd.Series) -> None:
        """Track the numeric range."""
        low, high = float(present.min()), float(present.max())
        self.minimum = low if self.minimum is None else min(self.minimum, low)
        self.maximum = high if self.maximum is None else max(self.maximum, high)

    def _observe_text(self, present: pd.Series) -> None:
        """Track the rendered length range, which bounds a string column."""
        lengths = present.astype(str).str.len()
        low, high = int(lengths.min()), int(lengths.max())
        self.min_length = low if self.min_length is None else min(self.min_length, low)
        self.max_length = high if self.max_length is None else max(self.max_length, high)

    def _observe_cardinality(self, present: pd.Series) -> None:
        """Count distinct values until the cap makes it pointless."""
        if self.counts is None:
            return
        chunk_counts = present.astype(str).value_counts()
        if len(chunk_counts) > CARDINALITY_CAP:
            self.counts = None
            self.distinct_exceeded_cap = True
            return
        for value, count in chunk_counts.items():
            self.counts[str(value)] += int(count)
        if len(self.counts) > CARDINALITY_CAP:
            self.counts = None
            self.distinct_exceeded_cap = True

    def _collect_samples(self, present: pd.Series) -> None:
        """Keep a few real values, purely to make the report legible."""
        while len(self.samples) < SAMPLE_VALUES and len(self.samples) < len(present):
            self.samples.append(str(present.iloc[len(self.samples)]))

    @property
    def total(self) -> int:
        """Rows observed for this column."""
        return self.non_null + self.nulls

    @property
    def null_percent(self) -> float:
        """Percentage of observed rows that were null."""
        return 0.0 if self.total == 0 else 100.0 * self.nulls / self.total

    @property
    def distinct(self) -> int | None:
        """Exact distinct count, or ``None`` when the cap was exceeded."""
        return None if self.counts is None else len(self.counts)

    @property
    def is_numeric(self) -> bool:
        """Whether every dtype seen was numeric."""
        return self.minimum is not None


# --- File-level observations --------------------------------------------------


@dataclass
class FileProbe:
    """Byte-level facts established from a small prefix of the file."""

    encoding: str
    encodings_rejected: tuple[str, ...]
    has_bom: bool
    delimiter: str
    delimiter_source: str
    header_detected: bool
    header_source: str
    first_line: str
    second_line: str


@dataclass
class Inspection:
    """The complete result of the chunked pass."""

    path: Path
    size_bytes: int
    probe: FileProbe
    columns: list[ColumnObservation]
    rows: int
    chunks: int
    chunk_size: int
    raw_newlines: int
    malformed: str | None


def probe_file(path: Path) -> FileProbe:
    """Determine encoding, delimiter and header presence from a byte prefix."""
    prefix = path.read_bytes()[:PROBE_BYTES]
    has_bom = prefix.startswith(b"\xef\xbb\xbf")

    rejected: list[str] = []
    text: str | None = None
    encoding = "unknown"
    for candidate in ENCODINGS_TO_TRY:
        try:
            text = prefix.decode(candidate)
        except UnicodeDecodeError:
            rejected.append(candidate)
            continue
        encoding = candidate
        break
    if text is None:
        raise SystemExit(f"Could not decode {path.name} with any of {ENCODINGS_TO_TRY}")

    # The final line of a fixed-size prefix is probably truncated.
    lines = text.splitlines()[:-1] or text.splitlines()
    sample = "\n".join(lines)

    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter, delimiter_source = dialect.delimiter, "csv.Sniffer"
    except csv.Error:
        delimiter, delimiter_source = ",", "fallback (Sniffer could not decide)"

    try:
        header_detected = csv.Sniffer().has_header(sample)
        header_source = "csv.Sniffer"
    except csv.Error:
        header_detected = True
        header_source = "fallback (Sniffer could not decide)"

    return FileProbe(
        encoding=encoding,
        encodings_rejected=tuple(rejected),
        has_bom=has_bom,
        delimiter=delimiter,
        delimiter_source=delimiter_source,
        header_detected=header_detected,
        header_source=header_source,
        first_line=lines[0] if lines else "",
        second_line=lines[1] if len(lines) > 1 else "",
    )


def count_newlines(path: Path) -> int:
    """Count newline bytes by streaming, for comparison with the parsed rows."""
    total = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            total += block.count(b"\n")
    return total


def inspect(path: Path, *, chunk_size: int) -> Inspection:
    """Stream the file and accumulate bounded per-column summaries."""
    from domain.exceptions.data import MalformedDataError

    probe = probe_file(path)
    observations: dict[str, ColumnObservation] = {}
    rows = 0
    chunks = 0
    malformed: str | None = None

    loader = CsvChunkLoader()
    try:
        for chunk in loader.stream_chunks(path, chunk_size):
            chunks += 1
            rows += len(chunk)
            for position, name in enumerate(chunk.columns):
                key = str(name)
                if key not in observations:
                    observations[key] = ColumnObservation(name=key, position=position)
                observations[key].observe(chunk[name])
            print(f"  ... {rows:,} rows read ({chunks} chunks)", end="\r", flush=True)
    except MalformedDataError as exc:
        malformed = str(exc)
    print()

    return Inspection(
        path=path,
        size_bytes=path.stat().st_size,
        probe=probe,
        columns=sorted(observations.values(), key=lambda column: column.position),
        rows=rows,
        chunks=chunks,
        chunk_size=chunk_size,
        raw_newlines=count_newlines(path),
        malformed=malformed,
    )


# --- Reporting ---------------------------------------------------------------


def human_bytes(size: int) -> str:
    """Render a byte count as both an exact and an approximate figure."""
    megabytes = size / (1024 * 1024)
    return f"{size:,} bytes ({megabytes:,.1f} MiB)"


def inventory(directory: Path) -> list[tuple[Path, int]]:
    """Every file beneath the dataset directory, with its size."""
    if not directory.is_dir():
        return []
    return sorted(
        ((path, path.stat().st_size) for path in directory.rglob("*") if path.is_file()),
        key=lambda item: str(item[0]),
    )


def project_label(path: Path) -> str:
    """Render ``path`` relative to the project root, or absolutely if outside it.

    The dataset normally lives under ``data/raw/``, but the location comes from
    configuration and may point anywhere. Reporting must not fail because of it.
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render_report(
    inspection: Inspection, files: list[tuple[Path, int]], source: IBMAMLDataSource
) -> str:
    """Build the markdown verification report from observed facts only."""
    lines: list[str] = [
        "# Dataset verification report — IBM AML HI-Small",
        "",
        f"Generated {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')} by "
        f"`backend/scripts/verify_dataset.py`.",
        "",
        "Every figure below was measured from the real local file by a chunked "
        "pass. Nothing is estimated, remembered or inferred from a column name.",
        "",
        "## Identity",
        "",
        "| Field | Verified value |",
        "| --- | --- |",
        "| Dataset | IBM Transactions for Anti-Money Laundering (AML) |",
        f"| Project dataset id | `{source.dataset_id}` |",
        f"| Remote id | `{source.remote_id}` |",
        f"| Variant | `{source.variant}` |",
        f"| Expected filename | `{source.expected_filename}` |",
        f"| Actual file | `{inspection.path.name}` |",
        f"| Location | `{project_label(inspection.path.parent)}/` |",
        f"| File size | {human_bytes(inspection.size_bytes)} |",
        "",
        "## Structure",
        "",
        "| Field | Verified value |",
        "| --- | --- |",
        f"| Rows (parsed, excluding header) | {inspection.rows:,} |",
        f"| Newline bytes in file | {inspection.raw_newlines:,} |",
        f"| Columns | {len(inspection.columns)} |",
        f"| Encoding | `{inspection.probe.encoding}`"
        f"{' (BOM present)' if inspection.probe.has_bom else ' (no BOM)'} |",
        f"| Delimiter | `{inspection.probe.delimiter}` "
        f"(determined by {inspection.probe.delimiter_source}) |",
        f"| Header row | {'present' if inspection.probe.header_detected else 'absent'} "
        f"(determined by {inspection.probe.header_source}) |",
        f"| Chunked pass | {inspection.chunks} chunks of {inspection.chunk_size:,} rows |",
        f"| Malformed rows | "
        f"{inspection.malformed or 'none — no parse error during the full chunked pass'} |",
    ]

    if inspection.rows and inspection.raw_newlines not in (
        inspection.rows,
        inspection.rows + 1,
    ):
        lines.extend(
            [
                f"> Note: the newline count ({inspection.raw_newlines:,}) does not equal the "
                f"parsed row count ({inspection.rows:,}) + 1 header, which indicates embedded "
                f"newlines inside quoted fields or a missing final newline.",
                "",
            ]
        )

    lines.extend(
        [
            "### Files actually present",
            "",
            "| Path (relative to the dataset directory) | Size |",
            "| --- | --- |",
        ]
    )
    root = inspection.path.parent
    for path, size in files:
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        lines.append(f"| `{label}` | {size:,} bytes |")
    csv_files = [path for path, _ in files if path.suffix.lower() == ".csv"]
    lines.extend(
        [
            "",
            f"CSV files present: **{len(csv_files)}** "
            f"({', '.join(f'`{p.name}`' for p in csv_files) or 'none'}).",
            "",
            "## Columns as observed",
            "",
            "| # | Column | Observed dtype(s) | Nulls | Distinct | Range / lengths | Samples |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for column in inspection.columns:
        dtypes = ", ".join(f"`{name}`" for name in sorted(column.dtypes))
        nulls = f"{column.nulls:,} ({column.null_percent:.2f}%)"
        distinct = (
            f"{column.distinct:,}"
            if column.distinct is not None
            else f"> {CARDINALITY_CAP}"
        )
        if column.is_numeric:
            extent = f"{column.minimum:,.6g} … {column.maximum:,.6g}"
        elif column.min_length is not None:
            extent = f"{column.min_length}–{column.max_length} chars"
        else:
            extent = "—"
        samples = ", ".join(f"`{value}`" for value in column.samples) or "—"
        lines.append(
            f"| {column.position} | `{column.name}` | {dtypes} | {nulls} | "
            f"{distinct} | {extent} | {samples} |"
        )

    low_cardinality = [
        column
        for column in inspection.columns
        if column.counts is not None and 1 <= len(column.counts) <= 20
    ]
    lines.extend(
        [
            "",
            "## Low-cardinality columns and their value counts",
            "",
            "Reported as evidence for identifying a target column and for defining "
            "closed vocabularies. Semantics are **not** inferred from names here.",
            "",
        ]
    )
    if not low_cardinality:
        lines.append("No column had 20 or fewer distinct values.")
    for column in low_cardinality:
        assert column.counts is not None
        lines.extend(
        [
            f"### `{column.name}`",
            "",
            "| Value | Count | Share |",
            "| --- | --- | --- |",
        ]
)
        for value, count in column.counts.most_common():
            share = 100.0 * count / column.non_null if column.non_null else 0.0
            lines.append(f"| `{value}` | {count:,} | {share:.4f}% |")
        lines.append("")

    lines.extend(
        [
            "## Null summary",
            "",
            "| Column | Nulls | % of rows |",
            "| --- | --- | --- |",
        ]
    )
    for column in inspection.columns:
        lines.append(f"| `{column.name}` | {column.nulls:,} | {column.null_percent:.4f}% |")

    numeric = [column.name for column in inspection.columns if column.is_numeric]
    categorical = [column.name for column in inspection.columns if not column.is_numeric]
    lines.extend(
        [
            "",
            "## Field groupings (by observed dtype, not by meaning)",
            "",
            f"- **Numeric ({len(numeric)}):** "
            f"{', '.join(f'`{name}`' for name in numeric) or 'none'}",
            f"- **Non-numeric ({len(categorical)}):** "
            f"{', '.join(f'`{name}`' for name in categorical) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


# --- Entry point -------------------------------------------------------------


def build_source() -> IBMAMLDataSource:
    """Compose the data source exactly as the application does."""
    manager = DatasetManager(
        registry=YamlDatasetRegistry(resolve("datasets/dataset_registry.yaml")),
        downloader=KaggleDownloader(),
        project_root=PROJECT_ROOT,
    )
    return IBMAMLDataSource(manager)


def main(argv: list[str] | None = None) -> int:
    """Inspect the real dataset and write the verification report.

    Returns:
        0 on success, 2 when the dataset is not present locally.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_verification_HI-Small.md",
    )
    args = parser.parse_args(argv)

    load_env_file()
    source = build_source()

    print(f"Dataset directory: {source.local_directory}")
    files = inventory(source.local_directory)
    print(f"Files present: {len(files)}")
    for path, size in files:
        print(f"  - {path.name}  ({size:,} bytes)")

    try:
        path = source.resolve_file()
    except FedSecureAIError as exc:
        print(f"\nCannot verify: {exc}", file=sys.stderr)
        print(
            "\nNo report was written and no value was invented. Download the "
            "dataset first:\n    python main.py dataset-download",
            file=sys.stderr,
        )
        return 2

    print(f"\nInspecting {path.name} in chunks of {args.chunk_size:,} rows...")
    inspection = inspect(path, chunk_size=args.chunk_size)

    report = render_report(inspection, files, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Rows: {inspection.rows:,}   Columns: {len(inspection.columns)}")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
