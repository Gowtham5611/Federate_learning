"""Tests for the dataset verification script's inspection logic.

The script's job is to state facts about a file that is far too large to hold in
memory. These tests run its machinery over *tiny invented CSV files* so that the
counting, probing and bounding logic is proven correct **before** it is pointed
at the real IBM AML dataset.

Every fixture here is TEST DATA: a handful of rows written by these tests. None
of it is IBM AML data, none of it resembles the real dataset's columns, and no
statistic produced here describes the real dataset. The real file is never read
by the test suite, which stays fully offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.verify_dataset import (
    CARDINALITY_CAP,
    ColumnObservation,
    count_newlines,
    human_bytes,
    inspect,
    inventory,
    probe_file,
    render_report,
)

pytestmark = pytest.mark.unit

# TEST DATA. Invented columns, deliberately unlike the real dataset's.
SAMPLE_CSV = (
    "alpha,beta,gamma\n"
    "1,red,10.5\n"
    "2,blue,20.5\n"
    "3,red,\n"
    "4,blue,40.5\n"
)


def write_csv(tmp_path: Path, body: str, name: str = "sample.csv") -> Path:
    """Write a small invented CSV. TEST DATA, not IBM AML data."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --- Byte-level probing -------------------------------------------------------


def test_probe_finds_a_comma_delimiter_and_a_header(tmp_path: Path) -> None:
    probe = probe_file(write_csv(tmp_path, SAMPLE_CSV))

    assert probe.delimiter == ","
    assert probe.header_detected is True
    assert probe.encoding == "utf-8"
    assert probe.has_bom is False


def test_probe_detects_a_semicolon_delimiter(tmp_path: Path) -> None:
    """The delimiter is observed, never assumed to be a comma."""
    body = "alpha;beta\n1;red\n2;blue\n3;green\n4;red\n"
    assert probe_file(write_csv(tmp_path, body)).delimiter == ";"


def test_probe_detects_a_tab_delimiter(tmp_path: Path) -> None:
    body = "alpha\tbeta\n1\tred\n2\tblue\n3\tgreen\n4\tred\n"
    assert probe_file(write_csv(tmp_path, body)).delimiter == "\t"


def test_probe_reports_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + SAMPLE_CSV.encode("utf-8"))

    probe = probe_file(path)

    assert probe.has_bom is True
    assert probe.encoding == "utf-8"


def test_probe_falls_back_through_the_encoding_list(tmp_path: Path) -> None:
    """A byte sequence that is not valid UTF-8 must still be readable."""
    path = tmp_path / "latin.csv"
    path.write_bytes("alpha,beta\n1,caf\xe9\n2,na\xefve\n3,r\xf4le\n".encode("cp1252"))

    probe = probe_file(path)

    assert probe.encoding != "utf-8"
    assert "utf-8" in probe.encodings_rejected


def test_probe_records_which_component_decided(tmp_path: Path) -> None:
    """Provenance is reported so a fallback is never mistaken for a measurement."""
    probe = probe_file(write_csv(tmp_path, SAMPLE_CSV))

    assert probe.delimiter_source
    assert probe.header_source


def test_probe_keeps_the_first_two_lines_verbatim(tmp_path: Path) -> None:
    probe = probe_file(write_csv(tmp_path, SAMPLE_CSV))

    assert probe.first_line == "alpha,beta,gamma"
    assert probe.second_line == "1,red,10.5"


# --- Newline counting ---------------------------------------------------------


def test_newlines_are_counted_by_streaming(tmp_path: Path) -> None:
    assert count_newlines(write_csv(tmp_path, SAMPLE_CSV)) == 5


def test_a_missing_final_newline_is_visible_in_the_count(tmp_path: Path) -> None:
    """A row count that disagrees with the newline count must be observable."""
    assert count_newlines(write_csv(tmp_path, SAMPLE_CSV.rstrip("\n"))) == 4


def test_an_empty_file_has_no_newlines(tmp_path: Path) -> None:
    assert count_newlines(write_csv(tmp_path, "")) == 0


# --- Bounded column accumulation ---------------------------------------------


def test_a_numeric_column_reports_its_observed_range() -> None:
    column = ColumnObservation(name="alpha", position=0)
    column.observe(pd.Series([1, 5, 3]))
    column.observe(pd.Series([9, 2]))

    assert column.minimum == 1.0
    assert column.maximum == 9.0
    assert column.is_numeric is True
    assert column.total == 5


def test_nulls_are_counted_across_chunks() -> None:
    column = ColumnObservation(name="gamma", position=2)
    column.observe(pd.Series([1.0, None]))
    column.observe(pd.Series([None, None]))

    assert column.nulls == 3
    assert column.non_null == 1
    assert column.null_percent == 75.0


def test_a_column_of_only_nulls_reports_no_range() -> None:
    column = ColumnObservation(name="empty", position=0)
    column.observe(pd.Series([None, None], dtype="object"))

    assert column.nulls == 2
    assert column.minimum is None
    assert column.is_numeric is False


def test_a_text_column_reports_its_length_range() -> None:
    column = ColumnObservation(name="beta", position=1)
    column.observe(pd.Series(["a", "abcd"]))

    assert column.min_length == 1
    assert column.max_length == 4
    assert column.is_numeric is False


def test_distinct_values_are_counted_exactly_below_the_cap() -> None:
    column = ColumnObservation(name="beta", position=1)
    column.observe(pd.Series(["red", "blue", "red"]))
    column.observe(pd.Series(["red"]))

    assert column.distinct == 2
    assert column.counts is not None
    assert column.counts["red"] == 3
    assert column.counts["blue"] == 1


def test_a_high_cardinality_column_is_abandoned_rather_than_enumerated() -> None:
    """This is what keeps memory constant on a column of unique identifiers."""
    column = ColumnObservation(name="id", position=0)
    column.observe(pd.Series([f"id-{index}" for index in range(CARDINALITY_CAP + 5)]))

    assert column.counts is None
    assert column.distinct is None
    assert column.distinct_exceeded_cap is True


def test_the_cap_can_be_exceeded_by_accumulation_across_chunks() -> None:
    """Each chunk stays under the cap; their union does not."""
    column = ColumnObservation(name="id", position=0)
    half = CARDINALITY_CAP // 2 + 1

    column.observe(pd.Series([f"a-{index}" for index in range(half)]))
    after_first = column.distinct
    column.observe(pd.Series([f"b-{index}" for index in range(half)]))
    after_second = column.distinct

    assert after_first == half
    assert after_second is None
    assert column.distinct_exceeded_cap is True


def test_an_abandoned_column_is_never_resurrected() -> None:
    column = ColumnObservation(name="id", position=0)
    column.observe(pd.Series([f"id-{index}" for index in range(CARDINALITY_CAP + 5)]))
    column.observe(pd.Series(["red", "red"]))

    assert column.counts is None


def test_a_boolean_column_is_not_treated_as_numeric() -> None:
    """A range of 0..1 would be a meaningless thing to report."""
    column = ColumnObservation(name="flag", position=0)
    column.observe(pd.Series([True, False, True]))

    assert column.is_numeric is False
    assert column.distinct == 2


def test_samples_are_bounded_and_real() -> None:
    column = ColumnObservation(name="beta", position=1)
    column.observe(pd.Series(["red", "blue", "green", "black", "white"]))

    assert len(column.samples) <= 3
    assert set(column.samples) <= {"red", "blue", "green", "black", "white"}


# --- The chunked pass ---------------------------------------------------------


def test_the_chunked_pass_counts_every_row(tmp_path: Path) -> None:
    result = inspect(write_csv(tmp_path, SAMPLE_CSV), chunk_size=2)

    assert result.rows == 4
    assert result.chunks == 2
    assert len(result.columns) == 3


def test_a_single_chunk_and_many_chunks_agree(tmp_path: Path) -> None:
    """The row count must not depend on how the file was sliced."""
    path = write_csv(tmp_path, SAMPLE_CSV)

    one = inspect(path, chunk_size=1000)
    many = inspect(path, chunk_size=1)

    assert one.rows == many.rows == 4
    assert [column.name for column in one.columns] == [
        column.name for column in many.columns
    ]
    assert [column.nulls for column in one.columns] == [
        column.nulls for column in many.columns
    ]


def test_columns_keep_their_file_order(tmp_path: Path) -> None:
    result = inspect(write_csv(tmp_path, SAMPLE_CSV), chunk_size=2)

    assert [column.name for column in result.columns] == ["alpha", "beta", "gamma"]
    assert [column.position for column in result.columns] == [0, 1, 2]


def test_nulls_survive_the_chunk_boundaries(tmp_path: Path) -> None:
    result = inspect(write_csv(tmp_path, SAMPLE_CSV), chunk_size=1)
    gamma = next(column for column in result.columns if column.name == "gamma")

    assert gamma.nulls == 1
    assert gamma.non_null == 3


def test_the_size_reported_is_the_size_on_disk(tmp_path: Path) -> None:
    path = write_csv(tmp_path, SAMPLE_CSV)
    result = inspect(path, chunk_size=2)

    assert result.size_bytes == path.stat().st_size


def test_a_malformed_row_is_recorded_not_swallowed(tmp_path: Path) -> None:
    """A parse failure must appear in the report rather than vanish."""
    body = "alpha,beta\n1,red\n2,blue,EXTRA\n3,green\n"
    result = inspect(write_csv(tmp_path, body), chunk_size=10)

    assert result.malformed is not None


def test_a_clean_file_reports_no_malformed_rows(tmp_path: Path) -> None:
    assert inspect(write_csv(tmp_path, SAMPLE_CSV), chunk_size=2).malformed is None


# --- Inventory ---------------------------------------------------------------


def test_inventory_lists_every_file_with_its_size(tmp_path: Path) -> None:
    write_csv(tmp_path, SAMPLE_CSV, name="one.csv")
    write_csv(tmp_path, SAMPLE_CSV, name="two.csv")
    (tmp_path / "nested").mkdir()
    write_csv(tmp_path / "nested", "x\n1\n", name="three.csv")

    files = inventory(tmp_path)

    assert len(files) == 3
    assert all(size > 0 for _, size in files)


def test_inventory_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    """An absent dataset directory is reported as empty, not as an error."""
    assert inventory(tmp_path / "nothing-here") == []


def test_inventory_ignores_directories_themselves(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    assert inventory(tmp_path) == []


# --- Rendering ---------------------------------------------------------------


def test_human_bytes_reports_the_exact_count_alongside_the_rounded_one() -> None:
    """The exact figure must always be present; MiB alone is not a measurement."""
    rendered = human_bytes(1024 * 1024 * 3)

    assert "3,145,728 bytes" in rendered
    assert "3.0 MiB" in rendered


class StubSource:
    """The identity fields the report quotes, without a dataset manager."""

    dataset_id = "test_dataset"
    remote_id = "example/test-dataset"
    variant = "TEST"
    expected_filename = "sample.csv"


def test_the_report_states_the_measured_figures(tmp_path: Path) -> None:
    path = write_csv(tmp_path, SAMPLE_CSV)
    result = inspect(path, chunk_size=2)

    report = render_report(result, inventory(tmp_path), StubSource())  # type: ignore[arg-type]

    assert "# Dataset verification report" in report
    assert f"{result.size_bytes:,} bytes" in report
    assert "| Rows (parsed, excluding header) | 4 |" in report
    assert "| Columns | 3 |" in report
    for name in ("alpha", "beta", "gamma"):
        assert f"`{name}`" in report


def test_the_report_enumerates_low_cardinality_values(tmp_path: Path) -> None:
    """The evidence a target column would be identified from."""
    path = write_csv(tmp_path, SAMPLE_CSV)
    result = inspect(path, chunk_size=2)

    report = render_report(result, inventory(tmp_path), StubSource())  # type: ignore[arg-type]

    assert "Low-cardinality columns" in report
    assert "`red`" in report
    assert "`blue`" in report


def test_the_report_never_claims_a_meaning_for_a_column(tmp_path: Path) -> None:
    """Groupings are by observed dtype; semantics are left to a human.

    The report may name the dataset and may say the words "target column" in its
    explanatory prose. What it must never do is attach an invented meaning to a
    column, so the check is scoped to the column rows themselves.
    """
    path = write_csv(tmp_path, SAMPLE_CSV)
    result = inspect(path, chunk_size=2)

    report = render_report(result, inventory(tmp_path), StubSource())  # type: ignore[arg-type]
    column_rows = [
        line for line in report.splitlines() if line.startswith(("| 0 |", "| 1 |", "| 2 |"))
    ]

    assert "by observed dtype, not by meaning" in report
    assert len(column_rows) == 3
    for row in column_rows:
        for invented in ("fraud", "launder", "target", "label", "amount", "timestamp"):
            assert invented not in row.lower()


def test_the_report_flags_a_row_count_that_disagrees_with_the_newlines(
    tmp_path: Path,
) -> None:
    """An embedded newline inside a quoted field must not pass unremarked."""
    body = 'alpha,beta\n1,"two\nlines"\n2,plain\n'
    path = write_csv(tmp_path, body)
    result = inspect(path, chunk_size=10)

    report = render_report(result, inventory(tmp_path), StubSource())  # type: ignore[arg-type]

    assert result.rows == 2
    assert result.raw_newlines == 4
    assert "does not equal the parsed row count" in report


def test_the_report_counts_the_csv_files_present(tmp_path: Path) -> None:
    path = write_csv(tmp_path, SAMPLE_CSV)
    write_csv(tmp_path, SAMPLE_CSV, name="other.csv")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    result = inspect(path, chunk_size=2)

    report = render_report(result, inventory(tmp_path), StubSource())  # type: ignore[arg-type]

    assert "CSV files present: **2**" in report
