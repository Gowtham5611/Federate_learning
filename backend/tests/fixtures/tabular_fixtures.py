"""Synthetic tabular fixtures for the data ingestion tests.

=============================================================================
SYNTHETIC TEST DATA - NOT IBM AML DATA
=============================================================================
Every schema, column name and value in this module is invented for testing the
ingestion machinery. None of it is derived from, or claims to represent, the IBM
Transactions for AML dataset:

* the column names (``record_id``, ``sender_ref``, ``unit`` ...) are generic and
  deliberately unlike any real dataset's headers;
* the ``unit`` vocabulary is ``AAA``/``BBB``, not currency codes, so no value here
  can be mistaken for a real observation;
* no row count, class ratio or file size here says anything about the real
  dataset.

The real IBM AML schema is defined in
``infrastructure/data/schemas/ibm_aml_schema.py``. These fixtures deliberately do
not use it: the loader and validator must be testable *without* the real schema
and without the 475 MB download, so a schema change cannot quietly rewrite what
they assert.

Files are written into pytest's ``tmp_path``, so nothing is committed and no file
is ever named ``HI-Small_Trans.csv``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.value_objects.column_spec import ColumnSpec
from domain.value_objects.table_schema import TableSchema

TEST_DATA_NOTICE = "SYNTHETIC TEST DATA - NOT IBM AML DATA"

SYNTHETIC_COLUMNS: tuple[str, ...] = (
    "record_id",
    "event_time",
    "sender_ref",
    "receiver_ref",
    "value",
    "unit",
    "flag",
)

SYNTHETIC_HEADER = ",".join(SYNTHETIC_COLUMNS)

SYNTHETIC_UNITS: tuple[str, ...] = ("AAA", "BBB")


def make_synthetic_schema(
    *, policy: UnexpectedColumnPolicy = UnexpectedColumnPolicy.REJECT
) -> TableSchema:
    """Build the invented schema that matches :data:`SYNTHETIC_HEADER`.

    Covers every constraint kind the validator implements - a required
    non-nullable integer, a datetime, a bounded float, a closed vocabulary, an
    optional nullable column, and a 0/1 flag - so one schema exercises all of
    them.
    """
    return TableSchema(
        name="synthetic_test_records",
        columns=(
            ColumnSpec(
                name="record_id",
                data_type=ColumnDataType.INTEGER,
                description="Invented row identifier.",
            ),
            ColumnSpec(
                name="event_time",
                data_type=ColumnDataType.DATETIME,
                description="Invented timestamp.",
            ),
            ColumnSpec(
                name="sender_ref",
                data_type=ColumnDataType.STRING,
                description="Invented sender reference.",
            ),
            ColumnSpec(
                name="receiver_ref",
                data_type=ColumnDataType.STRING,
                required=False,
                nullable=True,
                description="Invented receiver reference; optional and nullable.",
            ),
            ColumnSpec(
                name="value",
                data_type=ColumnDataType.FLOAT,
                minimum=0.0,
                description="Invented non-negative quantity.",
            ),
            ColumnSpec(
                name="unit",
                data_type=ColumnDataType.STRING,
                allowed_values=SYNTHETIC_UNITS,
                description="Invented closed vocabulary; not currency codes.",
            ),
            ColumnSpec(
                name="flag",
                data_type=ColumnDataType.INTEGER,
                minimum=0,
                maximum=1,
                description="Invented binary flag.",
            ),
        ),
        unexpected_column_policy=policy,
    )


def synthetic_row(index: int) -> str:
    """Build one obviously-invented CSV row."""
    day = (index % 28) + 1
    return ",".join(
        (
            str(1000 + index),
            f"2024-01-{day:02d}T10:{index % 60:02d}:00",
            f"REF-{index:04d}",
            f"REF-{index + 500:04d}",
            f"{100 + index}.50",
            SYNTHETIC_UNITS[index % len(SYNTHETIC_UNITS)],
            str(index % 2),
        )
    )


def write_csv(path: Path, lines: Sequence[str]) -> Path:
    """Write raw CSV lines verbatim.

    Deliberately raw rather than pandas-generated, so a test can produce input
    that pandas would refuse to write - a ragged row, an unterminated quote, an
    empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_synthetic_csv(path: Path, *, rows: int = 6) -> Path:
    """Write a well-formed file matching :func:`make_synthetic_schema`."""
    return write_csv(path, [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(rows))])
