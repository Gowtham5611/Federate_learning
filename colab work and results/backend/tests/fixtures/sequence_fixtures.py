"""Synthetic processed-split fixtures for the training tests.

=============================================================================
SYNTHETIC TEST DATA - NOT IBM AML DATA
=============================================================================
Every row produced here comes from a small deterministic function in this file.
The rows are **not** IBM AML data, not a sample of it and not derived from it. No
count, timestamp, amount, account, category code or class ratio produced by this
module says anything about the real dataset. Nothing here is ever written into
``data/`` — files go into pytest's ``tmp_path`` and vanish with the test.

The real ``data/processed/ibm_aml/HI-Small/`` tree is 23 source features wide,
57 model inputs after the nominal expansion, and over five million rows. It is
never read by a unit test: a suite that needed it could not run on a checkout
without it, and a 530 MB read per test is not a test.

What *is* copied from the real artefacts is their **shape**: the same metadata
filenames, the same keys, one nominal column recorded as an ordinal code with a
vocabulary and an unknown code, identifier columns listed separately from feature
columns, and splits that are chronologically ordered and non-overlapping. That is
deliberate — these fixtures exercise code written against those artefacts, and
inventing a different layout would test nothing.

The fixture is smaller in every dimension (2 continuous features, 1 nominal
column with a 3-value vocabulary, so 5 model inputs) precisely so that expected
tensors can be written out by hand in the tests that need them.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from domain.enums.split_name import ORDERED_SPLITS, SplitName
from infrastructure.data.sequences.processed_feature_metadata import (
    FEATURE_METADATA_FILE,
    METADATA_DIRECTORY,
    SPLIT_FILENAME,
    TRANSFORMERS_FILE,
)

TEST_DATA_NOTICE = "SYNTHETIC TEST DATA - NOT IBM AML DATA"

PREPROCESSING_VERSION = "test-0.0.0"
"""Deliberately not the real ``1.0.0``, so a fixture artefact cannot be mistaken
for one preprocessing actually wrote."""

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_COLUMN = "timestamp"
SENDER_COLUMN = "from_account"
RECEIVER_COLUMN = "to_account"
TARGET_COLUMN = "is_laundering"

CONTINUOUS_COLUMNS: tuple[str, ...] = ("amount_log", "same_bank")
NOMINAL_RAW_NAME = "Payment Format"
NOMINAL_COLUMN = "payment_format_code"
NOMINAL_VOCABULARY = 3
UNKNOWN_CODE = -1

FEATURE_COLUMNS: tuple[str, ...] = (*CONTINUOUS_COLUMNS, NOMINAL_COLUMN)
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    SENDER_COLUMN,
    RECEIVER_COLUMN,
)
COLUMN_ORDER: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)

SOURCE_DIMENSION = len(FEATURE_COLUMNS)
"""Model inputs with the nominal expansion off: the three source columns."""

EXPANDED_DIMENSION = len(CONTINUOUS_COLUMNS) + NOMINAL_VOCABULARY
"""Model inputs with the expansion on: 2 continuous plus a 3-wide indicator block."""

START = datetime(2020, 1, 1, 0, 0, 0)
"""Fixture epoch. 2020, not the real file's 2022, so no timestamp here can be
read as a real one."""

SPLIT_STARTS: dict[SplitName, datetime] = {
    SplitName.TRAIN: START,
    SplitName.VALIDATION: START + timedelta(days=30),
    SplitName.TEST: START + timedelta(days=60),
}
"""Non-overlapping, ascending split windows, as a chronological split produces."""

DEFAULT_SPLIT_ROWS: dict[SplitName, int] = {
    SplitName.TRAIN: 120,
    SplitName.VALIDATION: 60,
    SplitName.TEST: 60,
}


def make_rows(
    *,
    rows: int = 60,
    accounts: int = 6,
    positive_every: int = 7,
    start: datetime = START,
    minutes_apart: int = 5,
    codes: int = NOMINAL_VOCABULARY,
) -> list[dict[str, Any]]:
    """Generate deterministic synthetic processed rows in chronological order.

    Args:
        rows: How many rows to produce.
        accounts: Size of the sending-account pool. Rows are dealt round-robin,
            so every account accumulates a history and the per-account sequence
            index has something to link.
        positive_every: Every n-th row carries label 1. Zero produces none, which
            is what the degenerate-split tests need.
        start: Timestamp of the first row.
        minutes_apart: Gap between consecutive rows. Strictly positive, so the
            rows are already sorted and the order check passes.
        codes: Vocabulary size to spread the nominal codes over. Values at or
            above :data:`NOMINAL_VOCABULARY` exercise the out-of-vocabulary path.

    Returns:
        Rows keyed by processed column name, ascending in timestamp.
    """
    generated: list[dict[str, Any]] = []
    for index in range(rows):
        moment = start + timedelta(minutes=index * minutes_apart)
        sender = index % accounts
        receiver = (index * 3 + 1) % accounts
        generated.append(
            {
                TIMESTAMP_COLUMN: moment.strftime(TIMESTAMP_FORMAT),
                SENDER_COLUMN: f"TESTACCT{sender:04d}",
                RECEIVER_COLUMN: f"TESTACCT{receiver:04d}",
                "amount_log": round(1.0 + (index % 13) * 0.25, 4),
                "same_bank": index % 2,
                NOMINAL_COLUMN: index % codes,
                TARGET_COLUMN: (
                    1 if positive_every and index % positive_every == 0 else 0
                ),
            }
        )
    return generated


def feature_metadata_payload(**overrides: Any) -> dict[str, Any]:
    """The ``feature_metadata.json`` artefact, with the same keys as the real one.

    Args:
        **overrides: Keys to replace, so a test can remove or corrupt exactly one
            of them and assert on the resulting error.
    """
    payload: dict[str, Any] = {
        "notice": TEST_DATA_NOTICE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target_column": TARGET_COLUMN,
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "column_order": list(COLUMN_ORDER),
        "timestamp_format": TIMESTAMP_FORMAT,
        "raw_to_output_names": {
            "Timestamp": TIMESTAMP_COLUMN,
            "Account": SENDER_COLUMN,
            "Account.1": RECEIVER_COLUMN,
            f"{NOMINAL_RAW_NAME}_code": NOMINAL_COLUMN,
            "Is Laundering": TARGET_COLUMN,
        },
    }
    payload.update(overrides)
    return payload


def transformers_payload(**overrides: Any) -> dict[str, Any]:
    """The ``transformers.json`` artefact, carrying one ordinal-coded column."""
    payload: dict[str, Any] = {
        "notice": TEST_DATA_NOTICE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "categorical_encoding": {
            "scheme": "ordinal",
            "unknown_code": UNKNOWN_CODE,
            "codes_are_nominal": True,
            "columns": {
                NOMINAL_RAW_NAME: {
                    "vocabulary_size": NOMINAL_VOCABULARY,
                    "mapping": {"ACH": 0, "Cheque": 1, "Wire": 2},
                }
            },
        },
    }
    payload.update(overrides)
    return payload


def write_metadata(
    root: Path,
    *,
    features: Mapping[str, Any] | None = None,
    transformers: Mapping[str, Any] | None = None,
) -> Path:
    """Write both metadata artefacts under ``root/metadata/``.

    Args:
        root: Processed-root directory inside ``tmp_path``.
        features: Replacement ``feature_metadata.json`` payload.
        transformers: Replacement ``transformers.json`` payload.

    Returns:
        The metadata directory.
    """
    directory = root / METADATA_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / FEATURE_METADATA_FILE,
        dict(features) if features is not None else feature_metadata_payload(),
    )
    _write_json(
        directory / TRANSFORMERS_FILE,
        dict(transformers) if transformers is not None else transformers_payload(),
    )
    return directory


def write_split(
    root: Path,
    split: SplitName,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] = COLUMN_ORDER,
) -> Path:
    """Write one split's ``transactions.csv``.

    Args:
        root: Processed-root directory.
        split: Which split this is.
        rows: Rows to write, in the order they should appear in the file.
        columns: Header, so a test can write a file that disagrees with the
            recorded ``column_order``.

    Returns:
        The written path.
    """
    path = root / split.value / SPLIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([row.get(column, "") for column in columns] for row in rows)
    return path


def write_processed_tree(
    root: Path,
    *,
    splits: Mapping[SplitName, Sequence[Mapping[str, Any]]] | None = None,
    features: Mapping[str, Any] | None = None,
    transformers: Mapping[str, Any] | None = None,
    **row_options: Any,
) -> Path:
    """Write a complete synthetic processed tree: metadata plus three splits.

    Args:
        root: Directory to build under, normally a subdirectory of ``tmp_path``.
        splits: Explicit rows per split. Generated when omitted.
        features: Replacement ``feature_metadata.json`` payload.
        transformers: Replacement ``transformers.json`` payload.
        **row_options: Forwarded to :func:`make_rows` for the generated splits.

    Returns:
        ``root``, so a caller can pass it straight to a provider.
    """
    write_metadata(root, features=features, transformers=transformers)
    for split in ORDERED_SPLITS:
        if splits is not None:
            rows = splits.get(split)
            if rows is None:
                continue
        else:
            rows = make_rows(
                rows=DEFAULT_SPLIT_ROWS[split],
                start=SPLIT_STARTS[split],
                **row_options,
            )
        write_split(root, split, rows)
    return root


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialise one artefact the way the preprocessing writer does."""
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
