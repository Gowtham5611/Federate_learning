"""Reads the Phase 3 metadata artefacts that describe the processed splits.

Phase 4 does not restate which columns are features, which are identifiers or
which is the target. It reads them from the JSON artefacts the preprocessing
phase wrote next to the data. That costs a file read and a few key lookups, and
it buys the one property that matters here: the model cannot be trained on a
feature set that disagrees with the data on disk, because there is only one
description of that feature set and both phases use it.

The same artefacts also answer a question the preprocessing phase deliberately
left open. ``transformers.json`` records the three categorical columns as
*ordinal codes* and notes that expanding them "is deferred to the modelling
phase". This module supplies the vocabulary sizes needed for that expansion, from
the recorded vocabularies rather than from values observed in the data — so the
training and test splits produce identically shaped tensors even if some currency
never appears in one of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.exceptions.training import ProcessedMetadataError

METADATA_DIRECTORY = "metadata"
"""Subdirectory of the processed root holding the JSON artefacts."""

FEATURE_METADATA_FILE = "feature_metadata.json"
TRANSFORMERS_FILE = "transformers.json"
SPLIT_FILENAME = "transactions.csv"
"""Each split's single CSV, named as the processed-data writer names it."""

_RAW_TIMESTAMP = "Timestamp"
_RAW_SENDER_ACCOUNT = "Account"
"""Raw column names used only to look up their processed equivalents.

The raw dataset's sending-account column is ``Account`` and its receiving-account
column is ``Account.1``; the processed names are read from the recorded
``raw_to_output_names`` map rather than assumed to be ``from_account`` and
``to_account``.
"""


@dataclass(frozen=True, slots=True)
class NominalColumn:
    """One ordinal-coded categorical column and the size of its vocabulary.

    Attributes:
        column: Processed column name, e.g. ``payment_format_code``.
        vocabulary_size: Distinct categories the preprocessing phase recorded.
            Codes therefore run ``0 .. vocabulary_size - 1``.
        unknown_code: Code standing for a category absent from the fitted
            vocabulary. Expanded to an all-zero indicator block, which is the
            honest encoding: "none of the known categories" rather than a
            category of its own.
    """

    column: str
    vocabulary_size: int
    unknown_code: int

    def indicator_names(self) -> tuple[str, ...]:
        """Names of the indicator columns this one expands into."""
        stem = self.column.removesuffix("_code")
        return tuple(f"{stem}_is_{index}" for index in range(self.vocabulary_size))


@dataclass(frozen=True, slots=True)
class ProcessedFeatureMetadata:
    """What the processed splits contain, as recorded by the phase that wrote them.

    Attributes:
        root: Directory holding ``<split>/transactions.csv`` and ``metadata/``.
        preprocessing_version: Version stamped on the artefacts.
        target_column: Label column.
        identifier_columns: Traceability columns. Never model inputs.
        feature_columns: Source feature columns, in recorded order.
        column_order: Every column in the CSV, in file order.
        timestamp_format: Format the timestamp column was written in.
        timestamp_column: Processed name of the timestamp column.
        sender_column: Processed name of the sending-account column, which is what
            groups rows into one entity's history.
        nominal_columns: The ordinal-coded categorical columns.
    """

    root: Path
    preprocessing_version: str
    target_column: str
    identifier_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    column_order: tuple[str, ...]
    timestamp_format: str
    timestamp_column: str
    sender_column: str
    nominal_columns: tuple[NominalColumn, ...]

    # --- Construction --------------------------------------------------------

    @classmethod
    def read(cls, root: Path) -> ProcessedFeatureMetadata:
        """Load and cross-check the artefacts under ``root``.

        Args:
            root: Processed output directory for one dataset.

        Returns:
            The recorded description of the processed splits.

        Raises:
            ProcessedMetadataError: An artefact is missing, unparseable, or
                internally inconsistent.
        """
        features_path = root / METADATA_DIRECTORY / FEATURE_METADATA_FILE
        transformers_path = root / METADATA_DIRECTORY / TRANSFORMERS_FILE
        features = _read_json(features_path)
        transformers = _read_json(transformers_path)

        target = _require_str(features, "target_column", features_path)
        identifiers = _require_str_tuple(features, "identifier_columns", features_path)
        feature_columns = _require_str_tuple(features, "feature_columns", features_path)
        column_order = _require_str_tuple(features, "column_order", features_path)
        names = _require_mapping(features, "raw_to_output_names", features_path)

        overlap = set(identifiers) & set(feature_columns)
        if overlap:
            raise ProcessedMetadataError(
                f"columns {sorted(overlap)} are listed as both identifiers and "
                f"model features, so it is not recorded whether they may be fed "
                f"to a model",
                features_path,
            )
        if target in feature_columns:
            raise ProcessedMetadataError(
                f"the target column '{target}' also appears in feature_columns, "
                f"which would leak the label directly into the model input",
                features_path,
            )
        missing = [
            column
            for column in (target, *identifiers, *feature_columns)
            if column not in column_order
        ]
        if missing:
            raise ProcessedMetadataError(
                f"columns {missing} are described but absent from column_order, so "
                f"they cannot be read from the split files",
                features_path,
            )

        return cls(
            root=root,
            preprocessing_version=_require_str(
                features, "preprocessing_version", features_path
            ),
            target_column=target,
            identifier_columns=identifiers,
            feature_columns=feature_columns,
            column_order=column_order,
            timestamp_format=_require_str(features, "timestamp_format", features_path),
            timestamp_column=_processed_name(names, _RAW_TIMESTAMP, features_path),
            sender_column=_processed_name(names, _RAW_SENDER_ACCOUNT, features_path),
            nominal_columns=_read_nominal_columns(
                transformers, names, transformers_path, feature_columns
            ),
        )

    # --- Layout --------------------------------------------------------------

    def split_path(self, split: str) -> Path:
        """Where one split's rows live."""
        return self.root / split / SPLIT_FILENAME

    def nominal_column_names(self) -> tuple[str, ...]:
        """Processed names of the ordinal-coded categorical columns."""
        return tuple(nominal.column for nominal in self.nominal_columns)

    def continuous_columns(self) -> tuple[str, ...]:
        """Feature columns that are not ordinal category codes.

        These pass through unchanged whether or not the expansion is enabled, in
        the order Phase 3 recorded, so the feature axis is stable across runs.
        """
        nominal = set(self.nominal_column_names())
        return tuple(column for column in self.feature_columns if column not in nominal)

    def model_columns(self, *, one_hot_nominal_codes: bool) -> tuple[str, ...]:
        """Names of the columns on the model's feature axis, in order.

        With the expansion off, this is :attr:`feature_columns` unchanged — the
        code columns are fed as integers, which asserts an ordering that
        ``transformers.json`` explicitly says they do not have. With it on, each
        code column is replaced in place by its indicator block.
        """
        if not one_hot_nominal_codes:
            return self.feature_columns

        expansions = {
            nominal.column: nominal.indicator_names()
            for nominal in self.nominal_columns
        }
        columns: list[str] = []
        for column in self.feature_columns:
            columns.extend(expansions.get(column, (column,)))
        return tuple(columns)

    def feature_dimension(self, *, one_hot_nominal_codes: bool) -> int:
        """Width of the model's feature axis."""
        return len(self.model_columns(one_hot_nominal_codes=one_hot_nominal_codes))

    def columns_to_read(self) -> tuple[str, ...]:
        """Columns a split reader needs, and no others.

        The timestamp and the sending account are included because order has to be
        verified and rows have to be grouped into per-account histories. The
        receiving account is not: nothing in Phase 4 needs it, and reading a column
        that must never reach the model is a cost with no benefit.
        """
        needed = [
            self.timestamp_column,
            self.sender_column,
            *self.feature_columns,
            self.target_column,
        ]
        seen: dict[str, None] = {}
        for column in needed:
            seen.setdefault(column, None)
        return tuple(seen)


# --- Artefact reading helpers ------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Load one artefact, failing with the path rather than a bare OSError."""
    if not path.is_file():
        raise ProcessedMetadataError("the artefact does not exist", path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProcessedMetadataError(f"it could not be parsed ({error})", path) from error
    if not isinstance(payload, dict):
        raise ProcessedMetadataError("the top level is not a JSON object", path)
    return payload


def _require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    """Read a required string key."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ProcessedMetadataError(f"'{key}' is missing or not a string", path)
    return value


def _require_str_tuple(payload: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """Read a required non-empty list of strings."""
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ProcessedMetadataError(f"'{key}' is missing or not a non-empty list", path)
    if not all(isinstance(item, str) for item in value):
        raise ProcessedMetadataError(f"'{key}' contains a non-string entry", path)
    return tuple(str(item) for item in value)


def _require_mapping(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    """Read a required object key."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProcessedMetadataError(f"'{key}' is missing or not an object", path)
    return value


def _processed_name(names: dict[str, Any], raw: str, path: Path) -> str:
    """Translate a raw column name into its processed name."""
    value = names.get(raw)
    if not isinstance(value, str) or not value:
        raise ProcessedMetadataError(
            f"raw_to_output_names has no processed name for '{raw}'", path
        )
    return value


def _read_nominal_columns(
    transformers: dict[str, Any],
    names: dict[str, Any],
    path: Path,
    feature_columns: tuple[str, ...],
) -> tuple[NominalColumn, ...]:
    """Read the categorical vocabularies, skipping columns that are not features.

    Returns an empty tuple when the encoding block records a non-ordinal scheme:
    there is then nothing to expand, and inventing an expansion would contradict
    what the artefact says was done.
    """
    encoding = _require_mapping(transformers, "categorical_encoding", path)
    if encoding.get("scheme") != "ordinal":
        return ()

    unknown = encoding.get("unknown_code")
    if not isinstance(unknown, int):
        raise ProcessedMetadataError(
            "categorical_encoding.unknown_code is missing or not an integer", path
        )

    columns = _require_mapping(encoding, "columns", path)
    found: list[NominalColumn] = []
    for raw_name, block in sorted(columns.items()):
        if not isinstance(block, dict):
            raise ProcessedMetadataError(
                f"categorical_encoding.columns['{raw_name}'] is not an object", path
            )
        size = block.get("vocabulary_size")
        if not isinstance(size, int) or size <= 0:
            raise ProcessedMetadataError(
                f"categorical_encoding.columns['{raw_name}'].vocabulary_size is "
                f"missing or not a positive integer",
                path,
            )
        processed = _processed_name(names, f"{raw_name}_code", path)
        if processed not in feature_columns:
            continue
        found.append(
            NominalColumn(column=processed, vocabulary_size=size, unknown_code=unknown)
        )

    # Recorded in feature order rather than the artefact's alphabetical order, so
    # the expanded feature axis follows the same layout as the source columns.
    order = {column: index for index, column in enumerate(feature_columns)}
    return tuple(sorted(found, key=lambda nominal: order[nominal.column]))
