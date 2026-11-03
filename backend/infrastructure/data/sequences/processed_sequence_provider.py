"""Builds sequence datasets from the Phase 3 processed CSV splits.

Reads one split at a time, keeps the columns the model is allowed to see, and
hands the rest to :mod:`account_sequence_dataset`. Four things happen here that
are worth naming, because each is a place a temporal model can go quietly wrong:

**Identifier columns are dropped, not merely unused.** ``timestamp`` and
``from_account`` are read — order has to be verified and rows have to be grouped —
and then consumed into an integer index and a sortedness check. Neither reaches the
feature matrix. ``to_account`` is not read at all.

**Order is verified rather than trusted.** Sequence construction takes
chronological order from row position, so a split that is not sorted would build
sequences with a later transaction sitting before an earlier one. That is checked
on every load, and refused.

**Category codes are expanded.** ``transformers.json`` records the three
categorical columns as ordinal codes carrying no ordering, and records that the
expansion "is deferred to the modelling phase". This is that phase. The expansion
uses the recorded vocabulary sizes, not the values present in the split, so every
split produces the same feature width.

**Every architecture gets the same tensor.** Nothing here depends on which
architecture will consume the data. The GRU, LSTM and BiLSTM baselines are handed
byte-identical inputs from the same configuration, which is what makes comparing
them mean anything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from application.dto.experiment import DataProvenance, SequenceStrategy
from application.dto.training import TrainingConfig
from application.ports.output.sequence_data import (
    SequenceDataProviderPort,
    SequenceDataset,
)
from domain.enums.split_name import SplitName
from domain.exceptions.training import (
    EmptySequenceDatasetError,
    ProcessedMetadataError,
    UnsortedSplitError,
)
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
    build_previous_row_index,
)
from infrastructure.data.sequences.processed_feature_metadata import (
    ProcessedFeatureMetadata,
)

logger = logging.getLogger(__name__)

LEXICOGRAPHIC_TIMESTAMP_FORMATS: frozenset[str] = frozenset(
    {"%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"}
)
"""Timestamp formats whose text order is also their chronological order.

Zero-padded, most-significant-first and fixed-width, so the order check can
compare the strings and skip parsing three million datetimes. Any other recorded
format falls back to real parsing rather than to an assumption.
"""

TARGET_ASSIGNMENT = (
    "the label of the last (newest) transaction in the window, which is the "
    "transaction being classified"
)
PADDING_DESCRIPTION = (
    "zero-padded on the right for accounts with fewer than sequence_length prior "
    "transactions in the split; true lengths are supplied per sequence so the "
    "padding is masked rather than read as data"
)
BOUNDARY_POLICY = (
    "sequences are built from one split's rows only, so no window spans the "
    "train/validation or validation/test boundary"
)
ORDERING = (
    "row order in the processed file, which the chronological splitter wrote in "
    "ascending timestamp order and which is re-verified on every load"
)


class ProcessedSequenceProvider(SequenceDataProviderPort):
    """Reads processed splits and turns them into per-account sequences."""

    def __init__(self, processed_root: Path) -> None:
        """Read the preprocessing metadata that describes the splits.

        Args:
            processed_root: Directory holding ``<split>/transactions.csv`` and
                ``metadata/``.

        Raises:
            ProcessedMetadataError: The metadata artefacts are missing or unusable.
        """
        self._root = processed_root.resolve()
        self._metadata = ProcessedFeatureMetadata.read(self._root)
        self._rows_seen: dict[str, int] = {}
        self._positives_seen: dict[str, int] = {}

    @property
    def metadata(self) -> ProcessedFeatureMetadata:
        """The recorded description of the processed splits."""
        return self._metadata

    # --- Loading -------------------------------------------------------------

    def load(self, split: SplitName, config: TrainingConfig) -> SequenceDataset:
        """Read one split and build its sequences.

        Raises:
            ProcessedMetadataError: The split's file is missing.
            UnsortedSplitError: The split is not in chronological order.
            EmptySequenceDatasetError: No sequences could be built.
        """
        path = self._metadata.split_path(split.value)
        if not path.is_file():
            raise ProcessedMetadataError(
                f"the '{split.value}' split file is missing", path
            )

        frame = self._read(path, config)
        rows = len(frame)
        logger.info("Read %d row(s) from %s", rows, path)

        self._verify_order(frame, split)
        previous_row = self._link_accounts(frame)
        targets = frame.pop(self._metadata.target_column).to_numpy(dtype=np.float32)
        features = self._build_feature_matrix(frame, config, split)
        del frame

        dataset = AccountSequenceDataset(
            split=split,
            features=features,
            targets=targets,
            previous_row=previous_row,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
        )
        if dataset.sequence_count == 0:
            raise EmptySequenceDatasetError(split.value, rows)

        self._rows_seen[split.value] = dataset.row_count
        self._positives_seen[split.value] = dataset.positive_count
        return dataset

    def _read(self, path: Path, config: TrainingConfig) -> pd.DataFrame:
        """Read the columns Phase 4 is allowed to use, and no others."""
        columns = self._metadata.columns_to_read()
        frame = pd.read_csv(
            path,
            usecols=list(columns),
            dtype=self._read_dtypes(),
            nrows=config.row_limit,
        )
        # usecols does not fix column order; the metadata does.
        return frame[list(columns)]

    def _read_dtypes(self) -> dict[str, str]:
        """Narrow dtypes, chosen to keep a three-million-row split affordable.

        float32 rather than float64 halves the numeric footprint and is the
        precision the model trains in anyway. The category codes stay integral
        because they are about to be used as indices.
        """
        dtypes: dict[str, str] = {
            self._metadata.timestamp_column: "string",
            self._metadata.sender_column: "string",
            self._metadata.target_column: "int8",
        }
        nominal = set(self._metadata.nominal_column_names())
        for column in self._metadata.feature_columns:
            dtypes[column] = "int32" if column in nominal else "float32"
        return dtypes

    # --- Guards --------------------------------------------------------------

    def _verify_order(self, frame: pd.DataFrame, split: SplitName) -> None:
        """Refuse a split whose rows are not in chronological order."""
        stamps = frame.pop(self._metadata.timestamp_column)
        if self._metadata.timestamp_format not in LEXICOGRAPHIC_TIMESTAMP_FORMATS:
            stamps = pd.to_datetime(
                stamps, format=self._metadata.timestamp_format, errors="raise"
            )
        if stamps.is_monotonic_increasing:
            return

        values = stamps.to_numpy()
        offending = np.flatnonzero(values[1:] < values[:-1])
        raise UnsortedSplitError(split.value, int(offending[0]) + 1)

    def _link_accounts(self, frame: pd.DataFrame) -> NDArray[np.int64]:
        """Turn the sending-account column into a previous-row index.

        The account identifiers themselves go no further than this method. What
        survives is the *shape* of each account's history, which is what a
        sequence needs; the identity, which the spec forbids as a model input,
        does not.
        """
        accounts = frame.pop(self._metadata.sender_column)
        codes, _ = pd.factorize(accounts, sort=False)
        del accounts
        return build_previous_row_index(codes.astype(np.int64))

    # --- Feature matrix ------------------------------------------------------

    def _build_feature_matrix(
        self, frame: pd.DataFrame, config: TrainingConfig, split: SplitName
    ) -> NDArray[np.float32]:
        """Assemble ``(rows, feature_dimension)`` in the recorded column order."""
        rows = len(frame)
        width = self._metadata.feature_dimension(
            one_hot_nominal_codes=config.one_hot_nominal_codes
        )
        features = np.zeros((rows, width), dtype=np.float32)
        expansions = {
            nominal.column: nominal for nominal in self._metadata.nominal_columns
        }

        offset = 0
        for column in self._metadata.feature_columns:
            nominal = expansions.get(column) if config.one_hot_nominal_codes else None
            values = frame.pop(column)
            if nominal is None:
                features[:, offset] = values.to_numpy(dtype=np.float32)
                offset += 1
                continue

            codes = values.to_numpy(dtype=np.int64)
            known = (codes >= 0) & (codes < nominal.vocabulary_size)
            known_rows = np.flatnonzero(known)
            features[known_rows, offset + codes[known_rows]] = 1.0
            unknown = rows - int(known_rows.shape[0])
            if unknown:
                # An out-of-vocabulary code becomes an all-zero block: "none of
                # the known categories". Reported because a large count would mean
                # the split disagrees with the vocabulary it was encoded against.
                logger.info(
                    "Split '%s' column '%s': %d row(s) held a code outside the "
                    "recorded vocabulary of %d and were expanded to zeros",
                    split.value,
                    column,
                    unknown,
                    nominal.vocabulary_size,
                )
            offset += nominal.vocabulary_size

        if offset != width:
            raise ProcessedMetadataError(
                f"the expanded feature axis came out {offset} column(s) wide but "
                f"the metadata describes {width}",
                self._metadata.root,
            )
        return features

    # --- Description ---------------------------------------------------------

    def describe_features(self, config: TrainingConfig) -> DataProvenance:
        """Report the data's origin and the tensor width it produces.

        Row and positive counts cover the splits read so far, so calling this
        before loading yields the columns and the width, and calling it after
        yields the measured counts too. Nothing is filled in from the Phase 3
        artefacts: the counts reported are the ones this run actually read, which
        is what makes them a check on the run rather than a restatement.
        """
        return DataProvenance(
            processed_directory=str(self._root),
            preprocessing_version=self._metadata.preprocessing_version,
            feature_columns=self._metadata.feature_columns,
            identifier_columns=self._metadata.identifier_columns,
            target_column=self._metadata.target_column,
            feature_dimension=self._metadata.feature_dimension(
                one_hot_nominal_codes=config.one_hot_nominal_codes
            ),
            rows_per_split=dict(self._rows_seen),
            positives_per_split=dict(self._positives_seen),
        )

    def describe_strategy(self, config: TrainingConfig) -> SequenceStrategy:
        """Report how rows become sequences, in the experiment record's own words."""
        return SequenceStrategy(
            grouping_column=self._metadata.sender_column,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            target_assignment=TARGET_ASSIGNMENT,
            padding=PADDING_DESCRIPTION,
            boundary_policy=BOUNDARY_POLICY,
            ordering=ORDERING,
        )
