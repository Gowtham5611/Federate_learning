"""Builds temporal sequence datasets for partitioned client data.

Reuses the Phase 4 sequence methodology: account linking, chronological ordering,
one-hot categorical encoding from metadata, and padding handling.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from application.dto.training import TrainingConfig
from application.ports.output.sequence_data import SequenceDataset
from domain.enums.split_name import SplitName
from domain.exceptions.training import EmptySequenceDatasetError, ProcessedMetadataError, UnsortedSplitError
from infrastructure.data.sequences.account_sequence_dataset import (
    AccountSequenceDataset,
    build_previous_row_index,
)
from infrastructure.data.sequences.processed_feature_metadata import (
    ProcessedFeatureMetadata,
)
from infrastructure.data.sequences.processed_sequence_provider import (
    LEXICOGRAPHIC_TIMESTAMP_FORMATS,
)

logger = logging.getLogger(__name__)


class ClientSequenceProvider:
    """Builds an AccountSequenceDataset from an in-memory client DataFrame."""

    def __init__(self, metadata: ProcessedFeatureMetadata) -> None:
        """Initialize provider with metadata.

        Args:
            metadata: Preprocessing metadata containing schema and column definitions.
        """
        self._metadata = metadata

    @property
    def metadata(self) -> ProcessedFeatureMetadata:
        """Preprocessing metadata."""
        return self._metadata

    def build_client_dataset(
        self,
        frame: pd.DataFrame,
        config: TrainingConfig,
        *,
        client_id: str,
        split: SplitName = SplitName.TRAIN,
    ) -> SequenceDataset:
        """Transform a client's partition DataFrame into an AccountSequenceDataset.

        Args:
            frame: Client DataFrame with identical columns to the training split.
            config: Training configuration specifying sequence length and stride.
            client_id: Identifier of the client (for logging).
            split: Split name (defaults to SplitName.TRAIN).

        Returns:
            AccountSequenceDataset conforming to the standard SequenceDataset interface.

        Raises:
            UnsortedSplitError: If the client rows are not chronologically sorted.
            EmptySequenceDatasetError: If sequence generation produces zero windows.
        """
        working_frame = frame.copy()
        rows = len(working_frame)

        self._verify_order(working_frame, split)
        previous_row = self._link_accounts(working_frame)
        targets = working_frame.pop(self._metadata.target_column).to_numpy(
            dtype=np.float32
        )
        features = self._build_feature_matrix(working_frame, config, split)
        del working_frame

        dataset = AccountSequenceDataset(
            split=split,
            features=features,
            targets=targets,
            previous_row=previous_row,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
        )

        if dataset.sequence_count == 0:
            raise EmptySequenceDatasetError(f"{split.value}:{client_id}", rows)

        logger.info(
            "Built client %s sequence dataset: %d rows, %d sequences, %d positives",
            client_id,
            dataset.row_count,
            dataset.sequence_count,
            dataset.positive_count,
        )
        return dataset

    def _verify_order(self, frame: pd.DataFrame, split: SplitName) -> None:
        """Ensure rows are monotonic in chronological order."""
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
        """Convert sender account column into previous row indices."""
        accounts = frame.pop(self._metadata.sender_column)
        codes, _ = pd.factorize(accounts, sort=False)
        del accounts
        return build_previous_row_index(codes.astype(np.int64))

    def _build_feature_matrix(
        self, frame: pd.DataFrame, config: TrainingConfig, split: SplitName
    ) -> NDArray[np.float32]:
        """Assemble (rows, feature_dimension) array matching metadata."""
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
            offset += nominal.vocabulary_size

        if offset != width:
            raise ProcessedMetadataError(
                f"the expanded feature axis came out {offset} column(s) wide but "
                f"the metadata describes {width}",
                self._metadata.root,
            )
        return features

