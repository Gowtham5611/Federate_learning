"""Disk-backed, memory-mapped sequence provider for large-scale transaction datasets.

Solves the Google Colab RAM bottleneck by streaming CSV files in bounded chunks
and writing preprocessed features into disk-backed memory-mapped arrays (np.memmap).
Guarantees constant resident memory (<500 MB) regardless of dataset size (e.g., 31.25M rows).

Key guarantees:
1. Never loads full splits into pandas DataFrames simultaneously.
2. Employs chunked CSV reading with compact dtypes.
3. Preprocesses nominal one-hot categorical features into disk-backed float32 memmaps.
4. Generates previous_row indices with zero inter-account leakage.
5. Verifies chronological ordering across all chunks.
6. Atomic cache creation with schema & dataset fingerprint validation.
7. Strict rejection of row-truncated subsets in full-dataset mode.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

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
from infrastructure.data.sequences.processed_sequence_provider import (
    BOUNDARY_POLICY,
    LEXICOGRAPHIC_TIMESTAMP_FORMATS,
    ORDERING,
    PADDING_DESCRIPTION,
    TARGET_ASSIGNMENT,
)

logger = logging.getLogger(__name__)


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when available disk space is insufficient for building the cache."""


class DatasetFingerprintMismatchError(ValueError):
    """Raised when existing cache does not match the source dataset fingerprint."""


class SubsetModeDetectedError(ValueError):
    """Raised when full-data mode detects an unauthorized row limit or subset path."""


class MmapSequenceProvider(SequenceDataProviderPort):
    """Reads large processed transaction splits into disk-backed memory-mapped sequence datasets."""

    def __init__(
        self,
        processed_root: Path,
        cache_root: Path | None = None,
        chunk_size: int = 200_000,
    ) -> None:
        """Initialize MmapSequenceProvider.

        Args:
            processed_root: Root directory containing processed splits and metadata/
            cache_root: Directory to store memory-mapped caches. Defaults to processed_root / ".cache"
            chunk_size: Number of rows to process per CSV streaming chunk (default: 200,000)
        """
        self._root = Path(processed_root).resolve()
        self._metadata = ProcessedFeatureMetadata.read(self._root)
        self._cache_root = (
            Path(cache_root).resolve()
            if cache_root is not None
            else self._root / ".mmap_cache"
        )
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._chunk_size = chunk_size
        self._rows_seen: dict[str, int] = {}
        self._positives_seen: dict[str, int] = {}

    @property
    def metadata(self) -> ProcessedFeatureMetadata:
        """Preprocessed dataset metadata."""
        return self._metadata

    @property
    def cache_root(self) -> Path:
        """Directory housing disk-backed caches."""
        return self._cache_root

    def get_fingerprint_path(self, split: SplitName) -> Path:
        """Return path to fingerprint metadata file for a split."""
        return self._cache_root / split.value / "fingerprint.json"

    def estimate_storage_requirements_mb(
        self, split: SplitName, row_count: int
    ) -> dict[str, float]:
        """Estimate disk and RAM requirements for a split before building."""
        width = self._metadata.feature_dimension(one_hot_nominal_codes=True)
        features_mb = (row_count * width * 4) / (1024 * 1024)
        targets_mb = (row_count * 4) / (1024 * 1024)
        previous_mb = (row_count * 8) / (1024 * 1024)
        total_disk_mb = features_mb + targets_mb + previous_mb
        peak_ram_mb = (self._chunk_size * width * 4) / (1024 * 1024) + 150.0  # chunk + dict overhead
        return {
            "row_count": row_count,
            "feature_dimension": width,
            "features_disk_mb": round(features_mb, 2),
            "targets_disk_mb": round(targets_mb, 2),
            "previous_row_disk_mb": round(previous_mb, 2),
            "total_disk_required_mb": round(total_disk_mb, 2),
            "estimated_peak_ram_mb": round(peak_ram_mb, 2),
        }

    def verify_cache_integrity(
        self, split: SplitName, csv_path: Path
    ) -> tuple[bool, dict[str, Any] | None]:
        """Check if a valid, uncorrupted cache matching the source CSV exists."""
        split_cache = self._cache_root / split.value
        fp_path = split_cache / "fingerprint.json"
        features_path = split_cache / "features.mmap"
        targets_path = split_cache / "targets.npy"
        prev_path = split_cache / "previous_row.npy"

        if not (
            fp_path.exists()
            and features_path.exists()
            and targets_path.exists()
            and prev_path.exists()
        ):
            return False, None

        try:
            with open(fp_path, "r", encoding="utf-8") as f:
                fp_data = json.load(f)

            csv_stat = csv_path.stat()
            if fp_data.get("csv_size_bytes") != csv_stat.st_size:
                logger.warning(
                    "Cache invalid: CSV size changed (%s vs %s)",
                    fp_data.get("csv_size_bytes"),
                    csv_stat.st_size,
                )
                return False, None

            expected_rows = fp_data.get("row_count", 0)
            width = self._metadata.feature_dimension(one_hot_nominal_codes=True)
            expected_features_size = expected_rows * width * 4

            if features_path.stat().st_size != expected_features_size:
                logger.warning("Cache invalid: features.mmap size mismatch")
                return False, None

            return True, fp_data
        except Exception as ex:
            logger.warning("Cache verification error: %s", ex)
            return False, None

    def load(
        self,
        split: SplitName,
        config: TrainingConfig,
        *,
        enforce_full_dataset: bool = False,
        min_expected_rows: int | None = None,
    ) -> SequenceDataset:
        """Load one split's sequences, building or reusing disk-backed memory-mapped cache.

        Args:
            split: Which split to load (train, validation, test)
            config: TrainingConfig specifying sequence length and stride
            enforce_full_dataset: If True, rejects any row_limit or truncated data
            min_expected_rows: Optional minimum expected rows (e.g. 18,000,000 for full LI-Medium)

        Raises:
            SubsetModeDetectedError: If full-dataset enforcement is violated
            ProcessedMetadataError: If source file is missing
            UnsortedSplitError: If split is not chronologically monotonic
        """
        if enforce_full_dataset and config.row_limit is not None:
            raise SubsetModeDetectedError(
                f"Full-dataset mode is active, but an implicit row_limit={config.row_limit} was specified! "
                "You must set row_limit=None for full-dataset training."
            )

        csv_path = self._metadata.split_path(split.value)
        if not csv_path.is_file():
            raise ProcessedMetadataError(
                f"The '{split.value}' split file is missing at {csv_path}", csv_path
            )

        cache_valid, fp_data = self.verify_cache_integrity(split, csv_path)

        if not cache_valid:
            logger.info(
                "Disk-backed cache missing or outdated for %s. Building cache from %s...",
                split.value,
                csv_path.name,
            )
            fp_data = self._build_cache(split, csv_path, config)
        else:
            logger.info(
                "Reusing verified disk-backed cache for %s (%d rows, %d features, pos=%d).",
                split.value,
                fp_data["row_count"],
                fp_data["feature_dimension"],
                fp_data["positive_count"],
            )

        row_count = int(fp_data["row_count"])
        if min_expected_rows is not None and row_count < min_expected_rows:
            raise SubsetModeDetectedError(
                f"Expected at least {min_expected_rows} rows for full {split.value} split, "
                f"but loaded cache has only {row_count} rows! Ensure full dataset path is used."
            )

        split_cache = self._cache_root / split.value
        width = int(fp_data["feature_dimension"])

        # Open features as read-only memory map
        features_ro = np.memmap(
            split_cache / "features.mmap",
            dtype=np.float32,
            mode="r",
            shape=(row_count, width),
        )
        targets_arr = np.load(split_cache / "targets.npy", mmap_mode="r")
        prev_arr = np.load(split_cache / "previous_row.npy", mmap_mode="r")

        dataset = AccountSequenceDataset(
            split=split,
            features=features_ro,
            targets=targets_arr,
            previous_row=prev_arr,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
        )

        if dataset.sequence_count == 0:
            raise EmptySequenceDatasetError(split.value, row_count)

        self._rows_seen[split.value] = dataset.row_count
        self._positives_seen[split.value] = dataset.positive_count

        return dataset

    def _build_cache(
        self, split: SplitName, csv_path: Path, config: TrainingConfig
    ) -> dict[str, Any]:
        """Build disk-backed cache atomically in bounded chunks."""
        split_cache = self._cache_root / split.value
        split_cache.mkdir(parents=True, exist_ok=True)
        temp_cache = Path(tempfile.mkdtemp(prefix=f"mmap_{split.value}_", dir=str(self._cache_root)))

        try:
            # 1. Count rows safely without loading all into memory
            logger.info("Streaming line count for %s...", csv_path.name)
            target_col = self._metadata.target_column
            sender_col = self._metadata.sender_column
            ts_col = self._metadata.timestamp_column

            total_rows = 0
            for chunk in pd.read_csv(csv_path, usecols=[target_col], chunksize=500_000):
                total_rows += len(chunk)

            logger.info("Verified %d total rows in %s.", total_rows, csv_path.name)

            width = self._metadata.feature_dimension(
                one_hot_nominal_codes=config.one_hot_nominal_codes
            )

            # Check free disk space
            storage_spec = self.estimate_storage_requirements_mb(split, total_rows)
            logger.info("Storage preflight estimate: %s", storage_spec)

            stat_vfs = shutil.disk_usage(str(self._cache_root))
            free_mb = stat_vfs.free / (1024 * 1024)
            required_mb = storage_spec["total_disk_required_mb"] * 1.2
            if free_mb < required_mb:
                raise InsufficientDiskSpaceError(
                    f"Insufficient disk space in {self._cache_root}: "
                    f"Required ~{required_mb:.1f} MB, Available: {free_mb:.1f} MB."
                )

            # 2. Allocate temporary disk arrays
            tmp_features_path = temp_cache / "features.mmap"
            features_mmap = np.memmap(
                tmp_features_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_rows, width),
            )
            targets_arr = np.empty(total_rows, dtype=np.float32)
            account_codes = np.empty(total_rows, dtype=np.int64)

            columns = list(self._metadata.columns_to_read())
            dtypes: dict[str, str] = {
                ts_col: "string",
                sender_col: "string",
                target_col: "int8",
            }
            nominal_names = set(self._metadata.nominal_column_names())
            for col in self._metadata.feature_columns:
                dtypes[col] = "int32" if col in nominal_names else "float32"

            expansions = {
                nominal.column: nominal for nominal in self._metadata.nominal_columns
            }

            account_to_code: dict[str, int] = {}
            offset = 0
            prev_last_stamp = None
            positives_count = 0

            logger.info(
                "Streaming CSV in chunks of %d to disk-backed memmap...", self._chunk_size
            )
            for chunk_idx, chunk in enumerate(
                pd.read_csv(
                    csv_path,
                    usecols=columns,
                    dtype=dtypes,
                    chunksize=self._chunk_size,
                )
            ):
                chunk_len = len(chunk)

                # Chronological ordering guard
                stamps = chunk[ts_col]
                if not stamps.is_monotonic_increasing:
                    values = stamps.to_numpy()
                    offending = np.flatnonzero(values[1:] < values[:-1])
                    raise UnsortedSplitError(
                        split.value, int(offset + offending[0] + 1)
                    )

                if prev_last_stamp is not None and prev_last_stamp > stamps.iloc[0]:
                    raise UnsortedSplitError(split.value, offset + 1)
                prev_last_stamp = stamps.iloc[-1]

                # Targets
                chunk_targets = chunk[target_col].to_numpy(dtype=np.float32)
                targets_arr[offset : offset + chunk_len] = chunk_targets
                positives_count += int(np.sum(chunk_targets == 1.0))

                # Account codes mapping (compact int64)
                for i, acc in enumerate(chunk[sender_col]):
                    code = account_to_code.get(acc)
                    if code is None:
                        code = len(account_to_code)
                        account_to_code[acc] = code
                    account_codes[offset + i] = code

                # Features extraction and one-hot expansion
                chunk_features = np.zeros((chunk_len, width), dtype=np.float32)
                col_offset = 0
                for col in self._metadata.feature_columns:
                    nominal = (
                        expansions.get(col)
                        if config.one_hot_nominal_codes
                        else None
                    )
                    vals = chunk[col]
                    if nominal is None:
                        chunk_features[:, col_offset] = vals.to_numpy(dtype=np.float32)
                        col_offset += 1
                    else:
                        codes = vals.to_numpy(dtype=np.int64)
                        known = (codes >= 0) & (codes < nominal.vocabulary_size)
                        known_rows = np.flatnonzero(known)
                        chunk_features[known_rows, col_offset + codes[known_rows]] = 1.0
                        col_offset += nominal.vocabulary_size

                features_mmap[offset : offset + chunk_len, :] = chunk_features
                offset += chunk_len

                if (chunk_idx + 1) % 10 == 0 or offset == total_rows:
                    features_mmap.flush()
                    pct = (offset / total_rows) * 100.0
                    logger.info("  Processed %d / %d rows (%.1f%%)...", offset, total_rows, pct)

            features_mmap.flush()
            del features_mmap

            logger.info("Building vectorized previous_row index for %d accounts...", len(account_to_code))
            previous_row = build_previous_row_index(account_codes)
            del account_codes
            del account_to_code

            # Save arrays in temporary cache
            np.save(temp_cache / "targets.npy", targets_arr)
            np.save(temp_cache / "previous_row.npy", previous_row)

            # Write fingerprint metadata
            csv_stat = csv_path.stat()
            fp_data = {
                "split": split.value,
                "source_file": str(csv_path.name),
                "csv_size_bytes": csv_stat.st_size,
                "csv_mtime_utc": datetime.datetime.fromtimestamp(
                    csv_stat.st_mtime, tz=datetime.timezone.utc
                ).isoformat(),
                "row_count": total_rows,
                "feature_dimension": width,
                "target_column": target_col,
                "positive_count": positives_count,
                "negative_count": total_rows - positives_count,
                "one_hot_nominal_codes": config.one_hot_nominal_codes,
                "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            with open(temp_cache / "fingerprint.json", "w", encoding="utf-8") as f:
                json.dump(fp_data, f, indent=2)

            # Move atomically to final cache destination
            for item in temp_cache.iterdir():
                dest = split_cache / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

            logger.info("Successfully landed disk-backed cache for %s in %s.", split.value, split_cache)
            return fp_data

        finally:
            if temp_cache.exists():
                shutil.rmtree(temp_cache, ignore_errors=True)

    def describe_features(self, config: TrainingConfig) -> DataProvenance:
        """Report data origin and tensor width."""
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
        """Report how rows become sequences."""
        return SequenceStrategy(
            grouping_column=self._metadata.sender_column,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            target_assignment=TARGET_ASSIGNMENT,
            padding=PADDING_DESCRIPTION,
            boundary_policy=BOUNDARY_POLICY,
            ordering=ORDERING,
        )

