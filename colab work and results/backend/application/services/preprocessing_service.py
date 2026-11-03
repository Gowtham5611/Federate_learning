"""Preprocessing: the passes that turn the raw dataset into three usable splits.

The pipeline is three passes over the data, and each one exists because a
specific fact about this dataset makes a single pass impossible.

**Pass 1 — partition.** Stream the raw file in chunks, clean each chunk, and
append its rows to a file named after their calendar day. The raw file is *not*
in timestamp order, so this is what makes chronological processing possible at
all. The same pass accumulates the timestamp histogram, which is what the split
boundaries are computed from without sorting anything.

**Pass 2 — engineer.** Read the day partitions in date order. Each day is sorted
in memory (a day is a few hundred thousand rows), the features are built, and
the rows are assigned to splits. Training rows feed the encoder and the scaler
and are staged to disk; the moment the first non-training row appears, fitting
is frozen and every later row is transformed and written straight to its output.

**Pass 3 — finalise training.** Stream the staged training rows back and apply
the now-complete transformations. Only the training split needs this pass, and
only because its own rows are what the statistics were computed from.

The result is exact rather than approximate: the features and the split are
identical to what an in-memory pipeline with unlimited RAM would produce.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pandas import DataFrame

from application.dto.preprocessing import (
    CleaningReport,
    DatasetFingerprint,
    PerformanceReport,
    PreprocessingReport,
    RemovalRecord,
    SplitReport,
    SplitSummary,
    TargetReport,
)
from application.ports.output.data_loader import ChunkedDataLoaderPort
from application.ports.output.interim_storage import InterimStoragePort
from application.ports.output.processed_data_writer import ProcessedDataWriterPort
from application.ports.output.transaction_data_source import TransactionDataSourcePort
from domain.enums.split_name import ORDERED_SPLITS, SplitName
from domain.exceptions.preprocessing import (
    ChronologyViolationError,
    RawDatasetModifiedError,
)
from domain.value_objects.split_boundaries import SplitBoundaries
from domain.value_objects.table_schema import TableSchema
from infrastructure.data.preprocessors.aml_preprocessor import (
    POSITIVE_LABEL,
    PREPROCESSING_VERSION,
    RAW_DTYPES,
    TARGET_OUTPUT_COLUMN,
    AmlPreprocessor,
    count_duplicate_rows,
    removal_descriptions,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_IS_LAUNDERING,
    COL_TIMESTAMP,
    RAW_COLUMN_ORDER,
    TARGET_COLUMN,
)
from infrastructure.data.splitters.chronological_splitter import (
    ChronologicalSplitter,
    TimestampHistogram,
)

logger = logging.getLogger(__name__)

_HASH_BLOCK_BYTES = 8 * 1024 * 1024
"""Bytes per read when digesting the raw file. Bounded, like everything else."""


class _SplitAccumulator:
    """Running counts for one split, gathered as its rows go past."""

    def __init__(self, name: SplitName) -> None:
        """Start empty."""
        self.name = name
        self.rows = 0
        self.first_timestamp = ""
        self.last_timestamp = ""
        self.class_counts: Counter[int] = Counter()

    def observe(self, frame: DataFrame) -> None:
        """Fold one frame of this split's rows into the counts."""
        if frame.empty:
            return
        self.rows += len(frame)

        stamps = frame[COL_TIMESTAMP].astype(str)
        earliest, latest = str(stamps.min()), str(stamps.max())
        self.first_timestamp = (
            earliest if not self.first_timestamp else min(self.first_timestamp, earliest)
        )
        self.last_timestamp = max(self.last_timestamp, latest)

        labels = frame[COL_IS_LAUNDERING].astype("int64").value_counts()
        self.class_counts.update(
            {int(str(label)): int(count) for label, count in labels.items()}
        )

    def summarise(self) -> SplitSummary:
        """Freeze the counts into a reportable summary."""
        return SplitSummary(
            name=self.name,
            row_count=self.rows,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            positive_count=int(self.class_counts.get(POSITIVE_LABEL, 0)),
            negative_count=int(
                sum(count for label, count in self.class_counts.items() if label != POSITIVE_LABEL)
            ),
        )


class PreprocessingService:
    """Runs the three preprocessing passes and reports exactly what they did."""

    def __init__(
        self,
        source: TransactionDataSourcePort,
        loader: ChunkedDataLoaderPort,
        interim: InterimStoragePort,
        writer: ProcessedDataWriterPort,
        preprocessor: AmlPreprocessor,
        splitter: ChronologicalSplitter,
        schema: TableSchema,
        *,
        chunk_size: int,
        keep_interim: bool = False,
    ) -> None:
        """Wire the pipeline.

        Args:
            source: Locates the raw dataset. Never downloads it.
            loader: Streams files in bounded chunks.
            interim: Working storage for the partition and staging passes.
            writer: Destination for the processed splits and metadata.
            preprocessor: Per-chunk cleaning, features and fitted state.
            splitter: Computes and applies the chronological boundaries.
            schema: The verified raw schema, recorded in the artefacts.
            chunk_size: Rows per chunk when streaming.
            keep_interim: Leave the working files in place after the run.
                Off by default: they are a working copy of the dataset and
                keeping them silently would double its footprint on disk.
        """
        self._source = source
        self._loader = loader
        self._interim = interim
        self._writer = writer
        self._preprocessor = preprocessor
        self._splitter = splitter
        self._schema = schema
        self._chunk_size = chunk_size
        self._keep_interim = keep_interim

        self._removed: Counter[str] = Counter()
        self._duplicates = 0
        self._rows_read = 0
        self._timings: dict[str, float] = {}
        self._accumulators = {name: _SplitAccumulator(name) for name in ORDERED_SPLITS}

    def run(self) -> PreprocessingReport:
        """Execute the whole pipeline and return what it established.

        Returns:
            The measured report. Every number in it was counted during this run.

        Raises:
            DataFileNotFoundError: The raw dataset is not present locally.
            NotEnoughDataToSplitError: The data cannot be split chronologically.
            ChronologyViolationError: The produced splits overlap in time.
            RawDatasetModifiedError: The raw file changed during the run.
        """
        raw_path = self._source.resolve_file()
        logger.info(
            "Preprocessing starting: file=%s chunk_size=%d version=%s",
            raw_path.name,
            self._chunk_size,
            PREPROCESSING_VERSION,
        )

        with self._stage("fingerprint"):
            digest_before = _sha256(raw_path)
            size_bytes = raw_path.stat().st_size

        with self._stage("partition"):
            histogram = self._partition_pass(raw_path)

        with self._stage("boundaries"):
            boundaries = self._splitter.compute_boundaries(histogram)
            logger.info(
                "Chronological boundaries: train<=%s, validation<=%s (%s)",
                boundaries.train_last,
                boundaries.validation_last,
                boundaries.tie_policy,
            )

        try:
            with self._stage("engineer"):
                self._engineer_pass(boundaries)
            with self._stage("finalize_train"):
                self._finalize_train_pass()
        finally:
            self._writer.close()

        with self._stage("verify"):
            digest_after = _sha256(raw_path)
            if digest_before != digest_after:
                raise RawDatasetModifiedError(raw_path, digest_before, digest_after)

        report = self._build_report(
            raw_path,
            DatasetFingerprint(
                file_name=raw_path.name,
                size_bytes=size_bytes,
                sha256=digest_before,
                verified_unchanged=True,
            ),
            boundaries,
        )
        self._write_artefacts(report, boundaries)

        if not self._keep_interim:
            self._interim.cleanup()

        logger.info(
            "Preprocessing finished: %d row(s) read, %d written, %.1fs",
            report.cleaning.rows_read,
            report.cleaning.rows_written,
            sum(self._timings.values()),
        )
        return report

    # --- Pass 1 --------------------------------------------------------------

    def _partition_pass(self, raw_path: Path) -> TimestampHistogram:
        """Clean each chunk, bucket it by day, and build the timestamp histogram."""
        histogram = TimestampHistogram()
        self._interim.begin_partitions(RAW_COLUMN_ORDER)

        for chunk in self._loader.stream_chunks(
            raw_path, self._chunk_size, dtypes=RAW_DTYPES
        ):
            self._rows_read += len(chunk)
            cleaned, removed = self._preprocessor.clean(chunk)
            self._removed.update(removed)
            if cleaned.empty:
                continue

            histogram.update(cleaned[COL_TIMESTAMP])
            partitioned = cleaned[list(RAW_COLUMN_ORDER)]
            for day, rows in partitioned.groupby(
                self._preprocessor.day_keys(cleaned), sort=False
            ):
                self._interim.add_to_partition(str(day), rows)

        self._interim.end_partitions()
        logger.info(
            "Partitioned %d row(s) into %d day(s); %d distinct timestamp(s)",
            histogram.total_rows,
            len(self._interim.partition_keys()),
            histogram.distinct_timestamps,
        )
        return histogram

    # --- Pass 2 --------------------------------------------------------------

    def _engineer_pass(self, boundaries: SplitBoundaries) -> None:
        """Build features day by day, fitting on training rows and writing the rest."""
        for day in self._interim.partition_keys():
            partition = self._interim.read_partition(day)
            if partition.empty:
                self._interim.drop_partition(day)
                continue

            self._duplicates += count_duplicate_rows(partition)

            prepared = self._preprocessor.prepare_partition(partition)
            ordered = self._preprocessor.sort_chronologically(prepared)
            engineered = self._preprocessor.engineer(ordered)
            labels = self._splitter.assign(engineered[COL_TIMESTAMP], boundaries)

            for split in ORDERED_SPLITS:
                rows = engineered[labels == split]
                if rows.empty:
                    continue
                self._handle_split_rows(split, rows)

            self._interim.drop_partition(day)

        # A dataset with no rows outside training never triggers the lazy freeze
        # inside the loop; closing it here makes the final pass well-defined.
        self._preprocessor.freeze_fit()
        self._interim.end_staging()

    def _handle_split_rows(self, split: SplitName, rows: DataFrame) -> None:
        """Fit-and-stage training rows; transform-and-write everything else."""
        carried = rows[list(self._preprocessor.engineered_columns)]
        self._accumulators[split].observe(carried)

        if split.is_fitting_split:
            self._preprocessor.fit_on_training_chunk(carried)
            self._interim.stage(carried)
            return

        # The first non-training row is the point at which every training row
        # has been seen, so this is exactly when fitting must close.
        self._preprocessor.freeze_fit()
        self._writer.append(split, self._preprocessor.transform(carried))

    # --- Pass 3 --------------------------------------------------------------

    def _finalize_train_pass(self) -> None:
        """Apply the now-fitted transformations to the staged training rows."""
        for chunk in self._interim.read_staged(self._chunk_size):
            self._writer.append(SplitName.TRAIN, self._preprocessor.transform(chunk))

    # --- Reporting -----------------------------------------------------------

    def _build_report(
        self,
        raw_path: Path,
        fingerprint: DatasetFingerprint,
        boundaries: SplitBoundaries,
    ) -> PreprocessingReport:
        """Assemble the measured report and check the properties it must satisfy."""
        descriptions = removal_descriptions()
        cleaning = CleaningReport(
            rows_read=self._rows_read,
            removals=tuple(
                RemovalRecord(
                    reason=reason,
                    description=description,
                    rows_removed=int(self._removed.get(reason, 0)),
                )
                for reason, description in descriptions.items()
            ),
            rows_written=sum(
                self._writer.rows_written(split) for split in ORDERED_SPLITS
            ),
            duplicate_rows_detected=self._duplicates,
            duplicate_rows_removed=0,
        )

        splits = SplitReport(
            train_last_timestamp=boundaries.train_last,
            validation_last_timestamp=boundaries.validation_last,
            tie_policy=boundaries.tie_policy,
            summaries=tuple(
                self._accumulators[name].summarise() for name in ORDERED_SPLITS
            ),
        )
        if not splits.chronology_holds():
            raise ChronologyViolationError(
                SplitName.TRAIN.value,
                SplitName.TEST.value,
                "the produced splits overlap in time after writing",
            )

        totals: Counter[int] = Counter()
        for name in ORDERED_SPLITS:
            totals.update(self._accumulators[name].class_counts)

        target = TargetReport(
            source_column=TARGET_COLUMN,
            output_column=TARGET_OUTPUT_COLUMN,
            observed_values=tuple(sorted(str(label) for label in totals)),
            mapping=self._preprocessor.target_mapping(),
            positive_label=POSITIVE_LABEL,
            class_counts={int(label): int(count) for label, count in totals.items()},
        )

        return PreprocessingReport(
            version=PREPROCESSING_VERSION,
            source_file=str(raw_path),
            output_directory=str(self._writer.output_root),
            fingerprint=fingerprint,
            cleaning=cleaning,
            splits=splits,
            target=target,
            feature_columns=self._preprocessor.feature_columns,
            identifier_columns=self._preprocessor.identifier_columns,
            excluded_features=self._preprocessor.rejected_features(),
            performance=PerformanceReport(
                seconds_per_stage=dict(self._timings),
                peak_chunk_rows=self._chunk_size,
                distinct_accounts_tracked=self._preprocessor.distinct_accounts,
            ),
        )

    def _write_artefacts(
        self, report: PreprocessingReport, boundaries: SplitBoundaries
    ) -> None:
        """Write the metadata files that make the outputs reproducible."""
        described = self._preprocessor.describe()
        payload = report.to_dict()

        self._writer.write_metadata("schema", _schema_payload(self._schema))
        self._writer.write_metadata(
            "feature_metadata",
            {
                "preprocessing_version": described["preprocessing_version"],
                "timestamp_format": described["timestamp_format"],
                "column_order": described["column_order"],
                "identifier_columns": described["identifier_columns"],
                "feature_columns": described["feature_columns"],
                "target_column": described["target_column"],
                "raw_to_output_names": described["raw_to_output_names"],
                "numerical_transformations": described["numerical_transformations"],
                "feature_builders": described["feature_builders"],
                "excluded_features": report.excluded_features,
                "columns_not_encoded": described["columns_not_encoded"],
            },
        )
        self._writer.write_metadata(
            "transformers",
            {
                "preprocessing_version": described["preprocessing_version"],
                "fitted_on": described["fitted_on"],
                "categorical_encoding": described["categorical_encoding"],
                "scaling_parameters": described["scaling_parameters"],
                "scaled_columns": described["scaled_columns"],
                "unscaled_columns": described["unscaled_columns"],
                "unknown_category_counts": described["unknown_category_counts"],
            },
        )
        self._writer.write_metadata(
            "splits",
            {
                **payload["splits"],
                "boundaries": {
                    "train_last": boundaries.train_last,
                    "validation_last": boundaries.validation_last,
                    "tie_policy": boundaries.tie_policy,
                },
                "requested_fractions": dict(
                    zip(
                        [split.value for split in ORDERED_SPLITS],
                        self._splitter.fractions,
                        strict=True,
                    )
                ),
                "target": payload["target"],
            },
        )
        self._writer.write_metadata("preprocessing_report", payload)

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        """Time one stage and record it under ``name``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = time.perf_counter() - started
            logger.info("Stage '%s' finished in %.1fs", name, self._timings[name])


def _schema_payload(schema: TableSchema) -> dict[str, object]:
    """Serialise the verified schema for the artefacts."""
    return {
        "name": schema.name,
        "unexpected_column_policy": schema.unexpected_column_policy.value,
        "columns": [
            {
                "name": column.name,
                "data_type": column.data_type.value,
                "required": column.required,
                "nullable": column.nullable,
                "allowed_values": list(column.allowed_values),
                "minimum": column.minimum,
                "maximum": column.maximum,
                "description": column.description,
            }
            for column in schema.columns
        ],
    }


def _sha256(path: Path) -> str:
    """Digest a file without reading it whole.

    The digest is what turns "the pipeline does not write to its input" from a
    claim into a check: it is taken before the run and again after, and a
    difference aborts rather than being reported as a successful run over data
    that no longer exists.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
