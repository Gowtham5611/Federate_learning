"""Dataset ingestion: the composed source → loader → validator pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from pandas import DataFrame

from application.dto.validation import ChunkValidationResult
from application.ports.output.data_loader import DEFAULT_CHUNK_SIZE, ChunkedDataLoaderPort
from application.ports.output.schema_validator import SchemaValidatorPort
from application.ports.output.transaction_data_source import TransactionDataSourcePort
from domain.exceptions.data import SchemaValidationError

logger = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 5
"""Findings quoted in an abort message. The full report goes to the log."""


class DatasetIngestionService:
    """Streams validated chunks of a dataset, one chunk at a time.

    Composes the three ingestion ports and owns exactly one decision the parts
    cannot make individually: what to do when a chunk fails validation. Nothing
    else is added — no cleaning, no conversion, no feature construction.

    The service **never accumulates**. Chunks are yielded as they are read and no
    reference to a previous chunk is retained, so a consumer that processes and
    discards each chunk holds one chunk's worth of rows at a time regardless of
    file size. A caller that wants the whole file in memory must build it
    deliberately.

    All ingestion logging happens here rather than in the validator, so an
    invalid chunk is reported once instead of once per layer.
    """

    def __init__(
        self,
        source: TransactionDataSourcePort,
        loader: ChunkedDataLoaderPort,
        validator: SchemaValidatorPort,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        strict: bool = True,
    ) -> None:
        """Wire the pipeline.

        Args:
            source: Locates the file to read.
            loader: Streams the file in chunks.
            validator: Checks each chunk against a schema.
            chunk_size: Rows per chunk.
            strict: If ``True``, the first invalid chunk aborts the run. If
                ``False``, invalid chunks are logged and still yielded together
                with their report, leaving the decision to the consumer. Strict
                is the default because training on data that failed its schema
                produces results nobody can interpret.
        """
        self._source = source
        self._loader = loader
        self._validator = validator
        self._chunk_size = chunk_size
        self._strict = strict

    def stream_validated_chunks(self) -> Iterator[tuple[DataFrame, ChunkValidationResult]]:
        """Yield each chunk of the dataset together with its validation report.

        The file is located immediately, so an absent dataset fails at the call
        site rather than on first iteration. Reading and validating are deferred
        to the returned iterator.

        Returns:
            An iterator of ``(chunk, result)`` pairs in file order.

        Raises:
            DataFileNotFoundError: The dataset is not present locally.
        """
        file_path = self._source.resolve_file()
        logger.info(
            "Ingestion starting: dataset=%s variant=%s file=%s schema=%s "
            "chunk_size=%d strict=%s",
            self._source.dataset_id,
            self._source.variant,
            file_path.name,
            self._validator.schema.name,
            self._chunk_size,
            self._strict,
        )
        return self._ingest(file_path)

    def _ingest(self, file_path: Path) -> Iterator[tuple[DataFrame, ChunkValidationResult]]:
        """Read, validate and yield chunks, applying the strictness policy."""
        chunks = 0
        rows = 0
        invalid = 0
        for index, chunk in enumerate(self._loader.stream_chunks(file_path, self._chunk_size)):
            result = self._validator.validate_chunk(
                chunk, chunk_index=index, source=file_path.name
            )
            chunks += 1
            rows += result.row_count

            if not result.is_valid:
                invalid += 1
                if self._strict:
                    logger.error("Ingestion aborted by invalid chunk:\n%s", result.report())
                    raise SchemaValidationError(
                        file_path.name, index, self._abort_summary(result)
                    )
                logger.warning("Invalid chunk yielded in non-strict mode:\n%s", result.report())
            elif result.warnings:
                logger.warning("Chunk validated with warnings:\n%s", result.report())

            yield chunk, result

        logger.info(
            "Ingestion finished: file=%s chunks=%d rows=%d invalid_chunks=%d",
            file_path.name,
            chunks,
            rows,
            invalid,
        )

    @staticmethod
    def _abort_summary(result: ChunkValidationResult) -> str:
        """Condense a failed chunk's errors into one actionable line."""
        shown = result.errors[:_MAX_REPORTED_ERRORS]
        text = "; ".join(issue.describe() for issue in shown)
        remaining = len(result.errors) - len(shown)
        if remaining > 0:
            text += f"; and {remaining} further error(s) - see the log for the full report"
        return text
