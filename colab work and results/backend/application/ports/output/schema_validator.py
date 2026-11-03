"""Outbound port for checking a chunk of data against a schema."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pandas import DataFrame

from application.dto.validation import ChunkValidationResult
from domain.value_objects.table_schema import TableSchema


class SchemaValidatorPort(ABC):
    """Reports whether a chunk conforms to a schema, without changing it.

    Two constraints define this port, and both exist because violating them
    produces bugs that are almost impossible to see later:

    * **Validation must not mutate its input.** An implementation that "helpfully"
      coerces a column while checking it turns validation into an invisible
      preprocessing step, and the pipeline's behaviour then depends on whether
      validation ran. Type checks operate on temporary copies.
    * **Validation must not be preprocessing.** It reports; it never repairs,
      fills, drops or renames. What to do about invalid data is the caller's
      decision, made with the report in hand.
    """

    @property
    @abstractmethod
    def schema(self) -> TableSchema:
        """The schema being enforced."""

    @abstractmethod
    def validate_chunk(
        self, chunk: DataFrame, *, chunk_index: int, source: str
    ) -> ChunkValidationResult:
        """Check one chunk and return an aggregated report.

        Returning a result rather than raising is intentional: the caller decides
        whether an invalid chunk aborts the run or is recorded and skipped.

        Findings must be aggregated per ``(column, kind)`` with an occurrence
        count, never emitted once per offending row.

        Args:
            chunk: Rows to check. Must be left exactly as received.
            chunk_index: Zero-based position in the stream, for context.
            source: Label for the data's origin, for context.

        Returns:
            The findings, deterministically ordered.
        """
