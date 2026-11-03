"""CSV adapter for the processed-data writer port.

CSV, not Parquet, and the reason is environmental rather than architectural:
neither ``pyarrow`` nor ``fastparquet`` is installed here, and this project does
not add a dependency to a phase that does not require one. The consequences are
stated rather than hidden — CSV files are larger, carry no dtypes, and must be
re-typed on read. The metadata artefact written alongside them records the dtype
of every column precisely so that the next phase does not have to guess.

What CSV does give, and what the choice preserves, is streamability: each split
is one file that can be read back in bounded chunks by the same loader that read
the raw dataset.

The metadata writer refuses to serialise credential-looking keys. That is not a
theoretical guard: the pipeline's report includes configuration, and
configuration is where a Kaggle key would end up if one were ever added to it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from pandas import DataFrame

from application.ports.output.processed_data_writer import ProcessedDataWriterPort
from domain.enums.split_name import SplitName
from domain.exceptions.preprocessing import OutputWouldOverwriteInputError

logger = logging.getLogger(__name__)

SPLIT_FILENAME = "transactions.csv"
"""Name of each split's single output file."""

METADATA_DIRECTORY = "metadata"

FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_token",
        "credential",
        "credentials",
        "kaggle_key",
        "kaggle_username",
        "key",
        "password",
        "secret",
        "token",
    }
)
"""Keys that must never be serialised into a metadata artefact.

Matched case-insensitively against every key at every depth. A run aborts rather
than writing a file that might contain a secret, because a leaked artefact is
not something a later cleanup can undo.
"""


class CredentialInMetadataError(ValueError):
    """A metadata payload contained a key that could hold a secret."""

    def __init__(self, key: str, path: str) -> None:
        """Record the offending key and where it appeared."""
        self.key = key
        super().__init__(
            f"Refusing to write metadata: the key '{key}' at '{path}' is on the "
            f"forbidden list because it could carry a credential. Preprocessing "
            f"artefacts describe data, never authentication."
        )


class CsvProcessedDataWriter(ProcessedDataWriterPort):
    """Appends processed rows to one CSV per split, plus JSON metadata."""

    def __init__(self, output_root: Path, *, raw_directory: Path | None = None) -> None:
        """Prepare the output tree.

        Args:
            output_root: Directory to write ``<split>/transactions.csv`` and
                ``metadata/*.json`` under. Created if absent.
            raw_directory: The read-only raw dataset directory, if known.
                Supplied so the writer can refuse to write inside it.

        Raises:
            OutputWouldOverwriteInputError: ``output_root`` is inside
                ``raw_directory``.
        """
        resolved = output_root.resolve()
        if raw_directory is not None:
            raw_resolved = raw_directory.resolve()
            if resolved == raw_resolved or raw_resolved in resolved.parents:
                raise OutputWouldOverwriteInputError(resolved, raw_resolved)

        self._root = resolved
        self._handles: dict[SplitName, TextIO] = {}
        self._columns: dict[SplitName, tuple[str, ...]] = {}
        self._counts: dict[SplitName, int] = dict.fromkeys(SplitName, 0)
        self._closed = False

    @property
    def output_root(self) -> Path:
        """Directory the splits and metadata are written under."""
        return self._root

    def split_path(self, split: SplitName) -> Path:
        """Where ``split``'s rows are written."""
        return self._root / split.value / SPLIT_FILENAME

    def metadata_path(self, name: str) -> Path:
        """Where the artefact called ``name`` is written."""
        return self._root / METADATA_DIRECTORY / f"{name}.json"

    def append(self, split: SplitName, frame: DataFrame) -> int:
        """Append rows to one split's CSV, writing the header once.

        Raises:
            ValueError: The frame's columns differ from those already written
                for this split, which would make the file unparseable.
        """
        if frame.empty:
            return 0

        columns = tuple(frame.columns)
        handle = self._handle_for(split, columns)
        if self._columns[split] != columns:
            raise ValueError(
                f"column order changed for split '{split.value}': the file was "
                f"opened with {self._columns[split]} and this chunk has "
                f"{columns}. A CSV has one header, so every chunk must match it."
            )

        # index=False: the frame's positional index is an artefact of chunking
        # and would be meaningless - and misleading - in the output.
        frame.to_csv(handle, header=False, index=False, lineterminator="\n")
        self._counts[split] += len(frame)
        return len(frame)

    def write_metadata(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Write one JSON artefact after checking it holds no credentials.

        Raises:
            CredentialInMetadataError: A forbidden key was found.
        """
        _reject_credentials(payload, path=name)

        path = self.metadata_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys makes two runs over the same data produce byte-identical
        # artefacts, which is what lets reproducibility be checked by comparison.
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote preprocessing artefact '%s'", path.name)
        return path

    def rows_written(self, split: SplitName) -> int:
        """Rows appended to ``split`` so far."""
        return self._counts[split]

    def close(self) -> None:
        """Close every open split file. Idempotent."""
        if self._closed:
            return
        for split, handle in self._handles.items():
            handle.close()
            logger.info(
                "Closed split '%s': %d row(s) in %s",
                split.value,
                self._counts[split],
                self.split_path(split),
            )
        self._handles.clear()
        self._closed = True

    def _handle_for(self, split: SplitName, columns: tuple[str, ...]) -> TextIO:
        """Return the open file for ``split``, creating it on first use."""
        existing = self._handles.get(split)
        if existing is not None:
            return existing

        path = self.split_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
        handle.write(",".join(columns) + "\n")
        self._handles[split] = handle
        self._columns[split] = columns
        return handle


def _reject_credentials(payload: object, *, path: str) -> None:
    """Walk a payload and raise if any key could hold a secret."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            text = str(key).strip().lower()
            if text in FORBIDDEN_METADATA_KEYS:
                raise CredentialInMetadataError(str(key), path)
            _reject_credentials(value, path=f"{path}.{key}")
    elif isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            _reject_credentials(item, path=f"{path}[{index}]")
