"""Universal CSV Reader for FedSecureAI.

Supports:
- Encodings: UTF-8, UTF-8 with BOM, Latin-1, CP1252, ISO-8859-1.
- Delimiters: Comma (,), Semicolon (;), Tab (\\t), Pipe (|) with csv.Sniffer and robust fallback.
- Quoted strings, escaped delimiters, missing value variations.
- Variable row counts (from 1 to 100,000+ rows) and column counts.
- Chunked iteration for large files.
- Comprehensive structural metadata extraction.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1")
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
STANDARD_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
    "?",
    "none",
]


@dataclass
class CsvMetadata:
    """Detailed structural metadata extracted from an inspected CSV."""

    filename: str
    file_size_bytes: int
    total_rows: int
    total_columns: int
    column_names: List[str]
    detected_delimiter: str
    detected_encoding: str
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    duplicate_count: int
    warnings: List[str] = field(default_factory=list)
    is_empty: bool = False
    has_valid_header: bool = True
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return asdict(self)


class UniversalCsvReader:
    """Universal and resilient CSV reader with automatic encoding and delimiter detection."""

    def __init__(
        self,
        default_chunk_size: int = 5000,
        max_preview_rows: int = 10,
    ) -> None:
        self.default_chunk_size = default_chunk_size
        self.max_preview_rows = max_preview_rows

    def detect_encoding(self, raw_bytes: bytes) -> str:
        """Detect text encoding, handling BOM markers and common variants."""
        if not raw_bytes:
            return "utf-8"

        # Check for explicit BOMs
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
            return "utf-16"
        if raw_bytes.startswith(b"\xff\xfe\x00\x00") or raw_bytes.startswith(b"\x00\x00\xfe\xff"):
            return "utf-32"

        # Try decoding with supported encodings in priority order
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
            try:
                raw_bytes[: min(len(raw_bytes), 65536)].decode(enc)
                return enc
            except UnicodeDecodeError:
                continue

        return "latin-1"  # Safe fallback that accepts all byte values

    def decode_text(self, data: Union[bytes, str]) -> Tuple[str, str]:
        """Decode raw bytes or return string, returning (text, encoding)."""
        if isinstance(data, str):
            # Clean possible BOM in Python string
            cleaned = data.lstrip("\ufeff")
            return cleaned, "utf-8"

        encoding = self.detect_encoding(data)
        try:
            text = data.decode(encoding)
        except Exception:
            text = data.decode("latin-1", errors="replace")
            encoding = "latin-1"

        cleaned = text.lstrip("\ufeff")
        return cleaned, encoding

    def detect_delimiter(self, sample_text: str) -> str:
        """Detect CSV delimiter using csv.Sniffer with multi-line consistency heuristics."""
        if not sample_text or not sample_text.strip():
            return ","

        lines = [line for line in sample_text.strip().splitlines() if line.strip()][:15]
        if not lines:
            return ","

        first_line = lines[0]

        # 1. Try Python standard library csv.Sniffer
        try:
            sample_snippet = "\n".join(lines[:5])
            dialect = csv.Sniffer().sniff(sample_snippet, delimiters=CANDIDATE_DELIMITERS)
            if dialect.delimiter in CANDIDATE_DELIMITERS:
                return dialect.delimiter
        except Exception:
            pass

        # 2. Heuristic fallback: count consistent delimiters across sample lines
        best_delim = ","
        max_consistent_cols = 1

        for delim in CANDIDATE_DELIMITERS:
            col_counts = [len(line.split(delim)) for line in lines]
            # Check if column count is > 1 and consistent across rows
            if col_counts[0] > 1:
                # Calculate consistency (percentage of lines with same column count)
                mode_count = max(set(col_counts), key=col_counts.count)
                if mode_count > max_consistent_cols:
                    max_consistent_cols = mode_count
                    best_delim = delim

        return best_delim

    def inspect_csv(
        self,
        data: Union[bytes, str, Path],
        filename: str = "upload.csv",
    ) -> CsvMetadata:
        """Inspect CSV content and return comprehensive metadata without full dataset loading."""
        raw_bytes: bytes
        file_size: int

        if isinstance(data, Path) or (isinstance(data, str) and "\n" not in data and Path(data).exists()):
            path = Path(data)
            raw_bytes = path.read_bytes()
            filename = path.name
            file_size = len(raw_bytes)
        elif isinstance(data, str):
            raw_bytes = data.encode("utf-8")
            file_size = len(raw_bytes)
        else:
            raw_bytes = data
            file_size = len(raw_bytes)

        if not raw_bytes or len(raw_bytes.strip()) == 0:
            return CsvMetadata(
                filename=filename,
                file_size_bytes=0,
                total_rows=0,
                total_columns=0,
                column_names=[],
                detected_delimiter=",",
                detected_encoding="utf-8",
                dtypes={},
                missing_counts={},
                duplicate_count=0,
                warnings=["File is empty (0 bytes)."],
                is_empty=True,
                has_valid_header=False,
                sample_rows=[],
            )

        text, encoding = self.decode_text(raw_bytes)
        lines = [l for l in text.strip().splitlines() if l.strip()]

        if not lines:
            return CsvMetadata(
                filename=filename,
                file_size_bytes=file_size,
                total_rows=0,
                total_columns=0,
                column_names=[],
                detected_delimiter=",",
                detected_encoding=encoding,
                dtypes={},
                missing_counts={},
                duplicate_count=0,
                warnings=["File contains whitespace only with no readable lines."],
                is_empty=True,
                has_valid_header=False,
                sample_rows=[],
            )

        delimiter = self.detect_delimiter("\n".join(lines[:20]))

        # Read preview DataFrame safely
        warnings: List[str] = []
        try:
            preview_df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                nrows=min(len(lines), 100),
                na_values=STANDARD_NA_VALUES,
                keep_default_na=True,
            )
        except Exception as e:
            # Fallback using python engine
            try:
                preview_df = pd.read_csv(
                    io.StringIO(text),
                    sep=delimiter,
                    nrows=min(len(lines), 100),
                    engine="python",
                    on_bad_lines="skip",
                )
                warnings.append(f"CSV required permissive parsing: {str(e)}")
            except Exception as e2:
                return CsvMetadata(
                    filename=filename,
                    file_size_bytes=file_size,
                    total_rows=len(lines) - 1,
                    total_columns=0,
                    column_names=[],
                    detected_delimiter=delimiter,
                    detected_encoding=encoding,
                    dtypes={},
                    missing_counts={},
                    duplicate_count=0,
                    warnings=[f"Failed to parse CSV structure: {str(e2)}"],
                    is_empty=False,
                    has_valid_header=False,
                    sample_rows=[],
                )

        # Total rows approximation or exact count
        total_rows = max(0, len(lines) - 1)
        columns = [str(c).strip() for c in preview_df.columns]
        total_columns = len(columns)

        # Check if header appears valid (not all numeric or unnamed)
        has_valid_header = True
        numeric_headers = sum(1 for c in columns if re.match(r"^-?\d+(\.\d+)?$", c) or "unnamed" in c.lower())
        if total_columns > 0 and numeric_headers >= total_columns / 2:
            has_valid_header = False
            warnings.append("First row contains mostly numeric or unnamed values; header may be missing.")

        # Dtypes and missing counts
        dtypes = {col: str(preview_df[col].dtype) for col in columns}
        missing_counts = {col: int(preview_df[col].isna().sum()) for col in columns}
        duplicate_count = int(preview_df.duplicated().sum())

        # Sample preview records
        sample_records = preview_df.head(self.max_preview_rows).replace({np.nan: None}).to_dict(orient="records")

        return CsvMetadata(
            filename=filename,
            file_size_bytes=file_size,
            total_rows=total_rows,
            total_columns=total_columns,
            column_names=columns,
            detected_delimiter=delimiter,
            detected_encoding=encoding,
            dtypes=dtypes,
            missing_counts=missing_counts,
            duplicate_count=duplicate_count,
            warnings=warnings,
            is_empty=False,
            has_valid_header=has_valid_header,
            sample_rows=sample_records,
        )

    def read_dataframe(
        self,
        data: Union[bytes, str, Path],
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, CsvMetadata]:
        """Read CSV into a clean pandas DataFrame, automatically handling delimiters and encodings."""
        meta = self.inspect_csv(data)
        if meta.is_empty:
            return pd.DataFrame(), meta

        text, _ = self.decode_text(
            data if not isinstance(data, Path) else Path(data).read_bytes()
        )

        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=meta.detected_delimiter,
                nrows=nrows,
                usecols=usecols,
                na_values=STANDARD_NA_VALUES,
                keep_default_na=True,
            )
        except Exception:
            df = pd.read_csv(
                io.StringIO(text),
                sep=meta.detected_delimiter,
                nrows=nrows,
                usecols=usecols,
                engine="python",
                on_bad_lines="skip",
            )

        # Clean column names (strip whitespace and BOM residue)
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

        # Update metadata total_rows with exact parsed rows
        meta.total_rows = len(df)
        return df, meta

    def read_chunks(
        self,
        data: Union[bytes, str, Path],
        chunk_size: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        """Stream CSV in memory-bounded chunks."""
        meta = self.inspect_csv(data)
        if meta.is_empty:
            return

        c_size = chunk_size or self.default_chunk_size
        text, _ = self.decode_text(
            data if not isinstance(data, Path) else Path(data).read_bytes()
        )

        reader = pd.read_csv(
            io.StringIO(text),
            sep=meta.detected_delimiter,
            chunksize=c_size,
            na_values=STANDARD_NA_VALUES,
            keep_default_na=True,
        )

        for chunk in reader:
            chunk.columns = [str(c).strip().lstrip("\ufeff") for c in chunk.columns]
            yield chunk


