"""Preprocess LI-Medium dataset in bounded chunks and create 60:20:20 splits."""

import logging
import sys
import time
from pathlib import Path

# Ensure backend package is in python path
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/li_medium_run/logs/preprocessing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("li_medium_preprocessing")

from application.services.preprocessing_service import PreprocessingService
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from infrastructure.data.preprocessors.aml_preprocessor import (
    RAW_DTYPES,
    AmlPreprocessor,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    build_ibm_aml_transaction_schema,
)
from infrastructure.data.splitters.chronological_splitter import ChronologicalSplitter
from infrastructure.data.writers.csv_interim_storage import CsvInterimStorage
from infrastructure.data.writers.csv_processed_data_writer import CsvProcessedDataWriter


class DirectFileTransactionSource:
    """Transaction source pointing directly to a given raw file."""

    def __init__(self, raw_file_path: Path) -> None:
        self.raw_file_path = raw_file_path.resolve()

    def resolve_file(self) -> Path:
        if not self.raw_file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")
        return self.raw_file_path


def run_li_medium_preprocessing(
    raw_path: Path = Path("data/raw/ibm_aml/LI-Medium_Trans.csv"),
    output_root: Path = Path("data/processed/ibm_aml/LI-Medium"),
    interim_root: Path = Path("data/interim/ibm_aml/LI-Medium"),
    chunk_size: int = 200000,
) -> None:
    logger.info("Starting LI-Medium preprocessing pipeline...")
    logger.info("Raw input path: %s", raw_path.resolve())
    logger.info("Processed output path: %s", output_root.resolve())
    logger.info("Interim working path: %s", interim_root.resolve())
    logger.info("Chunk size: %d rows", chunk_size)

    output_root.mkdir(parents=True, exist_ok=True)
    interim_root.mkdir(parents=True, exist_ok=True)

    source = DirectFileTransactionSource(raw_path)
    loader = CsvChunkLoader()
    interim = CsvInterimStorage(interim_root, loader, partition_dtypes=RAW_DTYPES)
    writer = CsvProcessedDataWriter(output_root, raw_directory=raw_path.parent)
    preprocessor = AmlPreprocessor()
    splitter = ChronologicalSplitter(train_fraction=0.60, validation_fraction=0.20)
    schema = build_ibm_aml_transaction_schema()

    service = PreprocessingService(
        source=source,
        loader=loader,
        interim=interim,
        writer=writer,
        preprocessor=preprocessor,
        splitter=splitter,
        schema=schema,
        chunk_size=chunk_size,
        keep_interim=False,
    )

    t0 = time.perf_counter()
    report = service.run()
    elapsed = time.perf_counter() - t0

    logger.info("LI-Medium preprocessing completed successfully in %.1f seconds!", elapsed)
    logger.info("Rows read: %d, Rows written: %d, Rows removed: %d",
                report.cleaning.rows_read, report.cleaning.rows_written, report.cleaning.rows_removed)
    for sp in report.splits.per_split:
        logger.info("Split %s: %d rows (%.2f%%) | Pos: %d (%.4f%%) | Neg: %d | Time range: %s to %s",
                    sp.name.upper(), sp.rows, sp.share * 100, sp.positive, sp.positive_rate * 100,
                    sp.negative, sp.first_timestamp, sp.last_timestamp)


if __name__ == "__main__":
    run_li_medium_preprocessing()

