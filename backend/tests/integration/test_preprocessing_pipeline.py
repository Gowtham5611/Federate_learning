"""End-to-end tests for the preprocessing pipeline, on a synthetic dataset.

The unit tests check each piece against its contract. This file checks the
claims that only appear when the pieces run together, and that are exactly the
claims the final report makes:

* every input row is either written or counted as removed — the ledger reconciles
* the three splits do not overlap in time, measured from the written files
* the raw input file is byte-for-byte unchanged afterwards
* the same input twice produces byte-identical output
* nothing is left behind in the working directory

The input is a small synthetic CSV written into ``tmp_path``. It is emitted out
of timestamp order, exactly as the real file was measured to be, so a pipeline
that assumed sorted input would fail here rather than in production.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pandas import DataFrame, read_csv

from application.services.preprocessing_service import PreprocessingService
from domain.enums.split_name import ORDERED_SPLITS, SplitName
from domain.exceptions.data import DataFileNotFoundError
from domain.exceptions.preprocessing import NotEnoughDataToSplitError
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from infrastructure.data.preprocessors.aml_preprocessor import (
    RAW_DTYPES,
    TARGET_OUTPUT_COLUMN,
    AmlPreprocessor,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    RAW_COLUMN_ORDER,
    build_ibm_aml_transaction_schema,
)
from infrastructure.data.splitters.chronological_splitter import ChronologicalSplitter
from infrastructure.data.writers.csv_interim_storage import CsvInterimStorage
from infrastructure.data.writers.csv_processed_data_writer import CsvProcessedDataWriter
from tests.fixtures.ingestion_doubles import StubTransactionSource
from tests.fixtures.preprocessing_fixtures import (
    TEST_DATA_NOTICE,
    write_transactions_csv,
)

pytestmark = pytest.mark.integration


class _Harness:
    """One wired-up pipeline over one synthetic file, with its paths to hand."""

    def __init__(self, root: Path, service: PreprocessingService, raw: Path) -> None:
        self.root = root
        self.service = service
        self.raw = raw

    def split_frame(self, split: SplitName) -> DataFrame:
        """Read one written split back, everything as text."""
        return read_csv(
            self.root / "processed" / split.value / "transactions.csv", dtype=str
        )

    def metadata(self, name: str) -> dict[str, Any]:
        """Read one written artefact back."""
        text = (self.root / "processed" / "metadata" / f"{name}.json").read_text(
            encoding="utf-8"
        )
        loaded: dict[str, Any] = json.loads(text)
        return loaded

    @property
    def interim(self) -> Path:
        """Working directory the run used."""
        return self.root / "interim"


def _build(
    tmp_path: Path,
    *,
    rows: int = 240,
    days: int = 8,
    chunk_size: int = 25,
    keep_interim: bool = False,
    name: str = "run",
) -> _Harness:
    """Write a synthetic dataset and wire a pipeline over it."""
    root = tmp_path / name
    raw_directory = root / "raw"
    raw = write_transactions_csv(
        raw_directory / "synthetic_records.csv", rows=rows, days=days
    )
    loader = CsvChunkLoader()
    service = PreprocessingService(
        StubTransactionSource(raw, directory=raw_directory),
        loader,
        CsvInterimStorage(root / "interim", loader, partition_dtypes=RAW_DTYPES),
        CsvProcessedDataWriter(root / "processed", raw_directory=raw_directory),
        AmlPreprocessor(),
        ChronologicalSplitter(),
        build_ibm_aml_transaction_schema(),
        chunk_size=chunk_size,
        keep_interim=keep_interim,
    )
    return _Harness(root, service, raw)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- The run produces what it says it produces ---------------------------------


def test_the_pipeline_writes_one_file_per_split(tmp_path: Path) -> None:
    harness = _build(tmp_path)

    harness.service.run()

    for split in ORDERED_SPLITS:
        path = harness.root / "processed" / split.value / "transactions.csv"
        assert path.is_file(), split.value


def test_every_input_row_is_either_written_or_counted_as_removed(
    tmp_path: Path,
) -> None:
    """The ledger reconciles. Rule 5 in one assertion."""
    harness = _build(tmp_path, rows=240)

    report = harness.service.run()

    assert report.cleaning.rows_read == 240
    assert report.cleaning.rows_written == 240
    assert report.cleaning.rows_removed == 0
    assert report.cleaning.reconciles


def test_the_written_rows_add_up_to_the_reported_total(tmp_path: Path) -> None:
    """Counted from the files on disk, not from the counters that wrote them."""
    harness = _build(tmp_path, rows=240)

    report = harness.service.run()

    on_disk = sum(len(harness.split_frame(split)) for split in ORDERED_SPLITS)

    assert on_disk == report.cleaning.rows_written == 240


def test_the_splits_realise_the_requested_proportions(tmp_path: Path) -> None:
    harness = _build(tmp_path, rows=240, days=8)

    report = harness.service.run()

    assert report.splits.share(SplitName.TRAIN) == pytest.approx(0.60, abs=0.02)
    assert report.splits.share(SplitName.VALIDATION) == pytest.approx(0.20, abs=0.02)
    assert report.splits.share(SplitName.TEST) == pytest.approx(0.20, abs=0.02)


def test_the_written_splits_do_not_overlap_in_time(tmp_path: Path) -> None:
    """The central guarantee, measured from the output files themselves.

    ``max(train) <= min(validation)`` and ``max(validation) <= min(test)``, read
    back off disk so that a reporting bug cannot make it pass.
    """
    harness = _build(tmp_path)
    harness.service.run()

    windows = {}
    for split in ORDERED_SPLITS:
        stamps = list(harness.split_frame(split)["timestamp"])
        windows[split] = (min(stamps), max(stamps))

    assert windows[SplitName.TRAIN][1] <= windows[SplitName.VALIDATION][0]
    assert windows[SplitName.VALIDATION][1] <= windows[SplitName.TEST][0]


def test_the_report_agrees_with_the_files_about_the_time_windows(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    report = harness.service.run()

    for split in ORDERED_SPLITS:
        stamps = list(harness.split_frame(split)["timestamp"])
        summary = report.splits.summary_for(split)
        assert summary.first_timestamp == min(stamps), split.value
        assert summary.last_timestamp == max(stamps), split.value


def test_rows_within_each_split_are_written_in_chronological_order(
    tmp_path: Path,
) -> None:
    """The next phase builds sequences from these files and needs the order."""
    harness = _build(tmp_path)
    harness.service.run()

    for split in ORDERED_SPLITS:
        stamps = list(harness.split_frame(split)["timestamp"])
        assert stamps == sorted(stamps), split.value


def test_the_input_arrives_unsorted_so_the_ordering_is_earned(
    tmp_path: Path,
) -> None:
    """Guard on the fixture: if it were pre-sorted the test above would be empty.

    The real file was measured to be out of timestamp order, and the whole
    day-partitioning design exists because of that.
    """
    harness = _build(tmp_path)

    raw_stamps = list(read_csv(harness.raw, dtype=str)["Timestamp"])

    assert raw_stamps != sorted(raw_stamps)


def test_the_class_distribution_is_preserved_not_rebalanced(tmp_path: Path) -> None:
    """Rule 9: the natural distribution is reported, never resampled."""
    harness = _build(tmp_path, rows=240)
    expected = read_csv(harness.raw, dtype=str)["Is Laundering"].value_counts().to_dict()

    report = harness.service.run()

    assert report.target.class_counts == {
        int(str(label)): int(count) for label, count in expected.items()
    }


def test_the_split_class_counts_add_up_to_the_dataset_counts(tmp_path: Path) -> None:
    harness = _build(tmp_path)

    report = harness.service.run()

    positives = sum(
        summary.positive_count for summary in report.splits.summaries
    )

    assert positives == report.target.class_counts[report.target.positive_label]


def test_the_output_columns_are_identifiers_features_and_the_target(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    report = harness.service.run()
    columns = list(harness.split_frame(SplitName.TRAIN).columns)

    assert columns == [
        *report.identifier_columns,
        *report.feature_columns,
        TARGET_OUTPUT_COLUMN,
    ]


def test_all_three_splits_share_one_column_order(tmp_path: Path) -> None:
    """They are read by the same code in the next phase."""
    harness = _build(tmp_path)
    harness.service.run()

    orders = {
        split: tuple(harness.split_frame(split).columns) for split in ORDERED_SPLITS
    }

    assert len(set(orders.values())) == 1


def test_no_written_cell_is_empty(tmp_path: Path) -> None:
    harness = _build(tmp_path)
    harness.service.run()

    for split in ORDERED_SPLITS:
        assert not harness.split_frame(split).isna().to_numpy().any(), split.value


# --- The raw dataset is untouched ----------------------------------------------


def test_the_raw_file_is_byte_for_byte_unchanged(tmp_path: Path) -> None:
    """Rule 13, checked independently of the pipeline's own verification."""
    harness = _build(tmp_path)
    before = _digest(harness.raw)

    report = harness.service.run()

    assert _digest(harness.raw) == before
    assert report.fingerprint.sha256 == before
    assert report.fingerprint.verified_unchanged


def test_nothing_new_appears_in_the_raw_directory(tmp_path: Path) -> None:
    """No temporary file, no working copy, no output written beside the input."""
    harness = _build(tmp_path)
    before = sorted(path.name for path in harness.raw.parent.iterdir())

    harness.service.run()

    assert sorted(path.name for path in harness.raw.parent.iterdir()) == before


def test_the_reported_size_matches_the_file_on_disk(tmp_path: Path) -> None:
    harness = _build(tmp_path)

    report = harness.service.run()

    assert report.fingerprint.size_bytes == harness.raw.stat().st_size
    assert report.fingerprint.file_name == harness.raw.name


# --- Memory-bounded reading ----------------------------------------------------


def test_the_result_does_not_depend_on_the_chunk_size(tmp_path: Path) -> None:
    """The claim that makes chunking safe rather than merely cheap.

    A chunk size of 7 forces days to be split across many reads and history to
    be carried across many boundaries; a chunk size larger than the file reads
    it in one. Both must produce identical output.
    """
    small = _build(tmp_path, chunk_size=7, name="small")
    large = _build(tmp_path, chunk_size=10_000, name="large")

    small.service.run()
    large.service.run()

    for split in ORDERED_SPLITS:
        assert small.split_frame(split).equals(large.split_frame(split)), split.value


def test_two_identical_runs_produce_byte_identical_output(tmp_path: Path) -> None:
    """Reproducibility as a file comparison, which is the only convincing form."""
    first = _build(tmp_path, name="first")
    second = _build(tmp_path, name="second")

    first.service.run()
    second.service.run()

    for split in ORDERED_SPLITS:
        assert _digest(
            first.root / "processed" / split.value / "transactions.csv"
        ) == _digest(second.root / "processed" / split.value / "transactions.csv")


def test_the_metadata_artefacts_are_reproducible_apart_from_timings(
    tmp_path: Path,
) -> None:
    """Timings differ between runs by nature; nothing else may."""
    first = _build(tmp_path, name="first")
    second = _build(tmp_path, name="second")
    first.service.run()
    second.service.run()

    for name in ("schema", "feature_metadata", "transformers", "splits"):
        assert first.metadata(name) == second.metadata(name), name


# --- Leakage prevention, end to end --------------------------------------------


def test_the_scaling_parameters_were_fitted_on_the_training_rows_only(
    tmp_path: Path,
) -> None:
    """Rule 4, checked against the written artefact.

    The recorded ``count`` for each scaled column must equal the number of
    training rows. If validation or test rows had reached the scaler, it would
    be larger.
    """
    harness = _build(tmp_path)

    report = harness.service.run()
    transformers = harness.metadata("transformers")
    parameters = dict(transformers["scaling_parameters"])
    train_rows = report.splits.summary_for(SplitName.TRAIN).row_count

    assert parameters
    for column, values in parameters.items():
        assert values["count"] == pytest.approx(float(train_rows)), column


def test_the_category_vocabulary_holds_only_values_seen_in_training(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    harness.service.run()
    train = harness.split_frame(SplitName.TRAIN)
    encoding = dict(harness.metadata("transformers")["categorical_encoding"])

    assert encoding["scheme"] == "ordinal"
    assert train["payment_format_code"].astype(int).min() >= 0


def test_the_artefact_states_that_fitting_used_the_training_split(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    harness.service.run()

    assert harness.metadata("transformers")["fitted_on"] == "train split only"


def test_the_account_identifiers_are_recorded_as_deliberately_unencoded(
    tmp_path: Path,
) -> None:
    """Rule 7: the decision is in the artefact, where a later reader will look."""
    harness = _build(tmp_path)

    harness.service.run()
    not_encoded = harness.metadata("feature_metadata")["columns_not_encoded"]

    assert set(dict(not_encoded)) == {
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
    }


# --- Artefacts -----------------------------------------------------------------


def test_the_run_writes_the_five_metadata_artefacts(tmp_path: Path) -> None:
    harness = _build(tmp_path)

    harness.service.run()

    written = sorted(
        path.name for path in (harness.root / "processed" / "metadata").iterdir()
    )

    assert written == [
        "feature_metadata.json",
        "preprocessing_report.json",
        "schema.json",
        "splits.json",
        "transformers.json",
    ]


def test_the_report_artefact_records_the_boundaries_and_the_tie_policy(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    report = harness.service.run()
    splits = harness.metadata("splits")

    assert splits["train_last_timestamp"] == report.splits.train_last_timestamp
    assert splits["validation_last_timestamp"] == report.splits.validation_last_timestamp
    assert splits["tie_policy"] == "whole_timestamp_to_earlier_split"
    assert splits["chronology_holds"] is True


def test_the_report_artefact_records_the_fingerprint_and_the_ledger(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    harness.service.run()
    payload = harness.metadata("preprocessing_report")
    fingerprint = payload["dataset_fingerprint"]
    cleaning = payload["cleaning"]

    assert fingerprint["sha256"] == _digest(harness.raw)
    assert fingerprint["verified_unchanged_after_run"] is True
    assert cleaning["reconciles"] is True
    assert len(cleaning["removals"]) == 4


def test_the_schema_artefact_records_the_eleven_verified_columns(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path)

    harness.service.run()
    schema = harness.metadata("schema")
    names = [column["name"] for column in schema["columns"]]

    assert names == list(RAW_COLUMN_ORDER)


def test_the_artefacts_record_the_features_that_were_not_built(
    tmp_path: Path,
) -> None:
    """Rule 6: an unbuildable feature is documented, not silently absent."""
    harness = _build(tmp_path)

    harness.service.run()
    excluded = dict(harness.metadata("feature_metadata")["excluded_features"])

    assert "time_to_next_transaction" in excluded
    assert "is_fraud_ring_member" in excluded


def test_no_artefact_contains_anything_resembling_a_credential(
    tmp_path: Path,
) -> None:
    """The writer refuses forbidden keys; this checks the values too."""
    harness = _build(tmp_path)

    harness.service.run()

    for path in (harness.root / "processed" / "metadata").iterdir():
        text = path.read_text(encoding="utf-8").lower()
        for word in ("kaggle", "password", "api_key", "secret", "token"):
            assert word not in text, f"{path.name} mentions {word}"


# --- Working files -------------------------------------------------------------


def test_the_working_files_are_removed_when_the_run_finishes(
    tmp_path: Path,
) -> None:
    """They are a working copy of the dataset; keeping them doubles its footprint."""
    harness = _build(tmp_path)

    harness.service.run()

    assert not harness.interim.exists()


def test_the_working_files_are_kept_when_asked_for(tmp_path: Path) -> None:
    harness = _build(tmp_path, keep_interim=True)

    harness.service.run()

    assert harness.interim.exists()


def test_the_working_files_live_outside_the_processed_output(
    tmp_path: Path,
) -> None:
    harness = _build(tmp_path, keep_interim=True)

    harness.service.run()

    assert not harness.interim.is_relative_to(harness.root / "processed")


# --- Small and awkward inputs --------------------------------------------------


def test_a_dataset_too_small_to_split_is_refused_with_an_explanation(
    tmp_path: Path,
) -> None:
    """Two rows cannot make three splits, and saying so beats writing an empty one."""
    harness = _build(tmp_path, rows=2, days=1)

    with pytest.raises(NotEnoughDataToSplitError):
        harness.service.run()


def test_a_dataset_at_the_minimum_size_still_runs(tmp_path: Path) -> None:
    """Three rows on three distinct timestamps is the documented floor."""
    harness = _build(tmp_path, rows=3, days=3, chunk_size=2)

    report = harness.service.run()

    assert report.cleaning.rows_written == 3
    assert report.splits.chronology_holds()
    for split in ORDERED_SPLITS:
        assert report.splits.summary_for(split).row_count == 1


def test_a_chunk_size_larger_than_the_dataset_is_fine(tmp_path: Path) -> None:
    harness = _build(tmp_path, rows=30, days=6, chunk_size=1_000_000)

    assert harness.service.run().cleaning.rows_written == 30


def test_a_missing_dataset_is_reported_rather_than_downloaded(
    tmp_path: Path,
) -> None:
    """Rule 18: this command never acquires data. It fails and says how to get it."""
    loader = CsvChunkLoader()
    service = PreprocessingService(
        StubTransactionSource(None),
        loader,
        CsvInterimStorage(tmp_path / "interim", loader),
        CsvProcessedDataWriter(tmp_path / "processed"),
        AmlPreprocessor(),
        ChronologicalSplitter(),
        build_ibm_aml_transaction_schema(),
        chunk_size=100,
    )

    with pytest.raises(DataFileNotFoundError):
        service.run()


def test_malformed_rows_are_removed_counted_and_the_rest_still_run(
    tmp_path: Path,
) -> None:
    """A broken row must not abort a run over millions of good ones."""
    from tests.fixtures.preprocessing_fixtures import make_transactions

    records = make_transactions(rows=120, days=6)
    records[3]["Timestamp"] = "not a timestamp"
    records[9]["Amount Paid"] = -1.0
    records[15]["Is Laundering"] = 7

    root = tmp_path / "malformed"
    raw_directory = root / "raw"
    raw = write_transactions_csv(raw_directory / "synthetic_records.csv", records)
    loader = CsvChunkLoader()
    service = PreprocessingService(
        StubTransactionSource(raw, directory=raw_directory),
        loader,
        CsvInterimStorage(root / "interim", loader, partition_dtypes=RAW_DTYPES),
        CsvProcessedDataWriter(root / "processed", raw_directory=raw_directory),
        AmlPreprocessor(),
        ChronologicalSplitter(),
        build_ibm_aml_transaction_schema(),
        chunk_size=20,
    )

    report = service.run()

    assert report.cleaning.rows_read == 120
    assert report.cleaning.rows_removed == 3
    assert report.cleaning.rows_written == 117
    assert report.cleaning.reconciles
    removed = {record.reason: record.rows_removed for record in report.cleaning.removals}
    assert removed["unparseable_timestamp"] == 1
    assert removed["negative_amount"] == 1
    assert removed["unknown_target_value"] == 1


def test_duplicates_are_reported_and_kept(tmp_path: Path) -> None:
    """Rule 5: two identical transfers in one minute are plausible.

    Deleting real transactions from an AML dataset is the worse error.
    """
    from tests.fixtures.preprocessing_fixtures import make_transactions

    records = make_transactions(rows=120, days=6)
    records.append(dict(records[0]))
    records.append(dict(records[0]))

    root = tmp_path / "duplicates"
    raw_directory = root / "raw"
    raw = write_transactions_csv(raw_directory / "synthetic_records.csv", records)
    loader = CsvChunkLoader()
    service = PreprocessingService(
        StubTransactionSource(raw, directory=raw_directory),
        loader,
        CsvInterimStorage(root / "interim", loader, partition_dtypes=RAW_DTYPES),
        CsvProcessedDataWriter(root / "processed", raw_directory=raw_directory),
        AmlPreprocessor(),
        ChronologicalSplitter(),
        build_ibm_aml_transaction_schema(),
        chunk_size=50,
    )

    report = service.run()

    assert report.cleaning.duplicate_rows_detected == 2
    assert report.cleaning.duplicate_rows_removed == 0
    assert report.cleaning.rows_written == 122


# --- The fixtures say what they are --------------------------------------------


def test_the_fixture_module_labels_itself_as_synthetic() -> None:
    """It must never be mistaken for, or cited as, IBM AML data."""
    from tests.fixtures import preprocessing_fixtures

    assert TEST_DATA_NOTICE == "SYNTHETIC TEST DATA - NOT IBM AML DATA"
    assert TEST_DATA_NOTICE in (preprocessing_fixtures.__doc__ or "")


def test_no_test_writes_anything_named_like_the_real_dataset(
    tmp_path: Path,
) -> None:
    """Rule 14: no fabricated ``HI-Small_Trans.csv`` anywhere."""
    harness = _build(tmp_path)
    harness.service.run()

    assert not list(tmp_path.rglob("HI-Small*"))
