"""Tests for the two storage adapters: the processed writer and the interim store.

Two guarantees are checked here that no other test covers. The first is
immutability of the input: the writer refuses at construction to write anywhere
inside the raw dataset directory, which is a configuration mistake caught before
a single row exists rather than after the file it would overwrite is gone. The
second is that no artefact can carry a credential — a preprocessing report
includes configuration, and configuration is exactly where a Kaggle key would
end up if one were ever put there.

Everything is written into pytest's ``tmp_path``. Nothing here reads or writes
``data/raw``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from pandas import DataFrame

from domain.enums.split_name import SplitName
from domain.exceptions.preprocessing import OutputWouldOverwriteInputError
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from infrastructure.data.writers.csv_interim_storage import (
    PARTITIONS_SUBDIRECTORY,
    STAGED_FILENAME,
    CsvInterimStorage,
)
from infrastructure.data.writers.csv_processed_data_writer import (
    FORBIDDEN_METADATA_KEYS,
    METADATA_DIRECTORY,
    SPLIT_FILENAME,
    CredentialInMetadataError,
    CsvProcessedDataWriter,
)
from infrastructure.data.writers.interim_partition_store import remove_tree

pytestmark = pytest.mark.unit


def _rows(count: int = 3, start: int = 0) -> DataFrame:
    """A small synthetic frame with a stable column order."""
    return DataFrame(
        {
            "timestamp": [f"2020-01-01 00:{start + index:02d}:00" for index in range(count)],
            "amount_paid_log": [float(start + index) for index in range(count)],
            "is_laundering": [0] * count,
        }
    )


# --- The processed writer ------------------------------------------------------


def test_each_split_gets_one_streamable_file(tmp_path: Path) -> None:
    """One file per split, so the next phase reads it with the same loader."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, _rows())
    writer.close()

    path = writer.split_path(SplitName.TRAIN)

    assert path.name == SPLIT_FILENAME
    assert path.parent.name == SplitName.TRAIN.value
    assert path.is_file()


def test_the_header_is_written_once_however_many_chunks_arrive(tmp_path: Path) -> None:
    """Three appends must produce one header, or the file is unparseable."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    for index in range(3):
        writer.append(SplitName.TRAIN, _rows(2, start=index * 2))
    writer.close()

    lines = writer.split_path(SplitName.TRAIN).read_text(encoding="utf-8").splitlines()

    assert lines[0] == "timestamp,amount_paid_log,is_laundering"
    assert len(lines) == 7
    assert lines.count("timestamp,amount_paid_log,is_laundering") == 1


def test_appended_rows_are_counted_per_split(tmp_path: Path) -> None:
    """The counts are what the final report reconciles against the input."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, _rows(5))
    writer.append(SplitName.TEST, _rows(2))
    writer.close()

    assert writer.rows_written(SplitName.TRAIN) == 5
    assert writer.rows_written(SplitName.VALIDATION) == 0
    assert writer.rows_written(SplitName.TEST) == 2


def test_an_empty_chunk_writes_nothing_and_creates_no_file(tmp_path: Path) -> None:
    writer = CsvProcessedDataWriter(tmp_path / "out")

    assert writer.append(SplitName.TRAIN, _rows(0)) == 0

    writer.close()
    assert not writer.split_path(SplitName.TRAIN).exists()


def test_the_row_index_never_reaches_the_output(tmp_path: Path) -> None:
    """It is an artefact of chunking and would be misleading in the file."""
    frame = _rows(2)
    frame.index = [900, 901]
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, frame)
    writer.close()

    text = writer.split_path(SplitName.TRAIN).read_text(encoding="utf-8")

    assert "900" not in text


def test_a_chunk_whose_columns_changed_is_refused(tmp_path: Path) -> None:
    """A CSV has one header; a silently reordered chunk would corrupt the file."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, _rows())

    with pytest.raises(ValueError, match="column order changed"):
        writer.append(SplitName.TRAIN, _rows()[["is_laundering", "timestamp"]])

    writer.close()


def test_closing_twice_is_harmless(tmp_path: Path) -> None:
    """The service closes in a ``finally``; a second close must not raise."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, _rows())
    writer.close()
    writer.close()


def test_the_written_file_reads_back_with_the_project_loader(tmp_path: Path) -> None:
    """"Streamable" as a property of the output, not just an intention."""
    writer = CsvProcessedDataWriter(tmp_path / "out")
    writer.append(SplitName.TRAIN, _rows(10))
    writer.close()

    chunks = list(CsvChunkLoader().stream_chunks(writer.split_path(SplitName.TRAIN), 4))

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert list(chunks[0].columns) == ["timestamp", "amount_paid_log", "is_laundering"]


# --- Raw dataset immutability --------------------------------------------------


def test_an_output_directory_inside_the_raw_directory_is_refused(tmp_path: Path) -> None:
    """Rule 13 enforced at construction, before anything could be overwritten."""
    raw = tmp_path / "data" / "raw" / "ibm_aml"
    raw.mkdir(parents=True)

    with pytest.raises(OutputWouldOverwriteInputError):
        CsvProcessedDataWriter(raw / "processed", raw_directory=raw)


def test_the_raw_directory_itself_is_refused_as_an_output(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "ibm_aml"
    raw.mkdir(parents=True)

    with pytest.raises(OutputWouldOverwriteInputError):
        CsvProcessedDataWriter(raw, raw_directory=raw)


def test_a_sibling_output_directory_is_allowed(tmp_path: Path) -> None:
    """The check must reject containment, not any path that looks similar."""
    raw = tmp_path / "data" / "raw" / "ibm_aml"
    raw.mkdir(parents=True)

    writer = CsvProcessedDataWriter(tmp_path / "data" / "processed", raw_directory=raw)

    assert writer.output_root.is_relative_to(tmp_path)


def test_writing_leaves_files_in_the_raw_directory_untouched(tmp_path: Path) -> None:
    """The behavioural statement of the same rule."""
    raw = tmp_path / "data" / "raw" / "ibm_aml"
    raw.mkdir(parents=True)
    source = raw / "synthetic_records.csv"
    source.write_text("SYNTHETIC TEST DATA - NOT IBM AML DATA\n", encoding="utf-8")
    before = source.read_bytes()

    writer = CsvProcessedDataWriter(tmp_path / "processed", raw_directory=raw)
    writer.append(SplitName.TRAIN, _rows())
    writer.write_metadata("schema", {"columns": []})
    writer.close()

    assert source.read_bytes() == before
    assert list(raw.iterdir()) == [source]


# --- Metadata ------------------------------------------------------------------


def test_metadata_is_written_as_json_under_its_own_directory(tmp_path: Path) -> None:
    writer = CsvProcessedDataWriter(tmp_path / "out")

    path = writer.write_metadata("splits", {"train_rows": 60})

    assert path.parent.name == METADATA_DIRECTORY
    assert path.name == "splits.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"train_rows": 60}


def test_two_runs_produce_byte_identical_artefacts(tmp_path: Path) -> None:
    """Keys are sorted so reproducibility can be checked by comparing files."""
    payload = {"z": 1, "a": {"n": 2, "m": 3}}

    first = CsvProcessedDataWriter(tmp_path / "one").write_metadata("m", payload)
    second = CsvProcessedDataWriter(tmp_path / "two").write_metadata(
        "m", {"a": {"m": 3, "n": 2}, "z": 1}
    )

    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("key", sorted(FORBIDDEN_METADATA_KEYS))
def test_every_forbidden_key_is_refused_at_the_top_level(
    tmp_path: Path, key: str
) -> None:
    writer = CsvProcessedDataWriter(tmp_path / "out")

    with pytest.raises(CredentialInMetadataError):
        writer.write_metadata("report", {key: "value"})


def test_a_forbidden_key_nested_deep_in_the_payload_is_still_caught(
    tmp_path: Path,
) -> None:
    """The report nests configuration several levels down.

    A top-level-only check would miss exactly the case that matters.
    """
    writer = CsvProcessedDataWriter(tmp_path / "out")
    payload = {"run": {"config": {"sources": [{"kaggle_key": "would be a secret"}]}}}

    with pytest.raises(CredentialInMetadataError):
        writer.write_metadata("report", payload)


def test_the_refusal_names_the_key_but_never_prints_its_value(tmp_path: Path) -> None:
    """An error message that echoed the secret would defeat the purpose."""
    writer = CsvProcessedDataWriter(tmp_path / "out")

    with pytest.raises(CredentialInMetadataError) as raised:
        writer.write_metadata("report", {"api_token": "SECRET-VALUE-12345"})

    assert "api_token" in str(raised.value)
    assert "SECRET-VALUE-12345" not in str(raised.value)


def test_forbidden_keys_are_matched_regardless_of_case(tmp_path: Path) -> None:
    writer = CsvProcessedDataWriter(tmp_path / "out")

    with pytest.raises(CredentialInMetadataError):
        writer.write_metadata("report", {"KAGGLE_USERNAME": "x"})


def test_nothing_is_written_when_a_payload_is_refused(tmp_path: Path) -> None:
    """The check runs before the file is opened, so no partial file survives."""
    writer = CsvProcessedDataWriter(tmp_path / "out")

    with pytest.raises(CredentialInMetadataError):
        writer.write_metadata("report", {"password": "x"})

    assert not writer.metadata_path("report").exists()


def test_an_ordinary_payload_that_merely_mentions_a_key_is_allowed(
    tmp_path: Path,
) -> None:
    """The guard is on key names, so legitimate prose must not trip it."""
    writer = CsvProcessedDataWriter(tmp_path / "out")

    path = writer.write_metadata(
        "report", {"note": "credentials are never stored in artefacts"}
    )

    assert path.is_file()


# --- Interim storage -----------------------------------------------------------


def _storage(tmp_path: Path) -> CsvInterimStorage:
    return CsvInterimStorage(
        tmp_path / "interim", CsvChunkLoader(), partition_dtypes={"account": "string"}
    )


def test_rows_are_partitioned_under_the_key_they_were_added_with(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition("2020-01-01", DataFrame({"account": ["A"], "value": [1]}))
    storage.add_to_partition("2020-01-02", DataFrame({"account": ["B"], "value": [2]}))
    storage.end_partitions()

    assert storage.partition_keys() == ["2020-01-01", "2020-01-02"]
    assert list(storage.read_partition("2020-01-01")["account"]) == ["A"]


def test_partition_keys_come_back_in_ascending_order(tmp_path: Path) -> None:
    """This is what turns an unordered file into a chronological stream."""
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    for day in ("2020-01-03", "2020-01-01", "2020-01-02"):
        storage.add_to_partition(day, DataFrame({"account": ["A"], "value": [1]}))
    storage.end_partitions()

    assert storage.partition_keys() == ["2020-01-01", "2020-01-02", "2020-01-03"]


def test_a_partition_accumulates_rows_from_every_chunk_that_touched_it(
    tmp_path: Path,
) -> None:
    """The raw file is shuffled, so one day's rows arrive in many chunks."""
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    for index in range(4):
        storage.add_to_partition(
            "2020-01-01", DataFrame({"account": ["A"], "value": [index]})
        )
    storage.end_partitions()

    assert list(storage.read_partition("2020-01-01")["value"]) == [0, 1, 2, 3]


def test_identifier_columns_survive_the_partition_round_trip_as_text(
    tmp_path: Path,
) -> None:
    """A leading zero lost to type inference would silently change the account."""
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition(
        "2020-01-01", DataFrame({"account": ["000123"], "value": [1]})
    )
    storage.end_partitions()

    assert list(storage.read_partition("2020-01-01")["account"]) == ["000123"]


def test_a_consumed_partition_is_dropped_so_disk_does_not_grow(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition("2020-01-01", DataFrame({"account": ["A"], "value": [1]}))
    storage.end_partitions()

    storage.drop_partition("2020-01-01")

    assert storage.partition_keys() == []


def test_using_partitions_before_the_pass_starts_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="begin_partitions"):
        _storage(tmp_path).partition_keys()


def test_staged_rows_stream_back_in_the_order_they_were_written(
    tmp_path: Path,
) -> None:
    """Pass 3 depends on this: the training rows must return in canonical order."""
    storage = _storage(tmp_path)
    for index in range(3):
        storage.stage(DataFrame({"account": ["A"], "value": [index]}))
    storage.end_staging()

    values = [
        value for chunk in storage.read_staged(2) for value in chunk["value"]
    ]

    assert values == [0, 1, 2]


def test_staged_rows_come_back_in_bounded_chunks(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.stage(DataFrame({"account": ["A"] * 10, "value": list(range(10))}))
    storage.end_staging()

    assert [len(chunk) for chunk in storage.read_staged(4)] == [4, 4, 2]


def test_staging_an_empty_frame_writes_nothing(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    assert storage.stage(DataFrame({"account": [], "value": []})) == 0
    assert storage.staged_row_count() == 0


def test_reading_staged_rows_that_were_never_written_yields_nothing(
    tmp_path: Path,
) -> None:
    """A run with no training rows is a data problem, not a storage failure."""
    storage = _storage(tmp_path)
    storage.end_staging()

    assert list(storage.read_staged(100)) == []


def test_staged_columns_that_change_between_chunks_are_refused(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    storage.stage(DataFrame({"account": ["A"], "value": [1]}))

    with pytest.raises(ValueError, match="staged column order changed"):
        storage.stage(DataFrame({"value": [2], "account": ["B"]}))


def test_cleanup_removes_every_working_file(tmp_path: Path) -> None:
    """They are a working copy of the dataset; leaving them doubles its footprint."""
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition("2020-01-01", DataFrame({"account": ["A"], "value": [1]}))
    storage.end_partitions()
    storage.stage(DataFrame({"account": ["A"], "value": [1]}))
    storage.end_staging()

    storage.cleanup()

    assert not (tmp_path / "interim" / STAGED_FILENAME).exists()
    assert not (tmp_path / "interim" / PARTITIONS_SUBDIRECTORY).exists()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    """It runs at the end of a successful run and can be reached twice."""
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.end_partitions()

    storage.cleanup()
    storage.cleanup()


def test_a_locked_working_file_is_reported_rather_than_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A working file nothing can delete is reported, not raised.

    Observed on the real run: OneDrive held the partition directory open and
    ``rmdir`` raised ``WinError 5`` *after* every split and artefact had been
    written and the raw file re-verified. Cleanup is a tidy-up of scratch files,
    so failing there would discard a correct, finished result. It warns and
    names the leftovers instead.
    """
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition("2020-01-01", DataFrame({"account": ["A"], "value": [1]}))
    storage.end_partitions()

    def _denied(directory: Path) -> None:
        raise PermissionError(5, "Access is denied", str(directory))

    monkeypatch.setattr(
        "infrastructure.data.writers.interim_partition_store.shutil.rmtree", _denied
    )
    monkeypatch.setattr(
        "infrastructure.data.writers.interim_partition_store.REMOVAL_RETRY_SECONDS", 0.0
    )

    with caplog.at_level(logging.WARNING):
        storage.cleanup()

    assert PARTITIONS_SUBDIRECTORY in caplog.text
    assert "deleted by hand" in caplog.text


def test_a_transient_lock_is_retried_rather_than_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock that clears is retried rather than reported.

    The Windows handle is released a moment later, so the first failure is not
    the answer; retrying turns the common case back into a clean removal.
    """
    storage = _storage(tmp_path)
    storage.begin_partitions(("account", "value"))
    storage.add_to_partition("2020-01-01", DataFrame({"account": ["A"], "value": [1]}))
    storage.end_partitions()

    real_rmtree = shutil.rmtree
    partitions = tmp_path / "interim" / PARTITIONS_SUBDIRECTORY
    attempts: list[str] = []

    def _fails_once(directory: Path) -> None:
        attempts.append(str(directory))
        if attempts.count(str(partitions)) == 1:
            raise PermissionError(5, "Access is denied", str(directory))
        real_rmtree(directory)

    monkeypatch.setattr(
        "infrastructure.data.writers.interim_partition_store.shutil.rmtree", _fails_once
    )
    monkeypatch.setattr(
        "infrastructure.data.writers.interim_partition_store.REMOVAL_RETRY_SECONDS", 0.0
    )

    storage.cleanup()

    assert attempts.count(str(partitions)) == 2
    assert not partitions.exists()


def test_removing_a_directory_that_is_already_gone_counts_as_removed(
    tmp_path: Path,
) -> None:
    """Cleanup can be reached twice, so absence is success, not an error."""
    assert remove_tree(tmp_path / "never-existed") is True


def test_interim_files_live_outside_the_processed_output(tmp_path: Path) -> None:
    """Rule 11: working files are not an output and must not be mistaken for one."""
    storage = CsvInterimStorage(tmp_path / "interim", CsvChunkLoader())
    writer = CsvProcessedDataWriter(tmp_path / "processed")

    assert not storage.directory.is_relative_to(writer.output_root)
