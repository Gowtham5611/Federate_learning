"""Tests for the preprocessing CLI.

The service is never built here. ``preprocess_command`` takes a *factory*, and
these tests pass one that returns a recording double, which makes two things
checkable that an end-to-end test cannot state as cleanly: that the command
never constructs a service it does not intend to run, and that ``--chunk-size``
and ``--keep-interim`` reach the factory unchanged rather than being read from
configuration behind the caller's back.

The report the double returns is a synthetic :class:`PreprocessingReport` with
made-up numbers. It exists to exercise the formatter; none of its values
describe the real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

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
from application.services.dataset_manager import DatasetManager
from application.services.preprocessing_service import PreprocessingService
from domain.enums.split_name import SplitName
from domain.exceptions.data import DataFileNotFoundError
from interfaces.cli.app import build_parser, run
from interfaces.cli.commands.preprocess_commands import _render, preprocess_command
from tests.fixtures.dataset_registry_doubles import (
    InMemoryDatasetRegistry,
    make_descriptor,
)
from tests.fixtures.downloader_doubles import ExplodingDownloader

pytestmark = pytest.mark.unit


def _report(*, with_performance: bool = True) -> PreprocessingReport:
    """A synthetic report. The numbers are invented and describe no real run."""
    return PreprocessingReport(
        version="1.0.0",
        source_file="/synthetic/records.csv",
        output_directory="/synthetic/processed",
        fingerprint=DatasetFingerprint(
            file_name="synthetic_records.csv",
            size_bytes=4096,
            sha256="0" * 64,
            verified_unchanged=True,
        ),
        cleaning=CleaningReport(
            rows_read=100,
            removals=(
                RemovalRecord("unparseable_timestamp", "Timestamp did not parse", 3),
                RemovalRecord("negative_amount", "Amount below zero", 0),
            ),
            rows_written=97,
            duplicate_rows_detected=2,
            duplicate_rows_removed=0,
        ),
        splits=SplitReport(
            train_last_timestamp="2020-01-06 00:00:00",
            validation_last_timestamp="2020-01-08 00:00:00",
            tie_policy="whole_timestamp_to_earlier_split",
            summaries=(
                SplitSummary(
                    SplitName.TRAIN, 59, "2020-01-01 00:00:00", "2020-01-06 00:00:00", 4, 55
                ),
                SplitSummary(
                    SplitName.VALIDATION,
                    19,
                    "2020-01-06 00:01:00",
                    "2020-01-08 00:00:00",
                    1,
                    18,
                ),
                SplitSummary(
                    SplitName.TEST, 19, "2020-01-08 00:01:00", "2020-01-10 00:00:00", 2, 17
                ),
            ),
        ),
        target=TargetReport(
            source_column="Is Laundering",
            output_column="is_laundering",
            observed_values=("0", "1"),
            mapping={"0": 0, "1": 1},
            positive_label=1,
            class_counts={0: 90, 1: 7},
        ),
        feature_columns=("amount_paid_log", "hour_sin"),
        identifier_columns=("timestamp", "from_account"),
        excluded_features={"time_to_next_transaction": "would use future information"},
        performance=(
            PerformanceReport(
                seconds_per_stage={"scan": 1.5, "engineer": 2.5},
                peak_chunk_rows=50_000,
                distinct_accounts_tracked=42,
            )
            if with_performance
            else None
        ),
    )


class _RecordingFactory:
    """Records the overrides it was called with and returns a stub service."""

    def __init__(self, *, report: PreprocessingReport | None = None) -> None:
        self.calls: list[tuple[int | None, bool]] = []
        self.runs = 0
        self._report = report if report is not None else _report()

    def __call__(
        self, *, chunk_size: int | None, keep_interim: bool
    ) -> PreprocessingService:
        self.calls.append((chunk_size, keep_interim))
        return cast("PreprocessingService", self)

    def run(self) -> PreprocessingReport:
        self.runs += 1
        return self._report


class _ExplodingFactory:
    """Fails if the command ever builds a service it should not have built."""

    def __call__(
        self, *, chunk_size: int | None, keep_interim: bool
    ) -> PreprocessingService:
        raise AssertionError(
            f"a service must not be built here (chunk_size={chunk_size}, "
            f"keep_interim={keep_interim})"
        )


def _manager(tmp_path: Path) -> DatasetManager:
    """A manager with no network-capable collaborators."""
    return DatasetManager(
        registry=InMemoryDatasetRegistry((make_descriptor(),)),
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
    )


# --- Argument parsing ----------------------------------------------------------


def test_preprocess_is_a_registered_subcommand() -> None:
    args = build_parser().parse_args(["preprocess"])

    assert args.command == "preprocess"


def test_both_overrides_default_to_leaving_configuration_alone() -> None:
    """Absent flags must mean "use the configured value", not a hard-coded one."""
    args = build_parser().parse_args(["preprocess"])

    assert args.chunk_size is None
    assert args.keep_interim is False


def test_chunk_size_is_parsed_as_an_integer() -> None:
    args = build_parser().parse_args(["preprocess", "--chunk-size", "25000"])

    assert args.chunk_size == 25_000


def test_a_non_numeric_chunk_size_is_rejected_before_anything_runs() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["preprocess", "--chunk-size", "lots"])


def test_keep_interim_is_a_flag() -> None:
    args = build_parser().parse_args(["preprocess", "--keep-interim"])

    assert args.keep_interim is True


def test_the_preprocess_help_states_that_it_never_downloads() -> None:
    """Rule 18: the command operates on the local dataset and nothing else."""
    help_text = build_parser().format_help()

    assert "preprocess" in help_text
    assert "never downloads" in help_text


# --- Dispatch ------------------------------------------------------------------


def test_the_command_passes_both_overrides_to_the_factory_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = _RecordingFactory()

    exit_code = run(
        ["preprocess", "--chunk-size", "7", "--keep-interim"], _manager(tmp_path), factory
    )
    capsys.readouterr()

    assert exit_code == 0
    assert factory.calls == [(7, True)]


def test_the_defaults_reach_the_factory_as_none_and_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = _RecordingFactory()

    run(["preprocess"], _manager(tmp_path), factory)
    capsys.readouterr()

    assert factory.calls == [(None, False)]


def test_the_service_is_built_once_and_run_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = _RecordingFactory()

    preprocess_command(factory)
    capsys.readouterr()

    assert len(factory.calls) == 1
    assert factory.runs == 1


def test_a_successful_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = preprocess_command(_RecordingFactory())
    capsys.readouterr()

    assert exit_code == 0


def test_no_service_is_built_for_the_other_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Building one creates directories, so ``dataset-info`` must not trigger it."""
    run(["dataset-info"], _manager(tmp_path), _ExplodingFactory())
    capsys.readouterr()


def test_preprocessing_is_refused_when_no_factory_was_wired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller wiring only the dataset commands gets an explanation.

    Not a ``NoneType`` traceback.
    """
    with pytest.raises(SystemExit) as raised:
        run(["preprocess"], _manager(tmp_path))

    assert raised.value.code == 2
    assert "preprocessing is not available" in capsys.readouterr().err


def test_a_missing_dataset_propagates_as_the_domain_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The entry point turns it into an actionable message.

    The command does not swallow it and report success.
    """

    class _MissingDataset:
        def __call__(
            self, *, chunk_size: int | None, keep_interim: bool  # noqa: ARG002
        ) -> PreprocessingService:
            return cast("PreprocessingService", self)

        def run(self) -> PreprocessingReport:
            raise DataFileNotFoundError("records.csv")

    with pytest.raises(DataFileNotFoundError):
        preprocess_command(_MissingDataset())

    capsys.readouterr()


def test_the_banner_says_nothing_is_downloaded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    preprocess_command(_RecordingFactory())

    assert "Nothing is" in capsys.readouterr().out


# --- Report rendering ----------------------------------------------------------


def test_the_report_reconciles_rows_read_removed_and_written() -> None:
    """The ledger is the headline claim, so it is printed in full."""
    text = _render(_report())

    assert "read:         100" in text
    assert "removed:      3" in text
    assert "written:      97" in text
    assert "reconciles:   yes" in text


def test_a_broken_ledger_is_shouted_rather_than_quietly_shown() -> None:
    report = _report()
    broken = CleaningReport(
        rows_read=100, removals=(), rows_written=97, duplicate_rows_detected=0,
        duplicate_rows_removed=0,
    )
    text = _render(
        PreprocessingReport(
            version=report.version,
            source_file=report.source_file,
            output_directory=report.output_directory,
            fingerprint=report.fingerprint,
            cleaning=broken,
            splits=report.splits,
            target=report.target,
            feature_columns=report.feature_columns,
            identifier_columns=report.identifier_columns,
        )
    )

    assert "reconciles:   NO" in text


def test_only_the_rules_that_removed_something_are_listed() -> None:
    """A ledger of zeros would bury the line that matters."""
    text = _render(_report())

    assert "unparseable_timestamp" in text
    assert "negative_amount" not in text


def test_duplicates_are_reported_as_kept() -> None:
    """Rule 5: unusual rows are not silently deleted, and the report says so."""
    assert "2 (kept, not removed)" in _render(_report())


def test_every_split_is_reported_with_its_window_and_positives() -> None:
    text = _render(_report())

    for split in ("train", "validation", "test"):
        assert f"  {split}\n" in text
    assert "2020-01-01 00:00:00 .. 2020-01-06 00:00:00" in text
    assert "positives:  4" in text


def test_the_boundaries_and_tie_policy_are_printed() -> None:
    """Rule 10: the split is reproducible only if its boundaries are recorded."""
    text = _render(_report())

    assert "train <= 2020-01-06 00:00:00" in text
    assert "validation <= 2020-01-08 00:00:00" in text
    assert "whole_timestamp_to_earlier_split" in text


def test_chronology_is_reported_as_holding_when_it_does() -> None:
    assert "Chronology:     holds" in _render(_report())


def test_a_chronology_violation_is_reported_as_violated() -> None:
    """The check must be able to fail, or printing it proves nothing."""
    report = _report()
    text = _render(
        PreprocessingReport(
            version=report.version,
            source_file=report.source_file,
            output_directory=report.output_directory,
            fingerprint=report.fingerprint,
            cleaning=report.cleaning,
            splits=SplitReport(
                train_last_timestamp="2020-01-06 00:00:00",
                validation_last_timestamp="2020-01-08 00:00:00",
                tie_policy="whole_timestamp_to_earlier_split",
                summaries=(
                    SplitSummary(
                        SplitName.TRAIN,
                        10,
                        "2020-01-01 00:00:00",
                        "2020-01-09 00:00:00",
                        0,
                        10,
                    ),
                    SplitSummary(
                        SplitName.TEST,
                        10,
                        "2020-01-02 00:00:00",
                        "2020-01-10 00:00:00",
                        0,
                        10,
                    ),
                ),
            ),
            target=report.target,
            feature_columns=report.feature_columns,
            identifier_columns=report.identifier_columns,
        )
    )

    assert "Chronology:     VIOLATED" in text


def test_the_target_column_mapping_and_class_counts_are_printed() -> None:
    """Rule 9: the label is reported as found, never as assumed."""
    text = _render(_report())

    assert "Is Laundering -> is_laundering" in text
    assert "{'0': 0, '1': 1}" in text
    assert "class 0:      90" in text
    assert "class 1:      7" in text


def test_the_fingerprint_and_its_verification_are_printed() -> None:
    """Rule 13 is only credible if the check is visible in the output."""
    text = _render(_report())

    assert "0" * 64 in text
    assert "Raw unchanged:  yes" in text


def test_an_unverified_fingerprint_is_shouted() -> None:
    report = _report()
    text = _render(
        PreprocessingReport(
            version=report.version,
            source_file=report.source_file,
            output_directory=report.output_directory,
            fingerprint=DatasetFingerprint(
                file_name="synthetic_records.csv",
                size_bytes=4096,
                sha256="0" * 64,
                verified_unchanged=False,
            ),
            cleaning=report.cleaning,
            splits=report.splits,
            target=report.target,
            feature_columns=report.feature_columns,
            identifier_columns=report.identifier_columns,
        )
    )

    assert "Raw unchanged:  NO" in text


def test_feature_counts_distinguish_model_inputs_from_identifiers() -> None:
    text = _render(_report())

    assert "model inputs: 2" in text
    assert "identifiers:  2 (not model inputs)" in text
    assert "documented exclusions: 1" in text


def test_measured_performance_is_printed_when_it_was_measured() -> None:
    """Rule 16: reported, never estimated."""
    text = _render(_report())

    assert "total:        4.0s" in text
    assert "rows/s" in text
    assert "chunk size:   50,000 rows" in text
    assert "accounts:     42 tracked" in text
    assert "scan" in text


def test_the_performance_block_is_omitted_when_nothing_was_timed() -> None:
    """Absent measurements are absent, not zeros that look like measurements."""
    text = _render(_report(with_performance=False))

    assert "Performance" not in text


def test_the_report_names_the_files_it_wrote() -> None:
    text = _render(_report())

    for split in ("train", "validation", "test"):
        assert f"{split}/transactions.csv" in text
    assert "metadata/*.json" in text


def test_the_report_states_that_no_model_was_trained() -> None:
    """Rule 19: the output must not read as if a later phase had been done."""
    assert "No model was trained" in _render(_report())


def test_the_report_never_mentions_a_credential() -> None:
    text = _render(_report()).lower()

    for word in ("kaggle", "password", "token", "api_key", "secret"):
        assert word not in text
