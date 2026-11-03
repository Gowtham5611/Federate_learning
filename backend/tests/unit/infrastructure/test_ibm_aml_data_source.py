"""Tests for the IBM AML data source.

The source's job is narrow: say which dataset is being read and where its
transaction file is. These tests pin that identity, prove the source refuses to
read any other dataset, and prove it never downloads anything - an
:class:`ExplodingDownloader` is injected throughout, so any transfer attempt
fails the test.

No Kaggle credentials, no network and no real dataset are needed. Where a local
file is required, a small synthetic stand-in is written into ``tmp_path``; no
file is ever named after the real dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.ports.output.dataset_registry import DatasetRegistryPort
from application.services.dataset_manager import DatasetManager
from domain.exceptions.data import DataFileNotFoundError
from domain.exceptions.dataset import DatasetMismatchError, InvalidDatasetDescriptorError
from domain.value_objects.dataset_descriptor import DatasetDescriptor
from infrastructure.data.sources.ibm_aml_data_source import (
    IBM_AML_DATASET_ID,
    IBM_AML_REMOTE_ID,
    IBMAMLDataSource,
)
from tests.fixtures.dataset_registry_doubles import InMemoryDatasetRegistry, make_descriptor
from tests.fixtures.downloader_doubles import ExplodingDownloader

pytestmark = pytest.mark.unit

EXPECTED_DATASET_ID = "ibm_transactions_aml"
EXPECTED_KAGGLE_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
EXPECTED_VARIANT = "HI-Small"
EXPECTED_FILE = "HI-Small_Trans.csv"
EXPECTED_RELATIVE_DIR = Path("data/raw/ibm_aml")

SYNTHETIC_FILENAME = "synthetic_records.csv"
"""Stand-in file name. Deliberately not the real one, which is never created."""


class CountingRegistry(DatasetRegistryPort):
    """Wraps a registry and counts how often it is consulted."""

    def __init__(self, inner: DatasetRegistryPort) -> None:
        """Store the registry to delegate to."""
        self._inner = inner
        self.reads = 0

    def get(self, dataset_id: str) -> DatasetDescriptor:
        """Count and delegate."""
        self.reads += 1
        return self._inner.get(dataset_id)

    def primary(self) -> DatasetDescriptor:
        """Count and delegate."""
        self.reads += 1
        return self._inner.primary()

    def dataset_ids(self) -> tuple[str, ...]:
        """Delegate without counting: not a descriptor read."""
        return self._inner.dataset_ids()


def make_source(
    tmp_path: Path,
    *,
    descriptor: DatasetDescriptor | None = None,
    registry: DatasetRegistryPort | None = None,
) -> IBMAMLDataSource:
    """Build a source over an in-memory registry rooted at ``tmp_path``."""
    resolved = registry or InMemoryDatasetRegistry((descriptor or make_descriptor(),))
    manager = DatasetManager(
        registry=resolved,
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
    )
    return IBMAMLDataSource(manager)


def write_stand_in(tmp_path: Path, filename: str, *, subdirectory: str = "") -> Path:
    """Write a non-empty placeholder where the dataset directory would be."""
    directory = tmp_path / EXPECTED_RELATIVE_DIR / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text("placeholder,for,test\n1,2,3\n", encoding="utf-8")
    return path


# --- 1. The dataset id is the configured one ---------------------------------


def test_reports_the_configured_dataset_id(tmp_path: Path) -> None:
    assert make_source(tmp_path).dataset_id == EXPECTED_DATASET_ID


def test_the_modules_dataset_id_constant_matches_the_registry_entry(tmp_path: Path) -> None:
    """The constant and the registry must not be allowed to drift apart."""
    assert IBM_AML_DATASET_ID == EXPECTED_DATASET_ID
    assert make_source(tmp_path).dataset_id == IBM_AML_DATASET_ID


# --- 2. The dataset really is IBM Transactions for AML -----------------------


def test_reports_the_ibm_aml_kaggle_dataset(tmp_path: Path) -> None:
    source = make_source(tmp_path)

    assert source.remote_id == EXPECTED_KAGGLE_ID
    assert source.remote_id == IBM_AML_REMOTE_ID


def test_a_different_dataset_is_refused(tmp_path: Path) -> None:
    """Reading another dataset here would label its results as IBM AML."""
    other = make_descriptor(
        dataset_id="some_other_dataset",
        name="Some Other Dataset",
        remote_id="someone/some-other-dataset",
        source_url="https://www.kaggle.com/datasets/someone/some-other-dataset",
    )
    source = make_source(tmp_path, descriptor=other)

    with pytest.raises(DatasetMismatchError) as raised:
        _ = source.dataset_id

    assert raised.value.expected_remote_id == IBM_AML_REMOTE_ID
    assert raised.value.actual_remote_id == "someone/some-other-dataset"
    assert "IBMAMLDataSource" in str(raised.value)


@pytest.mark.parametrize(
    "substitute",
    [
        "someone/ibm-amlsim-example-data",
        "mlg-ulb/creditcardfraud",
        "someone/paysim1",
    ],
)
def test_named_substitute_datasets_are_refused(tmp_path: Path, substitute: str) -> None:
    """The three datasets the project explicitly forbids as replacements."""
    descriptor = make_descriptor(
        remote_id=substitute, source_url=f"https://www.kaggle.com/datasets/{substitute}"
    )

    with pytest.raises(DatasetMismatchError):
        _ = make_source(tmp_path, descriptor=descriptor).variant


# --- 3. The variant is HI-Small ----------------------------------------------


def test_reports_the_selected_variant(tmp_path: Path) -> None:
    assert make_source(tmp_path).variant == EXPECTED_VARIANT


def test_the_variant_comes_from_the_registry_rather_than_being_hardcoded(
    tmp_path: Path,
) -> None:
    """The registry stays the single source of truth for which variant is active."""
    descriptor = make_descriptor(
        selected_variant="HI-Medium", supported_variants=("HI-Small", "HI-Medium")
    )

    assert make_source(tmp_path, descriptor=descriptor).variant == "HI-Medium"


# --- 4. The expected file name -----------------------------------------------


def test_reports_the_expected_transaction_filename(tmp_path: Path) -> None:
    assert make_source(tmp_path).expected_filename == EXPECTED_FILE


def test_a_variant_declaring_no_csv_is_rejected(tmp_path: Path) -> None:
    descriptor = make_descriptor(expected_files=("readme.txt",))

    with pytest.raises(InvalidDatasetDescriptorError, match="exactly one"):
        _ = make_source(tmp_path, descriptor=descriptor).expected_filename


def test_a_variant_declaring_several_csvs_is_rejected(tmp_path: Path) -> None:
    """Which of two CSVs holds the transactions is not the source's guess to make."""
    descriptor = make_descriptor(expected_files=("one.csv", "two.csv"))

    with pytest.raises(InvalidDatasetDescriptorError, match="found 2"):
        _ = make_source(tmp_path, descriptor=descriptor).expected_filename


def test_the_csv_suffix_is_matched_case_insensitively(tmp_path: Path) -> None:
    descriptor = make_descriptor(expected_files=("Records.CSV",))

    assert make_source(tmp_path, descriptor=descriptor).expected_filename == "Records.CSV"


# --- 5. The local path -------------------------------------------------------


def test_reports_the_absolute_local_directory(tmp_path: Path) -> None:
    source = make_source(tmp_path)

    expected = (tmp_path / EXPECTED_RELATIVE_DIR).resolve()

    assert source.local_directory == expected
    assert source.local_directory.is_absolute()


def test_the_local_directory_is_reported_even_when_it_does_not_exist(
    tmp_path: Path,
) -> None:
    """Where the data belongs is a configuration fact, not an observation."""
    source = make_source(tmp_path)

    assert not source.local_directory.exists()
    assert source.local_directory.name == "ibm_aml"


def test_resolve_returns_the_file_inside_the_local_directory(tmp_path: Path) -> None:
    descriptor = make_descriptor(expected_files=(SYNTHETIC_FILENAME,))
    written = write_stand_in(tmp_path, SYNTHETIC_FILENAME)
    source = make_source(tmp_path, descriptor=descriptor)

    resolved = source.resolve_file()

    assert resolved == written
    assert resolved.parent == source.local_directory


def test_a_file_extracted_into_a_nested_directory_is_still_found(tmp_path: Path) -> None:
    """A Kaggle archive may unpack into a subdirectory; that is not a failure."""
    descriptor = make_descriptor(expected_files=(SYNTHETIC_FILENAME,))
    written = write_stand_in(tmp_path, SYNTHETIC_FILENAME, subdirectory="unpacked")
    source = make_source(tmp_path, descriptor=descriptor)

    assert source.resolve_file() == written


# --- 6. A missing local file -------------------------------------------------


def test_a_missing_file_is_reported_with_the_path_it_was_expected_at(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)

    with pytest.raises(DataFileNotFoundError) as raised:
        source.resolve_file()

    assert raised.value.path == source.local_directory / EXPECTED_FILE


def test_a_missing_file_is_reported_with_acquisition_instructions(tmp_path: Path) -> None:
    """On a fresh checkout this is the normal state, so the error must be a guide."""
    with pytest.raises(DataFileNotFoundError) as raised:
        make_source(tmp_path).resolve_file()

    message = str(raised.value)

    assert "KAGGLE_USERNAME" in message
    assert "KAGGLE_KEY" in message
    assert ".env.example" in message


def test_a_missing_file_is_never_substituted_with_a_generated_one(tmp_path: Path) -> None:
    source = make_source(tmp_path)

    with pytest.raises(DataFileNotFoundError):
        source.resolve_file()

    assert list(tmp_path.rglob("*.csv")) == [], "the source must not create data"


def test_an_empty_file_counts_as_missing(tmp_path: Path) -> None:
    """A zero-byte stub is not a dataset."""
    descriptor = make_descriptor(expected_files=(SYNTHETIC_FILENAME,))
    directory = tmp_path / EXPECTED_RELATIVE_DIR
    directory.mkdir(parents=True)
    (directory / SYNTHETIC_FILENAME).touch()

    with pytest.raises(DataFileNotFoundError):
        make_source(tmp_path, descriptor=descriptor).resolve_file()


# --- The source stays inside its responsibilities ----------------------------


def test_construction_reads_nothing(tmp_path: Path) -> None:
    """Building the source must not consult the registry or the filesystem."""
    registry = CountingRegistry(InMemoryDatasetRegistry())
    manager = DatasetManager(
        registry=registry, downloader=ExplodingDownloader(), project_root=tmp_path
    )

    IBMAMLDataSource(manager)

    assert registry.reads == 0


def test_the_registry_is_read_once_however_many_properties_are_used(
    tmp_path: Path,
) -> None:
    registry = CountingRegistry(InMemoryDatasetRegistry())
    source = make_source(tmp_path, registry=registry)

    for _ in range(2):
        _ = (source.dataset_id, source.remote_id, source.variant, source.expected_filename)

    assert registry.reads == 1, "the descriptor is cached by the manager"


def test_no_property_or_lookup_triggers_a_download(tmp_path: Path) -> None:
    """The injected downloader raises if called, so reaching the end is the assertion."""
    source = make_source(tmp_path)

    _ = (
        source.dataset_id,
        source.remote_id,
        source.variant,
        source.expected_filename,
        source.local_directory,
    )
    with pytest.raises(DataFileNotFoundError):
        source.resolve_file()


def test_the_source_holds_no_credentials(tmp_path: Path) -> None:
    """Authentication belongs to the downloader chain, not to the data source."""
    source = make_source(tmp_path)

    attributes = set(dir(source)) | set(vars(source))
    for forbidden in ("username", "key", "token", "password", "secret", "credential"):
        assert not any(forbidden in name.lower() for name in attributes), (
            f"the data source exposes a credential-like attribute: {forbidden}"
        )


def test_the_source_exposes_no_parsing_or_training_behaviour(tmp_path: Path) -> None:
    """Its surface is identity plus location; anything else is another layer's job."""
    public = {name for name in dir(make_source(tmp_path)) if not name.startswith("_")}

    assert public == {
        "dataset_id",
        "remote_id",
        "variant",
        "expected_filename",
        "local_directory",
        "resolve_file",
    }
