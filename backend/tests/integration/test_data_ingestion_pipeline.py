"""Integration tests for the data ingestion pipeline.

Exercises the real components together, in the order the project's data flows:

    Dataset Registry -> IBM AML Data Source -> Memory-Safe CSV Loader ->
    Schema Validator -> Validated Chunks

Only the *wiring* is under test. The real 530 MB dataset is neither downloaded
nor required, so where a file must exist on disk these tests write a small
synthetic stand-in whose columns are invented (see
``tests/fixtures/tabular_fixtures.py``). Nothing here describes the IBM AML
data: its schema is not established in this repository, and this suite does not
pretend otherwise.

For the stages that need a local file, the registry entry keeps the real dataset
*identity* - the data source refuses any other, by design - while its
``expected_files`` names the synthetic stand-in. That combination is what makes
the wiring testable offline; this suite never creates a file named
``HI-Small_Trans.csv`` and never writes into ``data/raw``, and a test at the end
of this module asserts that no fabricated file appears there.

No network access and no Kaggle credentials are used. An
:class:`ExplodingDownloader` is injected everywhere, so any attempted transfer
fails the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.dto.validation import ChunkValidationResult
from application.services.dataset_ingestion_service import DatasetIngestionService
from application.services.dataset_manager import DatasetManager
from domain.enums.validation_issue_kind import ValidationIssueKind
from domain.exceptions.data import DataFileNotFoundError, SchemaValidationError
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from infrastructure.data.sources.ibm_aml_data_source import (
    IBM_AML_DATASET_ID,
    IBM_AML_REMOTE_ID,
    IBMAMLDataSource,
)
from infrastructure.data.sources.yaml_dataset_registry import YamlDatasetRegistry
from infrastructure.data.validators.dataframe_schema_validator import DataFrameSchemaValidator
from infrastructure.utils.paths import PROJECT_ROOT, resolve
from tests.fixtures.dataset_registry_doubles import InMemoryDatasetRegistry, make_descriptor
from tests.fixtures.downloader_doubles import ExplodingDownloader
from tests.fixtures.tabular_fixtures import (
    SYNTHETIC_COLUMNS,
    SYNTHETIC_HEADER,
    make_synthetic_schema,
    synthetic_row,
    write_csv,
    write_synthetic_csv,
)

pytestmark = pytest.mark.integration

REGISTRY_PATH = "datasets/dataset_registry.yaml"
REAL_FILE = "HI-Small_Trans.csv"

STAND_IN_FILENAME = "synthetic_records.csv"
"""Name of the invented stand-in file. Never the real dataset's name."""

STAND_IN_VARIANT = "TEST-Small-SYNTHETIC"
"""Variant label chosen so it cannot be mistaken for a real IBM AML variant."""

DATASET_SUBDIRECTORY = Path("data/raw/ibm_aml")


def stand_in_source(tmp_path: Path, *, filename: str = STAND_IN_FILENAME) -> IBMAMLDataSource:
    """A real data source pointed at a synthetic local file inside ``tmp_path``.

    The descriptor keeps the approved dataset's identity because the source
    verifies it; the file it names is the invented stand-in.
    """
    descriptor = make_descriptor(
        selected_variant=STAND_IN_VARIANT,
        supported_variants=(STAND_IN_VARIANT,),
        expected_files=(filename,),
    )
    manager = DatasetManager(
        registry=InMemoryDatasetRegistry((descriptor,)),
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
    )
    return IBMAMLDataSource(manager)


def dataset_dir(tmp_path: Path) -> Path:
    """The directory the stand-in source will look in."""
    path = tmp_path / DATASET_SUBDIRECTORY
    path.mkdir(parents=True, exist_ok=True)
    return path


def corrupt_unit(row: str) -> str:
    """Put a value outside the synthetic closed vocabulary into one row."""
    fields = row.split(",")
    fields[SYNTHETIC_COLUMNS.index("unit")] = "CCC"
    return ",".join(fields)


# --- 27. Registry -> Data Source ------------------------------------------------


@pytest.fixture(scope="module")
def real_registry_source() -> IBMAMLDataSource:
    """A data source over the registry file the application actually ships."""
    path = resolve(REGISTRY_PATH)
    assert path.is_file(), f"{REGISTRY_PATH} is missing from the repository"
    manager = DatasetManager(
        registry=YamlDatasetRegistry(path),
        downloader=ExplodingDownloader(),
        project_root=PROJECT_ROOT,
    )
    return IBMAMLDataSource(manager)


def test_the_committed_registry_drives_the_data_source(
    real_registry_source: IBMAMLDataSource,
) -> None:
    """Every dataset fact the source reports comes from the shipped registry."""
    assert real_registry_source.dataset_id == IBM_AML_DATASET_ID
    assert real_registry_source.remote_id == IBM_AML_REMOTE_ID
    assert real_registry_source.variant == "HI-Small"
    assert real_registry_source.expected_filename == REAL_FILE


def test_the_data_source_resolves_the_real_datasets_location(
    real_registry_source: IBMAMLDataSource,
) -> None:
    directory = real_registry_source.local_directory

    assert directory.is_absolute()
    assert directory == (PROJECT_ROOT / DATASET_SUBDIRECTORY).resolve()


def test_the_real_dataset_is_reported_as_absent_rather_than_invented(
    real_registry_source: IBMAMLDataSource,
) -> None:
    """The expected state of a fresh checkout: absent, and clearly so."""
    if real_registry_source.local_directory.joinpath(REAL_FILE).is_file():
        pytest.skip("the real dataset is present on this machine")

    with pytest.raises(DataFileNotFoundError) as raised:
        real_registry_source.resolve_file()

    assert REAL_FILE in str(raised.value)
    assert "KAGGLE_USERNAME" in str(raised.value), "the error must say how to obtain it"


def test_reading_dataset_metadata_never_downloads_anything(
    real_registry_source: IBMAMLDataSource,
) -> None:
    """``ExplodingDownloader`` fails on any transfer, so completing is the proof."""
    _ = (
        real_registry_source.dataset_id,
        real_registry_source.remote_id,
        real_registry_source.variant,
        real_registry_source.expected_filename,
        real_registry_source.local_directory,
    )


# --- 28. Data Source -> Loader ---------------------------------------------------


def test_the_source_resolves_a_file_the_loader_can_stream(tmp_path: Path) -> None:
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=6)
    source = stand_in_source(tmp_path)

    chunks = list(CsvChunkLoader().stream_chunks(source.resolve_file(), chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 2, 2]
    assert tuple(chunks[0].columns) == SYNTHETIC_COLUMNS


def test_the_source_and_loader_agree_on_the_file(tmp_path: Path) -> None:
    written = write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=3)
    source = stand_in_source(tmp_path)

    resolved = source.resolve_file()

    assert resolved == written
    assert resolved.name == source.expected_filename


def test_a_file_extracted_into_a_subdirectory_is_still_streamed(tmp_path: Path) -> None:
    """The archive layout the Kaggle distribution may produce."""
    nested = dataset_dir(tmp_path) / "unpacked"
    nested.mkdir()
    write_synthetic_csv(nested / STAND_IN_FILENAME, rows=4)
    source = stand_in_source(tmp_path)

    chunks = list(CsvChunkLoader().stream_chunks(source.resolve_file(), chunk_size=4))

    assert len(chunks) == 1
    assert len(chunks[0]) == 4


def test_an_absent_file_stops_the_pipeline_at_the_source(tmp_path: Path) -> None:
    """The loader is never handed a path that does not exist."""
    dataset_dir(tmp_path)

    with pytest.raises(DataFileNotFoundError):
        stand_in_source(tmp_path).resolve_file()


# --- 29. Loader -> Validator ------------------------------------------------------


def stream_results(path: Path, *, chunk_size: int = 2) -> list[ChunkValidationResult]:
    """Validate every chunk of a file, returning the results in order."""
    validator = DataFrameSchemaValidator(make_synthetic_schema())
    return [
        validator.validate_chunk(chunk, chunk_index=index, source=path.name)
        for index, chunk in enumerate(CsvChunkLoader().stream_chunks(path, chunk_size))
    ]


def test_every_chunk_of_a_conforming_file_validates(tmp_path: Path) -> None:
    path = write_synthetic_csv(tmp_path / STAND_IN_FILENAME, rows=6)

    results = stream_results(path)

    assert len(results) == 3
    assert all(result.is_valid for result in results)
    assert [result.chunk_index for result in results] == [0, 1, 2]


def test_the_validator_accepts_whatever_dtypes_the_parser_inferred(tmp_path: Path) -> None:
    """The loader infers dtypes per chunk, so the validator must not rely on them."""
    path = write_synthetic_csv(tmp_path / STAND_IN_FILENAME, rows=4)

    chunk = next(CsvChunkLoader().stream_chunks(path, chunk_size=4))
    result = DataFrameSchemaValidator(make_synthetic_schema()).validate_chunk(
        chunk, chunk_index=0, source=path.name
    )

    inferred = dict(chunk.dtypes.astype(str))
    assert "int" in inferred["record_id"], "the parser did infer types of its own"
    assert "float" in inferred["value"]
    assert result.is_valid


def test_a_bad_value_is_reported_against_the_chunk_that_contained_it(
    tmp_path: Path,
) -> None:
    lines = [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(6))]
    lines[5] = corrupt_unit(lines[5])
    path = write_csv(tmp_path / STAND_IN_FILENAME, lines)

    results = stream_results(path, chunk_size=2)

    assert [result.is_valid for result in results] == [True, True, False]
    failing = results[2]
    assert failing.errors[0].kind is ValidationIssueKind.VALUE_NOT_ALLOWED
    assert failing.errors[0].example_rows == (4,), "the row's position in the file"


def test_validation_leaves_the_loaders_chunk_untouched(tmp_path: Path) -> None:
    """The chunk a caller receives is the parsed data, not a coerced copy."""
    lines = [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(4))]
    lines[2] = corrupt_unit(lines[2])
    path = write_csv(tmp_path / STAND_IN_FILENAME, lines)

    chunk = next(CsvChunkLoader().stream_chunks(path, chunk_size=4))
    before = chunk.copy(deep=True)
    result = DataFrameSchemaValidator(make_synthetic_schema()).validate_chunk(
        chunk, chunk_index=0, source=path.name
    )

    assert not result.is_valid
    assert chunk.equals(before)
    assert list(chunk.dtypes.astype(str)) == list(before.dtypes.astype(str))


def test_a_header_only_file_produces_no_chunks_to_validate(tmp_path: Path) -> None:
    path = write_csv(tmp_path / STAND_IN_FILENAME, [SYNTHETIC_HEADER])

    assert stream_results(path) == []


# --- 30. The whole pipeline -------------------------------------------------------


def build_pipeline(
    tmp_path: Path, *, chunk_size: int = 2, strict: bool = True
) -> DatasetIngestionService:
    """Compose the real source, loader and validator over the stand-in file."""
    return DatasetIngestionService(
        stand_in_source(tmp_path),
        CsvChunkLoader(),
        DataFrameSchemaValidator(make_synthetic_schema()),
        chunk_size=chunk_size,
        strict=strict,
    )


def test_the_pipeline_streams_validated_chunks_end_to_end(tmp_path: Path) -> None:
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=7)

    pairs = list(build_pipeline(tmp_path, chunk_size=3).stream_validated_chunks())

    assert [len(chunk) for chunk, _ in pairs] == [3, 3, 1]
    assert all(result.is_valid for _, result in pairs)
    assert [result.source for _, result in pairs] == [STAND_IN_FILENAME] * 3
    assert [result.chunk_index for _, result in pairs] == [0, 1, 2]


def test_the_pipeline_covers_every_row_exactly_once(tmp_path: Path) -> None:
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=10)

    pairs = list(build_pipeline(tmp_path, chunk_size=3).stream_validated_chunks())

    identifiers = [value for chunk, _ in pairs for value in chunk["record_id"].tolist()]

    assert identifiers == list(range(1000, 1010))
    assert sum(result.row_count for _, result in pairs) == 10


def test_the_pipeline_reads_one_chunk_at_a_time(tmp_path: Path) -> None:
    """Nothing between the file and the caller accumulates."""
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=200)

    chunk, result = next(build_pipeline(tmp_path, chunk_size=5).stream_validated_chunks())

    assert len(chunk) == 5
    assert result.row_count == 5


def test_the_pipeline_aborts_on_a_chunk_that_fails_validation(tmp_path: Path) -> None:
    lines = [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(6))]
    lines[3] = corrupt_unit(lines[3])
    write_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, lines)

    stream = build_pipeline(tmp_path, chunk_size=2).stream_validated_chunks()

    assert next(stream)[1].is_valid, "the first chunk was clean"
    with pytest.raises(SchemaValidationError) as raised:
        next(stream)

    assert raised.value.chunk_index == 1
    assert raised.value.source == STAND_IN_FILENAME
    assert "unit" in str(raised.value)


def test_a_non_strict_pipeline_reports_and_continues(tmp_path: Path) -> None:
    lines = [SYNTHETIC_HEADER, *(synthetic_row(i) for i in range(6))]
    lines[3] = corrupt_unit(lines[3])
    write_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, lines)

    pairs = list(
        build_pipeline(tmp_path, chunk_size=2, strict=False).stream_validated_chunks()
    )

    assert [result.is_valid for _, result in pairs] == [True, False, True]


def test_the_pipeline_fails_before_reading_when_the_dataset_is_absent(
    tmp_path: Path,
) -> None:
    dataset_dir(tmp_path)

    with pytest.raises(DataFileNotFoundError, match="KAGGLE_USERNAME"):
        build_pipeline(tmp_path).stream_validated_chunks()


def test_a_malformed_file_surfaces_as_a_project_error(tmp_path: Path) -> None:
    """Pandas' parser exceptions must not escape the infrastructure boundary."""
    write_csv(
        dataset_dir(tmp_path) / STAND_IN_FILENAME,
        [SYNTHETIC_HEADER, synthetic_row(0), '1,"unclosed'],
    )
    stream = build_pipeline(tmp_path, chunk_size=2).stream_validated_chunks()

    with pytest.raises(Exception) as raised:  # noqa: B017 - the type is the assertion
        list(stream)

    assert "pandas" not in type(raised.value).__module__


# --- The pipeline needs no dataset, no network and no credentials -----------------


def test_the_pipeline_runs_with_no_kaggle_credentials_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(variable, raising=False)
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=4)

    pairs = list(build_pipeline(tmp_path, chunk_size=2).stream_validated_chunks())

    assert len(pairs) == 2
    assert all(result.is_valid for _, result in pairs)


def test_no_file_named_after_the_real_dataset_is_ever_created(tmp_path: Path) -> None:
    """The fixture files must never be mistakable for the real download."""
    write_synthetic_csv(dataset_dir(tmp_path) / STAND_IN_FILENAME, rows=4)
    list(build_pipeline(tmp_path, chunk_size=2).stream_validated_chunks())

    assert list(tmp_path.rglob("HI-Small*")) == []
    assert [path.name for path in tmp_path.rglob("*.csv")] == [STAND_IN_FILENAME]


def test_the_repository_still_contains_no_fabricated_dataset_files() -> None:
    """This suite adds fixtures, not data: it must never write into ``data/raw``.

    A real ``HI-Small_Trans.csv`` may legitimately be present — an explicit
    ``dataset-download`` puts it there. What must never appear is a file this
    test suite invented: the stand-ins live in ``tmp_path`` and are named
    ``synthetic_records.csv``, never the real dataset's name. Anything else
    under ``data/raw`` means a test leaked, or a stand-in was passed off as the
    real thing.
    """
    raw = resolve("data/raw")
    if not raw.is_dir():
        pytest.skip("data/raw does not exist")

    permitted = {REAL_FILE}
    stray = [
        path.name
        for path in raw.rglob("*")
        if path.is_file() and path.name not in permitted and path.name != ".gitkeep"
    ]

    assert stray == [], (
        f"unexpected files under data/raw: {stray}. Only the real "
        f"{REAL_FILE} belongs there; synthetic stand-ins must stay in tmp_path"
    )
    assert STAND_IN_FILENAME not in stray, "a synthetic stand-in leaked into data/raw"


def test_the_synthetic_fixture_does_not_resemble_the_real_dataset() -> None:
    """Guards the honesty of every test above: these columns are invented."""
    header = SYNTHETIC_HEADER.lower()

    for real_sounding in ("amount", "bank", "currency", "payment", "laundering"):
        assert real_sounding not in header, (
            f"fixture column name '{real_sounding}' could be mistaken for real "
            f"IBM AML data, whose schema this repository does not know"
        )
