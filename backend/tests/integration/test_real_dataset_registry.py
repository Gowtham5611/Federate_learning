"""Integration tests for the real, committed dataset registry.

These tests read ``datasets/dataset_registry.yaml`` from the repository — the
file the application actually ships — and assert that it describes the approved
IBM AML dataset and contains no credentials.

They still perform no network access and require no Kaggle credentials: a
registry is metadata, and reading metadata never triggers a download.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from application.services.dataset_manager import DatasetManager
from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.value_objects.dataset_descriptor import DatasetDescriptor
from infrastructure.data.sources.yaml_dataset_registry import (
    FORBIDDEN_CREDENTIAL_KEYS,
    YamlDatasetRegistry,
)
from infrastructure.utils.config_loader import load_config
from infrastructure.utils.paths import PROJECT_ROOT, resolve
from tests.fixtures.downloader_doubles import ExplodingDownloader

pytestmark = pytest.mark.integration

REGISTRY_RELATIVE_PATH = "datasets/dataset_registry.yaml"
EXPECTED_DATASET_ID = "ibm_transactions_aml"
EXPECTED_NAME = "IBM Transactions for Anti-Money Laundering (AML)"
EXPECTED_KAGGLE_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
EXPECTED_URL = f"https://www.kaggle.com/datasets/{EXPECTED_KAGGLE_ID}"
EXPECTED_VARIANT = "HI-Small"
EXPECTED_FILE = "HI-Small_Trans.csv"
EXPECTED_LOCAL_PATH = Path("data/raw/ibm_aml")

FORBIDDEN_DATASET_SUBSTITUTES = ("amlsim", "creditcard", "credit-card", "credit_card", "paysim")


@pytest.fixture(scope="module")
def registry_path() -> Path:
    """Absolute path to the committed registry."""
    path = resolve(REGISTRY_RELATIVE_PATH)
    assert path.is_file(), f"{REGISTRY_RELATIVE_PATH} is missing from the repository"
    return path


@pytest.fixture(scope="module")
def descriptor(registry_path: Path) -> DatasetDescriptor:
    """The primary descriptor as loaded from the committed registry."""
    return YamlDatasetRegistry(registry_path).primary()


@pytest.fixture(scope="module")
def raw_registry(registry_path: Path) -> dict[str, Any]:
    """The registry as raw YAML, for structural assertions."""
    loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# --- The registry describes the approved dataset -----------------------------


def test_primary_dataset_id(descriptor: DatasetDescriptor) -> None:
    assert descriptor.dataset_id == EXPECTED_DATASET_ID


def test_dataset_name(descriptor: DatasetDescriptor) -> None:
    assert descriptor.name == EXPECTED_NAME
    assert descriptor.provider == "IBM"


def test_source_is_kaggle(descriptor: DatasetDescriptor) -> None:
    assert descriptor.source is DatasetSourceKind.KAGGLE


def test_source_url(descriptor: DatasetDescriptor) -> None:
    assert descriptor.source_url == EXPECTED_URL


def test_kaggle_dataset_id(descriptor: DatasetDescriptor) -> None:
    assert descriptor.remote_id == EXPECTED_KAGGLE_ID


def test_selected_variant(descriptor: DatasetDescriptor) -> None:
    assert descriptor.selected_variant == EXPECTED_VARIANT
    assert descriptor.selected_variant in descriptor.supported_variants


def test_expected_files(descriptor: DatasetDescriptor) -> None:
    assert descriptor.expected_files == (EXPECTED_FILE,)


def test_declared_status_is_configured_not_downloaded(descriptor: DatasetDescriptor) -> None:
    assert descriptor.status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED


def test_local_path(descriptor: DatasetDescriptor) -> None:
    assert descriptor.local_path == EXPECTED_LOCAL_PATH


def test_no_substitute_dataset_is_declared(
    raw_registry: dict[str, Any], descriptor: DatasetDescriptor
) -> None:
    """Only the approved dataset may be *declared*.

    Checks identity fields rather than the whole document: the entry's prose
    notes legitimately name the forbidden substitutes in order to prohibit them.
    """
    assert tuple(raw_registry["datasets"]) == (EXPECTED_DATASET_ID,)
    identity = " ".join(
        [descriptor.dataset_id, descriptor.name, descriptor.remote_id, descriptor.source_url]
    ).lower()
    for substitute in FORBIDDEN_DATASET_SUBSTITUTES:
        assert substitute not in identity, f"unapproved dataset reference: {substitute}"


def test_every_declared_status_definition_is_a_known_status(raw_registry: dict[str, Any]) -> None:
    declared = set(raw_registry["status_definitions"])
    assert declared == {status.value for status in DatasetStatus}


def test_local_metadata_is_unpopulated(raw_registry: dict[str, Any]) -> None:
    metadata = raw_registry["datasets"][EXPECTED_DATASET_ID]["local_metadata"]
    assert all(value is None for value in metadata.values()), (
        "local_metadata must stay null until a real download and validation run"
    )


# --- The registry contains no credentials ------------------------------------


def test_registry_contains_no_credential_keys(raw_registry: dict[str, Any]) -> None:
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).strip().lower() in FORBIDDEN_CREDENTIAL_KEYS:
                    found.append(str(key))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw_registry)
    assert found == []


def test_registry_text_contains_no_credential_assignment(registry_path: Path) -> None:
    text = registry_path.read_text(encoding="utf-8").lower()
    # The word may appear in a prose comment; an assignment must not exist.
    for key in ("kaggle_key:", "kaggle_username:", "api_token:", "password:", "secret:"):
        assert key not in text, f"registry appears to assign {key}"


# --- Configuration and registry agree ----------------------------------------


def test_active_dataset_id_in_config_exists_in_registry(registry_path: Path) -> None:
    active_id = load_config("base")["dataset"]["active_dataset_id"]
    assert active_id in YamlDatasetRegistry(registry_path).dataset_ids()


# --- The real registry does not imply a local dataset ------------------------


def test_real_registry_does_not_claim_a_local_dataset(registry_path: Path) -> None:
    """A committed registry must never make the manager report data it lacks."""
    manager = DatasetManager(
        registry=YamlDatasetRegistry(registry_path),
        downloader=ExplodingDownloader(),
        project_root=PROJECT_ROOT,
    )
    info = manager.get_dataset_info()

    if info.is_available:
        pytest.skip("the real dataset is present on this machine")

    assert info.effective_status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED
    assert info.download_performed is False
    assert info.missing_files == (EXPECTED_FILE,)


def test_no_dataset_csv_can_be_committed_to_the_repository() -> None:
    """Dataset files must be un-committable, whether or not they exist locally.

    The dataset is legitimately present on a machine that has run
    ``dataset-download``, so its *presence* proves nothing either way. What must
    hold unconditionally is that Git cannot pick it up: the ignore rules have to
    cover ``data/raw`` and the CSV extension.
    """
    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in ignore_rules]

    assert "data/raw/*" in stripped, "data/raw must be git-ignored"
    assert "*.csv" in stripped, "CSV files must be git-ignored repository-wide"

    raw_dir = resolve("data/raw")
    if not raw_dir.is_dir():
        pytest.skip("data/raw does not exist")

    # If Git is available, let it settle the question rather than inferring it
    # from the rules above.
    for csv_path in raw_dir.rglob("*.csv"):
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", str(csv_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 128:
            pytest.skip("not a Git repository; the rule assertions above still applied")
        assert completed.returncode == 0, f"{csv_path.name} is NOT git-ignored"
