"""Tests for the YAML dataset registry adapter."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.dataset import (
    DatasetRegistryNotFoundError,
    InvalidDatasetRegistryError,
    UnknownDatasetError,
)
from infrastructure.data.sources.yaml_dataset_registry import YamlDatasetRegistry

pytestmark = pytest.mark.unit

VALID_REGISTRY = """
primary_dataset_id: ibm_transactions_aml
datasets:
  ibm_transactions_aml:
    dataset_id: ibm_transactions_aml
    name: IBM Transactions for Anti-Money Laundering (AML)
    provider: IBM
    source: Kaggle
    source_url: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
    kaggle_dataset_id: ealtman2019/ibm-transactions-for-anti-money-laundering-aml
    purpose: Anti-Money Laundering transaction detection
    status: configured_not_downloaded
    local_path: data/raw/ibm_aml/
    selected_variant: HI-Small
    supported_variants:
      - HI-Small
    expected_files:
      - HI-Small_Trans.csv
    intended_usage: Primary training and evaluation data.
"""


def write_registry(tmp_path: Path, content: str) -> Path:
    """Write registry YAML to a temporary file."""
    path = tmp_path / "dataset_registry.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_loads_valid_registry(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(write_registry(tmp_path, VALID_REGISTRY))
    descriptor = registry.primary()
    assert descriptor.dataset_id == "ibm_transactions_aml"
    assert descriptor.source is DatasetSourceKind.KAGGLE
    assert descriptor.status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED
    assert descriptor.remote_id == "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    assert descriptor.selected_variant == "HI-Small"
    assert descriptor.expected_files == ("HI-Small_Trans.csv",)
    assert descriptor.local_path == Path("data/raw/ibm_aml")


def test_get_by_id(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(write_registry(tmp_path, VALID_REGISTRY))
    assert registry.get("ibm_transactions_aml").name.startswith("IBM Transactions")


def test_dataset_ids(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(write_registry(tmp_path, VALID_REGISTRY))
    assert registry.dataset_ids() == ("ibm_transactions_aml",)


def test_missing_registry_file_raises(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(tmp_path / "absent.yaml")
    with pytest.raises(DatasetRegistryNotFoundError, match="not found"):
        registry.primary()


def test_unknown_dataset_raises_with_known_ids(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(write_registry(tmp_path, VALID_REGISTRY))
    with pytest.raises(UnknownDatasetError, match="ibm_transactions_aml"):
        registry.get("credit_card_fraud")


def test_registry_without_datasets_section_raises(tmp_path: Path) -> None:
    registry = YamlDatasetRegistry(write_registry(tmp_path, "primary_dataset_id: x\n"))
    with pytest.raises(InvalidDatasetRegistryError, match="non-empty mapping"):
        registry.primary()


def test_registry_without_primary_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("primary_dataset_id: ibm_transactions_aml\n", "")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="'primary_dataset_id' is not declared"):
        registry.primary()


def test_primary_pointing_at_missing_dataset_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace(
        "primary_dataset_id: ibm_transactions_aml", "primary_dataset_id: nonexistent"
    )
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="no such dataset"):
        registry.primary()


def test_missing_required_field_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace(
        "    kaggle_dataset_id: ealtman2019/ibm-transactions-for-anti-money-laundering-aml\n", ""
    )
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="kaggle_dataset_id"):
        registry.primary()


def test_unknown_status_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("status: configured_not_downloaded", "status: probably_fine")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="unknown status"):
        registry.primary()


def test_unknown_source_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("source: Kaggle", "source: Dropbox")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="Unknown dataset source"):
        registry.primary()


def test_selected_variant_not_in_supported_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("selected_variant: HI-Small", "selected_variant: HI-Large")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="not supported"):
        registry.primary()


def test_absolute_local_path_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("local_path: data/raw/ibm_aml/", "local_path: /etc/ibm_aml")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="relative"):
        registry.primary()


def test_escaping_local_path_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace(
        "local_path: data/raw/ibm_aml/", "local_path: ../../elsewhere"
    )
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match=r"escape the project root"):
        registry.primary()


def test_empty_expected_files_raises(tmp_path: Path) -> None:
    content = VALID_REGISTRY.replace("      - HI-Small_Trans.csv\n", "")
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="expected_files"):
        registry.primary()


@pytest.mark.parametrize(
    "credential_key", ["kaggle_key", "api_token", "password", "secret", "credentials"]
)
def test_credential_like_key_in_registry_is_rejected(tmp_path: Path, credential_key: str) -> None:
    content = VALID_REGISTRY.replace(
        "    intended_usage: Primary training and evaluation data.",
        f"    intended_usage: Primary training and evaluation data.\n    {credential_key}: x",
    )
    registry = YamlDatasetRegistry(write_registry(tmp_path, content))
    with pytest.raises(InvalidDatasetRegistryError, match="credential-like key"):
        registry.primary()


def test_registry_is_parsed_once(tmp_path: Path) -> None:
    path = write_registry(tmp_path, VALID_REGISTRY)
    registry = YamlDatasetRegistry(path)
    first = registry.primary()
    path.unlink()  # cache must survive the file disappearing
    assert registry.primary() is first
