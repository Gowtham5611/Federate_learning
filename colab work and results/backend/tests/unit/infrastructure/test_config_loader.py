"""Tests for the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.utils.config_loader import ConfigurationError, load_config, load_yaml
from tests.conftest import YamlFileFactory

pytestmark = pytest.mark.unit

ENV_VAR = "FEDSECURE_TEST_ENV"


def test_loads_mapping(yaml_file: YamlFileFactory) -> None:
    path = yaml_file("app:\n  name: FedSecureAI\n  version: 0.1.0\n")
    assert load_yaml(path) == {"app": {"name": "FedSecureAI", "version": "0.1.0"}}


def test_empty_file_returns_empty_mapping(yaml_file: YamlFileFactory) -> None:
    assert load_yaml(yaml_file("")) == {}


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_yaml("does/not/exist.yaml")


def test_non_mapping_top_level_raises(yaml_file: YamlFileFactory) -> None:
    with pytest.raises(ConfigurationError, match="mapping"):
        load_yaml(yaml_file("- a\n- b\n"))


def test_invalid_yaml_raises(yaml_file: YamlFileFactory) -> None:
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_yaml(yaml_file("app:\n  - unbalanced: [\n"))


def test_expands_environment_placeholder(
    yaml_file: YamlFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, "production")
    path = yaml_file(f"app:\n  environment: ${{{ENV_VAR}}}\n")
    assert load_yaml(path)["app"]["environment"] == "production"


def test_placeholder_default_used_when_unset(
    yaml_file: YamlFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    path = yaml_file(f"app:\n  environment: ${{{ENV_VAR}:-development}}\n")
    assert load_yaml(path)["app"]["environment"] == "development"


def test_unset_placeholder_without_default_raises(
    yaml_file: YamlFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    path = yaml_file(f"app:\n  environment: ${{{ENV_VAR}}}\n")
    with pytest.raises(ConfigurationError, match="not set"):
        load_yaml(path)


def test_expands_inside_nested_lists(
    yaml_file: YamlFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, "x")
    path = yaml_file(f"items:\n  - a: ${{{ENV_VAR}}}\n")
    assert load_yaml(path)["items"] == [{"a": "x"}]


def test_load_config_reads_base_yaml(project_root: Path) -> None:
    config = load_config("base")
    assert config["app"]["name"] == "FedSecureAI"
    assert (project_root / config["paths"]["dataset_registry"]).is_file()
