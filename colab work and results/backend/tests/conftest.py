"""Shared pytest fixtures.

Fixtures are added here only when a test actually needs them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest

from infrastructure.utils.paths import PROJECT_ROOT


class YamlFileFactory(Protocol):
    """Writes YAML text to a temporary file and returns its path."""

    def __call__(self, content: str, name: str = ...) -> Path:
        """Write ``content`` to ``name`` inside the test's temporary directory."""
        ...


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture
def yaml_file(tmp_path: Path) -> YamlFileFactory:
    """Return a factory that writes YAML text to a temporary file."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write
