"""YAML configuration loading.

Deliberately thin: read a YAML file, expand ``${VAR}`` placeholders from the
environment, return a mapping. Typed settings objects are introduced per layer
as the modules that need them are implemented, rather than up front.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from infrastructure.utils.paths import CONFIG_DIR, resolve

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigurationError(RuntimeError):
    """Raised when configuration cannot be located, parsed or resolved."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary, expanding environment placeholders.

    Args:
        path: Absolute path, or a path relative to the project root.

    Raises:
        ConfigurationError: If the file is missing, unparseable, or does not
            contain a mapping at the top level.
    """
    resolved = resolve(path)
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file not found: {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {resolved}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Expected a mapping at the top level of {resolved}, got {type(raw).__name__}"
        )
    return {str(key): _expand(value) for key, value in raw.items()}


def load_config(name: str = "base") -> dict[str, Any]:
    """Load ``config/<name>.yaml``."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string leaves."""
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return _PLACEHOLDER.sub(_substitute, value)
    return value


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    resolved = os.environ.get(name, default)
    if resolved is None:
        raise ConfigurationError(
            f"Environment variable '{name}' is referenced by configuration but not set, "
            f"and no default was provided"
        )
    return resolved
