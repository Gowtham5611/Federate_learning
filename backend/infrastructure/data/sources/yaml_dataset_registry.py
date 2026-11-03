"""YAML-backed dataset registry adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from application.ports.output.dataset_registry import DatasetRegistryPort
from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.dataset import (
    DatasetError,
    DatasetRegistryNotFoundError,
    InvalidDatasetRegistryError,
    UnknownDatasetError,
)
from domain.value_objects.dataset_descriptor import DatasetDescriptor
from infrastructure.utils.config_loader import ConfigurationError, load_yaml
from infrastructure.utils.paths import PROJECT_ROOT

# Registry keys that must never hold a secret. Presence of any of these is a
# hard error: the registry is committed to Git, so a credential here would leak.
FORBIDDEN_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    {
        "kaggle_key",
        "kaggle_username",
        "api_key",
        "api_token",
        "token",
        "password",
        "secret",
        "credentials",
        "auth",
    }
)


class YamlDatasetRegistry(DatasetRegistryPort):
    """Reads dataset declarations from ``datasets/dataset_registry.yaml``.

    The file is parsed once and cached, since it is a hand-maintained
    declaration rather than changing runtime state. This adapter is read-only:
    it never writes a status back. A dataset's real state is observed from the
    filesystem by :class:`~application.services.dataset_manager.DatasetManager`,
    so the registry stays a human-authored declaration and cannot drift into
    being an unreliable cache of runtime facts.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        """Point the adapter at a registry file.

        Args:
            registry_path: Registry location. Defaults to the project's
                ``datasets/dataset_registry.yaml``.
        """
        self._path = (
            Path(registry_path)
            if registry_path is not None
            else PROJECT_ROOT / "datasets" / "dataset_registry.yaml"
        )
        self._descriptors: dict[str, DatasetDescriptor] | None = None
        self._primary_id: str | None = None

    @property
    def path(self) -> Path:
        """Location of the registry file this adapter reads."""
        return self._path

    def get(self, dataset_id: str) -> DatasetDescriptor:
        """Return the declaration for ``dataset_id``."""
        descriptors = self._load()
        try:
            return descriptors[dataset_id]
        except KeyError:
            raise UnknownDatasetError(dataset_id, tuple(sorted(descriptors))) from None

    def primary(self) -> DatasetDescriptor:
        """Return the declaration for the registry's primary dataset."""
        descriptors = self._load()
        if self._primary_id is None:
            raise InvalidDatasetRegistryError("'primary_dataset_id' is not declared", self._path)
        if self._primary_id not in descriptors:
            raise InvalidDatasetRegistryError(
                f"'primary_dataset_id' is '{self._primary_id}' but no such dataset is declared",
                self._path,
            )
        return descriptors[self._primary_id]

    def dataset_ids(self) -> tuple[str, ...]:
        """Return every declared dataset id."""
        return tuple(sorted(self._load()))

    # -- Parsing -----------------------------------------------------------

    def _load(self) -> dict[str, DatasetDescriptor]:
        """Parse and cache the registry."""
        if self._descriptors is not None:
            return self._descriptors

        if not self._path.is_file():
            raise DatasetRegistryNotFoundError(self._path)

        try:
            raw = load_yaml(self._path)
        except ConfigurationError as exc:
            raise InvalidDatasetRegistryError(str(exc), self._path) from exc

        entries = raw.get("datasets")
        if not isinstance(entries, dict) or not entries:
            raise InvalidDatasetRegistryError(
                "'datasets' must be a non-empty mapping of dataset_id to declaration",
                self._path,
            )

        descriptors: dict[str, DatasetDescriptor] = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                raise InvalidDatasetRegistryError(
                    f"entry '{key}' must be a mapping, got {type(entry).__name__}", self._path
                )
            self._reject_credentials(str(key), entry)
            descriptors[str(key)] = self._to_descriptor(str(key), entry)

        primary = raw.get("primary_dataset_id")
        self._primary_id = str(primary) if primary is not None else None
        self._descriptors = descriptors
        return descriptors

    def _reject_credentials(self, key: str, entry: dict[str, Any]) -> None:
        """Fail loudly if a dataset entry contains anything credential-shaped.

        The registry is version-controlled. A secret placed here would be
        committed, so this is treated as a configuration error rather than
        quietly ignored.
        """
        offending = sorted(k for k in entry if str(k).lower() in FORBIDDEN_CREDENTIAL_KEYS)
        if offending:
            raise InvalidDatasetRegistryError(
                f"dataset '{key}' declares credential-like key(s) {offending}. "
                f"Credentials must come from the environment, never from the "
                f"registry, which is committed to Git",
                self._path,
            )

    def _to_descriptor(self, dataset_id: str, entry: dict[str, Any]) -> DatasetDescriptor:
        """Map one raw registry entry onto a validated descriptor."""
        try:
            source = DatasetSourceKind.parse(self._require_str(dataset_id, entry, "source"))
        except ValueError as exc:
            raise InvalidDatasetRegistryError(str(exc), self._path) from exc

        status_value = self._require_str(dataset_id, entry, "status")
        try:
            status = DatasetStatus(status_value)
        except ValueError:
            known = ", ".join(member.value for member in DatasetStatus)
            raise InvalidDatasetRegistryError(
                f"dataset '{dataset_id}' has unknown status '{status_value}'. Known: {known}",
                self._path,
            ) from None

        try:
            return DatasetDescriptor(
                dataset_id=str(entry.get("dataset_id", dataset_id)),
                name=self._require_str(dataset_id, entry, "name"),
                provider=self._require_str(dataset_id, entry, "provider"),
                source=source,
                source_url=self._require_str(dataset_id, entry, "source_url"),
                # The registry key is provider-specific; the domain field is not.
                remote_id=self._require_str(dataset_id, entry, "kaggle_dataset_id"),
                purpose=self._require_str(dataset_id, entry, "purpose"),
                status=status,
                local_path=Path(self._require_str(dataset_id, entry, "local_path")),
                selected_variant=self._require_str(dataset_id, entry, "selected_variant"),
                supported_variants=self._require_str_tuple(
                    dataset_id, entry, "supported_variants"
                ),
                expected_files=self._require_str_tuple(dataset_id, entry, "expected_files"),
                intended_usage=self._require_str(dataset_id, entry, "intended_usage"),
            )
        except DatasetError as exc:
            # Re-raise descriptor-level validation failures with file context so
            # the operator knows which file to fix.
            raise InvalidDatasetRegistryError(str(exc), self._path) from exc

    def _require_str(self, dataset_id: str, entry: dict[str, Any], key: str) -> str:
        """Return a required non-empty scalar field as a string."""
        value = entry.get(key)
        if value is None or not str(value).strip():
            raise InvalidDatasetRegistryError(
                f"dataset '{dataset_id}' is missing required field '{key}'", self._path
            )
        return str(value).strip()

    def _require_str_tuple(
        self, dataset_id: str, entry: dict[str, Any], key: str
    ) -> tuple[str, ...]:
        """Return a required non-empty list field as a tuple of strings."""
        value = entry.get(key)
        if not isinstance(value, list) or not value:
            raise InvalidDatasetRegistryError(
                f"dataset '{dataset_id}' field '{key}' must be a non-empty list", self._path
            )
        items = tuple(str(item).strip() for item in value)
        if any(not item for item in items):
            raise InvalidDatasetRegistryError(
                f"dataset '{dataset_id}' field '{key}' contains an empty entry", self._path
            )
        return items
