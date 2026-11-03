"""Outbound port for reading dataset declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects.dataset_descriptor import DatasetDescriptor


class DatasetRegistryPort(ABC):
    """Read access to the project's dataset declarations.

    Defined as a port so the application layer never learns that the registry
    happens to be a YAML file. Tests substitute an in-memory implementation,
    which is what lets the dataset manager be exercised without touching disk.
    """

    @abstractmethod
    def get(self, dataset_id: str) -> DatasetDescriptor:
        """Return the declaration for ``dataset_id``.

        Raises:
            UnknownDatasetError: If the id is not declared.
            InvalidDatasetRegistryError: If the registry cannot be read.
        """

    @abstractmethod
    def primary(self) -> DatasetDescriptor:
        """Return the declaration for the registry's primary dataset.

        Raises:
            InvalidDatasetRegistryError: If no primary dataset is declared.
        """

    @abstractmethod
    def dataset_ids(self) -> tuple[str, ...]:
        """Return every declared dataset id, for diagnostics and error messages."""
