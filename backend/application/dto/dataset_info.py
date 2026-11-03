"""Dataset information returned across the application boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.value_objects.dataset_descriptor import DatasetDescriptor


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """A dataset's declared configuration together with observed local state.

    The distinction is the point of this DTO. ``declared_status`` is whatever
    the registry says; ``files_present`` and ``missing_files`` are measured from
    disk; ``effective_status`` is the two reconciled. A caller should report
    ``effective_status``, never ``declared_status``, so a stale registry cannot
    cause the project to claim data it does not have.

    Attributes:
        descriptor: The registry declaration this info was derived from.
        local_path: Absolute resolved directory the files belong in.
        declared_status: Status as recorded in the registry.
        effective_status: Status after reconciling with the filesystem.
        present_files: Expected files found locally and non-empty.
        missing_files: Expected files absent or zero-length.
        download_performed: Whether the call producing this info transferred
            data. A metadata query always reports ``False``.
        registry_disagrees_with_disk: Whether the registry's claim about local
            presence was contradicted by the filesystem.
    """

    descriptor: DatasetDescriptor
    local_path: Path
    declared_status: DatasetStatus
    effective_status: DatasetStatus
    present_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    download_performed: bool
    registry_disagrees_with_disk: bool

    @property
    def dataset_id(self) -> str:
        """Project-internal dataset identifier."""
        return self.descriptor.dataset_id

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self.descriptor.name

    @property
    def source(self) -> DatasetSourceKind:
        """Remote service hosting the dataset."""
        return self.descriptor.source

    @property
    def source_url(self) -> str:
        """Human-facing dataset landing page."""
        return self.descriptor.source_url

    @property
    def remote_id(self) -> str:
        """Provider-side dataset identifier."""
        return self.descriptor.remote_id

    @property
    def selected_variant(self) -> str:
        """Variant the project is currently configured to use."""
        return self.descriptor.selected_variant

    @property
    def is_available(self) -> bool:
        """Whether every expected file is present locally and non-empty."""
        return not self.missing_files
