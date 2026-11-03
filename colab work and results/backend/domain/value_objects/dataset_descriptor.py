"""Immutable description of a registered dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.dataset import (
    InvalidDatasetDescriptorError,
    InvalidDatasetPathError,
    UnsupportedVariantError,
)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """What the registry declares about one dataset.

    This is a *declaration*, not an observation. ``status`` records what the
    registry claims; whether the files actually exist is determined by
    inspecting the filesystem, never by reading this field.

    ``remote_id`` is the provider-side identifier (for Kaggle, the
    ``owner/slug`` pair). It is deliberately named generically so the domain
    does not acquire a Kaggle-shaped field; the YAML key remains
    ``kaggle_dataset_id`` and is mapped on load.

    Attributes:
        dataset_id: Project-internal identifier, unique within the registry.
        name: Human-readable dataset name.
        provider: Organisation that produced the data.
        source: Which remote service hosts it.
        source_url: Human-facing landing page.
        remote_id: Provider-side identifier used by the download adapter.
        purpose: What the dataset is for.
        status: Lifecycle state as declared by the registry.
        local_path: Project-relative directory the files belong in.
        selected_variant: The variant this project currently works with.
        supported_variants: Variants the registry declares as usable.
        expected_files: Filenames that must be present for
            :attr:`selected_variant` to count as downloaded.
        intended_usage: How the project intends to use the dataset.
    """

    dataset_id: str
    name: str
    provider: str
    source: DatasetSourceKind
    source_url: str
    remote_id: str
    purpose: str
    status: DatasetStatus
    local_path: Path
    selected_variant: str
    supported_variants: tuple[str, ...]
    expected_files: tuple[str, ...]
    intended_usage: str

    def __post_init__(self) -> None:
        """Reject entries that are syntactically fine but unusable.

        Raises:
            InvalidDatasetDescriptorError: If a required field is blank or a
                collection that must be populated is empty.
            UnsupportedVariantError: If ``selected_variant`` is not declared in
                ``supported_variants``.
            InvalidDatasetPathError: If ``local_path`` is absolute or escapes
                the project root.
        """
        for field_name in ("dataset_id", "name", "provider", "source_url", "remote_id"):
            if not str(getattr(self, field_name)).strip():
                raise InvalidDatasetDescriptorError(
                    f"'{field_name}' must not be empty", self.dataset_id or None
                )

        if not self.supported_variants:
            raise InvalidDatasetDescriptorError(
                "'supported_variants' must declare at least one variant", self.dataset_id
            )
        if self.selected_variant not in self.supported_variants:
            raise UnsupportedVariantError(
                self.dataset_id, self.selected_variant, self.supported_variants
            )
        if not self.expected_files:
            raise InvalidDatasetDescriptorError(
                "'expected_files' must list at least one file, otherwise local "
                "presence cannot be verified",
                self.dataset_id,
            )

        self._validate_local_path()

    def _validate_local_path(self) -> None:
        """Ensure ``local_path`` is a safe project-relative directory.

        The check is deliberately platform-independent. ``Path.is_absolute`` is
        not: on Windows ``/etc/data`` is "relative" (it has no drive), and on
        Linux ``C:/data`` is a directory literally named ``C:``. The registry is
        committed to Git and read on every platform, so an entry that is
        absolute *anywhere* must be rejected *everywhere*.
        """
        as_posix = self.local_path.as_posix()
        posix_form = PurePosixPath(as_posix)
        windows_form = PureWindowsPath(as_posix)

        if (
            self.local_path.is_absolute()
            or posix_form.is_absolute()
            or windows_form.drive
            or windows_form.root
        ):
            raise InvalidDatasetPathError(
                self.local_path,
                "must be relative to the project root so the layout is portable",
            )

        parts = posix_form.parts
        if ".." in parts:
            raise InvalidDatasetPathError(
                self.local_path, "must not contain '..' or escape the project root"
            )
        if not parts:
            raise InvalidDatasetPathError(self.local_path, "must not be empty")
