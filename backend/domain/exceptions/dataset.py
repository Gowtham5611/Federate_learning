"""Dataset configuration and registry errors."""

from __future__ import annotations

from pathlib import Path

from domain.exceptions.base import FedSecureAIError


class DatasetError(FedSecureAIError):
    """Base class for dataset configuration and availability problems."""


class DatasetRegistryNotFoundError(DatasetError):
    """The dataset registry file does not exist."""

    def __init__(self, path: Path | str) -> None:
        """Record the path that was searched."""
        self.path = Path(path)
        super().__init__(
            f"Dataset registry not found at '{self.path}'. "
            f"Expected a YAML registry; see datasets/dataset_registry.yaml."
        )


class InvalidDatasetRegistryError(DatasetError):
    """The dataset registry exists but is malformed or internally inconsistent."""

    def __init__(self, reason: str, path: Path | str | None = None) -> None:
        """Record why the registry was rejected, and where it came from."""
        self.path = Path(path) if path is not None else None
        location = f" in '{self.path}'" if self.path is not None else ""
        super().__init__(f"Invalid dataset registry{location}: {reason}")


class UnknownDatasetError(DatasetError):
    """The requested dataset id is not declared in the registry."""

    def __init__(self, dataset_id: str, known_ids: tuple[str, ...] = ()) -> None:
        """Record the rejected id and what the registry does contain."""
        self.dataset_id = dataset_id
        self.known_ids = known_ids
        known = ", ".join(known_ids) if known_ids else "none"
        super().__init__(
            f"Dataset '{dataset_id}' is not declared in the dataset registry. "
            f"Registered datasets: {known}. "
            f"Add it to datasets/dataset_registry.yaml before use."
        )


class UnsupportedVariantError(DatasetError):
    """The selected variant is not among the dataset's supported variants."""

    def __init__(self, dataset_id: str, variant: str, supported: tuple[str, ...]) -> None:
        """Record the rejected variant and the permitted set."""
        self.dataset_id = dataset_id
        self.variant = variant
        self.supported = supported
        allowed = ", ".join(supported) if supported else "none declared"
        super().__init__(
            f"Variant '{variant}' is not supported for dataset '{dataset_id}'. "
            f"Supported variants: {allowed}."
        )


class InvalidDatasetDescriptorError(DatasetError):
    """A dataset entry is structurally valid YAML but semantically unusable."""

    def __init__(self, reason: str, dataset_id: str | None = None) -> None:
        """Record why the entry was rejected."""
        self.dataset_id = dataset_id
        subject = f"Dataset '{dataset_id}'" if dataset_id else "Dataset entry"
        super().__init__(f"{subject} is invalid: {reason}")


class DatasetMismatchError(DatasetError):
    """A dataset-specific component was pointed at a different dataset.

    Guards the project's single-approved-dataset rule: a component written for
    one dataset's structure must refuse to operate on another's, rather than
    reading it and reporting results as if they described the approved data.
    """

    def __init__(self, component: str, expected_remote_id: str, actual_remote_id: str) -> None:
        """Record which component refused which dataset."""
        self.component = component
        self.expected_remote_id = expected_remote_id
        self.actual_remote_id = actual_remote_id
        super().__init__(
            f"{component} is defined for dataset '{expected_remote_id}' but the "
            f"registry's active dataset is '{actual_remote_id}'. Point the "
            f"component at the correct dataset id, or use a source built for "
            f"that dataset — this one must not read data it does not describe."
        )


class InvalidDatasetPathError(DatasetError):
    """A configured or requested dataset path is unusable or unsafe."""

    def __init__(self, path: Path | str, reason: str) -> None:
        """Record the offending path and why it was rejected."""
        self.path = Path(path)
        super().__init__(f"Invalid dataset path '{self.path}': {reason}")
