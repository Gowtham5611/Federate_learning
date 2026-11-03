"""Tests for the dataset descriptor value object.

The descriptor is the project's guard against a registry that lies: it refuses
to exist in an inconsistent state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.dataset import (
    InvalidDatasetDescriptorError,
    InvalidDatasetPathError,
    UnsupportedVariantError,
)
from domain.value_objects.dataset_descriptor import DatasetDescriptor
from tests.fixtures.dataset_registry_doubles import make_descriptor

pytestmark = pytest.mark.unit


def test_valid_descriptor_is_constructed() -> None:
    descriptor = make_descriptor()
    assert descriptor.source is DatasetSourceKind.KAGGLE
    assert descriptor.status is DatasetStatus.CONFIGURED_NOT_DOWNLOADED


def test_descriptor_is_immutable() -> None:
    descriptor = make_descriptor()
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.status = DatasetStatus.DOWNLOADED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "build"),
    [
        ("dataset_id", lambda: make_descriptor(dataset_id="   ")),
        ("name", lambda: make_descriptor(name="   ")),
        ("provider", lambda: make_descriptor(provider="   ")),
        ("source_url", lambda: make_descriptor(source_url="   ")),
        ("remote_id", lambda: make_descriptor(remote_id="   ")),
    ],
)
def test_blank_required_field_is_rejected(
    field: str, build: Callable[[], DatasetDescriptor]
) -> None:
    with pytest.raises(InvalidDatasetDescriptorError, match=field):
        build()


def test_variant_outside_supported_list_is_rejected() -> None:
    with pytest.raises(UnsupportedVariantError, match="HI-Large"):
        make_descriptor(selected_variant="HI-Large")


def test_empty_supported_variants_is_rejected() -> None:
    with pytest.raises(InvalidDatasetDescriptorError, match="supported_variants"):
        make_descriptor(supported_variants=())


def test_empty_expected_files_is_rejected() -> None:
    with pytest.raises(InvalidDatasetDescriptorError, match="expected_files"):
        make_descriptor(expected_files=())


def test_absolute_local_path_is_rejected() -> None:
    absolute = str(Path(Path.cwd().anchor) / "data")
    with pytest.raises(InvalidDatasetPathError, match="relative"):
        make_descriptor(local_path=absolute)


@pytest.mark.parametrize(
    "absolute_form",
    [
        "/etc/ibm_aml",  # POSIX-absolute (looks relative to Windows)
        "C:/data/ibm_aml",  # Windows-absolute (looks relative to POSIX)
        "C:\\data\\ibm_aml",
        "//server/share/ibm_aml",  # UNC
    ],
)
def test_absolute_path_of_any_platform_is_rejected(absolute_form: str) -> None:
    """The registry is committed and read on every OS, so the check must be too."""
    with pytest.raises(InvalidDatasetPathError, match="relative"):
        make_descriptor(local_path=absolute_form)


def test_local_path_escaping_the_project_root_is_rejected() -> None:
    with pytest.raises(InvalidDatasetPathError, match="escape the project root"):
        make_descriptor(local_path="../outside")


# --- Status semantics ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "implies_files"),
    [
        (DatasetStatus.CONFIGURED_NOT_DOWNLOADED, False),
        (DatasetStatus.DOWNLOADING, False),
        (DatasetStatus.FAILED, False),
        (DatasetStatus.DOWNLOADED, True),
        (DatasetStatus.VALIDATED, True),
        (DatasetStatus.PROCESSED, True),
    ],
)
def test_implies_local_files(status: DatasetStatus, implies_files: bool) -> None:
    assert status.implies_local_files is implies_files


def test_initial_status_label_is_explicit_about_absence() -> None:
    assert DatasetStatus.CONFIGURED_NOT_DOWNLOADED.label == "CONFIGURED - NOT DOWNLOADED"


def test_source_parsing_is_case_insensitive() -> None:
    assert DatasetSourceKind.parse("KAGGLE") is DatasetSourceKind.KAGGLE
    assert DatasetSourceKind.parse(" kaggle ") is DatasetSourceKind.KAGGLE


def test_unknown_source_lists_known_sources() -> None:
    with pytest.raises(ValueError, match="kaggle"):
        DatasetSourceKind.parse("dropbox")
