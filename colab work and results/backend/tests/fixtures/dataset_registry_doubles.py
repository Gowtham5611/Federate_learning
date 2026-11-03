"""Test doubles for the dataset registry port."""

from __future__ import annotations

from pathlib import Path

from application.ports.output.dataset_registry import DatasetRegistryPort
from domain.enums.dataset_source import DatasetSourceKind
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.dataset import InvalidDatasetRegistryError, UnknownDatasetError
from domain.value_objects.dataset_descriptor import DatasetDescriptor

IBM_AML_KAGGLE_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def make_descriptor(
    *,
    dataset_id: str = "ibm_transactions_aml",
    name: str = "IBM Transactions for Anti-Money Laundering (AML)",
    provider: str = "IBM",
    source: DatasetSourceKind = DatasetSourceKind.KAGGLE,
    source_url: str = f"https://www.kaggle.com/datasets/{IBM_AML_KAGGLE_ID}",
    remote_id: str = IBM_AML_KAGGLE_ID,
    purpose: str = "Anti-Money Laundering transaction detection",
    status: DatasetStatus = DatasetStatus.CONFIGURED_NOT_DOWNLOADED,
    local_path: str = "data/raw/ibm_aml/",
    selected_variant: str = "HI-Small",
    expected_files: tuple[str, ...] = ("HI-Small_Trans.csv",),
    supported_variants: tuple[str, ...] = ("HI-Small",),
    intended_usage: str = "Primary training and evaluation data.",
) -> DatasetDescriptor:
    """Build a descriptor mirroring the real registry entry, with overrides."""
    return DatasetDescriptor(
        dataset_id=dataset_id,
        name=name,
        provider=provider,
        source=source,
        source_url=source_url,
        remote_id=remote_id,
        purpose=purpose,
        status=status,
        local_path=Path(local_path),
        selected_variant=selected_variant,
        supported_variants=supported_variants,
        expected_files=expected_files,
        intended_usage=intended_usage,
    )


class InMemoryDatasetRegistry(DatasetRegistryPort):
    """Registry backed by a dict, so tests never touch the filesystem."""

    def __init__(
        self,
        descriptors: tuple[DatasetDescriptor, ...] = (),
        primary_id: str | None = None,
    ) -> None:
        """Store the given descriptors, defaulting the primary to the first."""
        if not descriptors:
            descriptors = (make_descriptor(),)
        self._descriptors = {d.dataset_id: d for d in descriptors}
        self._primary_id = primary_id if primary_id is not None else descriptors[0].dataset_id

    def get(self, dataset_id: str) -> DatasetDescriptor:
        """Return the named descriptor."""
        try:
            return self._descriptors[dataset_id]
        except KeyError:
            raise UnknownDatasetError(dataset_id, tuple(sorted(self._descriptors))) from None

    def primary(self) -> DatasetDescriptor:
        """Return the primary descriptor."""
        if self._primary_id not in self._descriptors:
            raise InvalidDatasetRegistryError(
                f"'primary_dataset_id' is '{self._primary_id}' but no such dataset is declared"
            )
        return self._descriptors[self._primary_id]

    def dataset_ids(self) -> tuple[str, ...]:
        """Return every stored id."""
        return tuple(sorted(self._descriptors))
