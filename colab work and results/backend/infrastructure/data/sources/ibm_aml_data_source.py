"""IBM AML transaction data source: identifies the dataset and locates its file.

Knows *which* dataset the project reads and *where* its transaction file is. It
does not know how to fetch it, how to parse it, or what is inside it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from application.ports.output.transaction_data_source import TransactionDataSourcePort
from application.services.dataset_manager import DatasetManager
from domain.exceptions.data import DataFileNotFoundError
from domain.exceptions.dataset import DatasetMismatchError, InvalidDatasetDescriptorError
from domain.value_objects.dataset_descriptor import DatasetDescriptor

logger = logging.getLogger(__name__)

IBM_AML_DATASET_ID = "ibm_transactions_aml"
"""Registry id of the project's only approved dataset."""

IBM_AML_REMOTE_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
"""Kaggle identifier of the approved dataset, duplicated here as a guard.

Not a second source of truth: the registry remains authoritative for every
dataset field. This constant exists only so that this source can refuse to read
a dataset it was not written for, which is what makes the project's
single-approved-dataset rule enforceable in code rather than by convention.
"""

_ACQUISITION_HINT = (
    "The IBM AML dataset is configured but not downloaded on this machine. "
    "Configure KAGGLE_USERNAME and KAGGLE_KEY (see .env.example) and run the "
    "dataset download step. No substitute or synthetic file is generated: an "
    "absent dataset is reported, never fabricated."
)


class IBMAMLDataSource(TransactionDataSourcePort):
    """Locates the transaction file of the configured IBM AML variant.

    Delegates every dataset fact to the :class:`DatasetManager` rather than
    restating any of it, so the registry stays the single source of truth for the
    dataset's identity, variant, filename and location.

    Constructing this source performs no filesystem or network access. Reading
    any property resolves the registry entry once (through the manager's cache)
    and verifies that the configured dataset really is the approved one.
    """

    def __init__(self, dataset_manager: DatasetManager) -> None:
        """Wire the source to the dataset lifecycle manager.

        Args:
            dataset_manager: Supplies the dataset's declaration and location.
        """
        self._manager = dataset_manager

    # -- Identity ----------------------------------------------------------

    @property
    def dataset_id(self) -> str:
        """Registry id of the dataset being read."""
        return self._descriptor.dataset_id

    @property
    def remote_id(self) -> str:
        """Kaggle identifier of the dataset being read.

        Dataset-specific information beyond the port's contract, kept because a
        report that names the data it used is more trustworthy than one that
        does not.
        """
        return self._descriptor.remote_id

    @property
    def variant(self) -> str:
        """Variant the project is configured to use."""
        return self._descriptor.selected_variant

    @property
    def expected_filename(self) -> str:
        """Transaction CSV the selected variant is expected to arrive as."""
        descriptor = self._descriptor
        candidates = tuple(
            name for name in descriptor.expected_files if name.lower().endswith(".csv")
        )
        if len(candidates) != 1:
            raise InvalidDatasetDescriptorError(
                f"variant '{descriptor.selected_variant}' must declare exactly one "
                f"CSV in expected_files for transactions to be read from it; "
                f"found {len(candidates)} in {list(descriptor.expected_files)}",
                descriptor.dataset_id,
            )
        return candidates[0]

    @property
    def local_directory(self) -> Path:
        """Absolute directory the variant's files belong in, existing or not."""
        return self._manager.get_local_path()

    # -- Location ----------------------------------------------------------

    def resolve_file(self) -> Path:
        """Absolute path to the transaction file, verified to exist.

        Returns:
            The existing, non-empty CSV.

        Raises:
            DataFileNotFoundError: The file is absent or empty locally. Carries
                the acquisition instructions, because on a fresh checkout this
                is the normal state rather than a fault.
        """
        filename = self.expected_filename
        located = self._manager.locate_file(filename)
        if located is None:
            expected_at = self.local_directory / filename
            logger.info(
                "Transaction file '%s' for variant '%s' is not present at %s",
                filename,
                self.variant,
                self.local_directory,
            )
            raise DataFileNotFoundError(expected_at, hint=_ACQUISITION_HINT)
        return located

    # -- Internals ---------------------------------------------------------

    @property
    def _descriptor(self) -> DatasetDescriptor:
        """The registry declaration, checked to be the approved dataset.

        Raises:
            DatasetMismatchError: The active dataset is not IBM AML. Reading a
                different dataset through this source would produce results
                labelled as IBM AML that describe something else.
        """
        descriptor = self._manager.descriptor
        if descriptor.remote_id != IBM_AML_REMOTE_ID:
            raise DatasetMismatchError(
                type(self).__name__, IBM_AML_REMOTE_ID, descriptor.remote_id
            )
        return descriptor
