"""Outbound port describing where a dataset's transaction file lives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TransactionDataSourcePort(ABC):
    """Identifies one dataset's transaction file and locates it on disk.

    This port sits between "which dataset is configured" and "read these bytes".
    Its whole job is to answer *what file, where* — so a loader receives a path
    and needs no knowledge of registries, variants or providers.

    Implementations must not: download anything, hold or read credentials,
    parse or transform data, or train models. A source that could download would
    make every metadata query a potential network call, which is exactly the
    property the dataset layer was built to avoid.
    """

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Project-internal identifier of the dataset being read."""

    @property
    @abstractmethod
    def variant(self) -> str:
        """Which variant of the dataset this source reads."""

    @property
    @abstractmethod
    def expected_filename(self) -> str:
        """Name of the file the variant is expected to arrive as."""

    @property
    @abstractmethod
    def local_directory(self) -> Path:
        """Absolute directory the dataset's files belong in.

        Returned whether or not the directory exists: it is a *declaration* of
        where the data goes, which is what an error message needs to be useful.
        """

    @abstractmethod
    def resolve_file(self) -> Path:
        """Absolute path to the transaction file, verified to exist.

        Returns:
            The existing, non-empty file to read.

        Raises:
            DataFileNotFoundError: The file is not present locally. The message
                must say how to obtain it, since an absent dataset is the
                expected state of a fresh checkout rather than a defect.
        """
