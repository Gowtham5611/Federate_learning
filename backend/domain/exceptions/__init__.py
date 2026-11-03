"""Domain-level exception hierarchy.

Import from this package rather than the individual modules::

    from domain.exceptions import DatasetError, MissingCredentialsError
"""

from __future__ import annotations

from domain.exceptions.base import FedSecureAIError
from domain.exceptions.data import (
    DataError,
    DataFileNotFoundError,
    EmptyDataFileError,
    InvalidChunkSizeError,
    MalformedDataError,
    SchemaDefinitionError,
    SchemaNotVerifiedError,
    SchemaValidationError,
    UnreadableDataFileError,
)
from domain.exceptions.dataset import (
    DatasetError,
    DatasetMismatchError,
    DatasetRegistryNotFoundError,
    InvalidDatasetDescriptorError,
    InvalidDatasetPathError,
    InvalidDatasetRegistryError,
    UnknownDatasetError,
    UnsupportedVariantError,
)
from domain.exceptions.download import (
    AuthenticationError,
    DownloadError,
    DownloadFailedError,
    IncompleteDownloadError,
    MissingCredentialsError,
    NetworkError,
)
from domain.exceptions.preprocessing import (
    ChronologyViolationError,
    InvalidSplitFractionsError,
    NotEnoughDataToSplitError,
    OutputWouldOverwriteInputError,
    PreprocessingError,
    RawDatasetModifiedError,
    TransformerAlreadyFittedError,
    TransformerNotFittedError,
)

__all__ = [
    "AuthenticationError",
    "ChronologyViolationError",
    "DataError",
    "DataFileNotFoundError",
    "DatasetError",
    "DatasetMismatchError",
    "DatasetRegistryNotFoundError",
    "DownloadError",
    "DownloadFailedError",
    "EmptyDataFileError",
    "FedSecureAIError",
    "IncompleteDownloadError",
    "InvalidChunkSizeError",
    "InvalidDatasetDescriptorError",
    "InvalidDatasetPathError",
    "InvalidDatasetRegistryError",
    "InvalidSplitFractionsError",
    "MalformedDataError",
    "MissingCredentialsError",
    "NetworkError",
    "NotEnoughDataToSplitError",
    "OutputWouldOverwriteInputError",
    "PreprocessingError",
    "RawDatasetModifiedError",
    "SchemaDefinitionError",
    "SchemaNotVerifiedError",
    "SchemaValidationError",
    "TransformerAlreadyFittedError",
    "TransformerNotFittedError",
    "UnknownDatasetError",
    "UnreadableDataFileError",
    "UnsupportedVariantError",
]
