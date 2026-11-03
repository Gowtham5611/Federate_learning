"""Dataset manager: the application service that owns dataset lifecycle state."""

from __future__ import annotations

import logging
from pathlib import Path

from application.dto.dataset_info import DatasetInfo
from application.dto.download import DownloadRequest
from application.ports.output.dataset_downloader import DatasetDownloaderPort
from application.ports.output.dataset_registry import DatasetRegistryPort
from domain.enums.dataset_status import DatasetStatus
from domain.exceptions.download import DownloadError, IncompleteDownloadError
from domain.value_objects.dataset_descriptor import DatasetDescriptor

logger = logging.getLogger(__name__)


class DatasetManager:
    """Coordinates dataset declaration, local presence and download.

    Responsibilities: resolve which dataset is active, determine where it
    belongs on disk, observe whether it is actually there, and — only when
    explicitly asked — delegate a download to the injected port.

    Deliberately *not* its responsibilities: parsing CSVs, preprocessing,
    feature engineering, training, and anything Kaggle-specific. It depends on
    :class:`DatasetDownloaderPort`, never on a concrete adapter, so it runs in
    tests with no network and no credentials.

    The critical invariant: **a metadata query never triggers a transfer.**
    Only :meth:`download_dataset` can cause bytes to move.
    """

    def __init__(
        self,
        registry: DatasetRegistryPort,
        downloader: DatasetDownloaderPort,
        *,
        project_root: Path,
        dataset_id: str | None = None,
    ) -> None:
        """Wire the manager to its dependencies.

        Args:
            registry: Source of dataset declarations.
            downloader: Adapter used only when a download is explicitly requested.
            project_root: Root that relative registry paths resolve against.
            dataset_id: Dataset to manage. Defaults to the registry's primary
                dataset.
        """
        self._registry = registry
        self._downloader = downloader
        self._project_root = project_root
        self._dataset_id = dataset_id
        self._descriptor: DatasetDescriptor | None = None

    # -- Declaration -------------------------------------------------------

    @property
    def descriptor(self) -> DatasetDescriptor:
        """The active dataset's registry declaration, resolved once and cached."""
        if self._descriptor is None:
            self._descriptor = (
                self._registry.get(self._dataset_id)
                if self._dataset_id is not None
                else self._registry.primary()
            )
            logger.info(
                "Dataset configuration loaded: id=%s variant=%s declared_status=%s",
                self._descriptor.dataset_id,
                self._descriptor.selected_variant,
                self._descriptor.status.value,
            )
        return self._descriptor

    def get_local_path(self) -> Path:
        """Absolute directory the dataset's files belong in."""
        return (self._project_root / self.descriptor.local_path).resolve()

    # -- Observation -------------------------------------------------------

    def get_dataset_info(self) -> DatasetInfo:
        """Return declared configuration reconciled with observed local state.

        Never downloads anything, regardless of the dataset's status.
        """
        info = self._build_info(download_performed=False)
        logger.info(
            "Dataset status checked: id=%s effective_status=%s files_present=%d/%d",
            info.dataset_id,
            info.effective_status.value,
            len(info.present_files),
            len(info.present_files) + len(info.missing_files),
        )
        if info.registry_disagrees_with_disk:
            logger.warning(
                "Registry declares '%s' as %s but %d expected file(s) are missing from %s; "
                "reporting %s instead. Update the registry or re-download.",
                info.dataset_id,
                info.declared_status.value,
                len(info.missing_files),
                info.local_path,
                info.effective_status.value,
            )
        elif info.is_available:
            logger.info(
                "Dataset found locally: id=%s path=%s", info.dataset_id, info.local_path
            )
        return info

    def is_dataset_available(self) -> bool:
        """Whether every expected file is present locally and non-empty."""
        return not self._inspect_files()[1]

    def get_status(self) -> DatasetStatus:
        """Effective status: the registry's claim reconciled with the filesystem."""
        return self._build_info(download_performed=False).effective_status

    def locate_file(self, filename: str) -> Path | None:
        """Absolute path to one expected file, if it is present and non-empty.

        Exposed so a data source can resolve a specific file without
        reimplementing the search — including the tolerance for archives that
        extract into a nested directory.

        Args:
            filename: Name of the file to find beneath the dataset directory.

        Returns:
            The absolute path, or ``None`` if the file is absent or empty.
        """
        return self._locate(self.get_local_path(), filename)

    # -- Mutation ----------------------------------------------------------

    def download_dataset(self, *, force: bool = False) -> DatasetInfo:
        """Download the dataset, unless it is already present.

        Calling this method *is* the explicit request to transfer data; no other
        method on this class can cause a download.

        Args:
            force: Re-download even if the expected files already exist.

        Returns:
            Info describing the dataset afterwards. ``download_performed`` is
            ``False`` when an existing local copy made the transfer unnecessary.

        Raises:
            DownloadError: If the transfer fails for any reason.
            IncompleteDownloadError: If the downloader reports success but the
                expected files are still missing.
        """
        descriptor = self.descriptor
        destination = self.get_local_path()

        if not force and self.is_dataset_available():
            logger.info(
                "Download skipped: '%s' already present at %s (pass force=True to re-download)",
                descriptor.dataset_id,
                destination,
            )
            return self._build_info(download_performed=False)

        request = DownloadRequest(
            dataset_id=descriptor.dataset_id,
            source=descriptor.source,
            remote_id=descriptor.remote_id,
            destination=destination,
            expected_files=descriptor.expected_files,
            variant=descriptor.selected_variant,
            overwrite=force,
        )

        logger.info(
            "Download requested: id=%s source=%s variant=%s -> %s (status=%s)",
            descriptor.dataset_id,
            descriptor.source.value,
            descriptor.selected_variant,
            destination,
            DatasetStatus.DOWNLOADING.value,
        )
        try:
            outcome = self._downloader.download(request)
        except DownloadError:
            logger.exception(
                "Download failed: id=%s (status=%s)",
                descriptor.dataset_id,
                DatasetStatus.FAILED.value,
            )
            raise

        _, missing = self._inspect_files()
        if missing:
            logger.error(
                "Download reported success but %d expected file(s) are missing: id=%s",
                len(missing),
                descriptor.dataset_id,
            )
            raise IncompleteDownloadError(descriptor.dataset_id, missing, str(destination))

        logger.info(
            "Download completed: id=%s files=%d destination=%s",
            descriptor.dataset_id,
            len(outcome.files),
            outcome.destination,
        )
        return self._build_info(download_performed=True)

    # -- Internals ---------------------------------------------------------

    def _build_info(self, *, download_performed: bool) -> DatasetInfo:
        """Assemble a :class:`DatasetInfo` from declaration plus observation."""
        descriptor = self.descriptor
        present, missing = self._inspect_files()
        declared = descriptor.status
        effective = self._reconcile(declared, has_missing=bool(missing))
        return DatasetInfo(
            descriptor=descriptor,
            local_path=self.get_local_path(),
            declared_status=declared,
            effective_status=effective,
            present_files=present,
            missing_files=missing,
            download_performed=download_performed,
            registry_disagrees_with_disk=declared.implies_local_files and bool(missing),
        )

    @staticmethod
    def _reconcile(declared: DatasetStatus, *, has_missing: bool) -> DatasetStatus:
        """Resolve the registry's claim against what is on disk.

        A registry claiming local files that are absent is downgraded — the
        project must never report data it does not have. A registry that has not
        caught up with a completed download is promoted to ``downloaded``, but
        no further: validation and processing are separate, evidenced steps.
        """
        if has_missing:
            return (
                DatasetStatus.CONFIGURED_NOT_DOWNLOADED
                if declared.implies_local_files
                else declared
            )
        if declared.implies_local_files:
            return declared
        return DatasetStatus.DOWNLOADED

    def _inspect_files(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Observe which expected files exist locally.

        Returns:
            ``(present, missing)`` filenames. A file that exists but is empty
            counts as missing: a zero-byte stub is not a dataset.
        """
        root = self.get_local_path()
        present: list[str] = []
        missing: list[str] = []
        for filename in self.descriptor.expected_files:
            if self._locate(root, filename) is not None:
                present.append(filename)
            else:
                missing.append(filename)
        return tuple(present), tuple(missing)

    @staticmethod
    def _locate(root: Path, filename: str) -> Path | None:
        """Find ``filename`` directly under ``root``, else anywhere beneath it.

        The recursive fallback tolerates archives that extract into a nested
        directory, which the Kaggle distribution may do.
        """
        if not root.is_dir():
            return None
        direct = root / filename
        if direct.is_file() and direct.stat().st_size > 0:
            return direct
        for candidate in root.rglob(filename):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None
