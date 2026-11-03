"""Dataset CLI commands."""

from __future__ import annotations

from application.dto.dataset_info import DatasetInfo
from application.services.dataset_manager import DatasetManager

_WIDTH = 40


def dataset_info_command(manager: DatasetManager) -> int:
    """Print the active dataset's configuration and local status.

    Read-only by construction: it calls only
    :meth:`~application.services.dataset_manager.DatasetManager.get_dataset_info`,
    which cannot trigger a transfer.

    Returns:
        Process exit code. Always 0 — an unconfigured dataset raises instead.
    """
    info = manager.get_dataset_info()
    print(_render(info))
    return 0


def dataset_download_command(manager: DatasetManager, *, force: bool = False) -> int:
    """Download the active dataset. This is the explicit transfer request.

    Invoking this command is the consent to move bytes; no other command in the
    CLI can cause a download. Credentials are read by the downloader's provider
    at this moment, from the environment, and are never printed here.

    Args:
        manager: Injected dataset service.
        force: Re-download even if the expected files are already present.

    Returns:
        Process exit code. Always 0 — a failure raises a
        :class:`~domain.exceptions.base.FedSecureAIError`, which the entry point
        reports.
    """
    rule = "=" * _WIDTH
    descriptor = manager.descriptor
    print(
        "\n".join(
            [
                rule,
                "FedSecureAI Dataset Download",
                rule,
                "",
                f"Dataset:  {descriptor.name}",
                f"Remote:   {descriptor.remote_id}",
                f"Variant:  {descriptor.selected_variant}",
                f"Files:    {', '.join(descriptor.expected_files)}",
                f"Target:   {manager.get_local_path()}",
                f"Force:    {'yes' if force else 'no'}",
                "",
                "Transferring from the remote provider. This may take a while and",
                "requires credentials in the environment. Nothing is printed that",
                "could disclose them.",
                "",
            ]
        )
    )

    info = manager.download_dataset(force=force)

    if info.download_performed:
        print(f"Download completed. Files are in {info.local_path}")
    else:
        print(
            f"Nothing transferred: the dataset is already present in "
            f"{info.local_path}. Pass --force to download it again."
        )
    print()
    print(_render(info))
    return 0


def _render(info: DatasetInfo) -> str:
    """Format dataset info as the CLI report block."""
    rule = "=" * _WIDTH
    sections: list[tuple[str, str]] = [
        ("Dataset", info.name),
        ("Source", info.source.label),
        ("Dataset ID", info.remote_id),
        ("Variant", info.selected_variant),
        ("Local Path", f"{info.descriptor.local_path.as_posix()}/"),
        ("Status", info.effective_status.label),
        ("Download", "Performed" if info.download_performed else "Not performed"),
    ]

    lines = [rule, "FedSecureAI Dataset Information", rule, ""]
    for title, value in sections:
        lines.extend([f"{title}:", value, ""])

    if info.missing_files:
        lines.extend(
            [
                f"Expected files not present locally ({len(info.missing_files)}):",
                *(f"  - {name}" for name in info.missing_files),
                "",
            ]
        )
    if info.present_files:
        lines.extend(
            [
                f"Expected files present locally ({len(info.present_files)}):",
                *(f"  - {name}" for name in info.present_files),
                "",
            ]
        )
    if info.registry_disagrees_with_disk:
        lines.extend(
            [
                f"WARNING: the registry declares this dataset as "
                f"'{info.declared_status.value}' but its files are not present. "
                f"Reporting the observed state instead.",
                "",
            ]
        )

    lines.extend(
        [
            "This command never downloads data. Use the explicit download",
            "operation to fetch the dataset.",
            rule,
        ]
    )
    return "\n".join(lines)
