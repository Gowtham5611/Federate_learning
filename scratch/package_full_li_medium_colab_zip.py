"""Builds the clean, production Colab ZIP package for Full LI-Medium training.

Excludes:
- Massive transaction CSV files (>8 GB)
- Virtual environments (.venv)
- Python cache directories (__pycache__, .pytest_cache)
- Temporary caches (.cache, .mmap_cache, scratch)
- Previous large artifact directories

Includes:
- Complete backend codebase
- Test suites
- Metadata directory for LI-Medium
- Colab runner notebook and markdown guides
- Requirements files and project configs
"""

import hashlib
import os
from pathlib import Path
import zipfile

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = WORKSPACE_ROOT / "FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip"

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".mmap_cache",
    "scratch",
    "artifacts",
    "Colab Work and Results",
    "audit_colab_results",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".mmap",
}

def should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Exclude directories
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    # Exclude CSV files in data/
    if rel_path.suffix.lower() == ".csv":
        return True
    if rel_path.suffix.lower() in EXCLUDE_EXTS:
        return True
    # Exclude zip files
    if rel_path.suffix.lower() == ".zip":
        return True
    return False

def build_zip():
    print(f"Packaging FedSecureAI codebase from {WORKSPACE_ROOT}...")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # Walk directory
        for root, dirs, files in os.walk(WORKSPACE_ROOT):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                file_path = root_path / file
                rel_path = file_path.relative_to(WORKSPACE_ROOT)

                if should_exclude(rel_path):
                    continue

                zipf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                total_files += 1
                total_uncompressed_bytes += file_path.stat().st_size

    # Calculate SHA-256
    hasher = hashlib.sha256()
    with open(OUTPUT_ZIP, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest()
    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    uncomp_mb = total_uncompressed_bytes / (1024 * 1024)

    print("=" * 60)
    print("ZIP PACKAGE SUCCESSFULLY CREATED")
    print(f"Target File     : {OUTPUT_ZIP.name}")
    print(f"File Path       : {OUTPUT_ZIP}")
    print(f"Total Files     : {total_files}")
    print(f"Uncompressed    : {uncomp_mb:.2f} MB")
    print(f"Compressed Size : {zip_size_mb:.2f} MB")
    print(f"SHA-256 Checksum: {zip_sha256}")
    print("=" * 60)

    # Verify zip integrity
    print("Testing ZIP integrity...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as test_zip:
        bad_file = test_zip.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupted file in ZIP: {bad_file}")
        namelist = test_zip.namelist()
        print(f"Integrity check passed! {len(namelist)} verified entries.")
        # Print key file checks
        key_files = [
            "backend/scripts/run_full_li_medium_training.py",
            "backend/infrastructure/data/sequences/mmap_sequence_provider.py",
            "backend/infrastructure/data/sequences/mmap_partition_manager.py",
            "tests/test_full_li_medium_pipeline.py",
            "colab_full_li_medium_runner.ipynb",
            "COLAB_FULL_LI_MEDIUM_GUIDE.md",
            "requirements_colab.txt",
            "data/processed/ibm_aml/LI-Medium/metadata/feature_metadata.json",
        ]
        print("\nVerifying key files present in ZIP:")
        for kf in key_files:
            present = kf in namelist
            print(f"  [{'OK' if present else 'MISSING'}] {kf}")

if __name__ == "__main__":
    build_zip()

