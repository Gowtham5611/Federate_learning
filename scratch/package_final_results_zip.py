import os
import sys
import zipfile
import hashlib
from pathlib import Path

ROOT = Path(".").resolve()
RESULTS_DIR = ROOT / "results"
OUT_ZIP = ROOT / "FedSecureAI_HI_Small_LI_Medium_All_Results.zip"

def main():
    print("=================================================================")
    print("PACKAGING FINAL RESULTS ARCHIVE")
    print(f"Target: {OUT_ZIP.name}")
    print("=================================================================")

    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist.")
        sys.exit(1)

    file_count = 0
    total_uncompressed_bytes = 0

    print("Building ZIP archive...")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(RESULTS_DIR.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(ROOT).as_posix() # e.g. results/...
            zf.write(p, rel)
            file_count += 1
            total_uncompressed_bytes += p.stat().st_size

    compressed_bytes = OUT_ZIP.stat().st_size
    print(f"\nArchive created: {OUT_ZIP.name}")
    print(f"Total files archived: {file_count}")
    print(f"Total uncompressed size: {total_uncompressed_bytes / (1024*1024):.2f} MB ({total_uncompressed_bytes} bytes)")
    print(f"Total compressed size: {compressed_bytes / (1024*1024):.2f} MB ({compressed_bytes} bytes)")

    # 1. Verify ZIP opens and run testzip()
    print("\nRunning testzip() archive integrity check...")
    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"FAILED: Archive corrupted at {bad_file}")
            sys.exit(1)
        print("PASS: testzip() verified 0 corrupted files, 100% CRC integrity.")
        namelist = zf.namelist()
        print(f"Entries in ZIP: {len(namelist)}")

        # Check required files
        required_patterns = [
            "results/HI_Small_Results/README.md",
            "results/HI_Small_Results/manifests/resume_manifest.json",
            "results/HI_Small_Results/verification/recalculated_metrics_full_test.json",
            "results/LI_Medium_Results/README.md",
            "results/LI_Medium_Results/verification/recalculated_metrics_full_test.json",
            "results/comparison/HI_Small_vs_LI_Medium.csv",
            "results/comparison/HI_Small_vs_LI_Medium.md",
            "results/comparison/comparison_methodology.md",
            "results/audit/file_inventory.csv",
            "results/audit/integrity_checksums.csv",
            "results/audit/verification_summary.json",
            "results/audit/verification_report.md",
            "results/README.md"
        ]
        for rp in required_patterns:
            assert rp in namelist, f"Missing required pattern in ZIP: {rp}"
            print(f"  Verified: {rp}")

        # Check checkpoint counts
        pts = [n for n in namelist if n.endswith(".pt")]
        print(f"  Verified Checkpoint (.pt) count: {len(pts)}")
        npzs = [n for n in namelist if n.endswith(".npz")]
        print(f"  Verified Prediction (.npz) count: {len(npzs)}")
        npys = [n for n in namelist if n.endswith(".npy")]
        print(f"  Verified Prediction (.npy) count: {len(npys)}")

    # 2. Compute SHA-256
    print("\nComputing SHA-256 Checksum...")
    hasher = hashlib.sha256()
    with open(OUT_ZIP, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()
    print(f"SHA-256: {sha256}")

    print("\nFINAL PACKAGING AND VERIFICATION COMPLETE.")

if __name__ == "__main__":
    main()

