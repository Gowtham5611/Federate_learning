"""Download LI-Medium_Trans.csv from Kaggle and extract if zipped."""

import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Set Kaggle environment variables
os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")

import kaggle

dataset_id = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
file_name = "LI-Medium_Trans.csv"
target_dir = Path("data/raw/ibm_aml")
target_dir.mkdir(parents=True, exist_ok=True)

target_csv = target_dir / file_name

if target_csv.exists():
    size = target_csv.stat().st_size
    print(f"Target file already exists: {target_csv} ({size:,} bytes)")
    if size == 2976436978:
        print("Size matches exactly (2,976,436,978 bytes). Ready.")
        sys.exit(0)
    else:
        print("Size differs from expected 2,976,436,978 bytes. Re-downloading...")

print(f"Authenticating Kaggle API...")
api = kaggle.KaggleApi()
api.authenticate()

print(f"Downloading {file_name} from {dataset_id} to {target_dir}...")
api.dataset_download_file(
    dataset_id,
    file_name=file_name,
    path=str(target_dir),
    force=True,
    quiet=False,
)

print("Download complete. Checking for zip archives...")
zip_candidates = list(target_dir.glob("LI-Medium_Trans*.zip")) + list(target_dir.glob("*.zip"))
for z in zip_candidates:
    print(f"Found archive: {z.name} ({z.stat().st_size:,} bytes). Extracting...")
    with zipfile.ZipFile(z, "r") as zf:
        zf.extractall(target_dir)
    print(f"Extracted {z.name}. Removing zip file...")
    z.unlink()

if target_csv.exists():
    size = target_csv.stat().st_size
    print(f"Extraction verified: {target_csv} exists with size {size:,} bytes.")
    if size == 2976436978:
        print("SUCCESS: Exact expected size matches: 2,976,436,978 bytes.")
    else:
        print(f"NOTE: Actual size is {size:,} bytes (expected 2,976,436,978 bytes).")
else:
    print("ERROR: LI-Medium_Trans.csv not found after extraction!")
    sys.exit(1)

