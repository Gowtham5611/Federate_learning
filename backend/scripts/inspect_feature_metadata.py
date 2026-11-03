import json
from pathlib import Path

p = Path("data/processed/ibm_aml/HI-Small/metadata/feature_metadata.json")
if p.exists():
    with open(p) as f:
        data = json.load(f)
    print("Feature Metadata keys:", list(data.keys()))
    print("Features list length:", len(data.get("features", [])))
    for idx, feat in enumerate(data.get("features", [])):
        print(f"  {idx}: {feat}")

