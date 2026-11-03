import json
from pathlib import Path

p = Path("data/processed/ibm_aml/HI-Small/metadata/feature_metadata.json")
with open(p) as f:
    data = json.load(f)

for k, v in data.items():
    if isinstance(v, list):
        print(f"{k} (len {len(v)}): {v[:10]}")
    elif isinstance(v, dict):
        print(f"{k} (len {len(v)}): {list(v.items())[:5]}")
    else:
        print(f"{k}: {v}")

