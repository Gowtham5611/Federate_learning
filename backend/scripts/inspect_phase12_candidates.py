import os
from pathlib import Path
import pandas as pd
import json

print("--- 1. PROVENANCE TABLE ---")
prov_path = Path("results/phase8/phase8_data_provenance.csv")
if prov_path.exists():
    df_prov = pd.read_csv(prov_path)
    for _, r in df_prov.iterrows():
        print(f"[{r['dataset']}] {r['config_id']}: {r['model_name']} -> {r['source_path']} (CKPT SHA: {str(r['checkpoint_sha256'])[:16]}...)")

print("\n--- 2. PHASE 7 SUMMARY AND FROZEN CONFIG ---")
p7_summary = Path("results/phase7/phase7_summary.csv")
if p7_summary.exists():
    df_p7 = pd.read_csv(p7_summary)
    print(df_p7.to_string())

p7_frozen = Path("results/phase7/frozen_ensemble_config.json")
if p7_frozen.exists():
    with open(p7_frozen) as f:
        print(json.dumps(json.load(f), indent=2))

print("\n--- 3. CHECKPOINTS ON DISK ---")
for ds in ["HI_Small_Results", "LI_Medium_Results"]:
    p = Path("results") / ds / "checkpoints"
    if p.exists():
        print(f"Found {p}:")
        for f in p.rglob("*.pt"):
            print(f"  {f} ({f.stat().st_size} bytes)")
    else:
        print(f"Not found: {p}")

print("\n--- 4. HISTORICAL REPO CHECKPOINTS (models/checkpoints) ---")
p_mc = Path("models/checkpoints")
if p_mc.exists():
    for f in p_mc.rglob("*.pt"):
        print(f"  {f} ({f.stat().st_size} bytes)")

