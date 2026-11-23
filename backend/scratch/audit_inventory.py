import os
import sys
import json
import hashlib
from pathlib import Path

def compute_sha256(filepath, max_bytes=None):
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        if max_bytes:
            h.update(f.read(max_bytes))
        else:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()

def main():
    root = Path('.').resolve()
    print("=================================================================")
    print("FEDSECUREAI AUDIT: FULL INVENTORY & ARTIFACT DISCOVERY")
    print("=================================================================")

    # 1. Inspect Raw and Processed Datasets
    print("\n--- 1. DATASET PATHS & METADATA ---")
    data_dir = root / "data"
    raw_dir = data_dir / "raw" / "ibm_aml"
    proc_dir = data_dir / "processed" / "ibm_aml"

    datasets = ["HI-Small", "LI-Medium", "Combined"]
    for d in datasets:
        print(f"\nDataset: {d}")
        raw_csv = raw_dir / f"{d}_Trans.csv"
        if raw_csv.exists():
            sz = raw_csv.stat().st_size / (1024*1024)
            print(f"  Raw CSV: {raw_csv} ({sz:.2f} MB)")
        else:
            print(f"  Raw CSV: NOT FOUND at {raw_csv}")

        p_split = proc_dir / d / "metadata" / "splits.json"
        if p_split.exists():
            s_data = json.loads(p_split.read_text(encoding='utf-8'))
            print(f"  Processed splits metadata ({p_split.name}):")
            splits = s_data.get('splits', {})
            for sname, sinfo in splits.items():
                print(f"    - {sname}: rows={sinfo.get('row_count')}, pos={sinfo.get('positive_count')}, neg={sinfo.get('negative_count')}")
        else:
            print(f"  Processed splits metadata NOT FOUND at {p_split}")

    # 2. Inspect All ZIP Archives
    print("\n--- 2. ALL ZIP ARCHIVES DISCOVERED ---")
    all_zips = sorted(root.rglob("*.zip"))
    for z in all_zips:
        rel = z.relative_to(root)
        sz_mb = z.stat().st_size / (1024*1024)
        sha = compute_sha256(z)
        print(f"  Archive: {rel}")
        print(f"    Size: {sz_mb:.2f} MB ({z.stat().st_size} bytes)")
        print(f"    SHA-256: {sha}")

    # 3. Inspect Colab Results vs Local Artifacts
    print("\n--- 3. DETAILED INSPECTION OF COLAB RESULTS ---")
    colab_results_dir = root / "Colab Results"
    if colab_results_dir.exists():
        for sub in colab_results_dir.iterdir():
            if sub.is_dir():
                sub_files = list(sub.rglob("*"))
                f_count = sum(1 for f in sub_files if f.is_file())
                print(f"  Subdirectory: {sub.name} ({f_count} files)")
                
                # Check for manifest
                for m in sub.rglob("*manifest*.json"):
                    print(f"    Manifest: {m.relative_to(root)}")
                    m_data = json.loads(m.read_text(encoding='utf-8'))
                    print(f"      Keys: {list(m_data.keys())[:6]}")
                    if "configurations" in m_data:
                        cfgs = m_data["configurations"]
                        if isinstance(cfgs, dict):
                            print(f"      Total configurations: {len(cfgs)}")
                            for cid, cval in cfgs.items():
                                status = cval.get("status")
                                arch = cval.get("architecture") or cval.get("model")
                                dist = cval.get("distribution")
                                algo = cval.get("algorithm")
                                fm = cval.get("final_metrics") or cval.get("test_metrics") or {}
                                f1 = fm.get("test_f1", fm.get("f1"))
                                roc = fm.get("test_roc_auc", fm.get("roc_auc"))
                                pr = fm.get("test_pr_auc", fm.get("pr_auc"))
                                print(f"        {cid}: arch={arch}, dist={dist}, algo={algo}, status={status}, test_f1={f1}, roc_auc={roc}, pr_auc={pr}")
                        elif isinstance(cfgs, list):
                            print(f"      Total configurations: {len(cfgs)}")
                            for cval in cfgs:
                                cid = cval.get("config_id") or cval.get("exp_id")
                                status = cval.get("status")
                                arch = cval.get("architecture")
                                dist = cval.get("partition_strategy") or cval.get("distribution")
                                algo = cval.get("algorithm")
                                print(f"        {cid}: arch={arch}, dist={dist}, algo={algo}, status={status}")

    # 4. Inspect fedavg_fedprox_full_comparison (HI-Small & Combined)
    print("\n--- 4. INSPECTION OF fedavg_fedprox_full_comparison (HI-Small & Combined) ---")
    full_comp_dir = root / "artifacts" / "fedavg_fedprox_full_comparison"
    if full_comp_dir.exists():
        manifest_p = full_comp_dir / "resume_manifest.json"
        if manifest_p.exists():
            m_data = json.loads(manifest_p.read_text(encoding='utf-8'))
            cfgs = m_data.get("configurations", [])
            print(f"  Total configs in manifest: {len(cfgs)}")
            hi_cfgs = [c for c in cfgs if c.get("dataset") == "HI-Small"]
            print(f"  HI-Small configurations: {len(hi_cfgs)}")
            for c in hi_cfgs:
                exp_id = c.get("exp_id")
                dist = c.get("partition_strategy")
                algo = c.get("algorithm")
                arch = c.get("architecture")
                status = c.get("status")
                val_roc = c.get("val_roc_auc")
                test_roc = c.get("test_roc_auc")
                test_pr = c.get("test_pr_auc")
                ckpt = c.get("best_checkpoint_path")
                ckpt_exists = (root / ckpt).exists() if ckpt else False
                preds = c.get("predictions_path")
                preds_exists = (root / preds).exists() if preds else False
                print(f"    {exp_id}: arch={arch}, dist={dist}, algo={algo}, status={status}, val_roc={val_roc}, test_roc={test_roc}, test_pr={test_pr}, ckpt_exists={ckpt_exists}, preds_exists={preds_exists}")

    print("\n--- INVENTORY COMPLETED ---")

if __name__ == "__main__":
    main()

