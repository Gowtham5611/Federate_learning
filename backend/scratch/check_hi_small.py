import json
from pathlib import Path

def main():
    paths = [
        Path("artifacts/fedavg_fedprox_full_comparison/resume_manifest.json"),
        Path("colab work and results/artifacts/fedavg_fedprox_full_comparison/resume_manifest.json"),
        Path("colab work and results/resume_manifest.json"),
        Path("resume_manifest.json")
    ]
    for p in paths:
        if p.exists():
            print(f"\n=== Manifest: {p} ===")
            data = json.loads(p.read_text(encoding="utf-8"))
            cfgs = data.get("configurations", [])
            if isinstance(cfgs, list):
                hi_cfgs = [c for c in cfgs if c.get("dataset") == "HI-Small"]
                print(f"  HI-Small count: {len(hi_cfgs)}")
                for c in hi_cfgs:
                    eid = c.get("exp_id")
                    st = c.get("status")
                    troc = c.get("test_roc_auc")
                    tpr = c.get("test_pr_auc")
                    tf1 = c.get("test_f1")
                    print(f"    {eid}: status={st}, roc={troc}, pr={tpr}, f1={tf1}")
            elif isinstance(cfgs, dict):
                hi_cfgs = {k: v for k, v in cfgs.items() if "hi_small" in str(v).lower() or "hi-small" in str(v).lower()}
                print(f"  HI-Small dict count: {len(hi_cfgs)}")
                for k, v in hi_cfgs.items():
                    print(f"    {k}: {v.get('status')}")

if __name__ == "__main__":
    main()

