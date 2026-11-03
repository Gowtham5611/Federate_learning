import re
from pathlib import Path

p = Path("colab work and results/artifacts/fedavg_fedprox_full_comparison/logs/suite_execution.log")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

records = {}
curr_target = None
curr_rounds = {}

for line in lines:
    if "Configuration Target:" in line:
        m = re.search(r"Configuration Target:\s+([^\s]+)", line)
        if m:
            curr_target = m.group(1).strip()
            curr_rounds = {}
    elif "Round " in line and "Val PR-AUC=" in line and curr_target:
        # Match Round X/5 ... Val Loss=..., Val PR-AUC=..., Val ROC-AUC=...
        m = re.search(r"Round\s+(\d+)/\d+.*?Val Loss=([0-9.]+).*?Val PR-AUC=([0-9.]+).*?Val ROC-AUC=([0-9.]+)", line)
        if m:
            rnd = int(m.group(1))
            vloss = float(m.group(2))
            vpr = float(m.group(3))
            vroc = float(m.group(4))
            curr_rounds[rnd] = {"val_loss": vloss, "val_pr_auc": vpr, "val_roc_auc": vroc}
    elif "Complete:" in line and "Tau=" in line and curr_target:
        m = re.search(r"Tau=([0-9.]+)", line)
        tau = float(m.group(1)) if m else None
        if curr_target not in records and curr_rounds:
            # find best round by max val_pr_auc
            best_rnd = max(curr_rounds.keys(), key=lambda r: curr_rounds[r]["val_pr_auc"])
            records[curr_target] = {
                "best_round": best_rnd,
                "val_loss": curr_rounds[best_rnd]["val_loss"],
                "val_pr_auc": curr_rounds[best_rnd]["val_pr_auc"],
                "val_roc_auc": curr_rounds[best_rnd]["val_roc_auc"],
                "tau": tau
            }
        curr_target = None

print(f"Extracted {len(records)} configurations with validation metrics:")
for tgt, val in sorted(records.items()):
    if "hi_small" in tgt:
        print(f"  {tgt:<42} | Best Rnd {val['best_round']} | Val PR: {val['val_pr_auc']:.6f} | Val ROC: {val['val_roc_auc']:.6f} | Tau: {val['tau']}")

