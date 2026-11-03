import re
from pathlib import Path

def inspect_logs(log_dir):
    p = Path(log_dir)
    if not p.exists():
        print(f"Log dir does not exist: {p}")
        return
    print(f"\n=======================================================")
    print(f"LOG DIRECTORY: {p}")
    print(f"=======================================================")
    log_files = sorted(p.glob("*.log"))
    print(f"Found {len(log_files)} log files")
    for lf in log_files:
        print(f"\n--- Log File: {lf.name} ({lf.stat().st_size} bytes) ---")
        lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
        print(f"Total lines: {len(lines)}")
        # Print first 25 lines
        for l in lines[:25]:
            print(f"  [START] {l}")
        # Search for key patterns
        keywords = ["device", "cuda", "row", "transaction", "batch", "round", "eval", "test", "roc", "f1", "pr_auc", "oom", "error", "client", "partition"]
        matches = [l for l in lines if any(k in l.lower() for k in ["dataset", "rows", "device", "client", "test pr_auc", "test roc_auc", "completed", "config_"])]
        print(f"Sample key lines ({len(matches)} matches):")
        for m in matches[:15]:
            print(f"  {m}")
        if len(matches) > 15:
            print("  ...")
            for m in matches[-10:]:
                print(f"  {m}")

def main():
    inspect_logs("Colab Results/artifacts/li_medium_experiment/logs")
    inspect_logs("Colab Results/FedSecureAI_LI_Medium_configs_7_12/logs")

if __name__ == "__main__":
    main()

