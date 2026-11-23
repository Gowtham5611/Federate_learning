from pathlib import Path

def main():
    p = Path("Colab Results/FedSecureAI_LI_Medium_configs_7_12/logs/full_combined_execution_20261020_154157.log")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, l in enumerate(lines[:60]):
        print(f"{i+1:3d}: {l}")

if __name__ == "__main__":
    main()

