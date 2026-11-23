import numpy as np
from pathlib import Path

def main():
    p = Path("Colab Results/FedSecureAI_LI_Medium_configs_7_12/predictions/config_7/test_predictions.npz")
    data = np.load(p)
    labels = data["labels"]
    pos = int(np.sum(labels == 1.0))
    neg = int(np.sum(labels == 0.0))
    print(f"File: {p}")
    print(f"Total rows: {len(labels)}")
    print(f"Positive labels (Laundering): {pos}")
    print(f"Negative labels (Legitimate): {neg}")
    print(f"Prevalence: {pos / len(labels):.8f}")

if __name__ == "__main__":
    main()

