import numpy as np
from pathlib import Path

def main():
    paths = [
        Path("Colab Results/FedSecureAI_LI_Medium_configs_7_12/predictions/config_7/test_predictions.npz"),
        Path("Colab Results/FedSecureAI_LI_Medium_configs_7_12/predictions/config_8/test_predictions.npz"),
        Path("Colab Results/FedSecureAI_LI_Medium_configs_7_12/predictions/config_9/test_predictions.npz"),
        Path("Colab Results/artifacts/li_medium_experiment/predictions/config_1/test_predictions.npz"),
        Path("Colab Results/artifacts/li_medium_experiment/predictions/config_2/test_predictions.npz"),
    ]
    for p in paths:
        if p.exists():
            data = np.load(p)
            print(f"=== {p} ===")
            print("  Files inside npz:", data.files)
            for k in data.files:
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}")

if __name__ == "__main__":
    main()

