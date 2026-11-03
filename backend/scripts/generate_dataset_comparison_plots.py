"""Generate Comparison Plots for HI-Small vs LI-Medium vs Combined."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plots_dir = Path("artifacts/dataset_comparison_combined/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("artifacts/dataset_comparison_combined/comparison_summary.csv")

# 1. Dataset Comparison (HI-Small vs LI-Medium vs Combined) by Architecture
plt.figure(figsize=(10, 6))
grouped = df.groupby(["training_dataset", "architecture"])["pr_auc"].mean().unstack()
grouped.plot(kind="bar", ax=plt.gca(), colormap="viridis", edgecolor="black")
plt.title("Mean Test PR-AUC: HI-Small vs. LI-Medium vs. Combined by Architecture", fontsize=12, fontweight="bold")
plt.ylabel("PR-AUC (Average Precision)", fontsize=11)
plt.xlabel("Training Dataset", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "dataset_comparison_by_architecture.png", dpi=300)
plt.close()

# 2. FedAvg vs FedProx (mu=0.1 vs mu=1.0)
plt.figure(figsize=(9, 5))
algo_group = df.groupby(["training_dataset", "algorithm", "mu"])["pr_auc"].mean().reset_index()
def make_label(r):
    if r["algorithm"] == "fedavg":
        return "FedAvg"
    return f"FedProx (mu={r['mu']})"

algo_group["algo_label"] = algo_group.apply(make_label, axis=1)
pivoted = algo_group.pivot(index="training_dataset", columns="algo_label", values="pr_auc")
pivoted.plot(kind="bar", ax=plt.gca(), colormap="plasma", edgecolor="black")
plt.title("FedAvg vs. FedProx (mu=0.1 vs. mu=1.0) Across Datasets", fontsize=12, fontweight="bold")
plt.ylabel("Mean PR-AUC", fontsize=11)
plt.xlabel("Training Dataset", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "fedavg_vs_fedprox_comparison.png", dpi=300)
plt.close()

# 3. IID vs Non-IID PR-AUC Degradation
plt.figure(figsize=(8, 5))
dist_group = df.groupby(["training_dataset", "distribution"])["pr_auc"].mean().unstack()
dist_group.plot(kind="bar", ax=plt.gca(), color=["#2b5c8f", "#d95f02"], edgecolor="black")
plt.title("Impact of Data Heterogeneity: IID vs. Non-IID Across Datasets", fontsize=12, fontweight="bold")
plt.ylabel("Mean PR-AUC", fontsize=11)
plt.xlabel("Training Dataset", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "iid_vs_non_iid_comparison.png", dpi=300)
plt.close()

print("Generated comparison plots under artifacts/dataset_comparison_combined/plots/")

