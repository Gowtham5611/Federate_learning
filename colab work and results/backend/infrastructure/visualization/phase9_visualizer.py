"""Publication-quality visualization engine for Phase 9 optimization experiments.

Generates 10 high-resolution (300 DPI) publication figures using Matplotlib Agg backend:
    01_fedprox_mu_vs_prauc.png
    02_fedprox_mu_vs_metrics.png
    03_fedavg_vs_fedprox_tuned.png
    04_scaffold_vs_fedavg_vs_fedprox.png
    05_convergence_curves_by_algorithm.png
    06_client_drift_comparison.png
    07_client_disagreement_trajectory.png
    08_model_architecture_sensitivity.png
    09_local_vs_global_update_norms.png
    10_phase9_comprehensive_radar_profile.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Scientific publication styling
PRIMARY_COLORS = {
    "fedavg": "#1f77b4",        # Steel Blue
    "fedprox": "#d62728",       # Crimson Red
    "scaffold": "#2ca02c",      # Forest Green
    "gru": "#9467bd",           # Purple
    "lstm": "#ff7f0e",          # Amber Orange
    "bilstm": "#17becf",        # Cyan Teal
}
DPI = 300


class Phase9Visualizer:
    """Renders 10 publication-quality comparison and ablation figures."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize visualizer with target directory."""
        self._output_dir = Path(output_dir).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        })

    def generate_all_plots(
        self,
        mu_ablation_df: pd.DataFrame,
        fedavg_df: pd.DataFrame,
        best_fedprox_df: pd.DataFrame,
        scaffold_df: pd.DataFrame,
        convergence_df: pd.DataFrame,
        drift_df: pd.DataFrame,
    ) -> list[Path]:
        """Generate all 10 Phase 9 publication figures."""
        saved_paths: list[Path] = []
        saved_paths.append(self.plot_01_mu_vs_prauc(mu_ablation_df))
        saved_paths.append(self.plot_02_mu_vs_metrics(mu_ablation_df))
        saved_paths.append(self.plot_03_fedavg_vs_fedprox(fedavg_df, best_fedprox_df))
        saved_paths.append(
            self.plot_04_three_way_comparison(fedavg_df, best_fedprox_df, scaffold_df)
        )
        saved_paths.append(self.plot_05_convergence_curves(convergence_df))
        saved_paths.append(self.plot_06_client_drift(drift_df))
        saved_paths.append(self.plot_07_client_disagreement(drift_df))
        saved_paths.append(self.plot_08_architecture_sensitivity(mu_ablation_df))
        saved_paths.append(self.plot_09_update_norms(drift_df))
        saved_paths.append(self.plot_10_radar_profile(fedavg_df, best_fedprox_df, scaffold_df))
        return saved_paths

    def plot_01_mu_vs_prauc(self, df: pd.DataFrame) -> Path:
        """Plot 01: FedProx proximal coefficient mu vs Validation PR-AUC across architectures."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for arch in ["gru", "lstm", "bilstm"]:
            sub = df[df["architecture"] == arch].sort_values("mu")
            if not sub.empty:
                color = PRIMARY_COLORS.get(arch, "#333333")
                ax.plot(
                    sub["mu"],
                    sub["val_pr_auc"],
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    label=arch.upper(),
                    color=color,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Proximal Coefficient $\\mu$ (log scale)")
        ax.set_ylabel("Validation PR-AUC")
        ax.set_title("FedProx: Effect of Proximal Coefficient $\\mu$ on Validation PR-AUC")
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
        ax.legend(title="Architecture")
        fig.tight_layout()
        out = self._output_dir / "01_fedprox_mu_vs_prauc.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_02_mu_vs_metrics(self, df: pd.DataFrame) -> Path:
        """Plot 02: Multi-metric ablation across mu for the overall best architecture."""
        fig, ax = plt.subplots(figsize=(8, 5))
        # Select best architecture based on mean validation PR-AUC
        best_arch = df.groupby("architecture")["val_pr_auc"].max().idxmax()
        sub = df[df["architecture"] == best_arch].sort_values("mu")

        ax.plot(sub["mu"], sub["val_pr_auc"], marker="o", label="PR-AUC", linewidth=2.2)
        ax.plot(sub["mu"], sub["val_f1"], marker="s", label="F1 Score", linewidth=2.0)
        ax.plot(sub["mu"], sub["val_recall"], marker="^", label="Recall", linewidth=2.0)
        ax.plot(sub["mu"], sub["val_precision"], marker="v", label="Precision", linewidth=2.0)

        ax.set_xscale("log")
        ax.set_xlabel("Proximal Coefficient $\\mu$ (log scale)")
        ax.set_ylabel("Validation Metric Value")
        ax.set_title(f"FedProx {str(best_arch).upper()}: Metric Response Across $\\mu$ Grid")
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "02_fedprox_mu_vs_metrics.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_03_fedavg_vs_fedprox(
        self, fedavg_df: pd.DataFrame, best_fedprox_df: pd.DataFrame
    ) -> Path:
        """Plot 03: Direct comparison between baseline FedAvg and best tuned FedProx."""
        fig, ax = plt.subplots(figsize=(8, 5))
        archs = ["gru", "lstm", "bilstm"]
        x = np.arange(len(archs))
        width = 0.35

        fedavg_vals = [
            fedavg_df[fedavg_df["architecture"] == a]["test_pr_auc"].values[0]
            if not fedavg_df[fedavg_df["architecture"] == a].empty else 0.0
            for a in archs
        ]
        fedprox_vals = [
            best_fedprox_df[best_fedprox_df["architecture"] == a]["test_pr_auc"].values[0]
            if not best_fedprox_df[best_fedprox_df["architecture"] == a].empty else 0.0
            for a in archs
        ]

        rects1 = ax.bar(
            x - width/2,
            fedavg_vals,
            width,
            label="FedAvg (Baseline)",
            color=PRIMARY_COLORS["fedavg"],
        )
        rects2 = ax.bar(
            x + width/2,
            fedprox_vals,
            width,
            label=r"FedProx (Tuned $\mu^*$)",
            color=PRIMARY_COLORS["fedprox"],
        )

        ax.set_ylabel("Test PR-AUC")
        ax.set_title("FedAvg Baseline vs Best Tuned FedProx (Test PR-AUC)")
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in archs])
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out = self._output_dir / "03_fedavg_vs_fedprox_tuned.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_04_three_way_comparison(
        self, fedavg_df: pd.DataFrame, best_fedprox_df: pd.DataFrame, scaffold_df: pd.DataFrame
    ) -> Path:
        """Plot 04: Three-way comparison between FedAvg, Best FedProx, and SCAFFOLD."""
        fig, ax = plt.subplots(figsize=(9, 5))
        archs = ["gru", "lstm", "bilstm"]
        x = np.arange(len(archs))
        width = 0.25

        fedavg_vals = [
            fedavg_df[fedavg_df["architecture"] == a]["test_pr_auc"].values[0]
            if not fedavg_df[fedavg_df["architecture"] == a].empty else 0.0
            for a in archs
        ]
        fedprox_vals = [
            best_fedprox_df[best_fedprox_df["architecture"] == a]["test_pr_auc"].values[0]
            if not best_fedprox_df[best_fedprox_df["architecture"] == a].empty else 0.0
            for a in archs
        ]
        scaffold_vals = [
            scaffold_df[scaffold_df["architecture"] == a]["test_pr_auc"].values[0]
            if not scaffold_df[scaffold_df["architecture"] == a].empty else 0.0
            for a in archs
        ]

        ax.bar(x - width, fedavg_vals, width, label="FedAvg", color=PRIMARY_COLORS["fedavg"])
        ax.bar(x, fedprox_vals, width, label="FedProx (Tuned)", color=PRIMARY_COLORS["fedprox"])
        ax.bar(x + width, scaffold_vals, width, label="SCAFFOLD", color=PRIMARY_COLORS["scaffold"])

        ax.set_ylabel("Test PR-AUC")
        ax.set_title("Three-Way Comparison: FedAvg vs Tuned FedProx vs SCAFFOLD")
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in archs])
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out = self._output_dir / "04_scaffold_vs_fedavg_vs_fedprox.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_05_convergence_curves(self, df: pd.DataFrame) -> Path:
        """Plot 05: Communication round convergence curves across algorithms."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].sort_values("round")
            color = PRIMARY_COLORS.get(algo.lower())
            ax.plot(
                sub["round"],
                sub["val_pr_auc"],
                marker="o",
                linewidth=2.2,
                label=algo.upper(),
                color=color,
            )
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Validation PR-AUC")
        ax.set_title("Convergence Trajectory Across Federated Rounds (Validation PR-AUC)")
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "05_convergence_curves_by_algorithm.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_06_client_drift(self, df: pd.DataFrame) -> Path:
        """Plot 06: Mean client drift ||w_client - w_global|| across communication rounds."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].sort_values("round")
            color = PRIMARY_COLORS.get(algo.lower())
            ax.plot(
                sub["round"],
                sub["mean_drift"],
                marker="o",
                linewidth=2.2,
                label=algo.upper(),
                color=color,
            )
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(r"Mean Client Parameter Drift $\|w_k - w_{\mathrm{global}}\|_2$")
        ax.set_title("Client Drift Comparison: FedAvg vs FedProx vs SCAFFOLD")
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "06_client_drift_comparison.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_07_client_disagreement(self, df: pd.DataFrame) -> Path:
        """Plot 07: Pairwise client update disagreement trajectory across rounds."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].sort_values("round")
            color = PRIMARY_COLORS.get(algo.lower())
            ax.plot(
                sub["round"],
                sub["pairwise_disagreement"],
                marker="s",
                linewidth=2.2,
                label=algo.upper(),
                color=color,
            )
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(r"Pairwise Client Disagreement $\frac{1}{3}\sum_{i<j} \|w_i - w_j\|_2$")
        ax.set_title("Client Disagreement Trajectory Across Rounds")
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "07_client_disagreement_trajectory.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_08_architecture_sensitivity(self, df: pd.DataFrame) -> Path:
        """Plot 08: Sensitivity of each architecture to the proximal penalty mu."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for arch in ["gru", "lstm", "bilstm"]:
            sub = df[df["architecture"] == arch].sort_values("mu")
            if not sub.empty:
                color = PRIMARY_COLORS.get(arch)
                baseline_val = sub[sub["mu"] == sub["mu"].min()]["val_pr_auc"].values[0]
                delta_prauc = sub["val_pr_auc"] - baseline_val
                ax.plot(
                    sub["mu"],
                    delta_prauc,
                    marker="^",
                    linewidth=2.0,
                    label=arch.upper(),
                    color=color,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Proximal Coefficient $\\mu$ (log scale)")
        ax.set_ylabel("$\\Delta$ Validation PR-AUC vs Weak Regularization")
        ax.set_title("Architecture Sensitivity to FedProx Regularization")
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "08_model_architecture_sensitivity.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_09_update_norms(self, df: pd.DataFrame) -> Path:
        """Plot 09: Local update norms across clients and rounds."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].sort_values("round")
            color = PRIMARY_COLORS.get(algo.lower())
            ax.plot(
                sub["round"],
                sub["weighted_drift"],
                marker="d",
                linewidth=2.0,
                label=f"{algo.upper()} (Weighted)",
                color=color,
            )
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(r"Sample-Weighted Update Norm $\sum_k p_k \|w_k - w_{\mathrm{global}}\|$")
        ax.set_title("Sample-Weighted Local Model Update Norms")
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out = self._output_dir / "09_local_vs_global_update_norms.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out

    def plot_10_radar_profile(
        self, fedavg_df: pd.DataFrame, best_fedprox_df: pd.DataFrame, scaffold_df: pd.DataFrame
    ) -> Path:
        """Plot 10: Multi-metric radar / bar profile across top configurations."""
        fig, ax = plt.subplots(figsize=(9, 5))
        metrics = ["test_pr_auc", "test_roc_auc", "test_f1", "test_recall", "test_precision"]
        metric_labels = ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
        x = np.arange(len(metrics))
        width = 0.25

        best_fedavg_row = fedavg_df.sort_values("test_pr_auc", ascending=False).iloc[0]
        best_fedprox_row = best_fedprox_df.sort_values("test_pr_auc", ascending=False).iloc[0]
        best_scaffold_row = scaffold_df.sort_values("test_pr_auc", ascending=False).iloc[0]

        v_fedavg = [best_fedavg_row[m] for m in metrics]
        v_fedprox = [best_fedprox_row[m] for m in metrics]
        v_scaffold = [best_scaffold_row[m] for m in metrics]

        lbl_avg = f"Best FedAvg ({best_fedavg_row['architecture'].upper()})"
        lbl_prox = (
            f"Best FedProx ({best_fedprox_row['architecture'].upper()}, "
            f"$\\mu={best_fedprox_row['mu']}$)"
        )
        lbl_scaf = f"Best SCAFFOLD ({best_scaffold_row['architecture'].upper()})"

        ax.bar(x - width, v_fedavg, width, label=lbl_avg, color=PRIMARY_COLORS["fedavg"])
        ax.bar(x, v_fedprox, width, label=lbl_prox, color=PRIMARY_COLORS["fedprox"])
        ax.bar(x + width, v_scaffold, width, label=lbl_scaf, color=PRIMARY_COLORS["scaffold"])

        ax.set_ylabel("Metric Value")
        ax.set_title("Comprehensive Performance Profile: Best Overall Configurations")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        fig.tight_layout()
        out = self._output_dir / "10_phase9_comprehensive_radar_profile.png"
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        return out
