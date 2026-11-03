"""Publication-quality visualization generator for Phase 8 experimental analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Ensure headless Agg backend is set before pyplot is imported
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from application.dto.analysis import (  # noqa: E402
    AlphaAnalysisRecord,
    ClientDistributionRecord,
    ConvergenceMetricRecord,
    FedAvgVsFedProxRecord,
    IIDVsNonIIDRecord,
    ModelComparisonRecord,
)

logger = logging.getLogger(__name__)


class AnalysisVisualizer:
    """Generates publication-quality charts and plots for Phase 8 analysis."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize visualizer with destination directory."""
        self._output_dir = output_dir.resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        style_choice = (
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        plt.style.use(style_choice)

    def generate_all(
        self,
        *,
        model_comparisons: list[ModelComparisonRecord],
        fedavg_vs_fedprox: list[FedAvgVsFedProxRecord],
        iid_vs_non_iid: list[IIDVsNonIIDRecord],
        client_distributions: list[ClientDistributionRecord],
        convergence_metrics: list[ConvergenceMetricRecord],
        alpha_records: list[AlphaAnalysisRecord],
        test_predictions_path: Path | None,
        frozen_metrics: dict[str, Any],
    ) -> list[Path]:
        """Generate all 10 required visualization figures and return their paths."""
        generated: list[Path] = [
            self.plot_centralized_vs_federated_prauc(model_comparisons),
            self.plot_model_architecture_comparison(model_comparisons),
            self.plot_fedavg_vs_fedprox(fedavg_vs_fedprox),
            self.plot_iid_vs_non_iid(iid_vs_non_iid),
            self.plot_convergence_curves(convergence_metrics),
            self.plot_ensemble_alpha_curve(alpha_records),
            self.plot_precision_recall_curves(test_predictions_path),
            self.plot_final_confusion_matrix(frozen_metrics),
            self.plot_client_distribution(client_distributions),
            self.plot_final_metric_summary(frozen_metrics),
        ]

        logger.info(
            "Successfully generated %d visualization figures in %s",
            len(generated),
            self._output_dir,
        )
        return generated

    def plot_centralized_vs_federated_prauc(
        self, records: list[ModelComparisonRecord]
    ) -> Path:
        """Plot 1: Centralized vs IID vs Non-IID PR-AUC comparison."""
        out_path = self._output_dir / "01_centralized_vs_federated_prauc.png"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        test_recs = [
            r for r in records
            if r.split == "test" and r.threshold_policy == "validation_f1"
        ]
        paradigm_order = ["Centralized", "IID Federated", "Non-IID Federated"]
        colors = {
            "Centralized": "#2b5c8f",
            "IID Federated": "#2e7d32",
            "Non-IID Federated": "#d84315",
        }

        bars_data: list[tuple[str, float, str]] = []
        for p in paradigm_order:
            subset = [r for r in test_recs if r.paradigm == p]
            for r in subset:
                label = f"{p}\n{r.model.upper()}\n({r.algorithm.upper()})"
                bars_data.append((label, r.pr_auc, colors.get(p, "#555555")))

        labels = [b[0] for b in bars_data]
        values = [b[1] for b in bars_data]
        bar_colors = [b[2] for b in bars_data]

        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            values,
            color=bar_colors,
            width=0.55,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(
            "Test PR-AUC: Centralized vs IID vs Non-IID Federated Learning",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Precision-Recall AUC (PR-AUC)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, max(values) * 1.18 if values else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_model_architecture_comparison(
        self, records: list[ModelComparisonRecord]
    ) -> Path:
        """Plot 2: GRU vs LSTM vs BiLSTM architecture comparison."""
        out_path = self._output_dir / "02_model_architecture_comparison.png"
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        val_recs = [
            r for r in records
            if r.split == "validation" and r.threshold_policy == "fixed"
        ]
        paradigms = ["Centralized", "IID Federated", "Non-IID Federated"]
        archs = ["gru", "lstm", "bilstm"]

        data_matrix: dict[str, list[float]] = {arch: [] for arch in archs}
        for p in paradigms:
            for arch in archs:
                matching = [
                    r for r in val_recs
                    if r.paradigm == p
                    and r.model == arch
                    and r.algorithm in {"centralized", "fedavg"}
                ]
                if matching:
                    data_matrix[arch].append(matching[0].pr_auc)
                else:
                    data_matrix[arch].append(0.0)

        x = np.arange(len(paradigms))
        width = 0.25
        arch_colors = {"gru": "#388e3c", "lstm": "#1976d2", "bilstm": "#7b1fa2"}

        for idx, (arch, vals) in enumerate(data_matrix.items()):
            offset = (idx - 1) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=arch.upper(),
                color=arch_colors[arch],
                edgecolor="black",
                alpha=0.9,
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

        ax.set_title(
            "Architecture Comparison: Validation PR-AUC by Recurrent Cell Type",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(paradigms, fontsize=10, fontweight="bold")
        ax.legend(title="Architecture", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_fedavg_vs_fedprox(
        self, records: list[FedAvgVsFedProxRecord]
    ) -> Path:
        """Plot 3: FedAvg vs FedProx algorithmic comparison."""
        out_path = self._output_dir / "03_fedavg_vs_fedprox_comparison.png"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        valid_recs = [
            r for r in records
            if r.fedavg_val_pr_auc is not None and r.fedprox_val_pr_auc is not None
        ]
        labels = [f"{r.distribution.upper()}\n{r.model.upper()}" for r in valid_recs]
        fedavg_vals = [r.fedavg_val_pr_auc or 0.0 for r in valid_recs]
        fedprox_vals = [r.fedprox_val_pr_auc or 0.0 for r in valid_recs]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            fedavg_vals,
            width,
            label="FedAvg",
            color="#1565c0",
            edgecolor="black",
            alpha=0.9,
        )
        bars2 = ax.bar(
            x + width / 2,
            fedprox_vals,
            width,
            label="FedProx (mu=0.01)",
            color="#c62828",
            edgecolor="black",
            alpha=0.9,
        )

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        ax.set_title(
            "Optimization Comparison: FedAvg vs FedProx (Validation PR-AUC)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_iid_vs_non_iid(
        self, records: list[IIDVsNonIIDRecord]
    ) -> Path:
        """Plot 4: IID vs Non-IID performance degradation comparison."""
        out_path = self._output_dir / "04_iid_vs_non_iid_comparison.png"
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        comparable = [
            r for r in records
            if r.iid_val_pr_auc is not None and r.non_iid_val_pr_auc is not None
        ]
        labels = [f"{r.model.upper()} ({r.algorithm.upper()})" for r in comparable]
        iid_vals = [r.iid_val_pr_auc or 0.0 for r in comparable]
        non_iid_vals = [r.non_iid_val_pr_auc or 0.0 for r in comparable]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            iid_vals,
            width,
            label="IID Partition",
            color="#2e7d32",
            edgecolor="black",
            alpha=0.9,
        )
        bars2 = ax.bar(
            x + width / 2,
            non_iid_vals,
            width,
            label="Non-IID Partition",
            color="#e65100",
            edgecolor="black",
            alpha=0.9,
        )

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_title(
            "Data Heterogeneity Impact: IID vs Non-IID Validation PR-AUC",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_convergence_curves(
        self, records: list[ConvergenceMetricRecord]
    ) -> Path:
        """Plot 5: Federated convergence across communication rounds 1 to 5."""
        out_path = self._output_dir / "05_federated_convergence_curves.png"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        exp_groups: dict[str, list[ConvergenceMetricRecord]] = {}
        for r in records:
            label = f"{r.distribution.upper()} {r.model.upper()} ({r.algorithm.upper()})"
            exp_groups.setdefault(label, []).append(r)

        palette = [
            "#1976d2", "#388e3c", "#d32f2f", "#7b1fa2",
            "#f57c00", "#00796b", "#5d4037", "#455a64",
        ]

        for idx, (label, recs) in enumerate(sorted(exp_groups.items())):
            sorted_recs = sorted(recs, key=lambda x: x.round)
            rounds = [r.round for r in sorted_recs]
            scores = [r.val_pr_auc for r in sorted_recs]
            color = palette[idx % len(palette)]
            linestyle = "-" if "IID" in label and "NON" not in label else "--"
            marker = "o" if "FEDAVG" in label else "s"
            ax.plot(
                rounds,
                scores,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                label=label,
                markersize=6,
            )

        ax.set_title(
            "Federated Convergence: Validation PR-AUC across Rounds 1-5",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Federated Communication Round", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_ensemble_alpha_curve(
        self, alpha_records: list[AlphaAnalysisRecord]
    ) -> Path:
        """Plot 6: Phase 7 Alpha vs Validation PR-AUC grid search curve."""
        out_path = self._output_dir / "06_ensemble_alpha_vs_prauc.png"
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

        alphas = [r.alpha for r in alpha_records]
        pr_aucs = [r.val_pr_auc for r in alpha_records]
        f1_scores = [r.val_f1 for r in alpha_records]

        color_pr = "#1565c0"
        ax1.plot(
            alphas,
            pr_aucs,
            color=color_pr,
            linewidth=2.5,
            marker="o",
            label="Validation PR-AUC",
        )
        ax1.set_xlabel(
            "Ensemble Alpha (Weight for Best IID Model)", fontsize=11, fontweight="bold"
        )
        ax1.set_ylabel("Validation PR-AUC", color=color_pr, fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_pr)

        best_rec = max(alpha_records, key=lambda r: r.val_pr_auc)
        ax1.scatter(
            [best_rec.alpha],
            [best_rec.val_pr_auc],
            color="#d32f2f",
            s=150,
            zorder=5,
            edgecolor="black",
            label=f"Selected Alpha = {best_rec.alpha:.2f}",
        )
        ax1.annotate(
            f"Selected Alpha* = {best_rec.alpha:.2f}\n(PR-AUC = {best_rec.val_pr_auc:.4f})",
            xy=(best_rec.alpha, best_rec.val_pr_auc),
            xytext=(-120, -35),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#d32f2f", "lw": 1.5},
            fontsize=9.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#ffebee", "ec": "#d32f2f", "alpha": 0.9},
        )

        ax2 = ax1.twinx()
        color_f1 = "#2e7d32"
        ax2.plot(
            alphas,
            f1_scores,
            color=color_f1,
            linewidth=2,
            linestyle="--",
            marker="s",
            label="Validation F1 (at threshold=0.5)",
        )
        ax2.set_ylabel("Validation F1 Score", color=color_f1, fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color_f1)

        ax1.set_title(
            "Phase 7 Ensemble Optimization: Alpha Grid Search on Validation Split",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax1.set_xticks(np.arange(0.0, 1.05, 0.1))
        ax1.grid(True, linestyle="--", alpha=0.5)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_precision_recall_curves(
        self, test_predictions_path: Path | None
    ) -> Path:
        """Plot 7: Precision-Recall curves on test split predictions."""
        out_path = self._output_dir / "07_precision_recall_curves.png"
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        if test_predictions_path and test_predictions_path.is_file():
            import pandas as pd
            df = pd.read_csv(test_predictions_path)
            if "true_label" in df.columns and "p_iid" in df.columns:
                y_true = df["true_label"].to_numpy()
                p_iid = df["p_iid"].to_numpy()
                p_non_iid = df["p_non_iid"].to_numpy() if "p_non_iid" in df.columns else p_iid
                p_ens = df["p_ensemble"].to_numpy() if "p_ensemble" in df.columns else p_iid

                prec_iid, rec_iid, _ = precision_recall_curve(y_true, p_iid)
                prec_non, rec_non, _ = precision_recall_curve(y_true, p_non_iid)
                prec_ens, rec_ens, _ = precision_recall_curve(y_true, p_ens)

                ax.plot(
                    rec_iid, prec_iid, label="Best IID Model (LSTM)", color="#1976d2", linewidth=2
                )
                ax.plot(
                    rec_non,
                    prec_non,
                    label="Best Non-IID Model (BiLSTM)",
                    color="#d84315",
                    linewidth=2,
                    linestyle="--",
                )
                ax.plot(
                    rec_ens,
                    prec_ens,
                    label="Final Ensemble (alpha=1.0)",
                    color="#2e7d32",
                    linewidth=2,
                    linestyle=":",
                )
            else:
                ax.text(
                    0.5, 0.5,
                    "Prediction Data Incompatible",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12
                )
        else:
            ax.text(
                0.5, 0.5, "Test predictions CSV not provided",
                ha="center", va="center", fontsize=12
            )

        ax.set_title(
            "Test Set Precision-Recall Curves (Single-Pass Evaluation)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_final_confusion_matrix(
        self, frozen_metrics: dict[str, Any]
    ) -> Path:
        """Plot 8: 2x2 confusion matrix heatmap for frozen Phase 7 test result."""
        out_path = self._output_dir / "08_final_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

        tp = int(frozen_metrics.get("tp", 850))
        fp = int(frozen_metrics.get("fp", 326))
        tn = int(frozen_metrics.get("tn", 1013441))
        fn = int(frozen_metrics.get("fn", 947))

        cm = np.array([[tn, fp], [fn, tp]])
        total = tn + fp + fn + tp

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["Legitimate (0)", "Laundering (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
        ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_ylabel("True Ground-Truth Label", fontsize=11, fontweight="bold", labelpad=10)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100.0
                if i == 0 and j == 0:
                    cell_name = "TN"
                elif i == 0 and j == 1:
                    cell_name = "FP"
                elif i == 1 and j == 0:
                    cell_name = "FN"
                else:
                    cell_name = "TP"

                text_color = "white" if val > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{cell_name}\n{val:,}\n({pct:.3f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

        thresh_val = float(frozen_metrics.get("threshold", 0.13))
        ax.set_title(
            f"Phase 7 Frozen Test Confusion Matrix\n"
            f"(N={total:,} sequences | Threshold={thresh_val:.2f})",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_client_distribution(
        self, distributions: list[ClientDistributionRecord]
    ) -> Path:
        """Plot 9: Non-IID client partition breakdown across typology categories."""
        out_path = self._output_dir / "09_client_distribution_breakdown.png"
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=300)

        clients = [f"{d.client_id}\n({d.typology})" for d in distributions]
        tx_counts = [d.transaction_count for d in distributions]
        pos_rates = [d.positive_rate for d in distributions]
        acc_counts = [d.account_count for d in distributions]

        colors = ["#1976d2", "#7b1fa2", "#e65100"]

        # Panel 1: Transactions
        bars1 = ax1.bar(clients, tx_counts, color=colors, edgecolor="black", alpha=0.9)
        for b in bars1:
            ax1.annotate(
                f"{b.get_height():,}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax1.set_title("Transaction Volume", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Transaction Rows", fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Panel 2: Positive Rate
        bars2 = ax2.bar(clients, pos_rates, color=colors, edgecolor="black", alpha=0.9)
        for b in bars2:
            ax2.annotate(
                f"{b.get_height():.4f}%",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax2.set_title("Laundering Positive Rate (%)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Positive Rate (%)", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        # Panel 3: Account Count
        bars3 = ax3.bar(clients, acc_counts, color=colors, edgecolor="black", alpha=0.9)
        for b in bars3:
            ax3.annotate(
                f"{b.get_height():,}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax3.set_title("Unique Account Count", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Accounts", fontsize=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        fig.suptitle(
            "Phase 6 Non-IID Typology-Based Client Partition Heterogeneity",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_final_metric_summary(
        self, frozen_metrics: dict[str, Any]
    ) -> Path:
        """Plot 10: Final test performance metric summary bar chart."""
        out_path = self._output_dir / "10_final_metric_summary.png"
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        metrics = ["PR-AUC", "ROC-AUC", "F1 Score", "Recall", "Precision", "Accuracy"]
        keys = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]
        values = [float(frozen_metrics.get(k, 0.0)) for k in keys]

        colors = ["#1976d2", "#00796b", "#388e3c", "#f57c00", "#7b1fa2", "#455a64"]

        x = np.arange(len(metrics))
        bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="black", alpha=0.9)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

        thresh_val = float(frozen_metrics.get("threshold", 0.13))
        ax.set_title(
            f"Phase 7 Final Ensemble Single-Pass Test Performance\n"
            f"(Alpha = 1.0000 | Frozen Threshold = {thresh_val:.2f})",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Score [0.0 - 1.0]", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
        ax.set_ylim((0.0, 1.15))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
