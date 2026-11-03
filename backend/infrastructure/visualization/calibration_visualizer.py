"""Plotting reliability diagrams and calibration curves."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from infrastructure.evaluation.calibration_metrics import compute_ece

logger = logging.getLogger(__name__)

ProbArray = NDArray[np.float32] | NDArray[np.float64]


def plot_reliability_diagram(
    y_true: ProbArray,
    p_pred: ProbArray,
    title: str,
    output_path: Path,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> None:
    """Generate and save a reliability diagram with confidence histogram."""
    ece, mce, data = compute_ece(y_true, p_pred, n_bins=n_bins, strategy=strategy)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # 1. Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    if len(data.bin_pred_prob) > 0:
        ax1.plot(
            data.bin_pred_prob,
            data.bin_true_prob,
            marker="s",
            color="royalblue",
            linewidth=2,
            label=f"Model (ECE={ece:.4f}, MCE={mce:.4f})",
        )
        ax1.bar(
            data.bin_centers,
            data.bin_true_prob,
            width=1.0 / n_bins,
            alpha=0.2,
            color="royalblue",
            edgecolor="black",
        )

    ax1.set_ylabel("Empirical Probability (True Positives)")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.0))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Prediction distribution histogram
    if len(data.bin_centers) > 0:
        ax2.bar(
            data.bin_centers,
            data.bin_counts,
            width=1.0 / n_bins,
            color="gray",
            alpha=0.7,
            edgecolor="black",
        )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved reliability diagram to %s", output_path)


def plot_calibration_comparison_curves(
    curves: dict[str, tuple[ProbArray, ProbArray]],
    title: str,
    output_path: Path,
    n_bins: int = 15,
) -> None:
    """Plot multiple calibration curves on a single plot for direct comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for i, (name, (y_true, p_pred)) in enumerate(curves.items()):
        ece, _, data = compute_ece(y_true, p_pred, n_bins=n_bins)
        color = colors[i % len(colors)]
        if len(data.bin_pred_prob) > 0:
            ax.plot(
                data.bin_pred_prob,
                data.bin_true_prob,
                marker="o",
                color=color,
                linewidth=1.8,
                label=f"{name} (ECE={ece:.4f})",
            )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved calibration comparison curve to %s", output_path)
