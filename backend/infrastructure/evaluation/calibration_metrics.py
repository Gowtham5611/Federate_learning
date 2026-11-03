"""Calibration metrics: ECE, MCE, Log Loss, Brier Score, and Reliability Diagrams."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from application.dto.calibration import CalibrationEvaluationMetrics
from infrastructure.evaluation.calibration_methods import (
    clip_probabilities,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReliabilityCurveData:
    """Bin-wise data for reliability diagrams."""

    bin_centers: NDArray[np.float64]
    bin_true_prob: NDArray[np.float64]
    bin_pred_prob: NDArray[np.float64]
    bin_counts: NDArray[np.int64]
    ece: float
    mce: float


def compute_ece(
    y_true: NDArray[np.float32] | NDArray[np.float64],
    p_pred: NDArray[np.float32] | NDArray[np.float64],
    *,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> tuple[float, float, ReliabilityCurveData]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Supports 'uniform' (equal width bins in [0, 1]) and 'quantile' (equal sample count bins).

    Returns:
        (ece, mce, reliability_data)
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    n = len(y)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return 0.0, 0.0, ReliabilityCurveData(
            empty, empty, empty, np.empty(0, dtype=np.int64), 0.0, 0.0
        )

    if strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(p, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, 1.0])
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers: list[float] = []
    bin_true_prob: list[float] = []
    bin_pred_prob: list[float] = []
    bin_counts: list[int] = []

    ece = 0.0
    mce = 0.0

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (p >= low) & (p <= high) if i == len(bin_edges) - 2 else (p >= low) & (p < high)

        count = int(np.count_nonzero(mask))
        if count > 0:
            avg_p = float(np.mean(p[mask]))
            acc = float(np.mean(y[mask]))
            err = abs(acc - avg_p)

            ece += (count / n) * err
            mce = max(mce, err)

            bin_centers.append(float((low + high) / 2.0))
            bin_true_prob.append(acc)
            bin_pred_prob.append(avg_p)
            bin_counts.append(count)

    curve_data = ReliabilityCurveData(
        bin_centers=np.array(bin_centers, dtype=np.float64),
        bin_true_prob=np.array(bin_true_prob, dtype=np.float64),
        bin_pred_prob=np.array(bin_pred_prob, dtype=np.float64),
        bin_counts=np.array(bin_counts, dtype=np.int64),
        ece=float(ece),
        mce=float(mce),
    )
    return float(ece), float(mce), curve_data


def evaluate_calibration_metrics(
    y_true: NDArray[np.float32] | NDArray[np.float64],
    p_pred: NDArray[np.float32] | NDArray[np.float64],
    *,
    threshold: float,
    historical_threshold: float = 0.13,
    n_bins_ece: int = 15,
    ece_strategy: str = "uniform",
) -> CalibrationEvaluationMetrics:
    """Compute complete suite of probability calibration and classification metrics."""
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(p_pred, dtype=np.float32)
    n = len(y)
    pos_count = int(np.count_nonzero(y == 1.0))
    prevalence = pos_count / n if n > 0 else 0.0

    # Probability quality metrics
    p_clipped = clip_probabilities(p)
    computed_log_loss = float(log_loss(y, p_clipped))
    computed_brier = float(brier_score_loss(y, p))
    ece_val, mce_val, _ = compute_ece(y, p, n_bins=n_bins_ece, strategy=ece_strategy)

    # Ranking metrics
    computed_pr_auc = float(average_precision_score(y, p)) if pos_count > 0 else 0.0
    computed_roc_auc = float(roc_auc_score(y, p)) if (0 < pos_count < n) else 0.5

    # Classification at chosen threshold
    pred_flag = p >= threshold
    tp = int(np.count_nonzero((y == 1.0) & pred_flag))
    fp = int(np.count_nonzero((y == 0.0) & pred_flag))
    fn = int(np.count_nonzero((y == 1.0) & ~pred_flag))
    tn = int(np.count_nonzero((y == 0.0) & ~pred_flag))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / n if n > 0 else 0.0

    # Classification at historical threshold
    hist_flag = p >= historical_threshold
    h_tp = int(np.count_nonzero((y == 1.0) & hist_flag))
    h_fp = int(np.count_nonzero((y == 0.0) & hist_flag))
    h_fn = int(np.count_nonzero((y == 1.0) & ~hist_flag))
    h_tn = int(np.count_nonzero((y == 0.0) & ~hist_flag))

    h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) > 0 else 0.0
    h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0.0
    h_f1 = (2.0 * h_prec * h_rec) / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0.0
    h_acc = (h_tp + h_tn) / n if n > 0 else 0.0

    return CalibrationEvaluationMetrics(
        log_loss=computed_log_loss,
        brier_score=computed_brier,
        ece=ece_val,
        mce=mce_val,
        pr_auc=computed_pr_auc,
        roc_auc=computed_roc_auc,
        threshold=float(threshold),
        f1=f1,
        precision=prec,
        recall=rec,
        accuracy=acc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        historical_threshold=float(historical_threshold),
        historical_f1=h_f1,
        historical_precision=h_prec,
        historical_recall=h_rec,
        historical_accuracy=h_acc,
        historical_tp=h_tp,
        historical_fp=h_fp,
        historical_tn=h_tn,
        historical_fn=h_fn,
        positive_count=pos_count,
        total_count=n,
        prevalence=prevalence,
    )


def select_optimal_f1_threshold(
    y_true: NDArray[np.float32],
    p_pred: NDArray[np.float32],
) -> float:
    """Find the threshold maximizing F1 on the given dataset (validation-only)."""
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(p_pred, dtype=np.float32)

    pos_count = np.count_nonzero(y == 1.0)
    if pos_count == 0:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y, p)
    f1_scores = np.zeros_like(thresholds)
    denom = precisions[:-1] + recalls[:-1]
    valid = denom > 0
    f1_scores[valid] = (2.0 * precisions[:-1][valid] * recalls[:-1][valid]) / denom[valid]

    if len(f1_scores) == 0 or np.all(f1_scores == 0):
        return 0.5

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])

