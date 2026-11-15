"""Unified AML Research Evaluation Suite.

Computes comprehensive regulatory compliance and operational metrics for highly imbalanced
transaction streams (0.1% positive prevalence):
1. Precision, Recall, F1, PR-AUC, ROC-AUC.
2. Confusion matrix counts (TP, FP, TN, FN).
3. Operational compliance trade-offs:
   - False Positives per 1,000 transactions (FP / 1k)
   - False Positives per 10,000 transactions (FP / 10k)
   - Recall at fixed low False Positive Rates: Recall@FPR=0.1% and Recall@FPR=1.0%
4. Probabilistic calibration quality:
   - Brier Score
   - Log Loss
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
5. Multi-client federated metrics aggregation:
   - Global model evaluation
   - Per-client local evaluations
   - Macro-averaged metrics
   - Sample-weighted averaged metrics
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

EPSILON = 1e-12


@dataclass
class AmlOperationalMetrics:
    """Standardized operational metrics for financial crime surveillance."""

    total_samples: int
    positive_samples: int
    negative_samples: int
    positive_prevalence_pct: float
    decision_threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_positives_per_1000: float
    false_positives_per_10000: float
    recall_at_fpr_0_1_pct: Optional[float]
    recall_at_fpr_1_pct: Optional[float]
    brier_score: float
    log_loss_value: float
    expected_calibration_error: float
    maximum_calibration_error: float
    edge_case_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedAmlEvaluationReport:
    """Multi-client federated performance report."""

    global_metrics: AmlOperationalMetrics
    client_metrics: Dict[str, AmlOperationalMetrics]
    macro_averages: Dict[str, float]
    sample_weighted_averages: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_metrics": self.global_metrics.to_dict(),
            "client_metrics": {k: v.to_dict() for k, v in self.client_metrics.items()},
            "macro_averages": self.macro_averages,
            "sample_weighted_averages": self.sample_weighted_averages,
        }


class AmlResearchEvaluator:
    """Computes publication-standard AML metrics adhering to regulatory requirements."""

    def __init__(self, n_calibration_bins: int = 15) -> None:
        self.n_bins = n_calibration_bins

    def evaluate(
        self,
        labels: NDArray[np.int64] | NDArray[np.float32] | List[int],
        probabilities: NDArray[np.float32] | NDArray[np.float64] | List[float],
        threshold: float,
    ) -> AmlOperationalMetrics:
        """Score predictions against true binary labels."""
        y = np.asarray(labels, dtype=np.int64).flatten()
        p = np.asarray(probabilities, dtype=np.float64).flatten()

        n = len(y)
        if n != len(p):
            raise ValueError(f"Length mismatch: {n} labels vs {len(p)} probabilities")

        if n == 0:
            raise ValueError("Cannot evaluate empty predictions array.")

        p_clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
        preds = (p >= threshold).astype(int)

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        prev_pct = round((pos_count / n) * 100.0, 4)

        tp = int(np.sum((y == 1) & (preds == 1)))
        fp = int(np.sum((y == 0) & (preds == 1)))
        tn = int(np.sum((y == 0) & (preds == 0)))
        fn = int(np.sum((y == 1) & (preds == 0)))

        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # Operational alert rates
        fp_per_1k = (fp / n) * 1000.0
        fp_per_10k = (fp / n) * 10000.0

        # Curve metrics & Recall at fixed FPRs
        pr_auc: Optional[float] = None
        roc_auc: Optional[float] = None
        rec_at_01: Optional[float] = None
        rec_at_1: Optional[float] = None
        note: Optional[str] = None

        if len(np.unique(y)) > 1:
            try:
                pr_auc = float(average_precision_score(y, p_clipped))
            except Exception:
                pr_auc = None

            try:
                roc_auc = float(roc_auc_score(y, p_clipped))
            except Exception:
                roc_auc = None

            try:
                fprs, tprs, _ = roc_curve(y, p_clipped)
                # Find maximum recall where FPR <= 0.001 (0.1%)
                mask_01 = fprs <= 0.001
                rec_at_01 = float(np.max(tprs[mask_01])) if np.any(mask_01) else 0.0

                # Find maximum recall where FPR <= 0.01 (1.0%)
                mask_1 = fprs <= 0.01
                rec_at_1 = float(np.max(tprs[mask_1])) if np.any(mask_1) else 0.0
            except Exception:
                rec_at_01 = None
                rec_at_1 = None
        else:
            note = "Single class evaluated; ranking curves (ROC, PR) require both classes."

        # Probabilistic calibration metrics
        brier = float(brier_score_loss(y, p_clipped))
        try:
            ll = float(log_loss(y, p_clipped))
        except Exception:
            ll = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))

        ece, mce = self._compute_ece_mce(y, p_clipped, n_bins=self.n_bins)

        return AmlOperationalMetrics(
            total_samples=n,
            positive_samples=pos_count,
            negative_samples=neg_count,
            positive_prevalence_pct=prev_pct,
            decision_threshold=round(float(threshold), 6),
            accuracy=round(float(acc), 6),
            precision=round(float(prec), 6),
            recall=round(float(rec), 6),
            f1_score=round(float(f1), 6),
            pr_auc=round(pr_auc, 6) if pr_auc is not None else None,
            roc_auc=round(roc_auc, 6) if roc_auc is not None else None,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            false_positives_per_1000=round(float(fp_per_1k), 3),
            false_positives_per_10000=round(float(fp_per_10k), 3),
            recall_at_fpr_0_1_pct=round(rec_at_01, 6) if rec_at_01 is not None else None,
            recall_at_fpr_1_pct=round(rec_at_1, 6) if rec_at_1 is not None else None,
            brier_score=round(brier, 6),
            log_loss_value=round(ll, 6),
            expected_calibration_error=round(float(ece), 6),
            maximum_calibration_error=round(float(mce), 6),
            edge_case_note=note,
        )

    def evaluate_federated(
        self,
        global_labels: NDArray[np.int64],
        global_probs: NDArray[np.float64],
        client_data: Dict[str, Tuple[NDArray[np.int64], NDArray[np.float64]]],
        threshold: float,
    ) -> FederatedAmlEvaluationReport:
        """Evaluate global model performance alongside per-client breakdown."""
        global_metrics = self.evaluate(global_labels, global_probs, threshold)

        client_metrics: Dict[str, AmlOperationalMetrics] = {}
        for cid, (c_lbls, c_probs) in client_data.items():
            client_metrics[cid] = self.evaluate(c_lbls, c_probs, threshold)

        # Compute macro and sample-weighted averages
        tracked_keys = ["precision", "recall", "f1_score", "brier_score", "false_positives_per_1000"]
        macro: Dict[str, float] = {}
        weighted: Dict[str, float] = {}

        total_client_samples = sum(m.total_samples for m in client_metrics.values())
        k_count = len(client_metrics)

        for k in tracked_keys:
            vals = [getattr(m, k) for m in client_metrics.values() if getattr(m, k) is not None]
            macro[k] = round(float(np.mean(vals)), 6) if vals else 0.0

            w_sum = sum(
                getattr(m, k) * (m.total_samples / total_client_samples)
                for m in client_metrics.values()
                if getattr(m, k) is not None
            )
            weighted[k] = round(float(w_sum), 6)

        return FederatedAmlEvaluationReport(
            global_metrics=global_metrics,
            client_metrics=client_metrics,
            macro_averages=macro,
            sample_weighted_averages=weighted,
        )

    def _compute_ece_mce(
        self, y: NDArray[np.int64], p: NDArray[np.float64], n_bins: int = 15
    ) -> Tuple[float, float]:
        """Compute Expected Calibration Error and Maximum Calibration Error."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(y)

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (p >= low) & (p <= high) if i == n_bins - 1 else (p >= low) & (p < high)
            bin_size = int(np.sum(mask))
            if bin_size > 0:
                bin_acc = float(np.mean(y[mask]))
                bin_conf = float(np.mean(p[mask]))
                diff = abs(bin_acc - bin_conf)
                ece += (bin_size / n) * diff
                if diff > mce:
                    mce = diff

        return ece, mce

