"""FedSecureAI Evaluation Metrics Module.

Provides robust, production-grade AML evaluation metrics handling arbitrary
transaction distributions, zero alerts, single-class slices, low-prevalence warnings,
multiple operating points (frozen, alert budget, best-F1 what-if), and bootstrap confidence intervals.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def fast_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision) identically to scikit-learn.
    
    Optimized for rapid execution in bootstrap resampling loops.
    """
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    recall = tps / pos
    precision = tps / (tps + fps)
    rec_diff = np.diff(np.r_[0, recall])
    return float(np.sum(rec_diff * precision))


def compute_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    n = len(y_true)
    if n == 0:
        return 0.0
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p >= bin_lower) & (p < bin_upper)
        
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_pred = np.mean(p[in_bin])
            avg_true = np.mean(y_true[in_bin])
            ece += np.abs(avg_pred - avg_true) * (bin_count / n)
            
    return float(ece)


def compute_recall_at_alerts(y_true: np.ndarray, p: np.ndarray, budgets: List[int] = [10, 50, 100]) -> Dict[int, Optional[float]]:
    """Compute Recall at fixed alert budgets (alerts per 10,000 transactions)."""
    n = len(y_true)
    total_positives = int(np.sum(y_true == 1))
    results: Dict[int, Optional[float]] = {}
    
    if total_positives == 0 or n == 0:
        return {b: None for b in budgets}
        
    sorted_indices = np.argsort(p)[::-1]
    sorted_y = y_true[sorted_indices]
    
    for budget in budgets:
        # budget is alerts per 10,000 transactions
        k = max(1, int(round(n * (budget / 10000.0))))
        k = min(k, n)
        detected_positives = int(np.sum(sorted_y[:k] == 1))
        results[budget] = float(detected_positives / total_positives)
        
    return results


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    thr: float,
    name: str = "Operating Point"
) -> Dict[str, Any]:
    """Compute exact confusion matrix and threshold metrics at a specific decision threshold."""
    n = len(y_true)
    if n == 0:
        return {
            "name": name, "threshold": thr, "alerts": 0, "alerts_per_10k": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": None, "recall": None, "specificity": None,
            "balanced_accuracy": None, "f1_score": None, "f2_score": None,
            "accuracy": 0.0, "mcc": 0.0, "cohen_kappa": 0.0
        }

    positives = int(np.sum(y_true == 1))
    negatives = n - positives

    is_alert = (p >= thr)
    alerts = int(np.sum(is_alert))
    alerts_per_10k = float((alerts / n) * 10000.0)

    tp = int(np.sum((y_true == 1) & is_alert))
    fp = int(np.sum((y_true == 0) & is_alert))
    fn = int(np.sum((y_true == 1) & (~is_alert)))
    tn = int(np.sum((y_true == 0) & (~is_alert)))

    accuracy = float((tp + tn) / n)

    if positives > 0 and negatives > 0:
        recall = float(tp / (tp + fn))
        specificity = float(tn / (tn + fp))
        balanced_accuracy = float(0.5 * (recall + specificity))

        # Precision and FDR are only undefined (0/0) when 0 alerts are generated
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None

        f1_denom = 2 * tp + fp + fn
        f1 = float((2.0 * tp) / f1_denom) if f1_denom > 0 else 0.0

        f2_denom = 5 * tp + 4 * fn + fp
        f2 = float((5.0 * tp) / f2_denom) if f2_denom > 0 else 0.0

        # Matthews Correlation Coefficient
        mcc_num = float(tp * tn - fp * fn)
        mcc_den = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        mcc = float(mcc_num / mcc_den) if mcc_den > 0.0 else 0.0

        # Cohen's Kappa
        po = float(tp + tn) / float(n)
        pe = float((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / float(n * n)
        cohen_kappa = float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0

    elif positives == 0:
        # All negatives
        recall = None
        specificity = float(tn / n)
        balanced_accuracy = None
        precision = 0.0 if (tp + fp) > 0 else None
        f1 = None
        f2 = None
        mcc = 0.0
        cohen_kappa = 0.0
    else:
        # All positives
        recall = float(tp / n)
        specificity = None
        balanced_accuracy = None
        precision = 1.0 if (tp + fp) > 0 else None
        f1_denom = 2 * tp + fn
        f1 = float((2.0 * tp) / f1_denom) if f1_denom > 0 else None
        f2_denom = 5 * tp + 4 * fn
        f2 = float((5.0 * tp) / f2_denom) if f2_denom > 0 else None
        mcc = 0.0
        cohen_kappa = 0.0

    return {
        "name": name,
        "threshold": round(float(thr), 6),
        "alerts": alerts,
        "alerts_per_10k": round(alerts_per_10k, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "specificity": round(specificity, 6) if specificity is not None else None,
        "balanced_accuracy": round(balanced_accuracy, 6) if balanced_accuracy is not None else None,
        "f1_score": round(f1, 6) if f1 is not None else None,
        "f2_score": round(f2, 6) if f2 is not None else None,
        "mcc": round(mcc, 6) if mcc is not None else None,
        "cohen_kappa": round(cohen_kappa, 6) if cohen_kappa is not None else None,
    }


def compute_operating_points(
    y_true: np.ndarray,
    p: np.ndarray,
    frozen_thr: float = 0.337784
) -> Dict[str, Any]:
    """Compute performance across three standard operational perspectives:
    1. Frozen Canonical threshold (e.g. 0.337784).
    2. Operational Alert Budgets (top-10, top-50, top-100 per 10k transactions).
    3. What-if Best F1 on this file (tuned on-file optimization, labeled as what-if).
    """
    n = len(y_true)
    if n == 0:
        return {}

    # 1. Frozen operating point
    frozen_op = compute_metrics_at_threshold(
        y_true, p, thr=frozen_thr, name=f"Frozen Canonical ({frozen_thr:.4f})"
    )
    frozen_op["description"] = "Production threshold calibrated on offline validation split"

    # 2. Alert budgets (10, 50, 100 per 10k)
    alert_budgets: Dict[str, Any] = {}
    sorted_indices = np.argsort(p)[::-1]

    for b in [10, 50, 100]:
        k = max(1, min(n, int(round(n * (b / 10000.0)))))
        cutoff_idx = k - 1
        b_thr = float(p[sorted_indices[cutoff_idx]])
        b_op = compute_metrics_at_threshold(
            y_true, p, thr=b_thr, name=f"Alert Budget: {b} / 10k"
        )
        b_op["budget_per_10k"] = b
        b_op["target_alerts"] = k
        alert_budgets[str(b)] = b_op

    # 3. Best F1 on file (what-if)
    positives = int(np.sum(y_true == 1))
    best_f1_op: Dict[str, Any] = {
        "threshold": frozen_thr,
        "f1_score": 0.0,
        "precision": None,
        "recall": 0.0,
        "f2_score": 0.0,
        "alerts": 0,
        "tp": 0, "fp": 0, "fn": positives, "tn": n - positives,
        "label": "what-if tuned on this file, not a performance claim"
    }

    if positives > 0:
        # Candidate thresholds from score percentiles or unique scores
        unique_probs = np.unique(p)
        if len(unique_probs) > 300:
            candidate_thrs = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 200)))
        else:
            candidate_thrs = unique_probs

        best_f1 = -1.0
        best_thr = float(frozen_thr)
        best_cand_op = None

        for cand_thr in candidate_thrs:
            op = compute_metrics_at_threshold(y_true, p, thr=float(cand_thr))
            f1_val = op["f1_score"]
            if f1_val is not None:
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_thr = float(cand_thr)
                    best_cand_op = op
                elif f1_val == best_f1 and op["precision"] is not None and (best_cand_op is None or (op["precision"] > (best_cand_op.get("precision") or 0))):
                    best_thr = float(cand_thr)
                    best_cand_op = op

        if best_cand_op is not None:
            best_f1_op = {
                **best_cand_op,
                "name": f"Best F1 on File ({best_thr:.4f})",
                "label": "what-if tuned on this file, not a performance claim"
            }

    return {
        "frozen": frozen_op,
        "alert_budgets": alert_budgets,
        "best_f1_on_file": best_f1_op
    }


def evaluate(
    y_true: Union[np.ndarray, List[int], List[float]],
    p: Union[np.ndarray, List[float]],
    thr: float = 0.337784,
    n_boot: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluate binary classification predictions with safe division and full diagnostics.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    p : array-like
        Predicted probabilities in [0, 1].
    thr : float
        Decision threshold for positive alert classification. Default is 0.337784.
    n_boot : int
        Number of bootstrap iterations for 95% confidence intervals.
    random_state : int
        RNG seed for bootstrap reproducibility.
        
    Returns
    -------
    dict
        Comprehensive evaluation metrics, uncertainty intervals, warnings, and status.
    """
    y_arr = np.asarray(y_true, dtype=np.int32).ravel()
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    
    if len(y_arr) != len(p_arr):
        raise ValueError(f"Shape mismatch: y_true has {len(y_arr)} elements but p has {len(p_arr)} elements.")
        
    n = len(y_arr)
    if n == 0:
        return {
            "n": 0, "positives": 0, "prevalence": 0.0, "alerts": 0, "alerts_per_10k": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.0,
            "precision": None, "recall": None, "specificity": None, "balanced_accuracy": None,
            "f1": None, "f2": None, "mcc": 0.0, "cohen_kappa": 0.0,
            "pr_auc": None, "roc_auc": None, "random_baseline_pr_auc": 0.0,
            "pr_auc_lift": None, "brier_score": 0.0, "log_loss": None, "ece": 0.0,
            "recall_at_alerts": {10: None, 50: None, 100: None},
            "pr_auc_ci_lower": None, "pr_auc_ci_upper": None,
            "operating_points": {},
            "warnings": ["empty dataset"], "status": "warning"
        }
        
    positives = int(np.sum(y_arr == 1))
    negatives = n - positives
    prevalence = float(positives / n)
    
    is_alert = (p_arr >= thr)
    alerts = int(np.sum(is_alert))
    alerts_per_10k = float((alerts / n) * 10000.0)
    
    tp = int(np.sum((y_arr == 1) & is_alert))
    fp = int(np.sum((y_arr == 0) & is_alert))
    fn = int(np.sum((y_arr == 1) & (~is_alert)))
    tn = int(np.sum((y_arr == 0) & (~is_alert)))
    
    accuracy = float((tp + tn) / n)
    
    unique_classes = np.unique(y_arr)
    is_single_class = (len(unique_classes) < 2)

    # Safe division for threshold-dependent metrics
    if positives > 0 and negatives > 0:
        recall: Optional[float] = float(tp / (tp + fn))
        specificity: Optional[float] = float(tn / (tn + fp))
        balanced_accuracy: Optional[float] = float(0.5 * (recall + specificity))
        precision: Optional[float] = float(tp / (tp + fp)) if (tp + fp) > 0 else None

        f1_denom = 2 * tp + fp + fn
        f1: Optional[float] = float((2.0 * tp) / f1_denom) if f1_denom > 0 else 0.0

        f2_denom = 5 * tp + 4 * fn + fp
        f2: Optional[float] = float((5.0 * tp) / f2_denom) if f2_denom > 0 else 0.0

        # Matthews Correlation Coefficient
        mcc_num = float(tp * tn - fp * fn)
        mcc_den = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        mcc: Optional[float] = float(mcc_num / mcc_den) if mcc_den > 0.0 else 0.0

        # Cohen's Kappa
        po = float(tp + tn) / float(n)
        pe = float((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / float(n * n)
        cohen_kappa: Optional[float] = float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0

    elif positives == 0:
        recall = None
        specificity = float(tn / n)
        balanced_accuracy = None
        precision = 0.0 if (tp + fp) > 0 else None
        f1 = None
        f2 = None
        mcc = 0.0
        cohen_kappa = 0.0
    else:
        recall = float(tp / n)
        specificity = None
        balanced_accuracy = None
        precision = 1.0 if (tp + fp) > 0 else None
        f1_denom = 2 * tp + fn
        f1 = float((2.0 * tp) / f1_denom) if f1_denom > 0 else None
        f2_denom = 5 * tp + 4 * fn
        f2 = float((5.0 * tp) / f2_denom) if f2_denom > 0 else None
        mcc = 0.0
        cohen_kappa = 0.0

    # Ranking metrics (PR-AUC, ROC-AUC)
    if is_single_class:
        pr_auc: Optional[float] = None
        roc_auc: Optional[float] = None
        pr_auc_lift: Optional[float] = None
    else:
        try:
            pr_auc = float(average_precision_score(y_arr, p_arr))
        except Exception:
            pr_auc = float(fast_pr_auc(y_arr, p_arr))
            
        try:
            roc_auc = float(roc_auc_score(y_arr, p_arr))
        except Exception:
            roc_auc = None
            
        if pr_auc is not None and prevalence > 0:
            pr_auc_lift = float(pr_auc / prevalence)
        else:
            pr_auc_lift = None
            
    random_baseline_pr_auc = prevalence
    
    # Brier score and Log Loss
    brier_score = float(np.mean((p_arr - y_arr) ** 2))
    
    eps = 1e-15
    p_clipped = np.clip(p_arr, eps, 1.0 - eps)
    log_loss_val = float(-np.mean(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped)))
    
    # Expected Calibration Error
    ece = compute_ece(y_arr, p_arr, n_bins=10)
    
    # Recall at alert budgets
    recall_at_alerts = compute_recall_at_alerts(y_arr, p_arr, budgets=[10, 50, 100])
    
    # Warnings and Diagnostics
    warnings: List[str] = []
    if is_single_class:
        if positives == 0:
            warnings.append("single_class: only negative (legitimate) class present")
        else:
            warnings.append("single_class: only positive (laundering) class present")
    if positives < 30 and not is_single_class:
        warnings.append("metrics unstable, sample has <30 positives")
    if alerts == 0 and not is_single_class:
        warnings.append("zero alerts generated at specified threshold")
        
    # Bootstrap Confidence Intervals for PR-AUC
    pr_auc_ci_lower: Optional[float] = None
    pr_auc_ci_upper: Optional[float] = None
    pr_auc_ci_reason: Optional[str] = None
    
    if is_single_class:
        pr_auc_ci_reason = "single class"
    elif positives < 2:
        pr_auc_ci_reason = "too few positives (< 2)"
        warnings.append("too few positives for bootstrap confidence intervals (< 2)")
    elif n_boot > 0:
        rng = np.random.default_rng(random_state)
        boot_scores = []
        
        for _ in range(n_boot):
            boot_idx = rng.integers(0, n, size=n)
            y_b = y_arr[boot_idx]
            if np.any(y_b == 1) and np.any(y_b == 0):
                score = fast_pr_auc(y_b, p_arr[boot_idx])
                boot_scores.append(score)
                
        if len(boot_scores) >= 20:
            pr_auc_ci_lower = float(np.percentile(boot_scores, 2.5))
            pr_auc_ci_upper = float(np.percentile(boot_scores, 97.5))
        else:
            pr_auc_ci_reason = "too few valid bootstrap resamples"

    # Compute operating points
    operating_points = compute_operating_points(y_arr, p_arr, frozen_thr=thr)
            
    if is_single_class:
        status = "single_class"
    elif len(warnings) > 0:
        status = "warning"
    else:
        status = "ok"
    
    return {
        "n": n,
        "positives": positives,
        "prevalence": prevalence,
        "alerts": alerts,
        "alerts_per_10k": alerts_per_10k,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "f2": f2,
        "f1_score": f1,
        "f2_score": f2,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "random_baseline_pr_auc": random_baseline_pr_auc,
        "pr_auc_lift": pr_auc_lift,
        "brier_score": brier_score,
        "log_loss": log_loss_val,
        "ece": ece,
        "recall_at_alerts": recall_at_alerts,
        "pr_auc_ci_lower": pr_auc_ci_lower,
        "pr_auc_ci_upper": pr_auc_ci_upper,
        "pr_auc_ci_reason": pr_auc_ci_reason,
        "operating_points": operating_points,
        "warnings": warnings,
        "status": status
    }
