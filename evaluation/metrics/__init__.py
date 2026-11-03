"""FedSecureAI Evaluation Metrics Module.

Provides robust, production-grade AML evaluation metrics handling arbitrary
transaction distributions, zero alerts, single-class slices, low-prevalence warnings,
and bootstrap confidence intervals.
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
            "precision": None, "recall": None, "specificity": None,
            "f1": None, "f2": None, "mcc": None, "cohen_kappa": None,
            "pr_auc": None, "roc_auc": None, "random_baseline_pr_auc": 0.0,
            "pr_auc_lift": None, "brier_score": 0.0, "log_loss": None, "ece": 0.0,
            "recall_at_alerts": {10: None, 50: None, 100: None},
            "pr_auc_ci_lower": None, "pr_auc_ci_upper": None,
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
    
    # Safe division for threshold-dependent metrics
    precision: Optional[float] = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall: Optional[float] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity: Optional[float] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    
    # F1 and F2 scores
    if precision is None or recall is None:
        f1: Optional[float] = None
        f2: Optional[float] = None
    elif (precision + recall) == 0.0:
        f1 = 0.0
        f2 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
        f2 = float(5.0 * precision * recall / (4.0 * precision + recall))
        
    # Matthews Correlation Coefficient (MCC)
    # Denominator = sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    denom_sq = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if denom_sq <= 0.0:
        mcc: Optional[float] = None
    else:
        mcc = float(((tp * tn) - (fp * fn)) / np.sqrt(denom_sq))
        
    # Cohen's Kappa
    # p_o = (tp + tn) / n
    # p_e = ((tp + fp)*(tp + fn) + (tn + fn)*(tn + fp)) / (n^2)
    po = float(tp + tn) / float(n)
    pe = float((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / float(n * n)
    if (1.0 - pe) <= 1e-12:
        cohen_kappa: Optional[float] = None
    else:
        cohen_kappa = float((po - pe) / (1.0 - pe))
        
    # Ranking metrics (PR-AUC, ROC-AUC)
    unique_classes = np.unique(y_arr)
    is_single_class = (len(unique_classes) < 2)
    
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
        warnings.append("dataset contains only one class")
    if positives < 30 and not is_single_class:
        warnings.append("metrics unstable, sample has <30 positives")
    if alerts == 0:
        warnings.append("zero alerts generated at specified threshold")
        
    # Bootstrap Confidence Intervals for PR-AUC
    pr_auc_ci_lower: Optional[float] = None
    pr_auc_ci_upper: Optional[float] = None
    
    if not is_single_class and n_boot > 0 and positives > 0:
        rng = np.random.default_rng(random_state)
        boot_scores = []
        
        # Optimize bootstrap speed for large datasets
        for _ in range(n_boot):
            boot_idx = rng.integers(0, n, size=n)
            y_b = y_arr[boot_idx]
            if np.any(y_b == 1) and np.any(y_b == 0):
                score = fast_pr_auc(y_b, p_arr[boot_idx])
                boot_scores.append(score)
                
        if len(boot_scores) >= 20:
            pr_auc_ci_lower = float(np.percentile(boot_scores, 2.5))
            pr_auc_ci_upper = float(np.percentile(boot_scores, 97.5))
            
    status = "warning" if len(warnings) > 0 else "ok"
    
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
        "f1": f1,
        "f2": f2,
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
        "warnings": warnings,
        "status": status
    }

