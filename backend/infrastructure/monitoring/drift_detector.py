"""Drift Detection and Monitoring Module (Phase M).

Provides statistical drift monitoring for anti-money laundering:
- Input Feature Drift: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test
- Concept / Performance Drift: Rolling PR-AUC, Recall, Precision, and F1 degradation
- Alert generation: Green (PSI < 0.1), Yellow (0.1 <= PSI < 0.25), Red (PSI >= 0.25)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureDriftResult:
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_level: str  # "NO_DRIFT", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass(frozen=True)
class PerformanceDriftResult:
    window_id: str
    sample_count: int
    baseline_pr_auc: float
    current_pr_auc: float
    pr_auc_drop: float
    baseline_f1: float
    current_f1: float
    f1_drop: float
    precision: float
    recall: float
    performance_status: str  # "STABLE", "DEGRADED", "CRITICAL"


@dataclass
class ComprehensiveDriftReport:
    timestamp: str
    model_version: str
    dataset_name: str
    overall_input_drift_status: str
    overall_performance_drift_status: str
    features_analyzed_count: int
    drifted_features_count: int
    feature_drift_details: Dict[str, FeatureDriftResult]
    performance_drift: Optional[PerformanceDriftResult]
    recommended_action: str


class AmlDriftDetector:
    """Detects statistical input distribution drift and operational performance decay."""

    def __init__(self, psi_moderate_threshold: float = 0.10, psi_critical_threshold: float = 0.25):
        self.psi_mod_th = psi_moderate_threshold
        self.psi_crit_th = psi_critical_threshold

    @staticmethod
    def calculate_psi(
        reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4
    ) -> float:
        """Calculate Population Stability Index between reference baseline and current stream."""
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)

        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 10 or len(cur) < 10:
            return 0.0

        # Create quantile-based bins from reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Ensure monotonic bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)

        ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * len(ref_counts))
        cur_pct = (cur_counts + epsilon) / (len(cur) + epsilon * len(cur_counts))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(0.0, psi))

    def evaluate_feature_drift(
        self,
        ref_values: np.ndarray,
        cur_values: np.ndarray,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Evaluate single numerical feature drift using PSI and Kolmogorov-Smirnov test."""
        psi = self.calculate_psi(ref_values, cur_values)
        
        # KS test
        clean_ref = ref_values[np.isfinite(ref_values)]
        clean_cur = cur_values[np.isfinite(cur_values)]

        if len(clean_ref) > 5 and len(clean_cur) > 5:
            ks_res = stats.ks_2samp(clean_ref, clean_cur)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)
        else:
            ks_stat = 0.0
            ks_pval = 1.0

        if psi >= self.psi_crit_th:
            drift_level = "SIGNIFICANT_DRIFT"
        elif psi >= self.psi_mod_th:
            drift_level = "MODERATE_DRIFT"
        else:
            drift_level = "NO_DRIFT"

        return FeatureDriftResult(
            feature_name=feature_name,
            psi_score=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_p_value=round(ks_pval, 6),
            drift_level=drift_level,
            reference_mean=round(float(np.mean(clean_ref)), 4) if len(clean_ref) else 0.0,
            current_mean=round(float(np.mean(clean_cur)), 4) if len(clean_cur) else 0.0,
            reference_std=round(float(np.std(clean_ref)), 4) if len(clean_ref) else 0.0,
            current_std=round(float(np.std(clean_cur)), 4) if len(clean_cur) else 0.0,
        )

    def evaluate_performance_drift(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        baseline_pr_auc: float,
        baseline_f1: float,
        decision_threshold: float = 0.337784,
        window_id: str = "monitoring_window_current",
    ) -> PerformanceDriftResult:
        """Evaluate concept drift when delayed ground-truth labels are provided."""
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_probs, dtype=float)

        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            return PerformanceDriftResult(
                window_id=window_id,
                sample_count=len(y_t),
                baseline_pr_auc=baseline_pr_auc,
                current_pr_auc=0.0,
                pr_auc_drop=0.0,
                baseline_f1=baseline_f1,
                current_f1=0.0,
                f1_drop=0.0,
                precision=0.0,
                recall=0.0,
                performance_status="INSUFFICIENT_LABELS",
            )

        cur_pr_auc = float(average_precision_score(y_t, y_p))
        preds = (y_p >= decision_threshold).astype(int)
        cur_f1 = float(f1_score(y_t, preds, zero_division=0))
        prec = float(precision_score(y_t, preds, zero_division=0))
        rec = float(recall_score(y_t, preds, zero_division=0))

        pr_auc_drop = float(baseline_pr_auc - cur_pr_auc)
        f1_drop = float(baseline_f1 - cur_f1)

        if pr_auc_drop >= 0.15 or f1_drop >= 0.15:
            status = "CRITICAL"
        elif pr_auc_drop >= 0.05 or f1_drop >= 0.05:
            status = "DEGRADED"
        else:
            status = "STABLE"

        return PerformanceDriftResult(
            window_id=window_id,
            sample_count=len(y_t),
            baseline_pr_auc=round(baseline_pr_auc, 4),
            current_pr_auc=round(cur_pr_auc, 4),
            pr_auc_drop=round(pr_auc_drop, 4),
            baseline_f1=round(baseline_f1, 4),
            current_f1=round(cur_f1, 4),
            f1_drop=round(f1_drop, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            performance_status=status,
        )

    def generate_drift_report(
        self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame,
        numerical_features: List[str],
        model_version: str = "HI-Small_LSTM_v1.0",
        dataset_name: str = "HI-Small",
        y_true: Optional[np.ndarray] = None,
        y_probs: Optional[np.ndarray] = None,
        baseline_pr_auc: float = 0.5902,
        baseline_f1: float = 0.5891,
        decision_threshold: float = 0.337784,
    ) -> ComprehensiveDriftReport:
        """Run full input and performance drift analysis and package report."""
        feature_results: Dict[str, FeatureDriftResult] = {}
        drifted_count = 0

        for col in numerical_features:
            if col in ref_df.columns and col in cur_df.columns:
                ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna().to_numpy()
                cur_vals = pd.to_numeric(cur_df[col], errors="coerce").dropna().to_numpy()
                res = self.evaluate_feature_drift(ref_vals, cur_vals, col)
                feature_results[col] = res
                if res.drift_level in ("MODERATE_DRIFT", "SIGNIFICANT_DRIFT"):
                    drifted_count += 1

        overall_input_status = "NO_DRIFT"
        if any(r.drift_level == "SIGNIFICANT_DRIFT" for r in feature_results.values()):
            overall_input_status = "SIGNIFICANT_DRIFT"
        elif drifted_count > 0:
            overall_input_status = "MODERATE_DRIFT"

        perf_result: Optional[PerformanceDriftResult] = None
        overall_perf_status = "UNKNOWN"
        if y_true is not None and y_probs is not None:
            perf_result = self.evaluate_performance_drift(
                y_true=y_true,
                y_probs=y_probs,
                baseline_pr_auc=baseline_pr_auc,
                baseline_f1=baseline_f1,
                decision_threshold=decision_threshold,
            )
            overall_perf_status = perf_result.performance_status

        # Action recommendation policy
        if overall_perf_status == "CRITICAL" or overall_input_status == "SIGNIFICANT_DRIFT":
            action = "ALERT: Significant shift detected. Initiate analyst audit and review federated aggregation weights."
        elif overall_perf_status == "DEGRADED" or overall_input_status == "MODERATE_DRIFT":
            action = "MONITOR: Moderate feature/performance variation within tolerance. Continue surveillance."
        else:
            action = "NOMINAL: Distribution stable within expected operational boundaries."

        return ComprehensiveDriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=model_version,
            dataset_name=dataset_name,
            overall_input_drift_status=overall_input_status,
            overall_performance_drift_status=overall_perf_status,
            features_analyzed_count=len(feature_results),
            drifted_features_count=drifted_count,
            feature_drift_details=feature_results,
            performance_drift=perf_result,
            recommended_action=action,
        )

