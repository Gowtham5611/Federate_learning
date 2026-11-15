"""Business-Cost-Aware Threshold Optimization Module (Phase F).

Implements rigorous, validation-only decision threshold selection balancing operational
financial compliance tradeoffs:
- Cost-Based Optimization: Minimizes Expected_Cost = FN * Cost_FN + FP * Cost_FP
- F1-Score Optimization: Maximizes harmonic mean of Precision and Recall
- Recall-at-Fixed-FPR: Maximizes recall subject to FPR <= alpha (e.g. 0.1% or 1.0%)
- Constrained Optimization: Maximize Recall subject to Precision >= P_min

Strict Safeguards:
- Thresholds are fitted exclusively on validation predictions.
- Selected threshold is frozen prior to test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class CostConfiguration:
    cost_fn: float = 1000.0  # Cost of an undetected money laundering event (regulatory fines, reputational loss)
    cost_fp: float = 25.0    # Cost of an analyst manually investigating a false alert
    benefit_tp: float = 0.0  # Optional financial recovery credit
    description: str = "Standard AML Tier-1 Bank Cost Structure ($1000 FN penalty, $25 analyst alert triage)"


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    selected_threshold: float
    optimization_method: str
    validation_f1: float
    validation_precision: float
    validation_recall: float
    validation_fpr: float
    validation_expected_cost: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    cost_configuration: Dict[str, Any]
    candidate_thresholds_evaluated: int


class CostAwareThresholdOptimizer:
    """Optimizes classification decision thresholds on validation probabilities."""

    def __init__(self, cost_config: Optional[CostConfiguration] = None):
        self.cost_config = cost_config or CostConfiguration()

    def optimize_threshold(
        self,
        y_val_true: np.ndarray,
        y_val_probs: np.ndarray,
        method: str = "cost_minimization",  # "cost_minimization", "f1_maximization", "recall_at_fpr", "precision_constrained"
        target_fpr: float = 0.01,
        min_precision: float = 0.20,
        num_candidates: int = 101,
    ) -> ThresholdOptimizationResult:
        """Find the optimal decision threshold on validation data using the specified criterion."""
        y_true = np.asarray(y_val_true, dtype=int)
        probs = np.asarray(y_val_probs, dtype=float)

        if len(y_true) != len(probs):
            raise ValueError(f"Shape mismatch: {len(y_true)} targets vs {len(probs)} probabilities.")

        # Discretize candidate thresholds in [0.01, 0.99]
        candidates = np.linspace(0.01, 0.99, num_candidates)

        best_th = 0.5
        best_score = float("inf") if method == "cost_minimization" else -1.0
        best_metrics: Dict[str, Any] = {}

        n_neg = int((y_true == 0).sum())

        for th in candidates:
            preds = (probs >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            fpr = float(fp / max(1, n_neg))
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))

            expected_cost = float(
                fn * self.cost_config.cost_fn
                + fp * self.cost_config.cost_fp
                - tp * self.cost_config.benefit_tp
            )

            is_better = False

            if method == "cost_minimization":
                if expected_cost < best_score:
                    best_score = expected_cost
                    is_better = True

            elif method == "f1_maximization":
                if f1 > best_score:
                    best_score = f1
                    is_better = True

            elif method == "recall_at_fpr":
                # Maximize recall subject to FPR <= target_fpr
                if fpr <= target_fpr:
                    if rec > best_score:
                        best_score = rec
                        is_better = True

            elif method == "precision_constrained":
                # Maximize recall subject to precision >= min_precision
                if prec >= min_precision:
                    if rec > best_score:
                        best_score = rec
                        is_better = True
            else:
                raise ValueError(f"Unsupported threshold optimization method: '{method}'")

            if is_better:
                best_th = float(th)
                best_metrics = {
                    "f1": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "fpr": round(fpr, 6),
                    "expected_cost": round(expected_cost, 2),
                    "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
                }

        # Fallback if no candidate satisfied constraint
        if not best_metrics:
            preds_50 = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds_50, labels=[0, 1]).ravel()
            best_th = 0.5
            best_metrics = {
                "f1": round(float(f1_score(y_true, preds_50, zero_division=0)), 4),
                "precision": round(float(precision_score(y_true, preds_50, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, preds_50, zero_division=0)), 4),
                "fpr": round(float(fp / max(1, n_neg)), 6),
                "expected_cost": round(float(fn * self.cost_config.cost_fn + fp * self.cost_config.cost_fp), 2),
                "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
            }

        return ThresholdOptimizationResult(
            selected_threshold=round(best_th, 6),
            optimization_method=method,
            validation_f1=best_metrics["f1"],
            validation_precision=best_metrics["precision"],
            validation_recall=best_metrics["recall"],
            validation_fpr=best_metrics["fpr"],
            validation_expected_cost=best_metrics["expected_cost"],
            confusion_matrix=best_metrics["cm"],
            cost_configuration={
                "cost_fn": self.cost_config.cost_fn,
                "cost_fp": self.cost_config.cost_fp,
                "benefit_tp": self.cost_config.benefit_tp,
                "description": self.cost_config.description,
            },
            candidate_thresholds_evaluated=len(candidates),
        )

