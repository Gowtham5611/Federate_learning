"""Dedicated persistence for Phase 7 IID + Non-IID Ensemble results and predictions.

Organizes results into:
    experiments/results/ensemble/<ensemble_experiment_id>/
        validation/
            model_selection.csv
            alpha_search.csv
            validation_metrics.json
        test/
            test_predictions.csv
            test_metrics.json
        report.json
        metadata.json
        results.csv
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from application.dto.ensemble import (
    AlphaCandidateRecord,
    EnsembleExperimentReport,
    ModelSelectionRecord,
)

logger = logging.getLogger(__name__)

RESULTS_COLUMNS = [
    "experiment_id",
    "model",
    "training_mode",
    "data_distribution",
    "federated_algorithm",
    "client_id",
    "round",
    "mu",
    "loss_type",
    "split",
    "threshold_policy",
    "recall",
    "f1",
    "pr_auc",
    "precision",
    "false_negatives",
    "roc_auc",
    "accuracy",
    "tp",
    "tn",
    "fp",
    "fn",
    "threshold",
    "positive_rate",
    "loss",
    "missed_positive_rate",
    "sequence_count",
    "training_time",
    "evaluation_time",
    "seed",
    "hyperparameters",
]


class EnsembleResultStore:
    """Manages isolated file-system storage for Phase 7 Ensemble experiments."""

    def __init__(self, root: Path) -> None:
        """Initialize store with results root directory."""
        self._root = root.resolve() / "ensemble"

    @property
    def root(self) -> Path:
        """Root directory where ensemble experiments reside."""
        return self._root

    def experiment_dir(self, experiment_id: str) -> Path:
        """Base directory for a specific ensemble experiment."""
        return self._root / experiment_id

    def save_model_selection(
        self,
        experiment_id: str,
        iid_candidates: list[ModelSelectionRecord],
        non_iid_candidates: list[ModelSelectionRecord],
        best_iid: ModelSelectionRecord,
        best_non_iid: ModelSelectionRecord,
    ) -> Path:
        """Save candidate model selection results to validation/model_selection.csv."""
        val_dir = self.experiment_dir(experiment_id) / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        csv_path = val_dir / "model_selection.csv"

        rows: list[dict[str, Any]] = []
        for c in [*iid_candidates, *non_iid_candidates]:
            d = c.to_dict()
            d["is_selected"] = (
                (c.experiment_id == best_iid.experiment_id)
                or (c.experiment_id == best_non_iid.experiment_id)
            )
            rows.append(d)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info("Saved model selection table to %s", csv_path)
        return csv_path

    def save_alpha_search(
        self,
        experiment_id: str,
        alpha_candidates: list[AlphaCandidateRecord],
    ) -> Path:
        """Save alpha search sweep results to validation/alpha_search.csv."""
        val_dir = self.experiment_dir(experiment_id) / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        csv_path = val_dir / "alpha_search.csv"

        rows = [c.to_dict() for c in alpha_candidates]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info("Saved alpha search table to %s", csv_path)
        return csv_path

    def save_validation_metrics(
        self,
        experiment_id: str,
        metrics_dict: dict[str, Any],
    ) -> Path:
        """Save detailed validation metrics to validation/validation_metrics.json."""
        val_dir = self.experiment_dir(experiment_id) / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        json_path = val_dir / "validation_metrics.json"

        content = json.dumps(metrics_dict, indent=2, sort_keys=True, default=str) + "\n"
        json_path.write_text(content, encoding="utf-8")
        logger.info("Saved validation metrics to %s", json_path)
        return json_path

    def save_test_predictions(
        self,
        experiment_id: str,
        row_indices: NDArray[np.int64],
        true_labels: NDArray[np.float32],
        p_iid: NDArray[np.float32],
        p_non_iid: NDArray[np.float32],
        p_ensemble: NDArray[np.float32],
        frozen_threshold: float,
    ) -> Path:
        """Save aligned test predictions to test/test_predictions.csv."""
        test_dir = self.experiment_dir(experiment_id) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        csv_path = test_dir / "test_predictions.csv"

        predicted_labels = (p_ensemble >= frozen_threshold).astype(int)

        df = pd.DataFrame({
            "row_index": row_indices,
            "true_label": true_labels.astype(int),
            "p_iid": p_iid,
            "p_non_iid": p_non_iid,
            "p_ensemble": p_ensemble,
            "frozen_threshold": frozen_threshold,
            "predicted_label": predicted_labels,
        })
        df.to_csv(csv_path, index=False)
        logger.info("Saved test predictions table (%d rows) to %s", len(df), csv_path)
        return csv_path

    def save_test_metrics(
        self,
        experiment_id: str,
        metrics_dict: dict[str, Any],
    ) -> Path:
        """Save detailed test metrics to test/test_metrics.json."""
        test_dir = self.experiment_dir(experiment_id) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        json_path = test_dir / "test_metrics.json"

        content = json.dumps(metrics_dict, indent=2, sort_keys=True, default=str) + "\n"
        json_path.write_text(content, encoding="utf-8")
        logger.info("Saved test metrics to %s", json_path)
        return json_path

    def save_report_and_metadata(
        self,
        report: EnsembleExperimentReport,
    ) -> tuple[Path, Path]:
        """Save report.json and metadata.json under the experiment root."""
        exp_dir = self.experiment_dir(report.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        report_path = exp_dir / "report.json"
        meta_path = exp_dir / "metadata.json"

        content = json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        report_path.write_text(content, encoding="utf-8")
        meta_path.write_text(content, encoding="utf-8")
        logger.info("Saved report and metadata to %s", exp_dir)
        return report_path, meta_path

    def save_results_csv(
        self,
        report: EnsembleExperimentReport,
        sequence_count_val: int,
        sequence_count_test: int,
    ) -> Path:
        """Write tabular summary to results.csv in the experiment directory."""
        exp_dir = self.experiment_dir(report.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = exp_dir / "results.csv"

        iid_model = report.selected_iid.model
        non_iid_model = report.selected_non_iid.model
        combined_model_name = f"ensemble_{iid_model}_{non_iid_model}"
        combined_algo = (
            f"{report.selected_iid.algorithm}+{report.selected_non_iid.algorithm}"
        )
        hp_json = json.dumps(report.config.to_dict(), sort_keys=True)

        rows: list[dict[str, Any]] = []

        # 1. Validation at fixed threshold 0.5
        v_fix = report.validation_metrics_at_fixed_threshold
        rows.append({
            "experiment_id": report.experiment_id,
            "model": combined_model_name,
            "training_mode": "ensemble",
            "data_distribution": "iid_and_non_iid",
            "federated_algorithm": combined_algo,
            "client_id": None,
            "round": None,
            "mu": None,
            "loss_type": "ensemble_probability",
            "split": "validation",
            "threshold_policy": "fixed",
            "recall": v_fix.recall,
            "f1": v_fix.f1,
            "pr_auc": v_fix.pr_auc,
            "precision": v_fix.precision,
            "false_negatives": v_fix.confusion.false_negatives,
            "roc_auc": v_fix.roc_auc,
            "accuracy": v_fix.accuracy,
            "tp": v_fix.confusion.true_positives,
            "tn": v_fix.confusion.true_negatives,
            "fp": v_fix.confusion.false_positives,
            "fn": v_fix.confusion.false_negatives,
            "threshold": 0.5,
            "positive_rate": v_fix.positive_rate,
            "loss": v_fix.loss,
            "missed_positive_rate": v_fix.confusion.missed_positive_rate,
            "sequence_count": sequence_count_val,
            "training_time": report.timing.get("tuning_seconds", 0.0),
            "evaluation_time": report.timing.get("val_eval_seconds", 0.0),
            "seed": report.config.seed,
            "hyperparameters": hp_json,
        })

        # 2. Validation at frozen validation_f1 threshold
        v_fro = report.validation_metrics_at_frozen_threshold
        rows.append({
            "experiment_id": report.experiment_id,
            "model": combined_model_name,
            "training_mode": "ensemble",
            "data_distribution": "iid_and_non_iid",
            "federated_algorithm": combined_algo,
            "client_id": None,
            "round": None,
            "mu": None,
            "loss_type": "ensemble_probability",
            "split": "validation",
            "threshold_policy": "validation_f1",
            "recall": v_fro.recall,
            "f1": v_fro.f1,
            "pr_auc": v_fro.pr_auc,
            "precision": v_fro.precision,
            "false_negatives": v_fro.confusion.false_negatives,
            "roc_auc": v_fro.roc_auc,
            "accuracy": v_fro.accuracy,
            "tp": v_fro.confusion.true_positives,
            "tn": v_fro.confusion.true_negatives,
            "fp": v_fro.confusion.false_positives,
            "fn": v_fro.confusion.false_negatives,
            "threshold": report.frozen_threshold,
            "positive_rate": v_fro.positive_rate,
            "loss": v_fro.loss,
            "missed_positive_rate": v_fro.confusion.missed_positive_rate,
            "sequence_count": sequence_count_val,
            "training_time": report.timing.get("tuning_seconds", 0.0),
            "evaluation_time": report.timing.get("val_eval_seconds", 0.0),
            "seed": report.config.seed,
            "hyperparameters": hp_json,
        })

        # 3. Test at fixed threshold 0.5
        t_fix = report.test_metrics_at_fixed_threshold
        rows.append({
            "experiment_id": report.experiment_id,
            "model": combined_model_name,
            "training_mode": "ensemble",
            "data_distribution": "iid_and_non_iid",
            "federated_algorithm": combined_algo,
            "client_id": None,
            "round": None,
            "mu": None,
            "loss_type": "ensemble_probability",
            "split": "test",
            "threshold_policy": "fixed",
            "recall": t_fix.recall,
            "f1": t_fix.f1,
            "pr_auc": t_fix.pr_auc,
            "precision": t_fix.precision,
            "false_negatives": t_fix.confusion.false_negatives,
            "roc_auc": t_fix.roc_auc,
            "accuracy": t_fix.accuracy,
            "tp": t_fix.confusion.true_positives,
            "tn": t_fix.confusion.true_negatives,
            "fp": t_fix.confusion.false_positives,
            "fn": t_fix.confusion.false_negatives,
            "threshold": 0.5,
            "positive_rate": t_fix.positive_rate,
            "loss": t_fix.loss,
            "missed_positive_rate": t_fix.confusion.missed_positive_rate,
            "sequence_count": sequence_count_test,
            "training_time": report.timing.get("tuning_seconds", 0.0),
            "evaluation_time": report.timing.get("test_eval_seconds", 0.0),
            "seed": report.config.seed,
            "hyperparameters": hp_json,
        })

        # 4. Test at frozen validation_f1 threshold
        t_fro = report.test_metrics_at_frozen_threshold
        rows.append({
            "experiment_id": report.experiment_id,
            "model": combined_model_name,
            "training_mode": "ensemble",
            "data_distribution": "iid_and_non_iid",
            "federated_algorithm": combined_algo,
            "client_id": None,
            "round": None,
            "mu": None,
            "loss_type": "ensemble_probability",
            "split": "test",
            "threshold_policy": "validation_f1",
            "recall": t_fro.recall,
            "f1": t_fro.f1,
            "pr_auc": t_fro.pr_auc,
            "precision": t_fro.precision,
            "false_negatives": t_fro.confusion.false_negatives,
            "roc_auc": t_fro.roc_auc,
            "accuracy": t_fro.accuracy,
            "tp": t_fro.confusion.true_positives,
            "tn": t_fro.confusion.true_negatives,
            "fp": t_fro.confusion.false_positives,
            "fn": t_fro.confusion.false_negatives,
            "threshold": report.frozen_threshold,
            "positive_rate": t_fro.positive_rate,
            "loss": t_fro.loss,
            "missed_positive_rate": t_fro.confusion.missed_positive_rate,
            "sequence_count": sequence_count_test,
            "training_time": report.timing.get("tuning_seconds", 0.0),
            "evaluation_time": report.timing.get("test_eval_seconds", 0.0),
            "seed": report.config.seed,
            "hyperparameters": hp_json,
        })

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        logger.info("Saved ensemble results.csv (4 rows) to %s", csv_path)
        return csv_path

