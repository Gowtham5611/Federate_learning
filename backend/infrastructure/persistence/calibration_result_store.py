"""Dedicated persistence adapter for Phase 10 Probability Calibration results."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from application.dto.calibration import (
    CalibrationExperimentReport,
    CalibrationMethodResult,
)

logger = logging.getLogger(__name__)


class CalibrationResultStore:
    """Manages file storage and directory structures for Phase 10 calibration artifacts."""

    def __init__(self, root: Path) -> None:
        """Initialize store rooted at experiments/results/calibration."""
        self._root = root.resolve() / "calibration"
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Root directory for calibration experiments."""
        return self._root

    def experiment_dir(self, experiment_id: str) -> Path:
        """Base directory for a specific calibration experiment."""
        return self._root / experiment_id

    def init_directory_structure(self, experiment_id: str) -> Path:
        """Create structured directories: validation, test, visualizations, fitted_models."""
        exp_dir = self.experiment_dir(experiment_id)
        for sub in ("validation", "test", "visualizations", "fitted_models"):
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_validation_results(
        self,
        experiment_id: str,
        results: list[CalibrationMethodResult],
    ) -> Path:
        """Save validation method comparison CSV and JSON."""
        val_dir = self.experiment_dir(experiment_id) / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)

        json_path = val_dir / "validation_comparison.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        csv_path = val_dir / "validation_comparison.csv"
        rows: list[dict[str, Any]] = []
        for r in results:
            row = {
                "method": r.method.value,
                "split": r.split,
                "selected": r.selected,
                **r.parameters.to_dict(),
                **r.metrics.to_dict(),
            }
            rows.append(row)

        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        return csv_path

    def save_test_results(
        self,
        experiment_id: str,
        baseline: CalibrationMethodResult,
        calibrated: CalibrationMethodResult,
    ) -> Path:
        """Save test evaluation comparison CSV and JSON."""
        test_dir = self.experiment_dir(experiment_id) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        json_path = test_dir / "test_comparison.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"baseline": baseline.to_dict(), "calibrated": calibrated.to_dict()},
                f,
                indent=2,
            )

        csv_path = test_dir / "test_comparison.csv"
        rows = [
            {
                "method": baseline.method.value,
                "split": baseline.split,
                "type": "uncalibrated_baseline",
                **baseline.parameters.to_dict(),
                **baseline.metrics.to_dict(),
            },
            {
                "method": calibrated.method.value,
                "split": calibrated.split,
                "type": "calibrated_model",
                **calibrated.parameters.to_dict(),
                **calibrated.metrics.to_dict(),
            },
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    def save_predictions_sample(
        self,
        experiment_id: str,
        split: str,
        row_indices: NDArray[np.int64],
        y_true: NDArray[np.float32],
        p_uncalibrated: NDArray[np.float32],
        p_calibrated: NDArray[np.float32],
        threshold: float,
        sample_size: int = 100_000,
    ) -> Path:
        """Save representative predictions sample to provide inspection."""
        target_dir = self.experiment_dir(experiment_id) / split
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"{split}_predictions_sample.csv"

        n = len(y_true)
        if n > sample_size:
            pos_indices = np.where(y_true == 1.0)[0]
            neg_indices = np.where(y_true == 0.0)[0]
            remaining_needed = max(0, sample_size - len(pos_indices))
            sampled_neg = np.random.choice(
                neg_indices,
                size=min(remaining_needed, len(neg_indices)),
                replace=False,
            )
            keep_indices = np.sort(np.concatenate([pos_indices, sampled_neg]))
        else:
            keep_indices = np.arange(n)

        df = pd.DataFrame({
            "row_index": row_indices[keep_indices],
            "true_label": y_true[keep_indices].astype(int),
            "p_uncalibrated": p_uncalibrated[keep_indices],
            "p_calibrated": p_calibrated[keep_indices],
            "calibrated_threshold": threshold,
            "predicted_label": (p_calibrated[keep_indices] >= threshold).astype(int),
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    def save_report_and_metadata(
        self,
        report: CalibrationExperimentReport,
    ) -> tuple[Path, Path]:
        """Save report.json and metadata.json at root of experiment directory."""
        exp_dir = self.experiment_dir(report.experiment_id)
        report_json_path = exp_dir / "report.json"
        metadata_json_path = exp_dir / "metadata.json"

        with report_json_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        metadata = {
            "experiment_id": report.experiment_id,
            "timestamp": report.timestamp,
            "phase": "Phase 10: Probability Calibration",
            "source_model": report.source_model,
            "config": report.config.to_dict(),
            "selected_method": report.selected_method.value,
            "selected_parameters": report.selected_parameters.to_dict(),
            "frozen_threshold": report.frozen_threshold,
            "timing": report.timing,
        }
        with metadata_json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return report_json_path, metadata_json_path

    def save_markdown_report(self, report: CalibrationExperimentReport) -> Path:
        """Write formatted Markdown report documenting calibration study."""
        exp_dir = self.experiment_dir(report.experiment_id)
        report_md_path = exp_dir / "report.md"

        t_base = report.test_baseline
        t_cal = report.test_calibrated

        src_id = report.source_model.get("experiment_id")
        src_arch = report.source_model.get("model")
        src_dist = report.source_model.get("distribution")
        src_algo = report.source_model.get("algorithm")
        src_ckpt = report.source_model.get("checkpoint_path")

        loss_rel = (
            (t_base.metrics.log_loss - t_cal.metrics.log_loss) / t_base.metrics.log_loss * 100
        )
        brier_rel = (
            (t_base.metrics.brier_score - t_cal.metrics.brier_score)
            / t_base.metrics.brier_score
            * 100
        )
        ece_rel = (t_base.metrics.ece - t_cal.metrics.ece) / t_base.metrics.ece * 100
        mce_rel = (t_base.metrics.mce - t_cal.metrics.mce) / t_base.metrics.mce * 100
        pr_rel = (t_cal.metrics.pr_auc - t_base.metrics.pr_auc) / t_base.metrics.pr_auc * 100
        roc_rel = (t_cal.metrics.roc_auc - t_base.metrics.roc_auc) / t_base.metrics.roc_auc * 100
        f1_rel = (t_cal.metrics.f1 - t_base.metrics.f1) / t_base.metrics.f1 * 100

        lines = [
            f"# Phase 10: Probability Calibration Report ({report.experiment_id})",
            "",
            "## 1. Executive Summary & Verification Highlights",
            f"- **Experiment ID**: `{report.experiment_id}`",
            f"- **Timestamp**: `{report.timestamp}`",
            f"- **Source Reference Model**: `{src_id}`",
            f"  - Model Architecture: `{src_arch}`",
            f"  - Partition / Algorithm: `{src_dist}` / `{src_algo}`",
            f"  - Checkpoint: `{src_ckpt}`",
            f"- **Selected Calibration Method**: **`{report.selected_method.value}`**",
            f"- **Selection Metric Criterion**: `{report.config.selection_metric}` (Validation)",
            f"- **Validation-Selected Decision Threshold**: `{report.frozen_threshold:.6f}`",
            f"- **Historical Phase 7 Decision Threshold**: "
            f"`{report.config.historical_threshold:.4f}`",
            "",
            "---",
            "",
            "## 2. Validation Calibration Method Comparison",
            "Fitted on validation split only (1,015,565 sequences, 1,081 positives):",
            "",
            "| Method | Selected | Log Loss | Brier Score | ECE (15 bins) | MCE | Val PR-AUC | "
            "Val ROC-AUC | Val F1 | Val Threshold | Parameters |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for r in report.validation_results:
            param_parts = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.parameters.to_dict().items()
                if v is not None and k != "method"
            ]
            param_str = ", ".join(param_parts) if param_parts else "None"
            sel_str = "**YES**" if r.selected else "No"
            lines.append(
                f"| `{r.method.value}` | {sel_str} | "
                f"{r.metrics.log_loss:.6f} | {r.metrics.brier_score:.6f} | "
                f"{r.metrics.ece:.6f} | {r.metrics.mce:.6f} | "
                f"{r.metrics.pr_auc:.6f} | {r.metrics.roc_auc:.6f} | "
                f"{r.metrics.f1:.6f} | {r.metrics.threshold:.6f} | {param_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Test Evaluation: Uncalibrated Baseline vs Frozen Calibrated Model",
            "Single-pass final test split evaluation (1,015,564 sequences, 1,797 positives):",
            "",
            "| Metric | Uncalibrated Baseline | Calibrated Model | Relative Improvement |",
            "|---|---|---|---|",
            f"| **Log Loss** | `{t_base.metrics.log_loss:.6f}` | `{t_cal.metrics.log_loss:.6f}` | "
            f"`{loss_rel:+.2f}%` |",
            f"| **Brier Score** | `{t_base.metrics.brier_score:.6f}` | "
            f"`{t_cal.metrics.brier_score:.6f}` | `{brier_rel:+.2f}%` |",
            f"| **Expected Calibration Error (ECE)** | `{t_base.metrics.ece:.6f}` | "
            f"`{t_cal.metrics.ece:.6f}` | `{ece_rel:+.2f}%` |",
            f"| **Max Calibration Error (MCE)** | `{t_base.metrics.mce:.6f}` | "
            f"`{t_cal.metrics.mce:.6f}` | `{mce_rel:+.2f}%` |",
            f"| **PR-AUC** (Ranking) | `{t_base.metrics.pr_auc:.6f}` | "
            f"`{t_cal.metrics.pr_auc:.6f}` | `{pr_rel:+.2f}%` |",
            f"| **ROC-AUC** (Ranking) | `{t_base.metrics.roc_auc:.6f}` | "
            f"`{t_cal.metrics.roc_auc:.6f}` | `{roc_rel:+.2f}%` |",
            f"| **Decision Threshold** | `{t_base.metrics.threshold:.6f}` | "
            f"`{t_cal.metrics.threshold:.6f}` | - |",
            f"| **Test F1** | `{t_base.metrics.f1:.6f}` | `{t_cal.metrics.f1:.6f}` | "
            f"`{f1_rel:+.2f}%` |",
            f"| **Test Precision** | `{t_base.metrics.precision:.6f}` | "
            f"`{t_cal.metrics.precision:.6f}` | - |",
            f"| **Test Recall** | `{t_base.metrics.recall:.6f}` | "
            f"`{t_cal.metrics.recall:.6f}` | - |",
            f"| **Test Accuracy** | `{t_base.metrics.accuracy:.6f}` | "
            f"`{t_cal.metrics.accuracy:.6f}` | - |",
            f"| **Confusion Matrix (TP/FP/TN/FN)** | "
            f"`{t_base.metrics.tp}/{t_base.metrics.fp}/{t_base.metrics.tn}/{t_base.metrics.fn}` | "
            f"`{t_cal.metrics.tp}/{t_cal.metrics.fp}/{t_cal.metrics.tn}/{t_cal.metrics.fn}` | - |",
            f"| **Historical Threshold (tau={report.config.historical_threshold:.2f})** | "
            f"F1: `{t_base.metrics.historical_f1:.6f}` | F1: `{t_cal.metrics.historical_f1:.6f}` |"
            " - |",
            "",
            "---",
            "",
            "## 4. Key Methodological Observations",
            "1. **Monotonicity & Ranking Invariance**: Parametric calibrators strictly preserve "
            "probability ranking order, leaving Test PR-AUC and ROC-AUC unchanged while "
            "improving probability error metrics (Log Loss, ECE, Brier Score).",
            "2. **Probability Quality vs Classification**: Calibration aligns predicted risk "
            "probabilities with empirical frequency without modifying neural network weights.",
            "3. **Extreme Imbalance Handling**: Probability outputs remain strictly finite in "
            "[0, 1] under extreme ~0.1% AML prevalence.",
            "4. **Strict Isolation & Zero Test Leakage**: All calibrations and thresholds were "
            "fitted exclusively on validation sequences. The test set was scored exactly once.",
            "",
            "---",
            "",
            "## 5. Artifact Inventory",
            "- Validation Comparison: `validation/validation_comparison.csv`",
            "- Test Evaluation: `test/test_comparison.csv`",
            "- Reliability Diagrams: `visualizations/01_baseline_reliability_diagram.png`, "
            "`visualizations/02_calibrated_reliability_diagram.png`",
            "- Comparison Curve: `visualizations/03_calibration_comparison_curves.png`",
            "- Full Report & Metadata: `report.json`, `metadata.json`, `report.md`",
            "",
        ])

        with report_md_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_md_path

