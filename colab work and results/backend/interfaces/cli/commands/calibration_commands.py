"""Phase 10 Probability Calibration CLI command handler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from application.dto.calibration import (
    CalibrationConfig,
    CalibrationExperimentReport,
)

if TYPE_CHECKING:
    from application.services.calibration_service import CalibrationService

_WIDTH = 65


class CalibrationSessionFactory(Protocol):
    """Factory protocol for building CalibrationService and CalibrationConfig."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[CalibrationService, CalibrationConfig]:
        """Construct calibration service and config with overrides applied."""
        ...


def calibration_command(
    factory: CalibrationSessionFactory,
    *,
    overrides: Mapping[str, object],
) -> int:
    """Run Phase 10 probability calibration workflow.

    Args:
        factory: Builds the calibration service and config.
        overrides: Command-line overrides.

    Returns:
        Process exit code (0 on success).
    """
    service, config = factory(overrides=overrides)
    rule = "=" * _WIDTH

    print(
        "\n".join([
            rule,
            "FedSecureAI Phase 10: Probability Calibration",
            rule,
            "",
            "Objective:          Evaluate probability calibration reliability for AML model",
            "Reference Model:    Official Phase 7 IID LSTM + FedAvg",
            f"Candidate Methods:  {[m.value for m in config.methods]}",
            f"Random Seed:        {config.seed}",
            f"Selection Metric:   {config.selection_metric.upper()} (Validation-only)",
            f"Threshold Policy:   {config.threshold_policy} (Validation-only)",
            "Test Set Status:    Untouched until final frozen evaluation",
            rule,
            "",
        ])
    )

    report: CalibrationExperimentReport = service.run(config)

    t_base = report.test_baseline.metrics
    t_cal = report.test_calibrated.metrics

    print(
        "\n".join([
            "",
            rule,
            "Phase 10 Calibration Experiment Summary",
            rule,
            f"Experiment ID:        {report.experiment_id}",
            f"Selected Method:      {report.selected_method.value}",
            f"Frozen Decision Tau:  {report.frozen_threshold:.6f}",
            "",
            "Validation Comparison:",
        ])
    )

    for r in report.validation_results:
        sel_tag = " [SELECTED]" if r.selected else ""
        print(
            f"  - {r.method.value:<20}: LogLoss={r.metrics.log_loss:.6f}, "
            f"ECE={r.metrics.ece:.6f}, Brier={r.metrics.brier_score:.6f}, "
            f"F1={r.metrics.f1:.4f}{sel_tag}"
        )

    print(
        "\n".join([
            "",
            "Final Single-Pass Test Split Evaluation:",
            f"  Log Loss:           {t_base.log_loss:.6f} -> {t_cal.log_loss:.6f} "
            f"({((t_base.log_loss - t_cal.log_loss) / t_base.log_loss * 100):+.2f}%)",
            f"  Brier Score:        {t_base.brier_score:.6f} -> {t_cal.brier_score:.6f} "
            f"({((t_base.brier_score - t_cal.brier_score) / t_base.brier_score * 100):+.2f}%)",
            f"  ECE (15 bins):      {t_base.ece:.6f} -> {t_cal.ece:.6f} "
            f"({((t_base.ece - t_cal.ece) / t_base.ece * 100):+.2f}%)",
            f"  MCE:                {t_base.mce:.6f} -> {t_cal.mce:.6f}",
            f"  PR-AUC:             {t_base.pr_auc:.6f} -> {t_cal.pr_auc:.6f} (Ranking)",
            f"  ROC-AUC:            {t_base.roc_auc:.6f} -> {t_cal.roc_auc:.6f} (Ranking)",
            f"  F1 Score:           {t_base.f1:.6f} -> {t_cal.f1:.6f} "
            f"(tau={report.frozen_threshold:.4f})",
            f"  Precision:          {t_base.precision:.6f} -> {t_cal.precision:.6f}",
            f"  Recall:             {t_base.recall:.6f} -> {t_cal.recall:.6f}",
            f"  Confusion (TP/FP):  {t_cal.tp} / {t_cal.fp} (Total pos: {t_cal.positive_count})",
            "",
            f"Artifacts saved in:   experiments/results/calibration/{report.experiment_id}/",
            rule,
        ])
    )

    return 0
