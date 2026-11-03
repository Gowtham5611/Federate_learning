"""Phase 7 IID + Non-IID Ensemble CLI command handler.

Executes ensemble discovery, validation alpha grid tuning, threshold selection,
and single-pass test evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from application.dto.ensemble import EnsembleConfig, EnsembleExperimentReport

if TYPE_CHECKING:
    from application.services.ensemble_service import EnsembleService

_WIDTH = 60


class EnsembleSessionFactory(Protocol):
    """Factory protocol for building EnsembleService and EnsembleConfig."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[EnsembleService, EnsembleConfig]:
        """Construct ensemble service and config with overrides applied."""
        ...


def ensemble_command(
    factory: EnsembleSessionFactory,
    *,
    overrides: Mapping[str, object],
) -> int:
    """Run Phase 7 IID + Non-IID ensemble modeling.

    Args:
        factory: Builds the ensemble service and config.
        overrides: Command-line overrides.

    Returns:
        Process exit code (0 on success).
    """
    service, config = factory(overrides=overrides)
    rule = "=" * _WIDTH

    print(
        "\n".join([
            rule,
            "FedSecureAI Phase 7: IID + Non-IID Ensemble Modeling",
            rule,
            "",
            "Ensemble Strategy:  P_ens = α * P_IID + (1 - α) * P_NonIID",
            f"Alpha Search Grid:  {len(config.alpha_grid)} points (step={config.alpha_step:.2f})",
            f"Random Seed:        {config.seed}",
            "Selection Metric:   PR-AUC (Validation-only)",
            "Threshold Policy:   F1-optimal (Validation-only)",
            "Test Set Status:    Untouched until final frozen evaluation",
            rule,
            "",
        ])
    )

    report: EnsembleExperimentReport = service.run(config)

    t_fro = report.test_metrics_at_frozen_threshold
    v_fro = report.validation_metrics_at_frozen_threshold

    print(
        "\n".join([
            "",
            rule,
            "Ensemble Experiment Results Summary",
            rule,
            f"Experiment ID:      {report.experiment_id}",
            "",
            "Selected Models (Validation PR-AUC):",
            (
                f"  IID Model:        {report.selected_iid.model.upper()} "
                f"({report.selected_iid.algorithm.upper()})"
            ),
            f"                    ID: {report.selected_iid.experiment_id}",
            f"                    Val PR-AUC: {report.selected_iid.validation_score:.6f}",
            (
                f"  Non-IID Model:    {report.selected_non_iid.model.upper()} "
                f"({report.selected_non_iid.algorithm.upper()})"
            ),
            f"                    ID: {report.selected_non_iid.experiment_id}",
            f"                    Val PR-AUC: {report.selected_non_iid.validation_score:.6f}",
            "",
            "Tuned Ensemble Parameters (Validation):",
            (
                f"  Best Alpha:       {report.best_alpha:.4f} "
                f"(IID weight = {report.best_alpha:.4f}, "
                f"Non-IID weight = {1.0 - report.best_alpha:.4f})"
            ),
            f"  Frozen Threshold: {report.frozen_threshold:.6f} (Fitted on validation F1)",
            "",
            "Validation Performance (at frozen threshold):",
            f"  PR-AUC:   {v_fro.pr_auc:.6f}",
            f"  ROC-AUC:  {v_fro.roc_auc:.6f}",
            f"  F1 Score: {v_fro.f1:.6f}",
            f"  Recall:   {v_fro.recall:.6f}",
            f"  Prec:     {v_fro.precision:.6f}",
            f"  Accuracy: {v_fro.accuracy:.6f}",
            "",
            "Final Single-Pass Test Performance (at frozen threshold):",
            f"  Test PR-AUC:   {t_fro.pr_auc:.6f}",
            f"  Test ROC-AUC:  {t_fro.roc_auc:.6f}",
            f"  Test F1 Score: {t_fro.f1:.6f}",
            f"  Test Recall:   {t_fro.recall:.6f}",
            f"  Test Prec:     {t_fro.precision:.6f}",
            f"  Test Accuracy: {t_fro.accuracy:.6f}",
            (
                f"  Confusion:     TP={t_fro.confusion.true_positives:,}, "
                f"FP={t_fro.confusion.false_positives:,}, "
                f"TN={t_fro.confusion.true_negatives:,}, "
                f"FN={t_fro.confusion.false_negatives:,}"
            ),
            "",
            f"Total Duration:     {report.timing.get('total_seconds', 0.0):.1f}s",
            rule,
        ])
    )

    return 0

