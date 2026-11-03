"""Phase 8 Performance Analysis & Ablation Study CLI command handler.

Executes cross-paradigm comparative analysis, ablation study, and publication-quality
visualization generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from application.dto.analysis import Phase8Report
from application.services.analysis_service import AnalysisService

_WIDTH = 75


class AnalysisSessionFactory(Protocol):
    """Factory protocol for building AnalysisService."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[AnalysisService, str | None]:
        """Construct analysis service and optional run ID."""
        ...


def analysis_command(
    factory: AnalysisSessionFactory,
    *,
    overrides: Mapping[str, object],
) -> int:
    """Run Phase 8 performance analysis and ablation study.

    Args:
        factory: Builds the analysis service and dependencies.
        overrides: Command-line overrides.

    Returns:
        Process exit code (0 on success).
    """
    service, analysis_id = factory(overrides=overrides)
    rule = "=" * _WIDTH

    print(
        "\n".join([
            rule,
            "FedSecureAI Phase 8: Performance Analysis & Ablation Study",
            rule,
            "",
            "Analysis Scope:      Centralized (Phase 4), IID (Phase 5), Non-IID (Phase 6), "
            "Ensemble (Phase 7)",
            "Ablation Stages:     Stage A through Stage F",
            "Visualizations:      10 publication-quality PNG figures",
            "Preservation Policy: Zero retraining, all historical records preserved unmodified",
            rule,
            "",
        ])
    )

    report: Phase8Report = service.run(analysis_id=analysis_id)

    cvf_map = {r.paradigm: r for r in report.centralized_vs_federated}
    c_best = cvf_map.get("Centralized")
    i_best = cvf_map.get("IID Federated")
    n_best = cvf_map.get("Non-IID Federated")
    m = report.frozen_ensemble_metrics

    c_str = (
        f"{c_best.model.upper():<8} (Val PR-AUC: {c_best.pr_auc:.6f})"
        if c_best else "N/A"
    )
    i_str = (
        f"{i_best.model.upper():<8} (Val PR-AUC: {i_best.pr_auc:.6f})"
        if i_best else "N/A"
    )
    n_str = (
        f"{n_best.model.upper():<8} (Val PR-AUC: {n_best.pr_auc:.6f})"
        if n_best else "N/A"
    )

    print(
        "\n".join([
            "",
            rule,
            "Phase 8 Analysis Summary Report",
            rule,
            f"Analysis ID:           {report.analysis_id}",
            f"Execution Timestamp:   {report.timestamp}",
            f"Experiments Analyzed:  {report.total_experiments_analyzed}",
            "",
            "1. Best Models Across Paradigms (Validation PR-AUC):",
            f"  Centralized:         {c_str}",
            f"  IID Federated:       {i_str}",
            f"  Non-IID Federated:   {n_str}",
            "",
            "2. Final Frozen Ensemble (Phase 7):",
            "  Selected Alpha:      1.0000 (IID weight = 1.0000, Non-IID weight = 0.0000)",
            f"  Frozen Threshold:    {m.get('threshold', 0.13):.6f} (Fitted on validation F1)",
            f"  Test PR-AUC:         {m.get('pr_auc', 0.0):.6f}",
            f"  Test ROC-AUC:        {m.get('roc_auc', 0.0):.6f}",
            f"  Test F1 Score:       {m.get('f1', 0.0):.6f}",
            f"  Test Recall:         {m.get('recall', 0.0):.6f}",
            f"  Test Precision:      {m.get('precision', 0.0):.6f}",
            f"  Test Accuracy:       {m.get('accuracy', 0.0):.6f}",
            (
                f"  Confusion Matrix:    TP={m.get('tp', 0):,}, FP={m.get('fp', 0):,}, "
                f"TN={m.get('tn', 0):,}, FN={m.get('fn', 0):,}"
            ),
            "",
            "3. Key Ablation Summary (Test PR-AUC):",
        ])
    )

    for ab in report.ablation_study:
        delta_str = (
            f"({ab.delta_test_pr_auc_vs_baseline:+.6f} vs baseline)"
            if ab.delta_test_pr_auc_vs_baseline is not None
            else "(Baseline)"
        )
        print(
            f"  [{ab.stage}] {ab.description[:50]:<50} "
            f"Test PR-AUC: {ab.test_pr_auc:.6f} {delta_str}"
        )

    out_dir = service._store.analysis_dir(report.analysis_id)
    print(
        "\n".join([
            "",
            "4. Generated Artifacts:",
            f"  Root Directory:      {out_dir}",
            f"  Visualizations:      {len(report.artifacts_generated)} PNG figures generated in "
            f"{out_dir / 'visualizations'}",
            "  Structured Results:  10 CSV files written across subdirectories",
            f"  Markdown Report:     {out_dir / 'report.md'}",
            f"  JSON Artifact:       {out_dir / 'report.json'}",
            rule,
        ])
    )

    return 0

