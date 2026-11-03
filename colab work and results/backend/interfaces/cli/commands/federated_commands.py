"""Federated learning CLI command handler.

Dispatches federated experiments across requested architectures, rendering
client partition statistics, communication rounds, and centralized validation outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from application.dto.federated import FederatedConfig, FederatedExperimentReport
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy

if TYPE_CHECKING:
    from application.services.federated_training_service import FederatedTrainingService

_WIDTH = 50


class FederatedSessionFactory(Protocol):
    """Factory protocol for building FederatedTrainingService and base FederatedConfig."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[FederatedTrainingService, FederatedConfig]:
        """Construct service and base configuration with overrides applied."""
        ...


def federated_command(
    factory: FederatedSessionFactory,
    *,
    architectures: Sequence[ModelArchitecture],
    overrides: Mapping[str, object],
) -> int:
    """Run IID federated learning experiments.

    Args:
        factory: Builds the federated training service and base config.
        architectures: Recurrent architectures to train.
        overrides: Command-line overrides.

    Returns:
        Process exit code (0 on success).
    """
    service, base_fed_config = factory(overrides=overrides)
    rule = "=" * _WIDTH
    names = ", ".join(arch.display_name for arch in architectures)

    print(
        "\n".join(
            [
                rule,
                "FedSecureAI IID Federated Learning",
                rule,
                "",
                f"Architectures:  {names}",
                f"Distribution:   {base_fed_config.distribution.upper()}",
                f"Algorithm:      {base_fed_config.algorithm.upper()}",
                f"Clients:        {base_fed_config.n_clients}",
                f"Rounds:         {base_fed_config.rounds}",
                f"Local Epochs:   {base_fed_config.local_epochs}",
                f"Mu (FedProx):   {base_fed_config.mu or 'N/A'}",
                f"Seed:           {base_fed_config.seed}",
                "Validation:     Centralized (HI-Small/validation)",
                "Loss:           BCEWithLogitsLoss (unweighted; natural imbalance preserved)",
                "",
                "Centralized validation runs after each federated round.",
                "The test split is not read until training has finished and the",
                "decision threshold has been frozen on validation.",
                "",
            ]
        )
    )

    reports: list[FederatedExperimentReport] = []
    for architecture in architectures:
        # Create config copy for this architecture
        t_cfg = base_fed_config.training_config.with_architecture(architecture)
        fed_cfg = FederatedConfig(
            training_config=t_cfg,
            n_clients=base_fed_config.n_clients,
            rounds=base_fed_config.rounds,
            local_epochs=base_fed_config.local_epochs,
            distribution=base_fed_config.distribution,
            algorithm=base_fed_config.algorithm,
            mu=base_fed_config.mu,
            seed=base_fed_config.seed,
        )

        print(f"--- Running {architecture.display_name} ({base_fed_config.algorithm.upper()}) ---")
        report = service.run(fed_cfg)
        reports.append(report)
        print(_render(report))

    return 0


def _render(report: FederatedExperimentReport) -> str:
    """Format experiment output."""
    rule = "-" * _WIDTH
    cfg = report.config
    t_cfg = cfg.training_config

    lines = [
        "",
        rule,
        f"Experiment:     {report.experiment_id}",
        rule,
        f"Model:          {t_cfg.architecture.display_name}",
        f"Distribution:   {cfg.distribution.upper()}",
        f"Algorithm:      {cfg.algorithm.upper()}",
        f"Mu:             {cfg.mu if cfg.mu is not None else 'None'}",
        f"Parameters:     {report.parameter_count:,}",
        f"Device:         {report.device}",
        f"Clients:        {cfg.n_clients}",
        f"Communication:  {cfg.rounds} rounds x {cfg.local_epochs} local epoch(s)",
        "",
        "Client Partition Statistics",
        f"  {'client_id':<12} {'samples':>10} {'positives':>10} {'negatives':>10} {'pos rate':>11}",
    ]

    for c in report.partition_report.clients:
        lines.append(
            f"  {c.client_id:<12} {c.sample_count:>10,} {c.positive_count:>10,} "
            f"{c.negative_count:>10,} {c.positive_rate * 100:>10.6f}%"
        )

    lines.extend(
        [
            "",
            "Federated Rounds Curve (Centralized Validation)",
            f"  {'round':>5}  {'val loss':>11}  {'val PR-AUC':>11}  {'val recall':>10} {'secs':>6}",
        ]
    )

    for r in report.rounds:
        marker = " *" if r.selected else "  "
        m = r.validation_metrics
        lines.append(
            f"  {r.round:>5}  {r.validation_loss:>11.6f}  {m.pr_auc:>11.6f}  "
            f"{m.recall:>11.6f}  {r.round_seconds:>7.1f}{marker}"
        )

    lines.extend(
        [
            "  (* selected best global checkpoint)",
            "",
            f"Selected round: {report.selected_round}",
            f"Frozen threshold: {report.frozen_threshold:.6f} "
            f"({report.threshold_selected_on or 'configured'})",
            "",
            "Results",
            f"  {'split':<11} {'policy':<14} {'recall':>9} {'F1':>9} {'PR-AUC':>9} "
            f"{'prec':>9} {'FN':>8} {'ROC-AUC':>9} {'acc':>9}",
        ]
    )

    for result in report.results:
        # Only render final validation and test rows (skip per-round rows in summary table)
        if result.split is SplitName.TEST or (
            result.split is SplitName.VALIDATION
            and result.threshold_policy is not ThresholdPolicy.FIXED
        ):
            metrics = result.metrics
            lines.append(
                f"  {result.split.value:<11} {result.threshold_policy.value:<14} "
                f"{metrics.recall:>9.6f} {metrics.f1:>9.6f} {metrics.pr_auc:>9.6f} "
                f"{metrics.precision:>9.6f} {metrics.false_negatives:>8,} "
                f"{metrics.roc_auc:>9.6f} {metrics.accuracy:>9.6f}"
            )

    lines.extend(
        [
            "",
            "Timing",
            f"  data prep:    {report.data_preparation_seconds:.1f}s",
            f"  training:     {report.training_seconds:.1f}s",
            f"  total:        {report.total_seconds:.1f}s",
            "",
        ]
    )

    return "\n".join(lines)
