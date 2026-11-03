"""Training CLI command.

Thin, like the preprocessing command: it builds nothing and decides nothing. The
service and the base configuration arrive from a factory, because constructing them
reads metadata from disk and creates output directories, and building the parser
should have no side effects.

``--model all`` runs the three architectures in one invocation, and the way it does
so is the point: one base configuration, :meth:`TrainingConfig.with_architecture`
applied three times, nothing else varied. Any per-architecture adjustment would have
to be written here, in the open, rather than emerging from three separate commands
run with three sets of flags.

There are no federated options here — no ``--iid``, ``--non-iid``, ``--client``,
``--fedavg``, ``--fedprox`` or ``--mu``. Those belong to phases that have not been
implemented, and a flag that accepted them now would imply behaviour that does not
exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from application.dto.experiment import ExperimentReport
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ORDERED_ARCHITECTURES, ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy

if TYPE_CHECKING:
    from application.services.centralized_training_service import CentralizedTrainingService

_WIDTH = 40

DEFERRED_NOTICE = (
    "Phase 4 implements centralized GRU, LSTM, and BiLSTM baselines and the "
    "reusable training/evaluation foundation. IID partitioning, typology-based "
    "Non-IID partitioning, FedAvg, FedProx, and federated aggregation are "
    "intentionally deferred to later phases."
)
"""The scope statement, printed by the command as well as documented.

Present in the tool's own output so that the boundary is visible to whoever runs it,
not only to whoever reads the documentation.
"""


class TrainingSessionFactory(Protocol):
    """Builds the training service and the base configuration on demand."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[CentralizedTrainingService, TrainingConfig]:
        """Construct the service and the configuration the overrides describe."""
        ...


def train_command(
    factory: TrainingSessionFactory,
    *,
    architectures: Sequence[ModelArchitecture],
    overrides: Mapping[str, object],
) -> int:
    """Train one or more centralized baselines on the processed splits.

    Args:
        factory: Builds the configured service and base configuration.
        architectures: Which architectures to train, in order.
        overrides: Command-line values that differ from the configured defaults.

    Returns:
        Process exit code. Always 0 — every failure raises a
        :class:`~domain.exceptions.base.FedSecureAIError`, which the entry point
        reports without a traceback.
    """
    service, base_config = factory(overrides=overrides)
    rule = "=" * _WIDTH
    names = ", ".join(architecture.display_name for architecture in architectures)
    print(
        "\n".join(
            [
                rule,
                "FedSecureAI Centralized Baseline Training",
                rule,
                "",
                f"Architectures: {names}",
                f"Seed:          {base_config.random_seed}",
                f"Sequences:     length {base_config.sequence_length}, "
                f"stride {base_config.sequence_stride}",
                "Loss:          BCEWithLogitsLoss (unweighted; natural imbalance "
                "preserved)",
                "",
                "The test split is not read until training has finished and the",
                "decision threshold has been frozen on validation.",
                "",
            ]
        )
    )

    reports: list[ExperimentReport] = []
    for architecture in architectures:
        config = base_config.with_architecture(architecture)
        print(f"--- {architecture.display_name} ---")
        report = service.run(config)
        reports.append(report)
        print(_render(report))

    if len(reports) > 1:
        print(_render_comparison(reports))
    return 0


def _render(report: ExperimentReport) -> str:
    """Format one experiment's measured outcome."""
    rule = "-" * _WIDTH
    config = report.config
    lines = [
        "",
        rule,
        f"Experiment:     {report.experiment_id}",
        rule,
        f"Model:          {config.architecture.display_name}",
        f"Parameters:     {report.parameter_count:,}",
        f"Device:         {report.device}",
        f"Torch:          {report.torch_version}",
        f"Input width:    {report.provenance.feature_dimension} "
        f"(from {len(report.provenance.feature_columns)} processed feature column(s))",
        f"Grouping:       {report.sequences.grouping_column}",
        f"Deterministic:  {'requested' if report.deterministic else 'NOT AVAILABLE'}",
        "",
        "Data",
    ]
    for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        rows = report.provenance.rows_per_split.get(split.value)
        positives = report.provenance.positives_per_split.get(split.value)
        if rows is None:
            continue
        share = (positives / rows * 100.0) if positives is not None and rows else 0.0
        lines.append(
            f"  {split.value:<11} {rows:>10,} rows, {positives or 0:>7,} positive(s) "
            f"({share:.6f}%)"
        )

    lines.extend(
        [
            "",
            "Training curve",
            f"  {'epoch':>5}  {'train loss':>11}  {'val loss':>11}  "
            f"{'val PR-AUC':>11}  {'val recall':>11}  {'secs':>7}",
        ]
    )
    for record in report.epochs:
        marker = " *" if record.selected else "  "
        lines.append(
            f"  {record.epoch:>5}  {record.train_loss:>11.6f}  "
            f"{record.validation_loss:>11.6f}  "
            f"{record.validation_metrics.get('pr_auc', 0.0):>11.6f}  "
            f"{record.validation_metrics.get('recall', 0.0):>11.6f}  "
            f"{record.seconds:>7.1f}{marker}"
        )

    selected_on = report.threshold_selected_on or "configuration (not fitted)"
    lines.extend(
        [
            "  (* selected checkpoint)",
            "",
            f"Selected epoch: {report.selected_epoch}"
            f"{' (early stopped)' if report.early_stopped else ''}",
            f"Selection by:   validation {config.selection_metric.value}",
            f"Threshold:      {report.frozen_threshold:.6f}, fitted on {selected_on}, "
            f"then frozen",
            "",
            "Results",
            f"  {'split':<11} {'policy':<14} {'recall':>9} {'F1':>9} {'PR-AUC':>9} "
            f"{'prec':>9} {'FN':>8} {'ROC-AUC':>9} {'acc':>9}",
        ]
    )
    for result in report.results:
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
            "  FN = laundering transactions the model let through.",
            "  Accuracy is reported last on purpose: flagging nothing scores "
            "~99.9% here.",
            "",
            "Timing",
            f"  data prep:    {report.data_preparation_seconds:.1f}s",
            f"  training:     {report.training_seconds:.1f}s",
            f"  total:        {report.total_seconds:.1f}s",
            "",
        ]
    )
    return "\n".join(lines)


def _render_comparison(reports: Sequence[ExperimentReport]) -> str:
    """Format the cross-architecture comparison, without naming a winner."""
    rule = "=" * _WIDTH
    lines = [
        rule,
        "Test-split comparison at the frozen threshold",
        rule,
        "",
        f"  {'model':<8} {'recall':>9} {'F1':>9} {'PR-AUC':>9} {'prec':>9} "
        f"{'FN':>8} {'ROC-AUC':>9} {'train s':>9}",
    ]
    ordered = sorted(
        reports, key=lambda report: ORDERED_ARCHITECTURES.index(report.config.architecture)
    )
    for report in ordered:
        result = report.result_for(
            SplitName.TEST, report.config.threshold_policy
        ) or report.result_for(SplitName.TEST, ThresholdPolicy.FIXED)
        if result is None:  # pragma: no cover - a test row is always emitted
            continue
        metrics = result.metrics
        lines.append(
            f"  {report.config.architecture.value:<8} {metrics.recall:>9.6f} "
            f"{metrics.f1:>9.6f} {metrics.pr_auc:>9.6f} {metrics.precision:>9.6f} "
            f"{metrics.false_negatives:>8,} {metrics.roc_auc:>9.6f} "
            f"{report.training_seconds:>9.1f}"
        )

    lines.extend(
        [
            "",
            "These are single-seed measurements on one dataset under one",
            "configuration. No architecture is declared best: separating a real",
            "difference from seed variance needs repeated runs, which this phase",
            "did not perform.",
            "",
            DEFERRED_NOTICE,
            rule,
        ]
    )
    return "\n".join(lines)
