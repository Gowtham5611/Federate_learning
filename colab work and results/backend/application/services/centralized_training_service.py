"""Centralized training: the order in which one baseline experiment happens.

This service owns the *sequence of steps*, not the arithmetic. It reads splits
through a provider, fits through an engine, scores through a metrics calculator
and persists through two stores — and it is the place where the rules that make
the result trustworthy are enforced rather than assumed:

- The test split is not loaded until training has finished and the decision
  threshold has been frozen. Not "not used" — not *read*.
- The threshold is either the configured constant or fitted on validation, and
  the split is passed to the calculator so an implementation can refuse anything
  else.
- Both checkpoints are written, best and final, to separate paths.
- Metrics are reported at both the fixed threshold and the frozen one, because at
  0.1% positives those two can tell very different stories about the same model
  and reporting only one of them would be a choice disguised as a measurement.

It knows nothing about PyTorch, and nothing about federated learning.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from application.dto.experiment import (
    ClassificationMetrics,
    ExperimentReport,
    ExperimentResult,
)
from application.dto.training import EpochRecord, TrainingConfig
from application.ports.output.experiment_store import (
    Checkpoint,
    CheckpointStorePort,
    ExperimentResultStorePort,
)
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    EpochCallback,
    FitOutcome,
    SplitPredictions,
    TrainingEnginePort,
)
from application.ports.output.sequence_data import (
    SequenceDataProviderPort,
    SequenceDataset,
)
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy, TrainingMode
from domain.exceptions.training import (
    FeatureDimensionMismatchError,
    NoCompletedEpochError,
)

logger = logging.getLogger(__name__)

LOSS_TYPE = "bce_with_logits"
"""The objective every Phase 4 baseline is trained with.

Recorded in every result row as a field rather than left implicit, because Phase 9
introduces a cost-sensitive variant and the two must be distinguishable in the
same table. Unweighted here: the natural class imbalance is the baseline's
subject, not a problem the baseline is allowed to correct.
"""


class CentralizedTrainingService:
    """Runs one centralized baseline experiment end to end."""

    def __init__(
        self,
        *,
        sequences: SequenceDataProviderPort,
        engine: TrainingEnginePort,
        metrics: MetricsCalculatorPort,
        checkpoints: CheckpointStorePort,
        results: ExperimentResultStorePort,
    ) -> None:
        """Wire the five collaborators.

        Args:
            sequences: Reads processed splits and builds temporal sequences.
            engine: Fits a model and scores a dataset.
            metrics: Computes metrics and selects thresholds.
            checkpoints: Stores the best and final parameters.
            results: Stores the report and the comparison rows.
        """
        self._sequences = sequences
        self._engine = engine
        self._metrics = metrics
        self._checkpoints = checkpoints
        self._results = results

    def run(self, config: TrainingConfig) -> ExperimentReport:
        """Train, select, freeze, evaluate and persist one experiment.

        Args:
            config: The experiment's hyperparameters.

        Returns:
            The report, already written to the result store.

        Raises:
            FeatureDimensionMismatchError: Two splits disagree on feature width.
            NoCompletedEpochError: Training produced no epoch to select from.
        """
        started = time.perf_counter()
        experiment_id = self._experiment_id(config)
        provenance = self._sequences.describe_features(config)
        strategy = self._sequences.describe_strategy(config)

        logger.info(
            "Experiment %s | %s | seed=%d | sequence_length=%d stride=%d | "
            "input_dim=%d | loss=%s",
            experiment_id,
            config.architecture.display_name,
            config.random_seed,
            config.sequence_length,
            config.sequence_stride,
            provenance.feature_dimension,
            LOSS_TYPE,
        )
        if config.is_reduced_run:
            logger.warning(
                "Experiment %s is a REDUCED run: at most %d row(s) per split. "
                "Its results describe that subset, not the full dataset.",
                experiment_id,
                config.row_limit,
            )

        # --- Data: train and validation only. Test stays on disk for now. -----
        prepare_started = time.perf_counter()
        train = self._sequences.load(SplitName.TRAIN, config)
        validation = self._sequences.load(SplitName.VALIDATION, config)
        preparation_seconds = time.perf_counter() - prepare_started
        self._require_matching_width(train, validation)
        self._log_split(train)
        self._log_split(validation)

        # --- Fit -------------------------------------------------------------
        outcome = self._engine.fit(
            config, train, validation, on_epoch=self._log_epoch(experiment_id)
        )
        if not outcome.epochs:
            raise NoCompletedEpochError(experiment_id)

        logger.info(
            "Experiment %s selected epoch %d of %d (%s)",
            experiment_id,
            outcome.best_epoch,
            len(outcome.epochs),
            "early stopped" if outcome.early_stopped else "full budget",
        )

        # Release the training sequences before the test split is read. On the
        # measured machine the training index and feature block are the largest
        # objects in the process, and holding them while test loads is the one
        # place peak memory would double for no benefit.
        del train

        # --- Score validation, then freeze the threshold ---------------------
        validation_predictions = self._engine.predict(
            config, validation, outcome.best_state
        )
        threshold, selected_on = self._freeze_threshold(config, validation_predictions)
        logger.info(
            "Experiment %s froze threshold %.6f (%s)",
            experiment_id,
            threshold,
            f"fitted on {selected_on}" if selected_on else "configured",
        )

        del validation

        # --- Only now is test read -------------------------------------------
        test = self._sequences.load(SplitName.TEST, config)
        test_predictions = self._engine.predict(config, test, outcome.best_state)
        self._log_split(test)
        test_rows = test.sequence_count
        del test

        # --- Persist ---------------------------------------------------------
        self._save_checkpoints(
            config, experiment_id, outcome, threshold, provenance.feature_dimension
        )

        results = (
            *self._rows_for(
                config,
                experiment_id,
                SplitName.VALIDATION,
                validation_predictions,
                outcome,
                threshold,
            ),
            *self._rows_for(
                config,
                experiment_id,
                SplitName.TEST,
                test_predictions,
                outcome,
                threshold,
            ),
        )

        report = ExperimentReport(
            experiment_id=experiment_id,
            config=config,
            # Re-read now that every split has been loaded: the provider counts
            # rows and positives as it reads them, so this call - unlike the one
            # before training - can report what the run actually consumed.
            provenance=self._sequences.describe_features(config),
            sequences=strategy,
            epochs=outcome.epochs,
            results=results,
            selected_epoch=outcome.best_epoch,
            frozen_threshold=threshold,
            threshold_selected_on=selected_on,
            device=outcome.device,
            parameter_count=outcome.parameter_count,
            training_seconds=outcome.training_seconds,
            data_preparation_seconds=preparation_seconds,
            total_seconds=time.perf_counter() - started,
            early_stopped=outcome.early_stopped,
            torch_version=str(outcome.extra.get("framework_version", "unknown")),
            deterministic=outcome.deterministic,
        )

        report_path = self._results.save_report(report)
        rows_path = self._results.append_rows(report)
        logger.info(
            "Experiment %s wrote %s and appended %d row(s) to %s",
            experiment_id,
            report_path,
            len(results),
            rows_path,
        )
        self._log_headline(experiment_id, report, test_rows)
        return report

    # --- Steps ---------------------------------------------------------------

    def _freeze_threshold(
        self, config: TrainingConfig, predictions: SplitPredictions
    ) -> tuple[float, str | None]:
        """Return the threshold to apply to test, and the split it came from.

        ``None`` for the split means the threshold was configured rather than
        fitted, which is a materially different claim and is recorded as such.
        """
        if not config.threshold_policy.is_learned:
            return config.threshold, None
        chosen = self._metrics.select_threshold(
            predictions.labels,
            predictions.probabilities,
            split=SplitName.VALIDATION,
        )
        return chosen, SplitName.VALIDATION.value

    def _rows_for(
        self,
        config: TrainingConfig,
        experiment_id: str,
        split: SplitName,
        predictions: SplitPredictions,
        outcome: FitOutcome,
        frozen_threshold: float,
    ) -> tuple[ExperimentResult, ...]:
        """Build one result row per threshold policy for a scored split.

        Always emits the fixed-threshold row. Emits the learned-threshold row too
        when the configured policy is a learned one, so the report shows what the
        frozen threshold bought relative to the default rather than only the
        number the policy happened to produce.
        """
        rows = [
            self._row(
                config,
                experiment_id,
                split,
                predictions,
                outcome,
                threshold=config.threshold,
                policy=ThresholdPolicy.FIXED,
            )
        ]
        if config.threshold_policy.is_learned:
            rows.append(
                self._row(
                    config,
                    experiment_id,
                    split,
                    predictions,
                    outcome,
                    threshold=frozen_threshold,
                    policy=config.threshold_policy,
                )
            )
        return tuple(rows)

    def _row(
        self,
        config: TrainingConfig,
        experiment_id: str,
        split: SplitName,
        predictions: SplitPredictions,
        outcome: FitOutcome,
        *,
        threshold: float,
        policy: ThresholdPolicy,
    ) -> ExperimentResult:
        """Score one split at one threshold and wrap it as a result row."""
        measured = self._metrics.evaluate(
            predictions.labels,
            predictions.probabilities,
            threshold=threshold,
            loss=predictions.mean_loss,
        )
        return ExperimentResult(
            experiment_id=experiment_id,
            architecture=config.architecture,
            training_mode=TrainingMode.CENTRALIZED,
            split=split,
            metrics=measured,
            threshold_policy=policy,
            seed=config.random_seed,
            loss_type=LOSS_TYPE,
            training_time_seconds=outcome.training_seconds,
            evaluation_time_seconds=predictions.seconds,
            sequence_count=len(predictions),
            hyperparameters=config.to_dict(),
        )

    def _save_checkpoints(
        self,
        config: TrainingConfig,
        experiment_id: str,
        outcome: FitOutcome,
        threshold: float,
        feature_dimension: int,
    ) -> None:
        """Write the best and the final parameters to separate paths."""
        best_record = outcome.epochs[outcome.best_epoch - 1]
        final_record = outcome.epochs[-1]
        for kind, state, record in (
            (CheckpointKind.BEST, outcome.best_state, best_record),
            (CheckpointKind.FINAL, outcome.final_state, final_record),
        ):
            path = self._checkpoints.save(
                Checkpoint(
                    experiment_id=experiment_id,
                    kind=kind,
                    architecture=config.architecture,
                    epoch=record.epoch,
                    parameters=state,
                    config=config,
                    feature_dimension=feature_dimension,
                    threshold=threshold,
                    validation_loss=record.validation_loss,
                    random_seed=config.random_seed,
                    validation_metrics=dict(record.validation_metrics),
                )
            )
            logger.info("Wrote %s checkpoint (epoch %d) to %s", kind.value, record.epoch, path)

    # --- Guards and logging --------------------------------------------------

    @staticmethod
    def _require_matching_width(train: SequenceDataset, other: SequenceDataset) -> None:
        """Refuse two splits whose feature widths disagree."""
        if train.feature_dimension != other.feature_dimension:
            raise FeatureDimensionMismatchError(
                train.feature_dimension,
                other.feature_dimension,
                f"the '{other.split.value}' split",
            )

    @staticmethod
    def _experiment_id(config: TrainingConfig) -> str:
        """Build a filesystem-safe identifier that says what the run was.

        Carries the architecture, the training mode, the seed and a UTC timestamp,
        so two runs of the same configuration are distinguishable and neither can
        overwrite the other's checkpoint.
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return (
            f"centralized_{config.architecture.value}"
            f"_seed{config.random_seed}_{stamp}"
        )

    @staticmethod
    def _log_split(dataset: SequenceDataset) -> None:
        """Record what a split actually contained, including its class balance."""
        positives = dataset.positive_count
        total = dataset.sequence_count
        share = (positives / total * 100.0) if total else 0.0
        discarded = dataset.row_count - total
        logger.info(
            "Split %-10s rows=%-10d sequences=%-10d positives=%-7d (%.6f%%)%s",
            dataset.split.value,
            dataset.row_count,
            total,
            positives,
            share,
            f" [stride discarded {discarded} labelled row(s)]" if discarded else "",
        )

    @staticmethod
    def _log_epoch(experiment_id: str) -> EpochCallback:
        """Return a callback that logs each epoch as it completes."""

        def record(entry: EpochRecord) -> None:
            measured = " ".join(
                f"val_{name}={value:.6f}"
                for name, value in sorted(entry.validation_metrics.items())
            )
            logger.info(
                "Experiment %s epoch %d | train_loss=%.6f val_loss=%.6f %s | %.1fs%s",
                experiment_id,
                entry.epoch,
                entry.train_loss,
                entry.validation_loss,
                measured,
                entry.seconds,
                " | best so far" if entry.selected else "",
            )

        return record

    @staticmethod
    def _log_headline(
        experiment_id: str, report: ExperimentReport, test_sequences: int
    ) -> None:
        """Log the test numbers that matter, at the frozen threshold."""
        primary = report.result_for(SplitName.TEST, report.config.threshold_policy)
        if primary is None:
            primary = report.result_for(SplitName.TEST, ThresholdPolicy.FIXED)
        if primary is None:  # pragma: no cover - a test row is always emitted
            return
        measured: ClassificationMetrics = primary.metrics
        logger.info(
            "Experiment %s TEST (%d sequences, threshold %.6f, policy %s): "
            "recall=%.6f f1=%.6f pr_auc=%.6f precision=%.6f "
            "false_negatives=%d roc_auc=%.6f accuracy=%.6f",
            experiment_id,
            test_sequences,
            measured.threshold,
            primary.threshold_policy.value,
            measured.recall,
            measured.f1,
            measured.pr_auc,
            measured.precision,
            measured.false_negatives,
            measured.roc_auc,
            measured.accuracy,
        )
