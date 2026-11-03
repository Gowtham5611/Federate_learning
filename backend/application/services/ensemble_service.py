"""Application service orchestrating Phase 7 IID + Non-IID Ensemble training.

Pipeline:
1. Automated discovery & selection of best IID and Non-IID models on validation PR-AUC.
2. Generation of validation predictions and explicit alignment verification.
3. Alpha grid tuning on centralized validation predictions to optimize validation PR-AUC.
4. Validation-only threshold selection maximizing F1.
5. Strict parameter freeze: selected models, checkpoints, best alpha, and classification threshold.
6. Single-pass evaluation on the untouched test split.
7. Persistence into experiments/results/ensemble/<experiment_id>/.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from application.dto.ensemble import (
    AlphaCandidateRecord,
    EnsembleConfig,
    EnsembleExperimentReport,
)
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    SplitPredictions,
    TrainingEnginePort,
)
from application.services.model_selector import ModelSelector
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.exceptions.training import PredictionAlignmentError
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.federated.server.federated_checkpoint_store import (
    FederatedCheckpointStore,
)
from infrastructure.models.pytorch.reproducibility import (
    describe_runtime,
    request_deterministic_kernels,
    resolve_device,
    seed_everything,
)
from infrastructure.persistence.ensemble_result_store import EnsembleResultStore

logger = logging.getLogger(__name__)


def verify_prediction_alignment(
    preds_iid: SplitPredictions,
    preds_non_iid: SplitPredictions,
    *,
    split: str,
) -> None:
    """Verify that predictions from IID and Non-IID models align deterministically.

    Raises:
        PredictionAlignmentError: If lengths, labels, or row indices do not match.
    """
    if len(preds_iid.labels) != len(preds_non_iid.labels):
        raise PredictionAlignmentError(
            f"{split.capitalize()} prediction length mismatch: IID has {len(preds_iid.labels)}, "
            f"Non-IID has {len(preds_non_iid.labels)}"
        )
    if not np.array_equal(preds_iid.labels, preds_non_iid.labels):
        raise PredictionAlignmentError(
            f"{split.capitalize()} prediction ground-truth labels do not align"
        )
    if not np.array_equal(preds_iid.row_indices, preds_non_iid.row_indices):
        raise PredictionAlignmentError(
            f"{split.capitalize()} prediction row indices do not align"
        )


class EnsembleService:
    """Coordinates the end-to-end execution of Phase 7 ensemble modeling."""

    def __init__(
        self,
        *,
        model_selector: ModelSelector,
        sequence_provider: ProcessedSequenceProvider,
        metrics: MetricsCalculatorPort,
        checkpoint_root: Path,
        result_store: EnsembleResultStore,
        training_engine: TrainingEnginePort,
    ) -> None:
        """Initialize service dependencies."""
        self._selector = model_selector
        self._sequences = sequence_provider
        self._metrics = metrics
        self._checkpoint_root = checkpoint_root.resolve()
        self._results = result_store
        self._engine = training_engine


    def run(self, config: EnsembleConfig) -> EnsembleExperimentReport:
        """Execute full Phase 7 ensemble workflow."""
        started_total = time.perf_counter()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        experiment_id = f"ensemble_iid_non_iid_seed{config.seed}_{timestamp}"

        seed_everything(config.seed)
        deterministic = request_deterministic_kernels()
        device = resolve_device(config.device)
        runtime = describe_runtime(device, deterministic=deterministic)
        logger.debug("Ensemble runtime environment: %s", runtime)

        logger.info(
            "=== Starting Phase 7 Ensemble Experiment: %s (seed=%d, alpha_step=%.2f) ===",
            experiment_id,
            config.seed,
            config.alpha_step,
        )

        # ---------------------------------------------------------------------
        # 1. Model Selection (Validation Only)
        # ---------------------------------------------------------------------
        best_iid, best_non_iid, iid_candidates, non_iid_candidates = (
            self._selector.select_pair()
        )

        self._results.save_model_selection(
            experiment_id,
            iid_candidates,
            non_iid_candidates,
            best_iid,
            best_non_iid,
        )

        # ---------------------------------------------------------------------
        # 2. Checkpoint Loading
        # ---------------------------------------------------------------------
        store_iid = FederatedCheckpointStore(
            self._checkpoint_root / "federated" / best_iid.distribution,
            algorithm=best_iid.algorithm,
        )
        ckpt_iid = store_iid.load(
            best_iid.experiment_id,
            CheckpointKind.GLOBAL_BEST,
            architecture=ModelArchitecture(best_iid.model),
        )

        store_non_iid = FederatedCheckpointStore(
            self._checkpoint_root / "federated" / best_non_iid.distribution,
            algorithm=best_non_iid.algorithm,
        )
        ckpt_non_iid = store_non_iid.load(
            best_non_iid.experiment_id,
            CheckpointKind.GLOBAL_BEST,
            architecture=ModelArchitecture(best_non_iid.model),
        )


        cfg_iid = replace(
            ckpt_iid.config,
            batch_size=config.batch_size,
            device=config.device,
            row_limit=config.row_limit,
        )
        cfg_non_iid = replace(
            ckpt_non_iid.config,
            batch_size=config.batch_size,
            device=config.device,
            row_limit=config.row_limit,
        )

        # ---------------------------------------------------------------------
        # 3. Centralized Validation Scoring & Alignment Check
        # ---------------------------------------------------------------------
        val_started = time.perf_counter()
        val_dataset = self._sequences.load(SplitName.VALIDATION, cfg_iid)
        val_prep_seconds = time.perf_counter() - val_started

        val_scoring_started = time.perf_counter()
        preds_iid_val = self._engine.predict(cfg_iid, val_dataset, ckpt_iid.parameters)
        preds_non_iid_val = self._engine.predict(cfg_non_iid, val_dataset, ckpt_non_iid.parameters)
        val_scoring_seconds = time.perf_counter() - val_scoring_started

        verify_prediction_alignment(preds_iid_val, preds_non_iid_val, split="validation")

        y_val = preds_iid_val.labels
        p_iid_val = preds_iid_val.probabilities
        p_non_iid_val = preds_non_iid_val.probabilities

        # ---------------------------------------------------------------------
        # 4. Alpha Grid Search on Validation
        # ---------------------------------------------------------------------
        alpha_tuning_started = time.perf_counter()
        alpha_records: list[AlphaCandidateRecord] = []

        for alpha in config.alpha_grid:
            p_ens_candidate = alpha * p_iid_val + (1.0 - alpha) * p_non_iid_val
            measured = self._metrics.evaluate(y_val, p_ens_candidate, threshold=0.5)
            alpha_records.append(
                AlphaCandidateRecord(
                    alpha=alpha,
                    pr_auc=measured.pr_auc,
                    roc_auc=measured.roc_auc,
                    f1=measured.f1,
                    recall=measured.recall,
                    precision=measured.precision,
                    accuracy=measured.accuracy,
                )
            )

        # Deterministic selection: max PR-AUC, tie-break closest to alpha=0.5
        best_candidate = max(
            alpha_records,
            key=lambda a: (a.pr_auc, -abs(a.alpha - 0.5)),
        )
        best_alpha = best_candidate.alpha

        marked_alpha_records = [
            replace(a, selected=(a.alpha == best_alpha)) for a in alpha_records
        ]
        alpha_tuning_seconds = time.perf_counter() - alpha_tuning_started

        self._results.save_alpha_search(experiment_id, marked_alpha_records)

        logger.info(
            "Selected Optimal Alpha: %.4f (Val PR-AUC=%.6f, Val ROC-AUC=%.6f)",
            best_alpha,
            best_candidate.pr_auc,
            best_candidate.roc_auc,
        )

        # ---------------------------------------------------------------------
        # 5. Validation Classification Threshold Selection
        # ---------------------------------------------------------------------
        p_ens_val = best_alpha * p_iid_val + (1.0 - best_alpha) * p_non_iid_val
        frozen_threshold = self._metrics.select_threshold(
            y_val, p_ens_val, split=SplitName.VALIDATION
        )

        val_metrics_frozen = self._metrics.evaluate(
            y_val, p_ens_val, threshold=frozen_threshold
        )
        val_metrics_fixed = self._metrics.evaluate(
            y_val, p_ens_val, threshold=0.5
        )

        val_metrics_dict: dict[str, Any] = {
            "fixed_threshold_0.5": val_metrics_fixed.to_dict(),
            "frozen_threshold": {
                "threshold": frozen_threshold,
                "metrics": val_metrics_frozen.to_dict(),
            },
            "sequence_count": val_dataset.sequence_count,
        }
        self._results.save_validation_metrics(experiment_id, val_metrics_dict)

        # ---------------------------------------------------------------------
        # 6. FREEZE STEP
        # ---------------------------------------------------------------------
        logger.info(
            "FROZEN PIPELINE: IID=%s (%s), Non-IID=%s (%s), Alpha=%.4f, Threshold=%.6f",
            best_iid.model.upper(),
            best_iid.experiment_id,
            best_non_iid.model.upper(),
            best_non_iid.experiment_id,
            best_alpha,
            frozen_threshold,
        )

        del val_dataset
        del p_iid_val
        del p_non_iid_val
        del p_ens_val

        # ---------------------------------------------------------------------
        # 7. Single-Pass Test Split Evaluation (Untouched until now)
        # ---------------------------------------------------------------------
        test_scoring_started = time.perf_counter()
        test_dataset = self._sequences.load(SplitName.TEST, cfg_iid)

        preds_iid_test = self._engine.predict(
            cfg_iid, test_dataset, ckpt_iid.parameters
        )
        preds_non_iid_test = self._engine.predict(
            cfg_non_iid, test_dataset, ckpt_non_iid.parameters
        )
        test_scoring_seconds = time.perf_counter() - test_scoring_started

        verify_prediction_alignment(preds_iid_test, preds_non_iid_test, split="test")

        y_test = preds_iid_test.labels
        p_iid_test = preds_iid_test.probabilities
        p_non_iid_test = preds_non_iid_test.probabilities
        p_ens_test = best_alpha * p_iid_test + (1.0 - best_alpha) * p_non_iid_test

        test_metrics_frozen = self._metrics.evaluate(
            y_test, p_ens_test, threshold=frozen_threshold
        )
        test_metrics_fixed = self._metrics.evaluate(
            y_test, p_ens_test, threshold=0.5
        )

        test_metrics_dict: dict[str, Any] = {
            "fixed_threshold_0.5": test_metrics_fixed.to_dict(),
            "frozen_threshold": {
                "threshold": frozen_threshold,
                "metrics": test_metrics_frozen.to_dict(),
            },
            "sequence_count": test_dataset.sequence_count,
        }
        self._results.save_test_metrics(experiment_id, test_metrics_dict)

        self._results.save_test_predictions(
            experiment_id=experiment_id,
            row_indices=preds_iid_test.row_indices,
            true_labels=y_test,
            p_iid=p_iid_test,
            p_non_iid=p_non_iid_test,
            p_ensemble=p_ens_test,
            frozen_threshold=frozen_threshold,
        )

        total_seconds = time.perf_counter() - started_total

        timing = {
            "val_prep_seconds": val_prep_seconds,
            "val_eval_seconds": val_scoring_seconds,
            "tuning_seconds": alpha_tuning_seconds,
            "test_eval_seconds": test_scoring_seconds,
            "total_seconds": total_seconds,
        }

        report = EnsembleExperimentReport(
            experiment_id=experiment_id,
            timestamp=timestamp,
            config=config,
            selected_iid=best_iid,
            selected_non_iid=best_non_iid,
            alpha_search=marked_alpha_records,
            best_alpha=best_alpha,
            frozen_threshold=frozen_threshold,
            validation_metrics_at_frozen_threshold=val_metrics_frozen,
            validation_metrics_at_fixed_threshold=val_metrics_fixed,
            test_metrics_at_frozen_threshold=test_metrics_frozen,
            test_metrics_at_fixed_threshold=test_metrics_fixed,
            timing=timing,
        )

        self._results.save_report_and_metadata(report)
        self._results.save_results_csv(
            report,
            sequence_count_val=len(y_val),
            sequence_count_test=test_dataset.sequence_count,
        )

        logger.info(
            "=== Phase 7 Ensemble Complete in %.1fs: Best Alpha=%.4f, Test PR-AUC=%.6f, "
            "Test F1=%.6f ===",
            total_seconds,
            best_alpha,
            test_metrics_frozen.pr_auc,
            test_metrics_frozen.f1,
        )

        return report
