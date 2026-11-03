"""Application service orchestrating Phase 10 Probability Calibration.

Strict sequence:
1. Locate official Phase 7 selected model (IID LSTM + FedAvg).
2. Generate/obtain validation predictions without mutating model weights.
3. Fit all candidate calibrators (None, Temperature Scaling, Platt Scaling, Isotonic Regression)
   using validation data only.
4. Compute validation probability quality (Log loss, Brier, ECE, MCE) and ranking metrics.
5. Select best calibration method using predefined validation selection rule
   (minimizing validation log loss).
6. Fit optimal decision threshold on validation data only and freeze it.
7. Single-pass evaluation on untouched test set using frozen calibrator and frozen threshold.
8. Persist isolated Phase 10 artifacts, reliability diagrams, JSON/CSV reports, and Markdown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from application.dto.calibration import (
    CalibrationConfig,
    CalibrationExperimentReport,
    CalibrationMethodResult,
)
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import SplitPredictions, TrainingEnginePort
from application.services.model_selector import ModelSelector
from domain.entities.calibration import CalibrationMethod, CalibrationParameters
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.evaluation.calibration_methods import (
    BaseCalibrator,
    build_calibrator,
)
from infrastructure.evaluation.calibration_metrics import (
    evaluate_calibration_metrics,
    select_optimal_f1_threshold,
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
from infrastructure.persistence.calibration_result_store import CalibrationResultStore
from infrastructure.visualization.calibration_visualizer import (
    plot_calibration_comparison_curves,
    plot_reliability_diagram,
)

logger = logging.getLogger(__name__)


class CalibrationService:
    """Coordinates the execution of Phase 10 Probability Calibration."""

    def __init__(
        self,
        *,
        model_selector: ModelSelector,
        sequence_provider: ProcessedSequenceProvider,
        metrics: MetricsCalculatorPort,
        checkpoint_root: Path,
        result_store: CalibrationResultStore,
        training_engine: TrainingEnginePort,
    ) -> None:
        """Initialize service dependencies."""
        self._selector = model_selector
        self._sequences = sequence_provider
        self._metrics = metrics
        self._checkpoint_root = checkpoint_root.resolve()
        self._results = result_store
        self._engine = training_engine

    def run(self, config: CalibrationConfig) -> CalibrationExperimentReport:
        """Execute Phase 10 probability calibration end-to-end."""
        started_total = time.perf_counter()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        experiment_id = f"calibration_seed{config.seed}_{timestamp}"

        seed_everything(config.seed)
        deterministic = request_deterministic_kernels()
        device = resolve_device(config.device)
        runtime = describe_runtime(device, deterministic=deterministic)
        logger.debug("Calibration runtime: %s", runtime)

        logger.info(
            "=== Starting Phase 10 Probability Calibration: %s (seed=%d) ===",
            experiment_id,
            config.seed,
        )

        self._results.init_directory_structure(experiment_id)

        # ---------------------------------------------------------------------
        # 1. Source Model Identification (Official Phase 7 IID LSTM + FedAvg)
        # ---------------------------------------------------------------------
        best_iid, _, _, _ = self._selector.select_pair()
        logger.info(
            "Official Reference Model: %s (%s, %s, %s)",
            best_iid.model.upper(),
            best_iid.algorithm.upper(),
            best_iid.distribution,
            best_iid.experiment_id,
        )

        source_model_info = {
            "model": best_iid.model,
            "algorithm": best_iid.algorithm,
            "distribution": best_iid.distribution,
            "experiment_id": best_iid.experiment_id,
            "checkpoint_kind": best_iid.checkpoint_kind,
            "checkpoint_path": best_iid.checkpoint_path,
            "validation_score": best_iid.validation_score,
            "selection_metric": best_iid.selection_metric,
        }

        # ---------------------------------------------------------------------
        # 2. Checkpoint Loading (Read-only)
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

        cfg_iid = replace(
            ckpt_iid.config,
            batch_size=config.batch_size,
            device=config.device,
            row_limit=config.row_limit,
        )

        # ---------------------------------------------------------------------
        # 3. Validation Predictions Generation (Validation Only)
        # ---------------------------------------------------------------------
        val_started = time.perf_counter()
        val_dataset = self._sequences.load(SplitName.VALIDATION, cfg_iid)
        val_prep_seconds = time.perf_counter() - val_started

        val_score_started = time.perf_counter()
        val_preds: SplitPredictions = self._engine.predict(
            cfg_iid, val_dataset, ckpt_iid.parameters
        )
        val_eval_seconds = time.perf_counter() - val_score_started

        y_val = val_preds.labels
        p_val = val_preds.probabilities
        logger.info(
            "Validation split predictions obtained: %d samples, %d positives",
            len(y_val),
            int(np.sum(y_val == 1.0)),
        )

        # ---------------------------------------------------------------------
        # 4. Calibration Methods Fitting (Validation Only)
        # ---------------------------------------------------------------------
        tuning_started = time.perf_counter()
        calibrator_instances: dict[CalibrationMethod, BaseCalibrator] = {}
        validation_results: list[CalibrationMethodResult] = []

        # Find validation threshold for uncalibrated baseline first
        val_baseline_threshold = select_optimal_f1_threshold(y_val, p_val)

        for method in config.methods:
            calibrator = build_calibrator(method)
            calibrator.fit(y_val, p_val)
            calibrator_instances[method] = calibrator

            p_val_calibrated = calibrator.calibrate(p_val)

            # Fit optimal threshold for this method on validation
            opt_thresh = select_optimal_f1_threshold(y_val, p_val_calibrated)

            val_metrics = evaluate_calibration_metrics(
                y_val,
                p_val_calibrated,
                threshold=opt_thresh,
                historical_threshold=config.historical_threshold,
                n_bins_ece=config.n_bins_ece,
                ece_strategy=config.ece_strategy,
            )

            res = CalibrationMethodResult(
                method=method,
                split="validation",
                parameters=calibrator.get_parameters(),
                metrics=val_metrics,
                selected=False,
            )
            validation_results.append(res)
            logger.info(
                "Method %s: Val Log Loss=%.6f, ECE=%.6f, Brier=%.6f, Val F1=%.6f (thresh=%.4f)",
                method.value,
                val_metrics.log_loss,
                val_metrics.ece,
                val_metrics.brier_score,
                val_metrics.f1,
                val_metrics.threshold,
            )

        # ---------------------------------------------------------------------
        # 5. Method Selection & Freezing (Validation Only)
        # ---------------------------------------------------------------------
        # Predefined rule: minimize validation log loss, breaking ties by ECE
        candidate_results = [r for r in validation_results if r.method != CalibrationMethod.NONE]
        if not candidate_results:
            selected_res = validation_results[0]
        else:
            selected_res = min(
                candidate_results,
                key=lambda r: (r.metrics.log_loss, r.metrics.ece),
            )

        selected_method = selected_res.method
        selected_calibrator = calibrator_instances[selected_method]
        selected_params = selected_calibrator.get_parameters()

        # Update selected flag in validation_results
        validation_results = [
            replace(r, selected=(r.method == selected_method)) for r in validation_results
        ]
        tuning_seconds = time.perf_counter() - tuning_started

        # Freeze selected decision threshold
        p_val_best = selected_calibrator.calibrate(p_val)
        frozen_threshold = select_optimal_f1_threshold(y_val, p_val_best)

        logger.info(
            "FROZEN CALIBRATOR: Method=%s, Threshold=%.6f, Parameters=%s",
            selected_method.value,
            frozen_threshold,
            selected_params.to_dict(),
        )

        # Save validation results
        self._results.save_validation_results(experiment_id, validation_results)

        # Free validation memory
        del val_dataset
        del p_val
        del p_val_best

        # ---------------------------------------------------------------------
        # 6. Single-Pass Test Evaluation (Untouched until now)
        # ---------------------------------------------------------------------
        test_score_started = time.perf_counter()
        test_dataset = self._sequences.load(SplitName.TEST, cfg_iid)
        test_preds: SplitPredictions = self._engine.predict(
            cfg_iid, test_dataset, ckpt_iid.parameters
        )
        test_eval_seconds = time.perf_counter() - test_score_started

        y_test = test_preds.labels
        p_test_uncal = test_preds.probabilities

        # 6.1 Baseline test metrics (uncalibrated)
        test_baseline_metrics = evaluate_calibration_metrics(
            y_test,
            p_test_uncal,
            threshold=val_baseline_threshold,  # Evaluated at its validation-selected threshold
            historical_threshold=config.historical_threshold,
            n_bins_ece=config.n_bins_ece,
            ece_strategy=config.ece_strategy,
        )
        test_baseline_result = CalibrationMethodResult(
            method=CalibrationMethod.NONE,
            split="test",
            parameters=CalibrationParameters(method=CalibrationMethod.NONE),
            metrics=test_baseline_metrics,
            selected=False,
        )

        # 6.2 Calibrated test metrics (using frozen calibrator and frozen threshold)
        p_test_calibrated = selected_calibrator.calibrate(p_test_uncal)
        test_calibrated_metrics = evaluate_calibration_metrics(
            y_test,
            p_test_calibrated,
            threshold=frozen_threshold,  # Strictly evaluated at frozen threshold
            historical_threshold=config.historical_threshold,
            n_bins_ece=config.n_bins_ece,
            ece_strategy=config.ece_strategy,
        )
        test_calibrated_result = CalibrationMethodResult(
            method=selected_method,
            split="test",
            parameters=selected_params,
            metrics=test_calibrated_metrics,
            selected=True,
        )

        logger.info(
            "Test Complete: Baseline Loss=%.6f (ECE=%.6f) -> Calibrated Loss=%.6f (ECE=%.6f)",
            test_baseline_metrics.log_loss,
            test_baseline_metrics.ece,
            test_calibrated_metrics.log_loss,
            test_calibrated_metrics.ece,
        )

        # Save test results and sample predictions
        self._results.save_test_results(
            experiment_id, test_baseline_result, test_calibrated_result
        )
        self._results.save_predictions_sample(
            experiment_id=experiment_id,
            split="test",
            row_indices=test_preds.row_indices,
            y_true=y_test,
            p_uncalibrated=p_test_uncal,
            p_calibrated=p_test_calibrated,
            threshold=frozen_threshold,
        )

        # ---------------------------------------------------------------------
        # 7. Visualizations
        # ---------------------------------------------------------------------
        vis_dir = self._results.experiment_dir(experiment_id) / "visualizations"
        base_title = (
            f"Test Reliability Diagram (Baseline - ECE={test_baseline_metrics.ece:.4f})"
        )
        plot_reliability_diagram(
            y_test,
            p_test_uncal,
            title=base_title,
            output_path=vis_dir / "01_baseline_reliability_diagram.png",
            n_bins=config.n_bins_ece,
            strategy=config.ece_strategy,
        )
        cal_title = (
            f"Test Reliability Diagram ({selected_method.value} - "
            f"ECE={test_calibrated_metrics.ece:.4f})"
        )
        plot_reliability_diagram(
            y_test,
            p_test_calibrated,
            title=cal_title,
            output_path=vis_dir / "02_calibrated_reliability_diagram.png",
            n_bins=config.n_bins_ece,
            strategy=config.ece_strategy,
        )

        # Multi-model calibration curves
        comparison_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "Uncalibrated": (y_test, p_test_uncal),
            f"Calibrated ({selected_method.value})": (y_test, p_test_calibrated),
        }
        plot_calibration_comparison_curves(
            comparison_dict,
            title="Probability Calibration Curves Comparison on Test Split",
            output_path=vis_dir / "03_calibration_comparison_curves.png",
            n_bins=config.n_bins_ece,
        )

        # ---------------------------------------------------------------------
        # 8. Report Compilation & Persistence
        # ---------------------------------------------------------------------
        total_seconds = time.perf_counter() - started_total
        timing = {
            "val_prep_seconds": val_prep_seconds,
            "val_eval_seconds": val_eval_seconds,
            "tuning_seconds": tuning_seconds,
            "test_eval_seconds": test_eval_seconds,
            "total_seconds": total_seconds,
        }

        report = CalibrationExperimentReport(
            experiment_id=experiment_id,
            timestamp=timestamp,
            source_model=source_model_info,
            config=config,
            validation_results=validation_results,
            selected_method=selected_method,
            selected_parameters=selected_params,
            frozen_threshold=frozen_threshold,
            test_baseline=test_baseline_result,
            test_calibrated=test_calibrated_result,
            timing=timing,
        )

        self._results.save_report_and_metadata(report)
        self._results.save_markdown_report(report)

        logger.info(
            "=== Phase 10 Probability Calibration Finished in %.1fs: Selected %s, "
            "Test Log Loss: %.6f -> %.6f ===",
            total_seconds,
            selected_method.value,
            test_baseline_metrics.log_loss,
            test_calibrated_metrics.log_loss,
        )

        return report
