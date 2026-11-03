"""Integration test for Phase 10 Probability Calibration pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from application.dto.calibration import CalibrationConfig
from application.dto.ensemble import ModelSelectionRecord
from application.ports.output.model_trainer import SplitPredictions
from application.services.calibration_service import CalibrationService
from application.services.model_selector import ModelSelector
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine
from infrastructure.persistence.calibration_result_store import (
    CalibrationResultStore,
)


class TestCalibrationPipelineIntegration:
    """Integration test verifying end-to-end Phase 10 execution and zero test-leakage."""

    def test_end_to_end_calibration_workflow_mocked_data(
        self, tmp_path: Path
    ) -> None:
        """Verify calibration runs without test leakage or source model mutation."""
        rng = np.random.default_rng(42)
        n_val, n_test = 2000, 2000

        y_val = np.zeros(n_val, dtype=np.float32)
        y_val[:10] = 1.0
        rng.shuffle(y_val)
        p_val = rng.uniform(0.0001, 0.05, size=n_val).astype(np.float32)
        p_val[y_val == 1.0] = rng.uniform(0.1, 0.9, size=10).astype(np.float32)

        y_test = np.zeros(n_test, dtype=np.float32)
        y_test[:15] = 1.0
        rng.shuffle(y_test)
        p_test = rng.uniform(0.0001, 0.05, size=n_test).astype(np.float32)
        p_test[y_test == 1.0] = rng.uniform(0.1, 0.9, size=15).astype(np.float32)

        preds_val = SplitPredictions(
            labels=y_val,
            probabilities=p_val,
            row_indices=np.arange(n_val, dtype=np.int64),
            mean_loss=0.01,
            seconds=0.1,
        )
        preds_test = SplitPredictions(
            labels=y_test,
            probabilities=p_test,
            row_indices=np.arange(n_test, dtype=np.int64),
            mean_loss=0.01,
            seconds=0.1,
        )

        mock_selector = MagicMock(spec=ModelSelector)
        best_iid_record = ModelSelectionRecord(
            experiment_id="federated_iid_fedavg_lstm_seed42_20260918T130515",
            model="lstm",
            algorithm="fedavg",
            distribution="iid",
            checkpoint_kind="global_best",
            checkpoint_path=str(tmp_path / "global_best.pt"),
            validation_score=0.367,
            selection_metric="pr_auc",
        )
        mock_selector.select_pair.return_value = (
            best_iid_record,
            best_iid_record,
            [],
            [],
        )

        # Create dummy checkpoint directory structure
        ckpt_dir = (
            tmp_path
            / "models"
            / "checkpoints"
            / "federated"
            / "iid"
            / "fedavg"
            / "lstm"
            / best_iid_record.experiment_id
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Dummy checkpoint file
        import torch

        from application.dto.training import TrainingConfig
        from domain.enums.model_architecture import ModelArchitecture
        dummy_config = TrainingConfig(architecture=ModelArchitecture.LSTM)
        torch.save(
            {
                "format_version": "1.0",
                "experiment_id": best_iid_record.experiment_id,
                "kind": "global_best",
                "architecture": "lstm",
                "epoch": 5,
                "parameters": {},
                "config": dummy_config.to_dict(),
                "feature_dimension": 57,
                "threshold": 0.13,
                "validation_loss": 0.01,
                "random_seed": 42,
                "validation_metrics": {},
            },
            ckpt_dir / "global_best.pt",
        )

        mock_sequences = MagicMock(spec=ProcessedSequenceProvider)
        mock_ds = MagicMock()
        mock_sequences.load.return_value = mock_ds

        mock_engine = MagicMock(spec=TorchTrainingEngine)
        mock_engine.predict.side_effect = [preds_val, preds_test]

        result_store = CalibrationResultStore(root=tmp_path / "experiments" / "results")
        metrics = SklearnMetricsCalculator()

        service = CalibrationService(
            model_selector=mock_selector,
            sequence_provider=mock_sequences,
            metrics=metrics,
            checkpoint_root=tmp_path / "models" / "checkpoints",
            result_store=result_store,
            training_engine=mock_engine,
        )

        config = CalibrationConfig(seed=42, n_bins_ece=10)
        report = service.run(config)

        assert report.selected_method in (
            "temperature_scaling",
            "platt_scaling",
            "isotonic_regression",
            "none",
        )
        assert report.test_baseline.metrics.pr_auc > 0.0
        assert report.test_calibrated.metrics.pr_auc > 0.0
        # Check files were generated
        exp_dir = result_store.experiment_dir(report.experiment_id)
        assert (exp_dir / "report.json").exists()
        assert (exp_dir / "metadata.json").exists()
        assert (exp_dir / "report.md").exists()
        assert (exp_dir / "validation" / "validation_comparison.csv").exists()
        assert (exp_dir / "test" / "test_comparison.csv").exists()
        assert (exp_dir / "visualizations" / "01_baseline_reliability_diagram.png").exists()
        assert (exp_dir / "visualizations" / "02_calibrated_reliability_diagram.png").exists()
        assert (exp_dir / "visualizations" / "03_calibration_comparison_curves.png").exists()
