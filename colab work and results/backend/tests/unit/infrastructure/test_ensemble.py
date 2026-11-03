"""Unit tests for Phase 7 IID + Non-IID Ensemble Modeling."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from application.dto.ensemble import (
    AlphaCandidateRecord,
    EnsembleConfig,
    EnsembleExperimentReport,
    ModelSelectionRecord,
)
from application.dto.experiment import (
    ClassificationMetrics,
    ConfusionCounts,
)
from application.ports.output.model_trainer import SplitPredictions
from application.services.ensemble_service import verify_prediction_alignment
from application.services.model_selector import ModelSelector
from domain.exceptions.training import PredictionAlignmentError
from infrastructure.persistence.ensemble_result_store import EnsembleResultStore


class TestAlphaCombination:
    """Test alpha combination formula and grid generation."""

    def test_default_alpha_grid(self) -> None:
        """Verify default alpha grid has 21 points with step size 0.05."""
        cfg = EnsembleConfig()
        grid = cfg.alpha_grid
        assert len(grid) == 21
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        assert 0.5 in grid
        for i in range(len(grid) - 1):
            assert round(grid[i + 1] - grid[i], 2) == 0.05

    def test_formula_boundary_zero(self) -> None:
        """Verify alpha=0.0 yields exactly the Non-IID predictions."""
        p_iid = np.array([0.2, 0.8, 0.4])
        p_non_iid = np.array([0.7, 0.1, 0.9])
        alpha = 0.0
        p_ens = alpha * p_iid + (1.0 - alpha) * p_non_iid
        np.testing.assert_allclose(p_ens, p_non_iid)

    def test_formula_boundary_one(self) -> None:
        """Verify alpha=1.0 yields exactly the IID predictions."""
        p_iid = np.array([0.2, 0.8, 0.4])
        p_non_iid = np.array([0.7, 0.1, 0.9])
        alpha = 1.0
        p_ens = alpha * p_iid + (1.0 - alpha) * p_non_iid
        np.testing.assert_allclose(p_ens, p_iid)

    def test_formula_midpoint(self) -> None:
        """Verify alpha=0.5 produces an equal blend of both model outputs."""
        p_iid = np.array([0.2, 0.8, 0.4])
        p_non_iid = np.array([0.6, 0.2, 0.8])
        alpha = 0.5
        p_ens = alpha * p_iid + (1.0 - alpha) * p_non_iid
        np.testing.assert_allclose(p_ens, [0.4, 0.5, 0.6])


class TestPredictionAlignment:
    """Test verification of deterministic prediction alignment."""

    def test_aligned_predictions_pass(self) -> None:
        """Verify that matching predictions pass alignment checks."""
        preds_iid = SplitPredictions(
            labels=np.array([0, 1, 0], dtype=np.float32),
            probabilities=np.array([0.1, 0.9, 0.3], dtype=np.float32),
            row_indices=np.array([0, 1, 2], dtype=np.int64),
            mean_loss=0.2,
            seconds=0.1,
        )
        preds_non_iid = SplitPredictions(
            labels=np.array([0, 1, 0], dtype=np.float32),
            probabilities=np.array([0.4, 0.6, 0.2], dtype=np.float32),
            row_indices=np.array([0, 1, 2], dtype=np.int64),
            mean_loss=0.3,
            seconds=0.1,
        )
        # Should not raise
        verify_prediction_alignment(preds_iid, preds_non_iid, split="validation")

    def test_length_mismatch_raises(self) -> None:
        """Verify that prediction length mismatches raise PredictionAlignmentError."""
        preds_iid = SplitPredictions(
            labels=np.array([0, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.9], dtype=np.float32),
            row_indices=np.array([0, 1], dtype=np.int64),
            mean_loss=0.2,
            seconds=0.1,
        )
        preds_non_iid = SplitPredictions(
            labels=np.array([0, 1, 0], dtype=np.float32),
            probabilities=np.array([0.1, 0.9, 0.5], dtype=np.float32),
            row_indices=np.array([0, 1, 2], dtype=np.int64),
            mean_loss=0.2,
            seconds=0.1,
        )
        with pytest.raises(PredictionAlignmentError, match="length mismatch"):
            verify_prediction_alignment(preds_iid, preds_non_iid, split="validation")

    def test_label_mismatch_raises(self) -> None:
        """Verify that ground-truth label discrepancies raise PredictionAlignmentError."""
        preds_iid = SplitPredictions(
            labels=np.array([0, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.9], dtype=np.float32),
            row_indices=np.array([0, 1], dtype=np.int64),
            mean_loss=0.2,
            seconds=0.1,
        )
        preds_non_iid = SplitPredictions(
            labels=np.array([0, 0], dtype=np.float32),  # Differing label
            probabilities=np.array([0.4, 0.6], dtype=np.float32),
            row_indices=np.array([0, 1], dtype=np.int64),
            mean_loss=0.3,
            seconds=0.1,
        )
        with pytest.raises(PredictionAlignmentError, match="ground-truth labels do not align"):
            verify_prediction_alignment(preds_iid, preds_non_iid, split="validation")

    def test_row_index_mismatch_raises(self) -> None:
        """Verify that row index sequence discrepancies raise PredictionAlignmentError."""
        preds_iid = SplitPredictions(
            labels=np.array([0, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.9], dtype=np.float32),
            row_indices=np.array([0, 1], dtype=np.int64),
            mean_loss=0.2,
            seconds=0.1,
        )
        preds_non_iid = SplitPredictions(
            labels=np.array([0, 1], dtype=np.float32),
            probabilities=np.array([0.4, 0.6], dtype=np.float32),
            row_indices=np.array([0, 2], dtype=np.int64),  # Differing index
            mean_loss=0.3,
            seconds=0.1,
        )
        with pytest.raises(PredictionAlignmentError, match="row indices do not align"):
            verify_prediction_alignment(preds_iid, preds_non_iid, split="validation")


class TestModelSelector:
    """Test model selection logic based on validation PR-AUC."""

    def test_select_best_picks_highest_validation_score(self, tmp_path: Path) -> None:
        """Verify that select_best selects the candidate with maximum validation PR-AUC."""
        selector = ModelSelector(results_root=tmp_path, checkpoints_root=tmp_path)
        candidates = [
            ModelSelectionRecord(
                distribution="iid",
                algorithm="fedavg",
                model="gru",
                experiment_id="exp_low",
                checkpoint_kind="global_best",
                checkpoint_path="dummy/path/low.pt",
                selection_metric="pr_auc",
                validation_score=0.250,
            ),
            ModelSelectionRecord(
                distribution="iid",
                algorithm="fedavg",
                model="lstm",
                experiment_id="exp_high",
                checkpoint_kind="global_best",
                checkpoint_path="dummy/path/high.pt",
                selection_metric="pr_auc",
                validation_score=0.450,
            ),
            ModelSelectionRecord(
                distribution="iid",
                algorithm="fedavg",
                model="bilstm",
                experiment_id="exp_mid",
                checkpoint_kind="global_best",
                checkpoint_path="dummy/path/mid.pt",
                selection_metric="pr_auc",
                validation_score=0.350,
            ),
        ]
        best = selector.select_best(candidates)
        assert best.experiment_id == "exp_high"
        assert best.model == "lstm"
        assert best.validation_score == 0.450

    def test_tie_breaking_deterministic(self, tmp_path: Path) -> None:
        """Verify that ties in validation score break deterministically by experiment_id."""
        selector = ModelSelector(results_root=tmp_path, checkpoints_root=tmp_path)
        candidates = [
            ModelSelectionRecord(
                distribution="non_iid",
                algorithm="fedavg",
                model="gru",
                experiment_id="exp_b",
                checkpoint_kind="global_best",
                checkpoint_path="dummy/b.pt",
                selection_metric="pr_auc",
                validation_score=0.300,
            ),
            ModelSelectionRecord(
                distribution="non_iid",
                algorithm="fedavg",
                model="lstm",
                experiment_id="exp_a",
                checkpoint_kind="global_best",
                checkpoint_path="dummy/a.pt",
                selection_metric="pr_auc",
                validation_score=0.300,
            ),
        ]
        # Highest (validation_score, experiment_id) wins
        best = selector.select_best(candidates)
        assert best.experiment_id == "exp_b"


class TestEnsembleResultStore:
    """Test writing of ensemble artifacts."""

    def test_save_all_artifacts(self, tmp_path: Path) -> None:
        """Verify that EnsembleResultStore writes all expected files and structure."""
        store = EnsembleResultStore(root=tmp_path)
        exp_id = "ensemble_test_exp"
        exp_dir = store.experiment_dir(exp_id)

        iid_rec = ModelSelectionRecord(
            distribution="iid",
            algorithm="fedavg",
            model="lstm",
            experiment_id="iid_exp",
            checkpoint_kind="global_best",
            checkpoint_path="dummy/iid.pt",
            selection_metric="pr_auc",
            validation_score=0.36,
        )
        non_iid_rec = ModelSelectionRecord(
            distribution="non_iid",
            algorithm="fedavg",
            model="bilstm",
            experiment_id="non_iid_exp",
            checkpoint_kind="global_best",
            checkpoint_path="dummy/non_iid.pt",
            selection_metric="pr_auc",
            validation_score=0.10,
        )

        store.save_model_selection(
            experiment_id=exp_id,
            iid_candidates=[iid_rec],
            non_iid_candidates=[non_iid_rec],
            best_iid=iid_rec,
            best_non_iid=non_iid_rec,
        )
        assert (exp_dir / "validation" / "model_selection.csv").exists()

        alpha_rec = AlphaCandidateRecord(
            alpha=0.5,
            pr_auc=0.38,
            roc_auc=0.88,
            f1=0.42,
            recall=0.50,
            precision=0.35,
            accuracy=0.99,
            selected=True,
        )
        store.save_alpha_search(exp_id, [alpha_rec])
        assert (exp_dir / "validation" / "alpha_search.csv").exists()

        store.save_validation_metrics(exp_id, {"best_alpha": 0.5, "pr_auc": 0.38})
        assert (exp_dir / "validation" / "validation_metrics.json").exists()

        row_indices = np.array([100, 101], dtype=np.int64)
        true_labels = np.array([0, 1], dtype=np.float32)
        p_iid = np.array([0.1, 0.9], dtype=np.float32)
        p_non_iid = np.array([0.2, 0.8], dtype=np.float32)
        p_ens = np.array([0.15, 0.85], dtype=np.float32)
        store.save_test_predictions(
            experiment_id=exp_id,
            row_indices=row_indices,
            true_labels=true_labels,
            p_iid=p_iid,
            p_non_iid=p_non_iid,
            p_ensemble=p_ens,
            frozen_threshold=0.35,
        )
        assert (exp_dir / "test" / "test_predictions.csv").exists()

        store.save_test_metrics(exp_id, {"pr_auc": 0.37})
        assert (exp_dir / "test" / "test_metrics.json").exists()

        dummy_metrics = ClassificationMetrics(
            precision=0.35,
            recall=0.5,
            f1=0.4,
            pr_auc=0.37,
            roc_auc=0.85,
            accuracy=0.99,
            confusion=ConfusionCounts(
                true_positives=1,
                true_negatives=99,
                false_positives=1,
                false_negatives=1,
            ),
            threshold=0.35,
            positive_rate=0.001,
            loss=0.05,
        )
        report = EnsembleExperimentReport(
            experiment_id=exp_id,
            timestamp="20260923T150000",
            config=EnsembleConfig(),
            selected_iid=iid_rec,
            selected_non_iid=non_iid_rec,
            alpha_search=[alpha_rec],
            best_alpha=0.5,
            frozen_threshold=0.35,
            validation_metrics_at_frozen_threshold=dummy_metrics,
            validation_metrics_at_fixed_threshold=dummy_metrics,
            test_metrics_at_frozen_threshold=dummy_metrics,
            test_metrics_at_fixed_threshold=dummy_metrics,
            timing={"total_seconds": 1.0},
        )
        store.save_report_and_metadata(report)
        assert (exp_dir / "report.json").exists()
        assert (exp_dir / "metadata.json").exists()

        store.save_results_csv(report, sequence_count_val=1000, sequence_count_test=1000)
        assert (exp_dir / "results.csv").exists()

        with (exp_dir / "results.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 4  # fixed/frozen validation + fixed/frozen test
            assert rows[0]["experiment_id"] == exp_id

