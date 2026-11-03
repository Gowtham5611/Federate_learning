"""Unit and integration tests for Phase 10 Probability Calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from application.dto.calibration import (
    CalibrationConfig,
    CalibrationEvaluationMetrics,
    CalibrationExperimentReport,
    CalibrationMethodResult,
)
from domain.entities.calibration import CalibrationMethod, CalibrationParameters
from infrastructure.evaluation.calibration_methods import (
    IdentityCalibrator,
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
    build_calibrator,
)
from infrastructure.evaluation.calibration_metrics import (
    compute_ece,
    evaluate_calibration_metrics,
    select_optimal_f1_threshold,
)
from infrastructure.persistence.calibration_result_store import CalibrationResultStore


class TestCalibratorFittingAndBounds:
    """Test mathematical properties, bounds, and fitting of calibrators."""

    @pytest.fixture
    def synthetic_imbalanced_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic imbalanced validation data (0.1% positives)."""
        rng = np.random.default_rng(42)
        n = 10_000
        y = np.zeros(n, dtype=np.float32)
        y[:10] = 1.0  # 10 positives in 10,000 samples
        rng.shuffle(y)

        # Uncalibrated probabilities with severe under/over confidence
        logits = rng.normal(-4.0, 1.5, size=n)
        p = 1.0 / (1.0 + np.exp(-logits))
        return y, p.astype(np.float32)

    def test_identity_calibrator_bounds(
        self, synthetic_imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify IdentityCalibrator leaves probabilities completely unchanged."""
        y, p = synthetic_imbalanced_data
        cal = IdentityCalibrator()
        cal.fit(y, p)
        cal_p = cal.calibrate(p)
        assert cal.method == CalibrationMethod.NONE
        np.testing.assert_allclose(cal_p, p)
        assert np.all(np.isfinite(cal_p))

    def test_temperature_scaling_bounds_and_finite(
        self, synthetic_imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify TemperatureScalingCalibrator produces valid probabilities in [0, 1]."""
        y, p = synthetic_imbalanced_data
        cal = TemperatureScalingCalibrator()
        cal.fit(y, p)
        assert cal.temperature > 0.0
        cal_p = cal.calibrate(p)
        assert np.all(np.isfinite(cal_p))
        assert np.all(cal_p >= 0.0)
        assert np.all(cal_p <= 1.0)
        # Check monotonicity: relative rankings preserved
        order_p = np.argsort(p)
        order_cal = np.argsort(cal_p)
        np.testing.assert_array_equal(order_p, order_cal)

    def test_platt_scaling_bounds_and_monotonicity(
        self, synthetic_imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify PlattScalingCalibrator keeps slope positive and bounds in [0, 1]."""
        y, p = synthetic_imbalanced_data
        cal = PlattScalingCalibrator()
        cal.fit(y, p)
        assert cal.a > 0.0  # Monotonicity preserved
        cal_p = cal.calibrate(p)
        assert np.all(np.isfinite(cal_p))
        assert np.all(cal_p >= 0.0)
        assert np.all(cal_p <= 1.0)
        # Check monotonic rank preservation
        import scipy.stats
        rho, _ = scipy.stats.spearmanr(p, cal_p)
        assert rho > 0.9999

    def test_isotonic_regression_bounds_and_fitting(
        self, synthetic_imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify IsotonicRegressionCalibrator outputs valid monotonic step probabilities."""
        y, p = synthetic_imbalanced_data
        cal = IsotonicRegressionCalibrator()
        cal.fit(y, p)
        cal_p = cal.calibrate(p)
        assert np.all(np.isfinite(cal_p))
        assert np.all(cal_p >= 0.0)
        assert np.all(cal_p <= 1.0)
        params = cal.get_parameters()
        assert params.isotonic_thresholds_count is not None
        assert params.isotonic_thresholds_count > 0

    def test_extreme_probabilities_no_overflow(self) -> None:
        """Verify extreme probability inputs near 0 and 1 do not produce NaN or Inf."""
        p_extreme = np.array([0.0, 1e-15, 0.5, 1.0 - 1e-15, 1.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

        for method in (
            CalibrationMethod.NONE,
            CalibrationMethod.TEMPERATURE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ):
            cal = build_calibrator(method)
            cal.fit(y, p_extreme)
            cal_p = cal.calibrate(p_extreme)
            assert np.all(np.isfinite(cal_p))
            assert np.all(cal_p >= 0.0)
            assert np.all(cal_p <= 1.0)


class TestCalibrationMetricsCalculation:
    """Test ECE, Log loss, Brier score, and threshold optimization."""

    def test_perfect_calibration_ece_near_zero(self) -> None:
        """Verify that perfectly calibrated predictions have ECE close to 0."""
        n = 10_000
        rng = np.random.default_rng(42)
        p = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        y = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float32)

        ece, mce, data = compute_ece(y, p, n_bins=10, strategy="uniform")
        assert ece < 0.05
        assert mce < 0.10
        assert len(data.bin_centers) > 0

    def test_log_loss_and_brier_score_positive(self) -> None:
        """Verify log loss and Brier score are non-negative and finite."""
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        p = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)

        metrics = evaluate_calibration_metrics(y, p, threshold=0.5, historical_threshold=0.13)
        assert metrics.log_loss > 0.0
        assert metrics.brier_score > 0.0
        assert metrics.ece >= 0.0
        assert metrics.prevalence == 0.5
        assert metrics.positive_count == 2
        assert metrics.total_count == 4

    def test_select_optimal_f1_threshold(self) -> None:
        """Verify that select_optimal_f1_threshold picks the best F1 cut."""
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        p = np.array([0.05, 0.10, 0.35, 0.80, 0.15], dtype=np.float32)

        thresh = select_optimal_f1_threshold(y, p)
        # Choosing threshold near 0.35 gives TP=2, FP=0 -> F1=1.0
        assert 0.15 < thresh <= 0.35


class TestResultStoreArtifactIsolation:
    """Test isolated directory creation and saving without affecting other phases."""

    def test_save_and_read_metadata_isolation(self, tmp_path: Path) -> None:
        """Verify CalibrationResultStore saves in dedicated root without modifying existing dirs."""
        store = CalibrationResultStore(root=tmp_path)
        exp_id = "test_calibration_exp_001"
        target_dir = store.init_directory_structure(exp_id)

        assert (target_dir / "validation").is_dir()
        assert (target_dir / "test").is_dir()
        assert (target_dir / "visualizations").is_dir()

        dummy_metrics = CalibrationEvaluationMetrics(
            log_loss=0.01,
            brier_score=0.001,
            ece=0.005,
            mce=0.01,
            pr_auc=0.5,
            roc_auc=0.9,
            threshold=0.15,
            f1=0.6,
            precision=0.7,
            recall=0.5,
            accuracy=0.99,
            tp=10,
            fp=4,
            tn=980,
            fn=6,
            historical_threshold=0.13,
            historical_f1=0.58,
            historical_precision=0.65,
            historical_recall=0.52,
            historical_accuracy=0.98,
            historical_tp=11,
            historical_fp=6,
            historical_tn=978,
            historical_fn=5,
            positive_count=16,
            total_count=1000,
            prevalence=0.016,
        )

        res_val = CalibrationMethodResult(
            method=CalibrationMethod.TEMPERATURE,
            split="validation",
            parameters=CalibrationParameters(method=CalibrationMethod.TEMPERATURE, temperature=1.2),
            metrics=dummy_metrics,
            selected=True,
        )

        val_csv = store.save_validation_results(exp_id, [res_val])
        assert val_csv.exists()

        test_base = CalibrationMethodResult(
            method=CalibrationMethod.NONE,
            split="test",
            parameters=CalibrationParameters(method=CalibrationMethod.NONE),
            metrics=dummy_metrics,
            selected=False,
        )
        test_cal = CalibrationMethodResult(
            method=CalibrationMethod.TEMPERATURE,
            split="test",
            parameters=CalibrationParameters(method=CalibrationMethod.TEMPERATURE, temperature=1.2),
            metrics=dummy_metrics,
            selected=True,
        )

        test_csv = store.save_test_results(exp_id, test_base, test_cal)
        assert test_csv.exists()

        report = CalibrationExperimentReport(
            experiment_id=exp_id,
            timestamp="20261013T120000",
            source_model={"model": "lstm", "algorithm": "fedavg"},
            config=CalibrationConfig(),
            validation_results=[res_val],
            selected_method=CalibrationMethod.TEMPERATURE,
            selected_parameters=res_val.parameters,
            frozen_threshold=0.15,
            test_baseline=test_base,
            test_calibrated=test_cal,
        )

        rep_json, meta_json = store.save_report_and_metadata(report)
        assert rep_json.exists()
        assert meta_json.exists()

        rep_md = store.save_markdown_report(report)
        assert rep_md.exists()
        content = rep_md.read_text(encoding="utf-8")
        assert "Phase 10: Probability Calibration Report" in content
        assert "temperature_scaling" in content
