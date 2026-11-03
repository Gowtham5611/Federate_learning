"""Unit tests for Phase 8 Performance Analysis & Ablation Study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from application.dto.analysis import (
    AblationRecord,
    AlphaAnalysisRecord,
    CentralizedVsFederatedRecord,
    ClientDistributionRecord,
    ConvergenceMetricRecord,
    FedAvgVsFedProxRecord,
    IIDVsNonIIDRecord,
    ModelComparisonRecord,
    Phase8Report,
)
from application.services.analysis_service import AnalysisService
from infrastructure.persistence.analysis_result_store import AnalysisResultStore
from infrastructure.visualization.analysis_visualizer import AnalysisVisualizer

# Force non-interactive Agg backend for test suite
matplotlib.use("Agg")


class TestAnalysisDTOs:
    """Test Phase 8 analysis DTO creation and serialization."""

    def test_centralized_vs_federated_record(self) -> None:
        """Verify CentralizedVsFederatedRecord field assignment and integrity."""
        rec = CentralizedVsFederatedRecord(
            paradigm="Centralized",
            model="bilstm",
            algorithm="centralized",
            experiment_id="centralized_bilstm_seed42_20260906T011749",
            split="validation",
            threshold_policy="fixed",
            threshold=0.5,
            pr_auc=0.4897,
            roc_auc=0.9850,
            f1=0.6402,
            recall=0.5501,
            precision=0.7654,
            accuracy=0.9995,
        )
        assert rec.paradigm == "Centralized"
        assert rec.model == "bilstm"
        assert rec.pr_auc == 0.4897

    def test_ablation_record(self) -> None:
        """Verify AblationRecord fields and confusion matrix counts."""
        rec = AblationRecord(
            stage="Stage A",
            description="Centralized baseline",
            experiment_id="centralized_bilstm_seed42",
            val_pr_auc=0.4897,
            test_pr_auc=0.4901,
            test_roc_auc=0.9850,
            test_f1=0.6402,
            test_recall=0.5501,
            test_precision=0.7654,
            test_accuracy=0.9995,
            tp=988,
            fp=303,
            tn=761195,
            fn=809,
            delta_test_pr_auc_vs_baseline=0.0,
        )
        assert rec.stage == "Stage A"
        assert rec.tp == 988

    def test_phase8_report_serialization(self) -> None:
        """Verify Phase8Report serialization to dictionary and JSON."""
        report = Phase8Report(
            analysis_id="test_run",
            timestamp="2026-10-03T21:00:00Z",
            seed=42,
            total_experiments_analyzed=10,
            centralized_records_count=3,
            iid_records_count=2,
            non_iid_records_count=3,
            ensemble_records_count=1,
            centralized_vs_federated=[],
            model_comparison=[],
            fedavg_vs_fedprox=[],
            iid_vs_non_iid=[],
            client_distribution=[],
            convergence_metrics=[],
            alpha_analysis=[],
            threshold_analysis=[],
            ablation_study=[],
            frozen_ensemble_metrics={
                "pr_auc": 0.3846,
                "roc_auc": 0.9789,
                "f1": 0.5718,
                "recall": 0.4730,
                "precision": 0.7234,
                "accuracy": 0.9994,
                "threshold": 0.13,
                "tp": 850,
                "fp": 325,
                "tn": 761173,
                "fn": 947,
            },
            key_findings=["Finding 1", "Finding 2"],
            limitations=["Limitation 1"],
            artifacts_generated=["plot1.png"],
        )
        data = report.to_dict()
        assert data["analysis_id"] == "test_run"
        assert data["seed"] == 42
        assert data["frozen_ensemble_metrics"]["tp"] == 850


class TestAnalysisResultStore:
    """Test directory tree creation and persistence of analysis artifacts."""

    def test_directory_structure_creation(self, tmp_path: Path) -> None:
        """Verify that all required subdirectories are created."""
        store = AnalysisResultStore(root=tmp_path)
        target = store.init_directory_structure("run_001")

        expected_dirs = [
            "summary",
            "comparisons",
            "ablations",
            "client_analysis",
            "convergence",
            "ensemble_analysis",
            "threshold_analysis",
            "visualizations",
        ]
        for name in expected_dirs:
            assert (target / name).is_dir()

    def test_save_csvs_and_markdown(self, tmp_path: Path) -> None:
        """Verify saving CSV files and rendering Markdown report."""
        store = AnalysisResultStore(root=tmp_path)
        analysis_id = "test_artifacts"
        store.init_directory_structure(analysis_id)

        report = Phase8Report(
            analysis_id=analysis_id,
            timestamp="2026-10-03T21:00:00Z",
            seed=42,
            total_experiments_analyzed=10,
            centralized_records_count=3,
            iid_records_count=2,
            non_iid_records_count=3,
            ensemble_records_count=1,
            centralized_vs_federated=[
                CentralizedVsFederatedRecord(
                    paradigm="Centralized",
                    model="bilstm",
                    algorithm="centralized",
                    experiment_id="cent_bilstm",
                    split="validation",
                    threshold_policy="fixed",
                    threshold=0.5,
                    pr_auc=0.4897,
                    roc_auc=0.9850,
                    f1=0.6402,
                    recall=0.5501,
                    precision=0.7654,
                    accuracy=0.9995,
                )
            ],
            model_comparison=[],
            fedavg_vs_fedprox=[],
            iid_vs_non_iid=[],
            client_distribution=[],
            convergence_metrics=[],
            alpha_analysis=[],
            threshold_analysis=[],
            ablation_study=[],
            frozen_ensemble_metrics={
                "pr_auc": 0.3846,
                "roc_auc": 0.9789,
                "f1": 0.5718,
                "recall": 0.4730,
                "precision": 0.7234,
                "accuracy": 0.9994,
                "threshold": 0.13,
                "tp": 850,
                "fp": 325,
                "tn": 761173,
                "fn": 947,
            },
            key_findings=["Finding 1", "Finding 2"],
            limitations=["Limitation 1"],
            artifacts_generated=["plot1.png"],
        )

        csv_paths = store.save_all_csvs(report)
        assert len(csv_paths) == 10
        for p in csv_paths.values():
            assert p.is_file()

        rep_path, meta_path, md_path = store.save_json_and_markdown(report)
        assert rep_path.is_file()
        assert meta_path.is_file()
        assert md_path.is_file()

        with md_path.open(encoding="utf-8") as f:
            md_text = f.read()
            assert "# Phase 8" in md_text
            assert "Finding 1" in md_text


class TestAnalysisVisualizer:
    """Test publication figure generation."""

    def test_visualizer_generates_all_plots(self, tmp_path: Path) -> None:
        """Verify that all 10 publication figures are generated and non-empty."""
        visualizer = AnalysisVisualizer(output_dir=tmp_path)

        models = [
            ModelComparisonRecord(
                paradigm="Centralized",
                algorithm="centralized",
                model="bilstm",
                split="validation",
                threshold_policy="fixed",
                pr_auc=0.4897,
                roc_auc=0.9850,
                f1=0.6402,
                recall=0.5501,
                precision=0.7654,
                accuracy=0.9995,
                experiment_id="cent_bilstm",
            ),
            ModelComparisonRecord(
                paradigm="IID Federated",
                algorithm="fedavg",
                model="lstm",
                split="validation",
                threshold_policy="fixed",
                pr_auc=0.3675,
                roc_auc=0.9789,
                f1=0.5718,
                recall=0.4730,
                precision=0.7234,
                accuracy=0.9994,
                experiment_id="iid_lstm",
            ),
        ]
        fed_comp = [
            FedAvgVsFedProxRecord(
                distribution="iid",
                model="gru",
                fedavg_exp_id="iid_gru",
                fedprox_exp_id="prox_gru",
                fedavg_val_pr_auc=0.3341,
                fedprox_val_pr_auc=0.0018,
                val_pr_auc_delta=-0.3323,
                fedavg_test_pr_auc=0.3477,
                fedprox_test_pr_auc=0.0027,
                fedavg_test_f1=0.5401,
                fedprox_test_f1=0.0003,
                notes="FedProx collapsed",
            )
        ]
        iid_non_iid = [
            IIDVsNonIIDRecord(
                model="lstm",
                algorithm="fedavg",
                iid_exp_id="iid_lstm",
                non_iid_exp_id="non_iid_lstm",
                iid_val_pr_auc=0.3675,
                non_iid_val_pr_auc=0.0657,
                val_pr_auc_degradation=-0.3018,
                iid_test_pr_auc=0.3846,
                non_iid_test_pr_auc=0.0620,
                iid_test_f1=0.5718,
                non_iid_test_f1=0.0805,
                notes="82% drop",
            )
        ]
        clients = [
            ClientDistributionRecord(
                client_id="client_1",
                typology="Retail POS High-Volume",
                transaction_count=539973,
                percentage_of_total=17.72,
                positive_count=359,
                negative_count=539614,
                positive_rate=0.000665,
                account_count=44345,
            ),
            ClientDistributionRecord(
                client_id="client_2",
                typology="Crypto/Wire/FX Cross-Border",
                transaction_count=981684,
                percentage_of_total=32.22,
                positive_count=965,
                negative_count=980719,
                positive_rate=0.000983,
                account_count=49299,
            ),
        ]
        conv = [
            ConvergenceMetricRecord(
                experiment_id="iid_lstm",
                distribution="iid",
                algorithm="fedavg",
                model="lstm",
                round=1,
                val_loss=0.005,
                val_pr_auc=0.20,
                val_roc_auc=0.85,
                val_f1=0.30,
                val_recall=0.25,
                val_precision=0.40,
            ),
            ConvergenceMetricRecord(
                experiment_id="iid_lstm",
                distribution="iid",
                algorithm="fedavg",
                model="lstm",
                round=5,
                val_loss=0.002,
                val_pr_auc=0.3675,
                val_roc_auc=0.9789,
                val_f1=0.5718,
                val_recall=0.4730,
                val_precision=0.7234,
            ),
        ]
        alphas = [
            AlphaAnalysisRecord(
                alpha=0.0,
                iid_weight=0.0,
                non_iid_weight=1.0,
                val_pr_auc=0.0955,
                val_roc_auc=0.9200,
                val_f1=0.1500,
                val_recall=0.2000,
                val_precision=0.1200,
                val_accuracy=0.9990,
                is_selected=False,
            ),
            AlphaAnalysisRecord(
                alpha=1.0,
                iid_weight=1.0,
                non_iid_weight=0.0,
                val_pr_auc=0.3675,
                val_roc_auc=0.9789,
                val_f1=0.5718,
                val_recall=0.4730,
                val_precision=0.7234,
                val_accuracy=0.9994,
                is_selected=True,
            ),
        ]
        frozen_metrics = {
            "tp": 850,
            "fp": 325,
            "tn": 761173,
            "fn": 947,
            "pr_auc": 0.3846,
            "roc_auc": 0.9789,
            "f1": 0.5718,
            "recall": 0.4730,
            "precision": 0.7234,
            "accuracy": 0.9994,
        }

        # Create dummy predictions CSV
        preds_file = tmp_path / "test_preds.csv"
        with preds_file.open("w", newline="", encoding="utf-8") as f:
            f.write("true_label,p_iid,p_non_iid,p_ensemble\n")
            for _ in range(50):
                f.write("0,0.01,0.02,0.01\n")
            for _ in range(10):
                f.write("1,0.85,0.70,0.85\n")

        paths = visualizer.generate_all(
            model_comparisons=models,
            fedavg_vs_fedprox=fed_comp,
            iid_vs_non_iid=iid_non_iid,
            client_distributions=clients,
            convergence_metrics=conv,
            alpha_records=alphas,
            test_predictions_path=preds_file,
            frozen_metrics=frozen_metrics,
        )

        assert len(paths) == 10
        for p in paths:
            fpath = Path(p)
            assert fpath.is_file(), f"Expected plot file {fpath} does not exist"
            assert fpath.stat().st_size > 0, f"Plot file {fpath} is empty"


class TestAnalysisServiceIntegration:
    """Test AnalysisService data loading and safe metric parsing on real workspace artifacts."""

    def test_analysis_service_loads_historical_artifacts(self, tmp_path: Path) -> None:
        """Verify loading historical Phase 4-7 CSV and JSON artifacts."""
        store = AnalysisResultStore(root=tmp_path)
        service = AnalysisService(
            results_root=Path("experiments/results"),
            analysis_store=store,
        )
        cent_df = service._load_centralized_results()
        iid_avg_df = service._load_iid_fedavg_results()
        iid_prox_df = service._load_iid_fedprox_results()
        non_iid_df, _ = service._load_non_iid_results()
        ens_df, ens_report, _ens_alpha_df, _ens_test_preds = service._load_ensemble_results()

        assert not cent_df.empty
        assert len(cent_df) == 24
        assert not iid_avg_df.empty
        assert len(iid_avg_df) == 21
        assert not iid_prox_df.empty
        assert len(iid_prox_df) == 13
        assert not non_iid_df.empty
        assert len(non_iid_df) >= 6
        assert not ens_df.empty
        assert ens_report.get("best_alpha") == 1.0

    def test_analysis_service_build_comparisons(self, tmp_path: Path) -> None:
        """Verify comparison and ablation records construction."""
        store = AnalysisResultStore(root=tmp_path)
        service = AnalysisService(
            results_root=Path("experiments/results"),
            analysis_store=store,
        )
        cent_df = service._load_centralized_results()
        iid_avg_df = service._load_iid_fedavg_results()
        iid_prox_df = service._load_iid_fedprox_results()
        non_iid_df, _ = service._load_non_iid_results()
        ens_df, _, _, _ = service._load_ensemble_results()

        cvf = service._build_centralized_vs_federated(
            cent_df, iid_avg_df, iid_prox_df, non_iid_df, ens_df
        )
        assert len(cvf) > 0
        paradigms = {r.paradigm for r in cvf}
        assert "Centralized" in paradigms
        assert "IID Federated" in paradigms
        assert "Non-IID Federated" in paradigms
        assert "Ensemble" in paradigms

        ablations = service._build_ablation_study(
            cent_df, iid_avg_df, non_iid_df, ens_df
        )
        assert len(ablations) == 6
        stages = [a.stage for a in ablations]
        assert stages == ["Stage A", "Stage B", "Stage C", "Stage D", "Stage E", "Stage F"]
