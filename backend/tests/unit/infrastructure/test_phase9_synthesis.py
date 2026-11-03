"""Unit test for Phase 9 result synthesis, tables, metadata, and visualizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from application.services.phase9_optimization_service import Phase9OptimizationService


def test_phase9_synthesis_generates_all_artifacts() -> None:
    """Verify that _synthesize_artifacts generates all 7 CSVs, 10 PNGs, metadata, and report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        svc = Phase9OptimizationService(
            results_root=tmp_path / "results",
            checkpoints_root=tmp_path / "checkpoints",
        )

        mock_fedprox = []
        for arch in ["gru", "lstm", "bilstm"]:
            for mu in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
                mock_fedprox.append({
                    "experiment_id": f"fedprox_{arch}_{mu}",
                    "architecture": arch,
                    "algorithm": "fedprox",
                    "mu": mu,
                    "seed": 42,
                    "selected_round": 5,
                    "frozen_threshold": 0.5,
                    "val_pr_auc": 0.15 - mu * 0.5 if mu < 0.01 else 0.002,
                    "val_roc_auc": 0.85,
                    "val_f1": 0.18,
                    "val_recall": 0.20,
                    "val_precision": 0.16,
                    "val_loss": 0.02,
                    "test_pr_auc": 0.20 - mu * 0.5 if mu < 0.01 else 0.002,
                    "test_roc_auc": 0.90,
                    "test_f1": 0.22,
                    "test_recall": 0.25,
                    "test_precision": 0.20,
                    "test_accuracy": 0.998,
                    "tp": 200,
                    "tn": 1000000,
                    "fp": 800,
                    "fn": 1500,
                    "rounds": [
                        {
                            "round": r,
                            "validation_loss": 0.03 - 0.002 * r,
                            "validation_metrics": {
                                "pr_auc": 0.10 + 0.01 * r,
                                "roc_auc": 0.80,
                                "f1": 0.15,
                            },
                            "drift_metrics": {
                                "mean_drift": 0.05 + 0.01 * r,
                                "weighted_drift": 0.048 + 0.01 * r,
                                "pairwise_disagreement": 0.07 + 0.02 * r,
                                "parameter_variance": 0.002 + 0.0005 * r,
                            },
                        }
                        for r in range(1, 6)
                    ],
                })

        mock_scaffold = []
        for arch in ["gru", "lstm", "bilstm"]:
            mock_scaffold.append({
                "experiment_id": f"scaffold_{arch}",
                "architecture": arch,
                "algorithm": "scaffold",
                "mu": None,
                "seed": 42,
                "selected_round": 5,
                "frozen_threshold": 0.5,
                "val_pr_auc": 0.14,
                "val_roc_auc": 0.86,
                "val_f1": 0.19,
                "val_recall": 0.21,
                "val_precision": 0.17,
                "val_loss": 0.022,
                "test_pr_auc": 0.21,
                "test_roc_auc": 0.91,
                "test_f1": 0.23,
                "test_recall": 0.26,
                "test_precision": 0.21,
                "test_accuracy": 0.998,
                "tp": 210,
                "tn": 1000000,
                "fp": 790,
                "fn": 1490,
                "rounds": [
                    {
                        "round": r,
                        "validation_loss": 0.032 - 0.002 * r,
                        "validation_metrics": {
                            "pr_auc": 0.11 + 0.01 * r,
                            "roc_auc": 0.81,
                            "f1": 0.16,
                        },
                        "drift_metrics": {
                            "mean_drift": 0.02 + 0.005 * r,
                            "weighted_drift": 0.019 + 0.004 * r,
                            "pairwise_disagreement": 0.03 + 0.01 * r,
                            "parameter_variance": 0.001 + 0.0002 * r,
                        },
                    }
                    for r in range(1, 6)
                ],
            })

        mock_baselines = {
            f"fedavg_{arch}": {
                "architecture": arch,
                "algorithm": "fedavg",
                "val_pr_auc": 0.16,
                "test_pr_auc": 0.23,
                "test_roc_auc": 0.92,
                "test_f1": 0.24,
                "test_recall": 0.27,
                "test_precision": 0.22,
                "test_accuracy": 0.998,
                "tp": 220,
                "tn": 1000000,
                "fp": 780,
                "fn": 1480,
                "rounds": [
                    {
                        "round": r,
                        "validation_loss": 0.028 - 0.002 * r,
                        "validation_metrics": {
                            "pr_auc": 0.12 + 0.01 * r,
                            "roc_auc": 0.82,
                            "f1": 0.17,
                        },
                    }
                    for r in range(1, 6)
                ],
            }
            for arch in ["gru", "lstm", "bilstm"]
        }

        synthesis = svc._synthesize_artifacts(
            fedprox_summaries=mock_fedprox,
            scaffold_summaries=mock_scaffold,
            baselines=mock_baselines,
            hybrid_summary=None,
            elapsed_seconds=120.5,
        )
        assert synthesis["metadata_path"].is_file()

        # Check all CSV tables exist
        res_dir = tmp_path / "results" / "phase9_federated_optimization"
        assert (res_dir / "fedprox" / "mu_ablation" / "fedprox_mu_ablation.csv").is_file()
        assert (res_dir / "fedprox" / "best_model" / "best_fedprox_summary.csv").is_file()
        assert (res_dir / "scaffold" / "reports" / "scaffold_results.csv").is_file()
        assert (res_dir / "comparisons" / "fedavg_vs_fedprox.csv").is_file()
        assert (res_dir / "comparisons" / "three_way_comparison.csv").is_file()
        assert (res_dir / "client_drift" / "client_drift_analysis.csv").is_file()
        assert (res_dir / "convergence" / "convergence_trajectory.csv").is_file()

        # Check all 10 plots exist
        plots = list((res_dir / "visualizations").glob("*.png"))
        assert len(plots) == 10

        # Check metadata.json and report.md
        meta_file = res_dir / "metadata.json"
        report_file = res_dir / "report.md"
        assert meta_file.is_file()
        assert report_file.is_file()
        assert "PHASE 9 COMPLETION REPORT" in report_file.read_text(encoding="utf-8")
        assert "Q1:" in report_file.read_text(encoding="utf-8")
        assert "Q9:" in report_file.read_text(encoding="utf-8")
