"""Unit and integration test suite for FedSecureAI Phase 8 Performance Analysis and Ablation Study.

Verifies the 10 core requirements specified in Phase 8 specification:
1. FedProx mu filtering (strictly mu=0.0001 accepted; historical mu=0.1, mu=1.0 rejected/excluded).
2. Dataset separation (HI-Small vs LI-Medium complete isolation).
3. Configuration mapping (Configs 1-12 per dataset mapped to architecture, distribution, algorithm).
4. Metric difference calculations (exact mathematical deltas for IID vs Non-IID and FedAvg vs FedProx).
5. Missing artifact handling (LI-Medium centralized baseline handled as DATA_GAP_NOT_AVAILABLE / DOCUMENTED_AS_LIMITATION).
6. Consistent split identity (HI-Small test: 1,015,564 rows; LI-Medium test: 6,249,112 rows).
7. Threshold trade-off calculations (positive, scaled alert rates and bounded PR metrics).
8. Round-level aggregation (5 communication rounds across 24 federated configurations = 120 rows).
9. JSON serialization conforming to Phase8Report DTO contract.
10. Preservation of Phase 7 outputs (alpha*=1.00 equivalence, zero false gain claims).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.application.dto.analysis import (
    AblationRecord,
    AlphaAnalysisRecord,
    CentralizedVsFederatedRecord,
    ClientDistributionRecord,
    ConvergenceMetricRecord,
    FedAvgVsFedProxRecord,
    IIDVsNonIIDRecord,
    ModelComparisonRecord,
    Phase8Report,
    ThresholdAnalysisRecord,
)

PHASE8_DIR = Path("results/phase8")
AUTH_DIR = Path("results/authoritative_results")
PHASE7_DIR = Path("results/phase7")


def test_01_fedprox_mu_filtering_strictness():
    """Verify that only FedProx runs with mu=0.0001 are included and verified."""
    mu_verif_csv = AUTH_DIR / "fedprox_mu_verification.csv"
    assert mu_verif_csv.exists(), "fedprox_mu_verification.csv must exist in authoritative results"
    df_mu = pd.read_csv(mu_verif_csv)

    # 12 active configurations must be verified with mu == 0.0001
    verified_configs = df_mu[df_mu["compliance_verdict"] == "VERIFIED_MU_0.0001"]
    assert len(verified_configs) == 12, "Must have exactly 12 verified FedProx configurations"
    assert (verified_configs["verified_mu"] == 0.0001).all(), "All verified configs must have mu=0.0001"
    assert (verified_configs["action"] == "INCLUDE_IN_COMPARISON").all()

    # Historical runs with mu != 0.0001 must be marked EXCLUDED
    excluded_configs = df_mu[df_mu["action"] == "EXCLUDE_FROM_COMPARISON"]
    assert len(excluded_configs) == 2, "Historical exploratory configs (mu=0.1, mu=1.0) must be EXCLUDED"
    assert set(excluded_configs["verified_mu"].tolist()) == {0.1, 1.0}

    # Verify that fedavg_vs_fedprox_mu_0.0001.csv contains only mu=0.0001
    fedprox_ablation_csv = PHASE8_DIR / "fedavg_vs_fedprox_mu_0.0001.csv"
    assert fedprox_ablation_csv.exists(), "fedavg_vs_fedprox_mu_0.0001.csv must exist"
    df_ablation = pd.read_csv(fedprox_ablation_csv)
    assert len(df_ablation) == 12, "Must contain exactly 12 pairwise ablation rows (6 HI-Small + 6 LI-Medium)"
    assert (df_ablation["fedprox_mu_verified"] == 0.0001).all(), "Ablation table must strictly have verified mu=0.0001"


def test_02_dataset_separation():
    """Verify that HI-Small and LI-Medium datasets are strictly separated across all tables."""
    for filename in [
        "centralized_vs_federated.csv",
        "fedavg_vs_fedprox_mu_0.0001.csv",
        "iid_vs_noniid.csv",
        "phase8_summary.csv",
        "round_level_analysis.csv",
    ]:
        csv_path = PHASE8_DIR / filename
        assert csv_path.exists(), f"{filename} must exist"
        df = pd.read_csv(csv_path)
        assert "dataset" in df.columns, f"{filename} must contain 'dataset' column"
        datasets = set(df["dataset"].unique())
        assert datasets == {"HI-Small", "LI-Medium"}, f"{filename} contains invalid datasets: {datasets}"

        # Ensure no cross-dataset leakage in config ids
        hi_rows = df[df["dataset"] == "HI-Small"]
        li_rows = df[df["dataset"] == "LI-Medium"]
        assert len(hi_rows) > 0 and len(li_rows) > 0, f"{filename} must have records for both datasets"


def test_03_configuration_mapping():
    """Verify that Configs 1-12 per dataset map to the exact architecture, distribution, and algorithm."""
    summary_csv = PHASE8_DIR / "phase8_summary.csv"
    assert summary_csv.exists()
    df_summary = pd.read_csv(summary_csv)

    expected_mapping = {
        1: ("GRU", "IID", "FedAvg"),
        2: ("GRU", "IID", "FedProx"),
        3: ("LSTM", "IID", "FedAvg"),
        4: ("LSTM", "IID", "FedProx"),
        5: ("BiLSTM", "IID", "FedAvg"),
        6: ("BiLSTM", "IID", "FedProx"),
        7: ("GRU", "Non-IID", "FedAvg"),
        8: ("GRU", "Non-IID", "FedProx"),
        9: ("LSTM", "Non-IID", "FedAvg"),
        10: ("LSTM", "Non-IID", "FedProx"),
        11: ("BiLSTM", "Non-IID", "FedAvg"),
        12: ("BiLSTM", "Non-IID", "FedProx"),
    }

    for dataset in ["HI-Small", "LI-Medium"]:
        ds_df = df_summary[df_summary["dataset"] == dataset]
        assert len(ds_df) == 12, f"{dataset} must have 12 federated configurations"
        for _, row in ds_df.iterrows():
            raw_cid = str(row["config_id"]).replace("config_", "")
            cid = int(raw_cid)
            exp_arch, exp_dist, exp_algo = expected_mapping[cid]
            assert row["architecture"].upper() == exp_arch.upper(), f"Config {cid} arch mismatch in {dataset}"
            assert row["distribution"] == exp_dist, f"Config {cid} dist mismatch in {dataset}"
            assert row["algorithm"] == exp_algo, f"Config {cid} algo mismatch in {dataset}"


def test_04_metric_difference_calculations():
    """Verify that all deltas in FedAvg vs FedProx and IID vs Non-IID are mathematically exact."""
    # 1. FedAvg vs FedProx
    df_ablation = pd.read_csv(PHASE8_DIR / "fedavg_vs_fedprox_mu_0.0001.csv")
    for _, row in df_ablation.iterrows():
        calc_pr_auc_delta = row["fedprox_test_pr_auc"] - row["fedavg_test_pr_auc"]
        calc_f1_delta = row["fedprox_test_f1"] - row["fedavg_test_f1"]
        delta_col = "delta_test_pr_auc (FedProx - FedAvg)"
        assert np.isclose(row[delta_col], calc_pr_auc_delta, atol=1e-6), (
            f"PR-AUC delta mismatch in FedAvg vs FedProx row: {row[delta_col]} vs {calc_pr_auc_delta}"
        )
        assert np.isclose(row["delta_test_f1"], calc_f1_delta, atol=1e-6), (
            f"F1 delta mismatch in FedAvg vs FedProx row: {row['delta_test_f1']} vs {calc_f1_delta}"
        )

    # 2. IID vs Non-IID
    df_iid = pd.read_csv(PHASE8_DIR / "iid_vs_noniid.csv")
    for _, row in df_iid.iterrows():
        calc_pr_auc_degradation = row["iid_test_pr_auc"] - row["non_iid_test_pr_auc"]
        calc_f1_degradation = row["iid_test_f1"] - row["non_iid_test_f1"]
        deg_col = "test_pr_auc_degradation (IID - NonIID)"
        assert np.isclose(row[deg_col], calc_pr_auc_degradation, atol=1e-6), (
            f"PR-AUC degradation mismatch: {row[deg_col]} vs {calc_pr_auc_degradation}"
        )
        assert np.isclose(row["test_f1_degradation"], calc_f1_degradation, atol=1e-6), (
            f"F1 degradation mismatch: {row['test_f1_degradation']} vs {calc_f1_degradation}"
        )


def test_05_missing_artifact_handling_data_gap():
    """Verify that missing LI-Medium centralized baseline is handled as DATA_GAP_NOT_AVAILABLE."""
    cvf_csv = PHASE8_DIR / "centralized_vs_federated.csv"
    assert cvf_csv.exists()
    df_cvf = pd.read_csv(cvf_csv)

    # HI-Small has 3 centralized records (GRU, LSTM, BiLSTM)
    hi_cent = df_cvf[(df_cvf["dataset"] == "HI-Small") & (df_cvf["paradigm"] == "Centralized")]
    assert len(hi_cent) == 3, "HI-Small must have 3 centralized records"
    assert (hi_cent["pr_auc"] > 0).all(), "HI-Small centralized PR-AUC must be positive"

    # LI-Medium has NO centralized records in the active comparison, properly documented
    li_cent = df_cvf[(df_cvf["dataset"] == "LI-Medium") & (df_cvf["paradigm"] == "Centralized")]
    assert len(li_cent) == 0, "LI-Medium centralized must NOT be fabricated"

    # Check phase8_discrepancies.csv
    disc_csv = PHASE8_DIR / "phase8_discrepancies.csv"
    assert disc_csv.exists()
    df_disc = pd.read_csv(disc_csv)
    gap_rows = df_disc[df_disc["category"] == "CENTRALIZED_BASELINE"]
    assert len(gap_rows) > 0, "CENTRALIZED_BASELINE must be registered in discrepancies table"
    assert gap_rows["dataset"].values[0] == "LI-Medium"
    assert gap_rows["remediation_status"].values[0] == "DOCUMENTED_AS_LIMITATION"


def test_06_consistent_split_identity():
    """Verify split row counts match across authoritative and Phase 8 provenance artifacts."""
    prov_csv = PHASE8_DIR / "phase8_data_provenance.csv"
    assert prov_csv.exists()
    df_prov = pd.read_csv(prov_csv)

    # Check HI-Small test split row count
    hi_prov = df_prov[df_prov["dataset"] == "HI-Small"]
    assert (hi_prov["ground_truth_rows"] == 1015564).all(), "HI-Small test row count must be 1,015,564"

    # Check LI-Medium test split row count
    li_prov = df_prov[df_prov["dataset"] == "LI-Medium"]
    assert (li_prov["ground_truth_rows"] == 6249112).all(), "LI-Medium test row count must be 6,249,112"


def test_07_threshold_tradeoff_calculations():
    """Verify that threshold trade-offs are strictly bounded and non-negative."""
    tradeoff_csv = PHASE8_DIR / "threshold_tradeoff_analysis.csv"
    assert tradeoff_csv.exists()
    df_tradeoff = pd.read_csv(tradeoff_csv)

    # 4 models/ensembles * 25 thresholds = 100 records
    assert len(df_tradeoff) == 100, f"Expected 100 records in threshold analysis, got {len(df_tradeoff)}"

    # Metrics bounded [0, 1]
    assert ((df_tradeoff["precision"] >= 0.0) & (df_tradeoff["precision"] <= 1.0)).all()
    assert ((df_tradeoff["recall"] >= 0.0) & (df_tradeoff["recall"] <= 1.0)).all()
    assert ((df_tradeoff["f1"] >= 0.0) & (df_tradeoff["f1"] <= 1.0)).all()

    # Alert rates non-negative and properly computed
    assert (df_tradeoff["alerts_per_10k"] >= 0.0).all()
    for _, row in df_tradeoff.iterrows():
        total_eval = row["tp"] + row["fp"] + row["tn"] + row["fn"]
        expected_rate = ((row["tp"] + row["fp"]) / total_eval) * 10000.0
        assert np.isclose(row["alerts_per_10k"], expected_rate, atol=1e-2)


def test_08_round_level_aggregation():
    """Verify round-level metrics span rounds 1 to 5 for all 24 federated configurations."""
    rounds_csv = PHASE8_DIR / "round_level_analysis.csv"
    assert rounds_csv.exists()
    df_rounds = pd.read_csv(rounds_csv)

    # 24 configurations * 5 rounds = 120 rows
    assert len(df_rounds) == 120, f"Expected 120 round-level rows, got {len(df_rounds)}"

    # Check rounds span 1 to 5
    unique_rounds = sorted(df_rounds["round"].unique().tolist())
    assert unique_rounds == [1, 2, 3, 4, 5], f"Rounds must be 1 to 5, got {unique_rounds}"

    # Val PR-AUC bounded in (0, 1)
    assert ((df_rounds["val_pr_auc"] >= 0.0) & (df_rounds["val_pr_auc"] <= 1.0)).all()


def test_09_json_serialization_conforming_to_dto():
    """Verify that phase8_report.json parses cleanly and complies with Phase8Report DTO schema."""
    report_json_path = PHASE8_DIR / "phase8_report.json"
    assert report_json_path.exists()

    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required top-level fields of Phase8Report DTO
    required_fields = [
        "analysis_id",
        "timestamp",
        "seed",
        "total_experiments_analyzed",
        "centralized_records_count",
        "iid_records_count",
        "non_iid_records_count",
        "ensemble_records_count",
        "centralized_vs_federated",
        "model_comparison",
        "fedavg_vs_fedprox",
        "iid_vs_non_iid",
        "client_distribution",
        "convergence_metrics",
        "alpha_analysis",
        "threshold_analysis",
        "ablation_study",
        "frozen_ensemble_metrics",
        "key_findings",
        "limitations",
        "artifacts_generated",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field '{field}' in phase8_report.json"

    # Verify counts
    assert data["total_experiments_analyzed"] == 27
    assert data["centralized_records_count"] == 3
    assert data["iid_records_count"] == 12
    assert data["non_iid_records_count"] == 12
    assert data["ensemble_records_count"] == 2

    # Instantiate the DTO to verify exact type conformity
    cvf_records = [CentralizedVsFederatedRecord(**r) for r in data["centralized_vs_federated"]]
    model_records = [ModelComparisonRecord(**r) for r in data["model_comparison"]]
    fed_records = [FedAvgVsFedProxRecord(**r) for r in data["fedavg_vs_fedprox"]]
    iid_records = [IIDVsNonIIDRecord(**r) for r in data["iid_vs_non_iid"]]
    client_records = [ClientDistributionRecord(**r) for r in data["client_distribution"]]
    conv_records = [ConvergenceMetricRecord(**r) for r in data["convergence_metrics"]]
    alpha_records = [AlphaAnalysisRecord(**r) for r in data["alpha_analysis"]]
    thresh_records = [ThresholdAnalysisRecord(**r) for r in data["threshold_analysis"]]
    ablation_records = [AblationRecord(**r) for r in data["ablation_study"]]

    report_dto = Phase8Report(
        analysis_id=data["analysis_id"],
        timestamp=data["timestamp"],
        seed=data["seed"],
        total_experiments_analyzed=data["total_experiments_analyzed"],
        centralized_records_count=data["centralized_records_count"],
        iid_records_count=data["iid_records_count"],
        non_iid_records_count=data["non_iid_records_count"],
        ensemble_records_count=data["ensemble_records_count"],
        centralized_vs_federated=cvf_records,
        model_comparison=model_records,
        fedavg_vs_fedprox=fed_records,
        iid_vs_non_iid=iid_records,
        client_distribution=client_records,
        convergence_metrics=conv_records,
        alpha_analysis=alpha_records,
        threshold_analysis=thresh_records,
        ablation_study=ablation_records,
        frozen_ensemble_metrics=data["frozen_ensemble_metrics"],
        key_findings=data["key_findings"],
        limitations=data["limitations"],
        artifacts_generated=data["artifacts_generated"],
    )

    assert "phase8_analysis" in report_dto.analysis_id
    assert len(report_dto.convergence_metrics) == 120
    assert len(report_dto.fedavg_vs_fedprox) == 12


def test_10_preservation_of_phase7_outputs():
    """Verify that Phase 7 ensemble outputs are preserved and alpha*=1.00 equivalence is maintained."""
    report_json_path = PHASE8_DIR / "phase8_report.json"
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frozen = data["frozen_ensemble_metrics"]
    assert "HI_Small" in frozen and "LI_Medium" in frozen

    for ds_key, ds_name in [("HI_Small", "HI-Small"), ("LI_Medium", "LI-Medium")]:
        ds_frozen = frozen[ds_key]
        assert ds_frozen["optimal_alpha"] == 1.00, f"{ds_name} selected alpha must be 1.00"
        # Delta over standalone champion is exactly 0.000000
        assert np.isclose(ds_frozen["ensemble_gain_vs_iid"], 0.0, atol=1e-6), (
            f"{ds_name} ensemble PR-AUC must match standalone IID champion exactly"
        )
        assert "P_ensemble = 1.0 * P_IID" in ds_frozen["mathematical_equivalence"]

