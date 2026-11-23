"""Comprehensive Test Suite for Master Research-Grade Upgrade.

Covers Phases B through T:
1. Universal CSV Data Quality Auditor (Phase B)
2. Causal Graph-Based Features & Zero Future Leakage (Phases C & D)
3. Class Imbalance Strategies & Focal Loss (Phase E)
4. Business-Cost-Aware Threshold Optimization (Phase F)
5. Federated Communication Efficiency & Quantization (Phase H)
6. Post-Hoc Drift Detection & PSI Calculation (Phase M)
7. Deterministic AML Typology Analysis (Phase N)
8. SAR-Style Draft Investigation Report Export (Phase P)
9. Differential Privacy Rényi Accountant (Phase Q)
10. Temporal sequence boundary & anti-leakage guarantee (Phase C)
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure backend on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from infrastructure.aml.investigation_report_generator import InvestigationReportGenerator
from infrastructure.aml.typology_analyzer import AmlTypologyAnalyzer
from infrastructure.data.features.graph_features import CausalGraphFeatureExtractor, TimeAwareAmlGraph
from infrastructure.data.validators.csv_data_quality_auditor import CsvDataQualityAuditor
from infrastructure.evaluation.cost_aware_threshold import CostAwareThresholdOptimizer, CostConfiguration
from infrastructure.federated.communication.communication_profiler import (
    CommunicationCompressor,
    FederatedCommunicationProfiler,
)
from infrastructure.federated.privacy.differential_privacy_accountant import (
    RenyiDifferentialPrivacyAccountant,
)
from infrastructure.models.training_strategies import FocalLoss, TrainingStrategyFactory
from infrastructure.monitoring.drift_detector import AmlDriftDetector


# ==============================================================================
# 1. Universal Data Quality Auditor (Phase B)
# ==============================================================================


def test_csv_data_quality_auditor_profiling():
    """Verify data quality auditor reports duplicates, missing cells, and schema status."""
    auditor = CsvDataQualityAuditor()
    df = pd.DataFrame({
        "timestamp": ["2022-09-01 09:00:00", "2022-09-01 09:10:00", "2022-09-01 09:10:00", "bad_date"],
        "from_account": ["ACC_1", "ACC_2", "ACC_2", "ACC_3"],
        "to_account": ["ACC_2", "ACC_3", "ACC_3", "ACC_4"],
        "amount": [1000.0, 2500.0, 2500.0, np.nan],
        "is_laundering": [0, 1, 1, 0],
    })

    report = auditor.audit_dataframe(df, filename="test_batch.csv")

    assert report.total_rows == 4
    assert report.total_columns == 5
    assert report.duplicate_rows_count == 1  # Rows 1 and 2 are identical
    assert report.total_missing_cells == 1   # 1 NaN in amount
    assert report.has_labels is True
    assert report.detected_target_column == "is_laundering"
    assert report.schema_status == "COMPATIBLE"
    assert report.is_ready_for_inference is True
    assert len(report.column_profiles) == 5


# ==============================================================================
# 2. Causal Graph Features & Anti-Leakage (Phases C & D)
# ==============================================================================


def test_time_aware_graph_zero_future_leakage():
    """Verify that transactions at time t cannot observe edges occurring at t + delta."""
    graph = TimeAwareAmlGraph()

    t0 = pd.Timestamp("2022-09-01 10:00:00")
    t1 = pd.Timestamp("2022-09-01 11:00:00")
    t2 = pd.Timestamp("2022-09-01 12:00:00")

    # Edge 1 at t0
    graph.add_transaction("TX_1", "ACC_A", "ACC_B", 5000.0, t0)
    # Edge 2 at t2 (future relative to t1)
    graph.add_transaction("TX_2", "ACC_B", "ACC_A", 4800.0, t2)

    # Query graph at t1: should see TX_1, but MUST NOT see TX_2
    features_t1 = graph.extract_features_for_transaction("ACC_A", "ACC_B", 1000.0, t1)

    assert features_t1.sender_out_degree == 1  # only TX_1
    assert features_t1.receiver_in_degree == 1  # only TX_1
    assert features_t1.is_immediate_round_trip == 0  # TX_2 has not happened yet!

    # Query graph at t2 + 1s: now TX_2 has happened, round-trip must be detected
    features_t2 = graph.extract_features_for_transaction("ACC_A", "ACC_B", 1000.0, t2 + timedelta(seconds=1))
    assert features_t2.is_immediate_round_trip == 1  # B -> A occurred at t2


def test_circular_flow_3hop_detection():
    """Verify 3-hop circular flow detection: A -> B -> C -> A."""
    graph = TimeAwareAmlGraph()

    base_time = pd.Timestamp("2022-09-01 08:00:00")
    graph.add_transaction("TX_1", "ACC_A", "ACC_B", 10000.0, base_time)
    graph.add_transaction("TX_2", "ACC_B", "ACC_C", 9500.0, base_time + timedelta(hours=2))
    graph.add_transaction("TX_3", "ACC_C", "ACC_A", 9000.0, base_time + timedelta(hours=4))

    # Evaluate transaction A -> B shortly after circular completion
    features = graph.extract_features_for_transaction(
        from_account="ACC_A",
        to_account="ACC_B",
        amount=500.0,
        timestamp=base_time + timedelta(hours=5),
    )
    assert features.is_circular_flow_3hop == 1


# ==============================================================================
# 3. Class Imbalance Strategies & Focal Loss (Phase E)
# ==============================================================================


def test_focal_loss_forward_and_gradient():
    """Verify Focal Loss calculates loss on raw logits and generates valid gradients."""
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    logits = torch.tensor([[-2.0], [3.0], [0.5]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0]])

    loss = focal(logits, targets)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_training_strategy_factory_pos_weight():
    """Verify Factory creates class-weighted BCE loss with exact train-only positive weights."""
    targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9 neg, 1 pos
    criterion, meta = TrainingStrategyFactory.create_loss_function("class_weighted", train_targets=targets)

    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    assert meta["pos_weight"] == 9.0
    assert meta["n_pos"] == 1
    assert meta["n_neg"] == 9


# ==============================================================================
# 4. Business-Cost-Aware Threshold Optimization (Phase F)
# ==============================================================================


def test_cost_aware_threshold_optimization():
    """Verify validation-only threshold tuning minimizes Expected_Cost."""
    np.random.seed(42)
    y_val = np.array([0] * 90 + [1] * 10)
    # Simulate decent probabilities
    probs = np.zeros(100)
    probs[:90] = np.random.uniform(0.01, 0.40, size=90)
    probs[90:] = np.random.uniform(0.50, 0.95, size=10)

    config = CostConfiguration(cost_fn=1000.0, cost_fp=25.0)
    optimizer = CostAwareThresholdOptimizer(cost_config=config)

    res = optimizer.optimize_threshold(y_val, probs, method="cost_minimization")

    assert 0.40 <= res.selected_threshold <= 0.55
    assert res.validation_f1 > 0.80
    assert res.validation_recall >= 0.90
    assert res.validation_expected_cost < 200.0  # Minimal FN and FP cost


# ==============================================================================
# 5. Federated Communication Efficiency & Quantization (Phase H)
# ==============================================================================


def test_communication_quantization_and_profiler():
    """Verify Int8 quantization compression and profiler accounting."""
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(57, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    state = dummy_model.state_dict()
    quant_dict, comp_bytes = CommunicationCompressor.compress_int8_quantization(state)
    uncomp_bytes = sum(p.numel() * p.element_size() for p in dummy_model.parameters())

    assert comp_bytes < uncomp_bytes * 0.35  # ~3x-4x reduction

    decomp = CommunicationCompressor.decompress_int8(quant_dict)
    assert len(decomp) == len(state)
    for k in state:
        assert decomp[k].shape == state[k].shape

    profiler = FederatedCommunicationProfiler("Federated_LSTM", dummy_model)
    profiler.record_round(1, num_clients=3, compression_strategy="int8", local_train_sec=1.2, aggregation_sec=0.1)
    summary = profiler.generate_summary(compression_strategy="int8")

    assert summary.total_rounds == 1
    assert summary.overall_compression_reduction_pct > 70.0


# ==============================================================================
# 6. Drift Detection & PSI Calculation (Phase M)
# ==============================================================================


def test_psi_and_feature_drift_detection():
    """Verify PSI accurately flags significant distribution shifts."""
    np.random.seed(42)
    ref = np.random.normal(loc=100.0, scale=15.0, size=1000)
    stable_cur = np.random.normal(loc=100.5, scale=15.0, size=1000)
    drifted_cur = np.random.normal(loc=180.0, scale=35.0, size=1000)

    detector = AmlDriftDetector()

    psi_stable = detector.calculate_psi(ref, stable_cur)
    assert psi_stable < 0.10  # Nominal / No drift

    psi_drifted = detector.calculate_psi(ref, drifted_cur)
    assert psi_drifted > 0.25  # Significant drift

    drift_report = detector.evaluate_feature_drift(ref, drifted_cur, "amount_paid")
    assert drift_report.drift_level == "SIGNIFICANT_DRIFT"


# ==============================================================================
# 7. AML Typology Analysis (Phase N)
# ==============================================================================


def test_structuring_and_velocity_typology_rules():
    """Verify rule-based typology indicator flags structuring and bursts."""
    analyzer = AmlTypologyAnalyzer()

    t0 = pd.Timestamp("2022-09-01 09:00:00")
    account_hist = [
        {"tx_id": "T0", "from_account": "ACC_SMURF", "to_account": "B1", "amount": 9200.0, "timestamp": str(t0 - timedelta(hours=3))},
        {"tx_id": "T1", "from_account": "ACC_SMURF", "to_account": "B2", "amount": 9500.0, "timestamp": str(t0 - timedelta(hours=1))},
    ]

    current_tx = {
        "tx_id": "T2",
        "from_account": "ACC_SMURF",
        "to_account": "B3",
        "amount": 9800.0,
        "timestamp": str(t0),
    }

    profile = analyzer.analyze_transaction(
        tx_row=current_tx,
        account_history=account_hist,
        model_risk_score=0.88,
        model_threshold=0.337784,
    )

    assert profile.has_any_rule_flag is True
    assert profile.overall_suspicion_tier == "HIGH_PRIORITY_INVESTIGATION"
    typo_names = [t.typology_name for t in profile.detected_typologies]
    assert "Structuring / Smurfing" in typo_names


# ==============================================================================
# 8. SAR-Style Draft Investigation Report Export (Phase P)
# ==============================================================================


def test_investigation_report_generation_and_csv_export():
    """Verify draft SAR report generates with mandatory regulatory disclaimer and compliance CSV."""
    generator = InvestigationReportGenerator()

    txs = [
        {
            "transaction_id": "TX_9901",
            "from_account": "ACC_ALPHA",
            "to_account": "ACC_BETA",
            "amount": 45000.0,
            "timestamp": "2022-09-01 10:15:00",
            "predicted_probability": 0.9412,
            "detected_typologies": [{"typology_name": "Rapid Fund Movement"}],
        }
    ]

    report = generator.generate_report(txs, analyst_id="compliance_officer_42")

    assert report["summary"]["status"] == "DRAFT_INTERNAL_INVESTIGATION"
    assert "NOT constitute an officially submitted Suspicious Activity Report" in report["summary"]["regulatory_disclaimer"]
    assert report["summary"]["flagged_transactions_count"] == 1
    assert report["summary"]["high_priority_count"] == 1

    csv_out = generator.export_csv_format(report)
    assert "Suspicious Activity Investigation Report — Draft" in csv_out
    assert "TX_9901" in csv_out
    assert "DRAFT_INTERNAL_ONLY" in csv_out


# ==============================================================================
# 9. Differential Privacy Rényi Accountant (Phase Q)
# ==============================================================================


def test_renyi_dp_accountant_bounds():
    """Verify RDP accountant computes finite, strictly positive epsilon given noise."""
    accountant = RenyiDifferentialPrivacyAccountant(delta=1e-5)

    # 1. Unnoised training provides no DP guarantee
    zero_dp = accountant.compute_epsilon(noise_multiplier=0.0, clipping_norm=1.0, sample_rate=0.01, total_steps=100)
    assert zero_dp.is_dp_guarantee_valid is False
    assert zero_dp.epsilon == float("inf")

    # 2. Gaussian mechanism with sigma=1.2, q=0.01 over 100 rounds
    dp_budget = accountant.compute_epsilon(noise_multiplier=1.2, clipping_norm=1.0, sample_rate=0.01, total_steps=100)
    assert dp_budget.is_dp_guarantee_valid is True
    assert 0.1 < dp_budget.epsilon < 5.0  # Rigorous bounded epsilon
    assert dp_budget.delta == 1e-5
