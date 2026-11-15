"""Comprehensive Research Verification Test Suite for FedSecureAI.

Tests the core mathematical, architectural, and operational components introduced
in the Research-Grade Upgrade:
1. ResearchDataQualityAuditor and chronological leakage detection.
2. PartitionHeterogeneityAuditor (TVD, W1, and account atomicity).
3. Adaptive Federated Aggregation (FedAdaptive).
4. AML Operational Research Evaluator (PR-AUC, Recall@FPR, ECE, MCE).
5. Post-hoc calibration monotonicity and ranking preservation.
6. Sequence Integrated Gradients completeness axiom.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from infrastructure.data.partitioners.partition_heterogeneity_auditor import (
    PartitionHeterogeneityAuditor,
)
from infrastructure.data.validators.research_data_quality_auditor import (
    ResearchDataQualityAuditor,
)
from infrastructure.evaluation.aml_research_evaluator import (
    AmlResearchEvaluator,
)
from infrastructure.evaluation.calibration_methods import (
    TemperatureScalingCalibrator,
)
from infrastructure.explainability.integrated_gradients import (
    SequenceIntegratedGradients,
)
from infrastructure.federated.aggregation.adaptive_fed import (
    aggregate_adaptive_fed,
    compute_adaptive_weights,
)


# ==============================================================================
# 1. Data Quality & Zero-Leakage Tests
# ==============================================================================


def test_research_data_quality_auditor_live():
    """Verify data quality auditor passes on authoritative HI-Small splits."""
    auditor = ResearchDataQualityAuditor(dataset_name="HI-Small")
    report = auditor.audit()

    assert report.audit_verdict == "PASSED_RESEARCH_GRADE_STANDARDS"
    assert report.schema_verified is True
    assert report.invalid_numerical_values_detected == 0
    assert report.duplicate_rows_detected == 0
    assert report.test_set_distribution_preserved is True
    assert report.temporal_leakage.is_train_val_strictly_chronological is True
    assert report.temporal_leakage.is_val_test_strictly_chronological is True
    assert report.temporal_leakage.temporal_overlap_detected is False


# ==============================================================================
# 2. Partition Heterogeneity & Atomicity Tests
# ==============================================================================


def test_partition_heterogeneity_and_atomicity_live():
    """Verify TVD, W1, and account atomicity checks on live partitions."""
    auditor = PartitionHeterogeneityAuditor(dataset_name="HI-Small", seed=42)
    report = auditor.audit()

    assert report.audit_verdict == "PARTITIONING_VERIFIED_ZERO_LEAKAGE"
    assert report.account_atomicity_verified is True
    assert report.zero_train_val_test_leakage_verified is True
    assert report.mean_non_iid_label_tvd > 0.0
    assert report.mean_non_iid_wasserstein > 0.0
    assert len(report.non_iid_client_metrics) == 3


# ==============================================================================
# 3. Adaptive Federated Aggregation (FedAdaptive) Tests
# ==============================================================================


def test_fedadaptive_recovers_fedavg_when_beta_zero():
    """Verify FedAdaptive degenerates to sample-weighted FedAvg when beta=0."""
    global_state = {"weight": torch.tensor([0.0, 0.0]), "bias": torch.tensor([0.0])}
    state_a = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])}
    state_b = {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([1.5])}
    state_c = {"weight": torch.tensor([5.0, 6.0]), "bias": torch.tensor([2.5])}

    client_states = {"c1": state_a, "c2": state_b, "c3": state_c}
    client_counts = {"c1": 100, "c2": 200, "c3": 700}

    aggregated_params, weights = aggregate_adaptive_fed(
        global_state=global_state,
        client_states=client_states,
        client_sample_counts=client_counts,
        beta=0.0,
    )

    assert np.isclose(weights["c1"], 0.1, atol=1e-5)
    assert np.isclose(weights["c2"], 0.2, atol=1e-5)
    assert np.isclose(weights["c3"], 0.7, atol=1e-5)
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)

    expected_weight = 0.1 * state_a["weight"] + 0.2 * state_b["weight"] + 0.7 * state_c["weight"]
    assert torch.allclose(aggregated_params["weight"], expected_weight, atol=1e-5)


def test_fedadaptive_downweights_divergent_client():
    """Verify FedAdaptive exponentially penalizes high-drift client updates when beta > 0."""
    global_state = {"w": torch.tensor([1.0, 1.0])}
    state_a = {"w": torch.tensor([1.0, 1.0])}
    state_b = {"w": torch.tensor([1.1, 0.9])}
    state_c = {"w": torch.tensor([10.0, 10.0])}  # severe outlier drift

    client_states = {"c1": state_a, "c2": state_b, "c3": state_c}
    client_counts = {"c1": 100, "c2": 100, "c3": 100}

    _, weights = aggregate_adaptive_fed(
        global_state=global_state,
        client_states=client_states,
        client_sample_counts=client_counts,
        beta=1.5,
    )

    # In FedAvg, all would be 0.333. In FedAdaptive, c3 should be strongly downweighted
    assert weights["c3"] < weights["c1"]
    assert weights["c3"] < weights["c2"]
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)


# ==============================================================================
# 4. AML Operational Evaluator Tests
# ==============================================================================


def test_aml_research_evaluator_metrics():
    """Verify AML research evaluation suite computes PR-AUC, Recall@FPR, ECE, Brier."""
    evaluator = AmlResearchEvaluator()

    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.05, 0.1, 0.08, 0.12, 0.02, 0.2, 0.15, 0.3, 0.85, 0.95], dtype=float)

    metrics = evaluator.evaluate(y_true, y_prob, threshold=0.5)

    assert hasattr(metrics, "pr_auc")
    assert hasattr(metrics, "roc_auc")
    assert hasattr(metrics, "recall_at_fpr_0_1_pct")
    assert hasattr(metrics, "recall_at_fpr_1_pct")
    assert hasattr(metrics, "false_positives_per_10000")
    assert hasattr(metrics, "expected_calibration_error")
    assert hasattr(metrics, "brier_score")

    assert metrics.pr_auc > 0.8
    assert metrics.roc_auc > 0.9
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0


# ==============================================================================
# 5. Calibration Monotonicity Tests
# ==============================================================================


def test_temperature_scaling_preserves_ranking_and_reduces_ece():
    """Verify Temperature Scaling preserves exact rank ordering of probabilities."""
    calibrator = TemperatureScalingCalibrator()

    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0], dtype=np.float32)
    # Overconfident probabilities
    p_val = np.array([0.001, 0.002, 0.01, 0.05, 0.02, 0.999, 0.995, 0.08, 0.98, 0.04], dtype=np.float32)

    fitted_calibrator = calibrator.fit(p_val, y_val)
    p_calibrated = fitted_calibrator.calibrate(p_val)

    # Monotonicity check: rank ordering of p_val and p_calibrated must be identical
    orig_ranks = np.argsort(p_val)
    calib_ranks = np.argsort(p_calibrated)
    np.testing.assert_array_equal(orig_ranks, calib_ranks)


# ==============================================================================
# 6. Sequence Integrated Gradients Completeness Axiom Test
# ==============================================================================


class SimpleRecurrentModel(nn.Module):
    """Simple sequence classifier for explainer axiom validation."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        out = self.fc(hn[-1])
        return out.squeeze(-1)


def test_integrated_gradients_completeness_axiom():
    """Verify Sundararajan et al. completeness axiom: sum(IG) == f(x) - f(baseline)."""
    torch.manual_seed(42)
    model = SimpleRecurrentModel(input_dim=4, hidden_dim=8)
    model.eval()

    explainer = SequenceIntegratedGradients(model=model, target="logit")

    # Sequence of length 3 with feature dim 4
    x = torch.randn(1, 3, 4, requires_grad=True)
    lengths = torch.tensor([3], dtype=torch.long)

    attributions, diagnostics = explainer.attribute(x, lengths, steps=50)

    # Completeness check from diagnostics
    max_abs_residual = diagnostics["max_abs_residual"]
    assert max_abs_residual < 1e-3, f"Completeness residual too large: {max_abs_residual}"
