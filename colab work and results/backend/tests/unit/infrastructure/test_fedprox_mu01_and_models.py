"""Unit tests verifying FedProx with mu=0.1, model architectures, and metric evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import aggregate_fedavg, compute_aggregation_weights
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine, compute_proximal_loss
from infrastructure.models.pytorch.sequence_classifier import build_classifier


class TestFedProxMu01AndArchitectures:
    """Tests for FedProx mu=0.1, recurrent architectures, and evaluation scores."""

    def test_proximal_loss_at_mu_0_1(self) -> None:
        """Verify FedProx objective with mu=0.1: (mu / 2) * ||w - w_t||^2."""
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))

        global_params = {"weight": torch.tensor([[0.0, 0.0, 0.0]])}
        device = torch.device("cpu")

        # ||w - w_t||^2 = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        # (mu / 2) * 14 with mu = 0.1 -> 0.05 * 14 = 0.70
        loss = compute_proximal_loss(model, global_params, mu=0.1, device=device)
        assert float(loss.detach()) == pytest.approx(0.70, rel=1e-5)

    def test_proximal_loss_mu_zero_equivalence(self) -> None:
        """When mu=0.0, proximal loss must be exactly 0."""
        model = nn.Linear(4, 2)
        device = torch.device("cpu")
        global_params = {k: torch.randn_like(v) for k, v in model.named_parameters()}
        loss = compute_proximal_loss(model, global_params, mu=0.0, device=device)
        assert float(loss.detach()) == 0.0

    @pytest.mark.parametrize("arch", [ModelArchitecture.GRU, ModelArchitecture.LSTM, ModelArchitecture.BILSTM])
    def test_recurrent_architectures_separate_and_valid(self, arch: ModelArchitecture) -> None:
        """Verify each architecture builds properly, has distinct parameter counts, and outputs scalar logits."""
        config = TrainingConfig(architecture=arch, hidden_size=64, num_layers=1)
        feature_dim = 57
        model = build_classifier(config, feature_dim)

        # Feed batch of 4 sequences with length 10
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, feature_dim)
        lengths = torch.tensor([10, 8, 6, 4], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            logits = model(x, lengths)

        assert logits.shape == (batch_size,)
        assert not torch.isnan(logits).any()

        # Architecture-specific expected parameters
        if arch == ModelArchitecture.GRU:
            assert model.parameter_count() == 23681
        elif arch == ModelArchitecture.LSTM:
            assert model.parameter_count() == 31553
        elif arch == ModelArchitecture.BILSTM:
            assert model.parameter_count() == 63105

    def test_metrics_use_continuous_probabilities(self) -> None:
        """Verify that PR-AUC and ROC-AUC are computed from continuous probabilities."""
        calculator = SklearnMetricsCalculator()
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        probs = np.array([0.05, 0.12, 0.20, 0.85, 0.90], dtype=np.float32)

        metrics = calculator.evaluate(labels, probs, threshold=0.5)
        assert metrics.pr_auc > 0.90
        assert metrics.roc_auc == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.confusion.true_positives == 2
        assert metrics.confusion.false_negatives == 0

    def test_fedavg_sample_weighted_aggregation(self) -> None:
        """Verify server aggregation correctly weights by client sample count."""
        state_1 = {"weight": torch.tensor([1.0, 1.0])}
        state_2 = {"weight": torch.tensor([2.0, 2.0])}
        state_3 = {"weight": torch.tensor([4.0, 4.0])}

        counts = {"client_1": 100, "client_2": 200, "client_3": 100}
        weights = compute_aggregation_weights(counts)
        # client_1: 100/400 = 0.25, client_2: 200/400 = 0.50, client_3: 100/400 = 0.25
        assert weights["client_1"] == pytest.approx(0.25)
        assert weights["client_2"] == pytest.approx(0.50)
        assert weights["client_3"] == pytest.approx(0.25)

        agg = aggregate_fedavg({"client_1": state_1, "client_2": state_2, "client_3": state_3}, weights)
        # Expected: 0.25*1 + 0.5*2 + 0.25*4 = 0.25 + 1.0 + 1.0 = 2.25
        assert torch.allclose(agg["weight"], torch.tensor([2.25, 2.25]))
