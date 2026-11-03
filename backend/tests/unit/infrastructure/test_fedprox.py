"""Unit tests for FedProx proximal loss and training engine."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedprox import (
    FedProxTrainingEngine,
    compute_proximal_loss,
)


def test_proximal_loss_zero_when_identical() -> None:
    model = nn.Linear(4, 2)
    device = torch.device("cpu")
    global_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    loss = compute_proximal_loss(model, global_params, mu=0.01, device=device)
    assert float(loss.detach()) == pytest.approx(0.0)


def test_proximal_loss_calculation() -> None:
    model = nn.Linear(2, 1, bias=False)
    # Set weights to [2.0, 3.0]
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))

    # Global weights [0.0, 0.0]
    global_params = {"weight": torch.tensor([[0.0, 0.0]])}
    device = torch.device("cpu")

    # ||w - w_global||^2 = 2^2 + 3^2 = 4 + 9 = 13
    # (mu / 2) * 13 with mu = 0.02 -> 0.01 * 13 = 0.13
    loss = compute_proximal_loss(model, global_params, mu=0.02, device=device)
    assert float(loss.detach()) == pytest.approx(0.13)


def test_proximal_loss_scales_with_mu() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 1.0]]))

    global_params = {"weight": torch.tensor([[0.0, 0.0]])}
    device = torch.device("cpu")

    loss_mu1 = compute_proximal_loss(model, global_params, mu=0.01, device=device)
    loss_mu2 = compute_proximal_loss(model, global_params, mu=0.02, device=device)

    assert float(loss_mu2.detach()) == pytest.approx(float(loss_mu1.detach()) * 2.0)


def test_fedprox_engine_mu_validation() -> None:
    metrics = SklearnMetricsCalculator()
    engine = FedProxTrainingEngine(metrics, mu=0.05)
    assert engine.mu == 0.05

    with pytest.raises(ValueError, match="non-negative"):
        FedProxTrainingEngine(metrics, mu=-0.01)
