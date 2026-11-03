"""Unit tests for FedProx proximal regularization and mu sensitivity."""

from __future__ import annotations

import torch
from torch import nn

from infrastructure.federated.aggregation.fedprox import compute_proximal_loss


def test_proximal_loss_zero_when_mu_zero() -> None:
    """When mu is 0, proximal penalty is identically zero."""
    model = nn.Linear(4, 1)
    global_params = {k: v.clone() for k, v in model.state_dict().items()}
    loss = compute_proximal_loss(model, global_params, mu=0.0, device=torch.device("cpu"))
    assert float(loss) == 0.0


def test_proximal_loss_zero_when_parameters_match() -> None:
    """When local parameters equal global parameters, proximal penalty is zero."""
    model = nn.Linear(4, 1)
    global_params = {k: v.clone() for k, v in model.state_dict().items()}
    loss = compute_proximal_loss(model, global_params, mu=0.01, device=torch.device("cpu"))
    assert float(loss.detach()) == 0.0


def test_proximal_loss_scales_linearly_with_mu() -> None:
    """Verify (mu/2) * ||w - w_global||^2 scales linearly with mu."""
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)

    global_params = {
        "weight": torch.zeros(1, 4),
        "bias": torch.zeros(1),
    }

    # ||w - w_global||^2 = 4 * (1.0)^2 + (0.0)^2 = 4.0
    loss_001 = compute_proximal_loss(model, global_params, mu=0.01, device=torch.device("cpu"))
    loss_01 = compute_proximal_loss(model, global_params, mu=0.1, device=torch.device("cpu"))

    # Expected: 0.01 / 2 * 4.0 = 0.02
    assert abs(float(loss_001.detach()) - 0.02) < 1e-6
    # Expected: 0.1 / 2 * 4.0 = 0.20
    assert abs(float(loss_01.detach()) - 0.20) < 1e-6
    assert abs(float(loss_01.detach()) / float(loss_001.detach()) - 10.0) < 1e-5
