"""Phase C Audit: FedAvg and FedProx mathematical unit tests.

Verifies:
1. Proximal penalty is exactly 0.0 when model parameters equal reference parameters.
2. Proximal penalty gradient exactly matches the analytical derivative: mu * (w - w_ref).
3. Proximal penalty scaling at fixed mu=0.1.
4. FedAvg sample-weighted aggregation: weights sum to 1.0, parameters match established sample counts.
"""

import pytest
import torch
import torch.nn as nn
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import compute_proximal_loss


def test_fedprox_penalty_zero_at_reference():
    """Verify that proximal penalty is exactly 0 when w == w_global."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    device = torch.device("cpu")
    # Reference is identical to current parameters
    global_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    loss = compute_proximal_loss(model, global_params, mu=0.1, device=device)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)


def test_fedprox_gradient_matches_analytical_derivative():
    """Verify that d/dw [(mu/2) * ||w - w_ref||^2] == mu * (w - w_ref)."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2, bias=True)
    device = torch.device("cpu")
    
    # Create fixed reference parameters
    ref_params = {name: torch.randn_like(param) for name, param in model.named_parameters()}
    
    mu = 0.1
    loss = compute_proximal_loss(model, ref_params, mu=mu, device=device)
    loss.backward()
    
    for name, param in model.named_parameters():
        expected_grad = mu * (param.detach() - ref_params[name])
        actual_grad = param.grad
        assert torch.allclose(actual_grad, expected_grad, atol=1e-6), (
            f"Gradient mismatch for {name}: max diff {torch.max(torch.abs(actual_grad - expected_grad))}"
        )


def test_fedavg_sample_weighted_aggregation_math():
    """Verify sample-weighted FedAvg aggregation mathematics."""
    counts = {"client_1": 539973, "client_2": 981684, "client_3": 1525559}
    total = sum(counts.values())
    weights = compute_aggregation_weights(counts)
    
    # Weights sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    
    # Established sample count matches exactly
    for cid, n_k in counts.items():
        assert weights[cid] == pytest.approx(n_k / total, abs=1e-6)
    
    # Parameter aggregation w_global = sum(p_k * w_k)
    p1 = torch.tensor([1.0, 2.0, 3.0])
    p2 = torch.tensor([4.0, 5.0, 6.0])
    p3 = torch.tensor([7.0, 8.0, 9.0])
    client_states = {
        "client_1": {"weight": p1},
        "client_2": {"weight": p2},
        "client_3": {"weight": p3},
    }
    
    agg = aggregate_fedavg(client_states, weights)
    expected = weights["client_1"] * p1 + weights["client_2"] * p2 + weights["client_3"] * p3
    assert torch.allclose(agg["weight"], expected, atol=1e-6)

