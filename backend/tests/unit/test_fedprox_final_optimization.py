"""Unit tests for FedProx Final Optimization and Verification.

Tests:
1. mu=0 yields exactly zero proximal penalty.
2. w=w_ref yields exactly zero proximal penalty.
3. Penalty scales linearly with mu.
4. Proximal penalty gradient matches analytical derivative mu * (w - w_ref).
5. Global reference parameters remain detached and unmutated during backward pass.
6. Weighted BCE handles zero-positive batches cleanly without NaN or zero division.
7. Operational threshold selection enforces alert budget <= 50 per 10k on validation data.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve

from infrastructure.federated.aggregation.fedprox import compute_proximal_loss
from infrastructure.models.pytorch.loss_functions import build_loss_criterion


def test_fedprox_mu_zero_penalty():
    """Verify that proximal penalty is exactly zero when mu=0.0."""
    model = nn.Linear(5, 2)
    device = torch.device("cpu")
    ref_params = {name: torch.randn_like(param) for name, param in model.named_parameters()}

    loss = compute_proximal_loss(model, ref_params, mu=0.0, device=device)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)


def test_fedprox_zero_drift_penalty():
    """Verify that proximal penalty is exactly zero when w == w_ref."""
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    device = torch.device("cpu")
    ref_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    loss = compute_proximal_loss(model, ref_params, mu=1.0, device=device)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)


def test_fedprox_linear_scaling_with_mu():
    """Verify that proximal penalty scales strictly linearly with mu."""
    model = nn.Linear(6, 3)
    device = torch.device("cpu")
    ref_params = {name: torch.randn_like(param) for name, param in model.named_parameters()}

    loss_mu_1 = compute_proximal_loss(model, ref_params, mu=1.0, device=device)
    loss_mu_2 = compute_proximal_loss(model, ref_params, mu=2.0, device=device)
    loss_mu_half = compute_proximal_loss(model, ref_params, mu=0.5, device=device)

    val_1 = float(loss_mu_1.detach())
    val_2 = float(loss_mu_2.detach())
    val_half = float(loss_mu_half.detach())

    assert val_2 == pytest.approx(2.0 * val_1, rel=1e-5)
    assert val_half == pytest.approx(0.5 * val_1, rel=1e-5)


def test_fedprox_gradient_analytical_derivative():
    """Verify that d/dw [(mu/2) * ||w - w_ref||^2] == mu * (w - w_ref)."""
    torch.manual_seed(123)
    model = nn.Linear(4, 2, bias=True)
    device = torch.device("cpu")

    ref_params = {name: torch.randn_like(param) for name, param in model.named_parameters()}
    mu = 1.0

    loss = compute_proximal_loss(model, ref_params, mu=mu, device=device)
    loss.backward()

    for name, param in model.named_parameters():
        expected_grad = mu * (param.detach() - ref_params[name])
        actual_grad = param.grad
        assert torch.allclose(actual_grad, expected_grad, atol=1e-6)


def test_fedprox_reference_parameters_gradient_isolation():
    """Verify reference parameters remain detached and unmutated during backward pass."""
    model = nn.Linear(3, 1)
    device = torch.device("cpu")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False)
    ref_params = {"weight": ref_tensor, "bias": torch.tensor([0.5], requires_grad=False)}

    loss = compute_proximal_loss(model, ref_params, mu=1.0, device=device)
    loss.backward()

    # Model parameters have gradients
    assert model.weight.grad is not None
    # Reference parameters do not have grad
    assert ref_tensor.grad is None


def test_weighted_bce_zero_positive_batch_handling():
    """Verify Weighted BCE handles batches with zero positive labels without NaN or zero division."""
    pos_weight_val = 2303.2774
    criterion = build_loss_criterion("weighted_bce", pos_weight_value=pos_weight_val)

    logits = torch.randn(1024, requires_grad=True)
    labels = torch.zeros(1024, dtype=torch.float32)  # Zero positives

    loss = criterion(logits, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert float(loss.detach()) > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_operational_threshold_budget_constraint():
    """Verify that operational threshold search strictly respects the <= 50 alerts/10k budget."""
    np.random.seed(42)
    n_samples = 100000
    n_pos = 100
    labels = np.zeros(n_samples, dtype=int)
    labels[:n_pos] = 1

    # Simulated model probabilities
    probs = np.random.beta(0.1, 10.0, size=n_samples)
    probs[:n_pos] += np.random.uniform(0.1, 0.4, size=n_pos)
    probs = np.clip(probs, 0.0, 1.0)

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    p = precisions[:-1]
    r = recalls[:-1]
    th = thresholds
    tp = r * n_pos
    alerts = np.where(p > 0, tp / p, 0.0)
    alerts_per_10k = (alerts / n_samples) * 10000.0

    alert_budget = 50.0
    under_budget_mask = alerts_per_10k <= alert_budget
    assert np.any(under_budget_mask), "Should have thresholds satisfying budget"

    best_idx = np.flatnonzero(under_budget_mask)[np.argmax(r[under_budget_mask])]
    chosen_thresh = th[best_idx]

    # Verify predictions at chosen threshold
    preds = (probs >= chosen_thresh).astype(int)
    actual_alerts = np.sum(preds)
    actual_alerts_per_10k = (actual_alerts / n_samples) * 10000.0

    assert actual_alerts_per_10k <= alert_budget + 1e-4

