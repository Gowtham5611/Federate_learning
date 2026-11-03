"""Unit tests for client drift tracker and parameter distance."""

from __future__ import annotations

import torch

from infrastructure.federated.metrics.drift_tracker import (
    compute_drift_metrics,
    compute_parameter_distance,
)


def test_parameter_distance_zero_for_identical_states() -> None:
    """Distance between identical parameter states must be exactly zero."""
    state = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([0.5, 0.5])}
    assert compute_parameter_distance(state, state) == 0.0


def test_parameter_distance_euclidean() -> None:
    """Distance between different states matches Euclidean norm."""
    state_a = {"w": torch.tensor([0.0, 0.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    assert abs(compute_parameter_distance(state_a, state_b) - 5.0) < 1e-6


def test_compute_drift_metrics() -> None:
    """Verify mean drift, weighted drift, and pairwise disagreement."""
    global_state = {"w": torch.zeros(2)}
    client_states = {
        "client_1": {"w": torch.tensor([1.0, 0.0])},  # dist = 1.0
        "client_2": {"w": torch.tensor([0.0, 2.0])},  # dist = 2.0
        "client_3": {"w": torch.tensor([0.0, 0.0])},  # dist = 0.0
    }
    weights = {"client_1": 0.2, "client_2": 0.3, "client_3": 0.5}

    metrics = compute_drift_metrics(global_state, client_states, weights)

    # Mean drift = (1 + 2 + 0) / 3 = 1.0
    assert abs(metrics["mean_drift"] - 1.0) < 1e-5
    # Weighted drift = 0.2 * 1 + 0.3 * 2 + 0.5 * 0 = 0.8
    assert abs(metrics["weighted_drift"] - 0.8) < 1e-5
    # Max drift = 2.0, min drift = 0.0
    assert abs(metrics["max_drift"] - 2.0) < 1e-5
    assert abs(metrics["min_drift"] - 0.0) < 1e-5

    # Individual client keys
    assert abs(metrics["drift_client_1"] - 1.0) < 1e-5
    assert abs(metrics["drift_client_2"] - 2.0) < 1e-5
    assert abs(metrics["drift_client_3"] - 0.0) < 1e-5
    assert metrics["pairwise_disagreement"] > 0.0

