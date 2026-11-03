"""Unit tests for FedAvg parameter aggregation."""

from __future__ import annotations

import pytest
import torch

from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)


def test_compute_aggregation_weights() -> None:
    counts = {"client_1": 100, "client_2": 200, "client_3": 100}
    weights = compute_aggregation_weights(counts)

    assert pytest.approx(weights["client_1"]) == 0.25
    assert pytest.approx(weights["client_2"]) == 0.50
    assert pytest.approx(weights["client_3"]) == 0.25
    assert pytest.approx(sum(weights.values())) == 1.0


def test_compute_aggregation_weights_equal() -> None:
    counts = {"client_1": 1000, "client_2": 1000, "client_3": 1000}
    weights = compute_aggregation_weights(counts)

    for cid in ("client_1", "client_2", "client_3"):
        assert pytest.approx(weights[cid]) == 1.0 / 3.0


def test_aggregate_fedavg_weighted_average() -> None:
    client_states = {
        "client_1": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.0])},
        "client_2": {"weight": torch.tensor([3.0, 6.0]), "bias": torch.tensor([2.0])},
        "client_3": {"weight": torch.tensor([5.0, 10.0]), "bias": torch.tensor([4.0])},
    }
    # Weights: 0.25, 0.5, 0.25
    weights = {"client_1": 0.25, "client_2": 0.50, "client_3": 0.25}

    aggregated = aggregate_fedavg(client_states, weights)

    # weight: 0.25*[1, 2] + 0.5*[3, 6] + 0.25*[5, 10] = [0.25+1.5+1.25, 0.5+3.0+2.5] = [3.0, 6.0]
    expected_weight = torch.tensor([3.0, 6.0])
    # bias: 0.25*0 + 0.5*2 + 0.25*4 = 1.0 + 1.0 = 2.0
    expected_bias = torch.tensor([2.0])

    torch.testing.assert_close(aggregated["weight"], expected_weight)
    torch.testing.assert_close(aggregated["bias"], expected_bias)


def test_aggregate_fedavg_mismatched_keys_raises() -> None:
    client_states = {
        "client_1": {"w1": torch.tensor([1.0])},
        "client_2": {"w2": torch.tensor([2.0])},
    }
    weights = {"client_1": 0.5, "client_2": 0.5}

    with pytest.raises(ValueError, match="missing from client"):
        aggregate_fedavg(client_states, weights)


def test_aggregate_fedavg_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_fedavg({}, {})

