"""Unit tests for SCAFFOLD algorithm, control variates, and aggregation."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.scaffold import (
    SCAFFOLDTrainingEngine,
    aggregate_scaffold,
    init_control_variate,
)


class SimpleLinearClassifier(nn.Module):
    """Minimal model for testing SCAFFOLD updates."""

    def __init__(self, in_features: int = 4) -> None:
        """Initialize classifier."""
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: Tensor, lengths: Any = None) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self.fc(x).squeeze(-1))


def test_init_control_variate_matches_parameter_shapes() -> None:
    """Verify control variate initializes to zeros matching all floating point parameters."""
    model = SimpleLinearClassifier(in_features=8)
    params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cv = init_control_variate(params)

    assert "fc.weight" in cv
    assert "fc.bias" in cv
    assert cv["fc.weight"].shape == (1, 8)
    assert cv["fc.bias"].shape == (1,)
    assert torch.all(cv["fc.weight"] == 0.0)
    assert torch.all(cv["fc.bias"] == 0.0)


def test_client_control_variates_remain_isolated() -> None:
    """Verify each client maintains its own distinct control variate."""
    metrics = SklearnMetricsCalculator()
    engine_1 = SCAFFOLDTrainingEngine(metrics)
    engine_2 = SCAFFOLDTrainingEngine(metrics)

    cv_1 = {"fc.weight": torch.ones(1, 4), "fc.bias": torch.ones(1)}
    cv_2 = {"fc.weight": torch.full((1, 4), 2.0), "fc.bias": torch.full((1,), 2.0)}
    global_cv = {"fc.weight": torch.zeros(1, 4), "fc.bias": torch.zeros(1)}

    engine_1.set_control_variates(cv_1, global_cv)
    engine_2.set_control_variates(cv_2, global_cv)

    # Modifying cv_1 should not alter engine_2
    assert not torch.allclose(
        engine_1.client_control_variate["fc.weight"],
        engine_2.client_control_variate["fc.weight"],
    )
    assert torch.allclose(engine_1.client_control_variate["fc.weight"], torch.ones(1, 4))
    assert torch.allclose(engine_2.client_control_variate["fc.weight"], torch.full((1, 4), 2.0))


def test_scaffold_control_variate_update_computation() -> None:
    """Verify Option II control variate update formula: c_k^+ = c_k - c + (1/(K*eta))*(x - y)."""
    metrics = SklearnMetricsCalculator()
    engine = SCAFFOLDTrainingEngine(metrics)

    initial_x = {"fc.weight": torch.ones(1, 4), "fc.bias": torch.zeros(1)}
    c_k = {"fc.weight": torch.zeros(1, 4), "fc.bias": torch.zeros(1)}
    c = {"fc.weight": torch.zeros(1, 4), "fc.bias": torch.zeros(1)}

    engine.set_control_variates(c_k, c)
    engine._initial_global_parameters = initial_x
    engine._total_steps_taken = 2
    lr = 0.1

    final_y = {"fc.weight": torch.full((1, 4), 0.8), "fc.bias": torch.zeros(1)}
    new_c_k, delta_c = engine.compute_control_variate_update(final_y, learning_rate=lr)

    # Expected: delta = 1.0 / (2 * 0.1) * (1.0 - 0.8) - 0.0 = 5.0 * 0.2 = 1.0
    expected_delta_weight = torch.ones(1, 4)
    assert torch.allclose(delta_c["fc.weight"], expected_delta_weight, atol=1e-5)
    assert torch.allclose(new_c_k["fc.weight"], expected_delta_weight, atol=1e-5)


def test_aggregate_scaffold_updates_model_and_control_variate() -> None:
    """Verify SCAFFOLD aggregation computes sample-weighted averages."""
    client_states = {
        "client_1": {"fc.weight": torch.tensor([[1.0, 2.0]]), "fc.bias": torch.tensor([0.0])},
        "client_2": {"fc.weight": torch.tensor([[3.0, 4.0]]), "fc.bias": torch.tensor([2.0])},
    }
    client_deltas = {
        "client_1": {"fc.weight": torch.tensor([[0.2, 0.4]]), "fc.bias": torch.tensor([0.1])},
        "client_2": {"fc.weight": torch.tensor([[0.6, 0.8]]), "fc.bias": torch.tensor([0.3])},
    }
    global_cv = {
        "fc.weight": torch.tensor([[1.0, 1.0]]),
        "fc.bias": torch.tensor([0.5]),
    }
    weights = {"client_1": 0.5, "client_2": 0.5}

    agg_params, updated_cv = aggregate_scaffold(
        client_states, client_deltas, global_cv, weights
    )

    # Aggregated weight: 0.5 * [1,2] + 0.5 * [3,4] = [2, 3]
    assert torch.allclose(agg_params["fc.weight"], torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(agg_params["fc.bias"], torch.tensor([1.0]))

    # Updated global CV: [1,1] + 0.5 * [0.2, 0.4] + 0.5 * [0.6, 0.8] = [1.4, 1.6]
    assert torch.allclose(updated_cv["fc.weight"], torch.tensor([[1.4, 1.6]]))
    assert torch.allclose(updated_cv["fc.bias"], torch.tensor([0.7]))

