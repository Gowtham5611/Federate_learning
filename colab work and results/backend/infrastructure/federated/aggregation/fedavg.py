"""Sample-count weighted Federated Averaging (FedAvg) aggregation.

Computes:
    w_global = sum_{k=1}^K (n_k / N) * w_k
where n_k is the number of local training samples on client k,
N = sum_k n_k, and w_k is the client parameter state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch import Tensor

from application.ports.output.model_trainer import ParameterState

logger = logging.getLogger(__name__)


def compute_aggregation_weights(
    client_sample_counts: Mapping[str, int],
) -> dict[str, float]:
    """Compute sample-count weights for each client.

    Args:
        client_sample_counts: Mapping from client_id to sample count n_k.

    Returns:
        Mapping from client_id to normalized weight (n_k / N).

    Raises:
        ValueError: If total sample count is zero or client_sample_counts is empty.
    """
    if not client_sample_counts:
        raise ValueError("client_sample_counts cannot be empty")

    total_samples = sum(client_sample_counts.values())
    if total_samples <= 0:
        raise ValueError(
            f"Total sample count must be positive; got {total_samples}"
        )

    return {
        cid: count / total_samples for cid, count in client_sample_counts.items()
    }


def aggregate_fedavg(
    client_states: Mapping[str, ParameterState],
    weights: Mapping[str, float],
) -> ParameterState:
    """Perform sample-count weighted FedAvg parameter aggregation.

    Args:
        client_states: Mapping from client_id to its local ParameterState.
        weights: Mapping from client_id to its aggregation weight.

    Returns:
        Aggregated global ParameterState on CPU.

    Raises:
        ValueError: If client_states is empty or keys don't match weights.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    missing = set(client_states.keys()) ^ set(weights.keys())
    if missing:
        raise ValueError(
            f"Mismatch between client_states and weights keys: {missing}"
        )

    # Normalize weights just in case of float precision discrepancies
    total_weight = sum(weights.values())
    normalized_weights = {cid: w / total_weight for cid, w in weights.items()}

    first_client = next(iter(client_states.values()))
    param_keys = list(first_client.keys())

    aggregated_state: dict[str, Tensor] = {}

    for key in param_keys:
        # Check all clients have this parameter
        for cid, state in client_states.items():
            if key not in state:
                raise ValueError(
                    f"Parameter '{key}' missing from client '{cid}' parameter state"
                )

        first_tensor = first_client[key]
        if not isinstance(first_tensor, Tensor):
            # Pass through non-tensor states if any
            aggregated_state[key] = first_tensor
            continue

        if not first_tensor.is_floating_point():
            # For non-floating point tensors (e.g. tracking buffers), take first client's
            aggregated_state[key] = first_tensor.detach().cpu().clone()
            continue

        # Weighted sum across all clients
        weighted_sum = torch.zeros_like(first_tensor, dtype=first_tensor.dtype, device="cpu")
        for cid, state in client_states.items():
            w = normalized_weights[cid]
            tensor = state[key]
            if isinstance(tensor, Tensor):
                weighted_sum += tensor.detach().to(device="cpu", dtype=first_tensor.dtype) * w
            else:
                raise TypeError(
                    f"Expected Tensor for key '{key}' in client '{cid}', got {type(tensor)}"
                )

        aggregated_state[key] = weighted_sum

    logger.debug(
        "Aggregated %d parameters across %d clients with weights: %s",
        len(aggregated_state),
        len(client_states),
        {cid: f"{w:.4f}" for cid, w in normalized_weights.items()},
    )

    return aggregated_state

