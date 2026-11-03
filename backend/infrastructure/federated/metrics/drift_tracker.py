r"""Quantitative Client Drift and Disagreement Tracker.

Computes client drift metrics across federated communication rounds:
1. Distance between local model and global model: ||w_k - w_global||_2.
2. Mean and sample-weighted drift across all participating clients.
3. Client update disagreement: pairwise parameter distance:
   (1 / C(K,2)) * sum_{i < j} ||w_i - w_j||_2.
4. Client variance: (1 / K) * sum_{k=1}^K ||w_k - \bar{w}||_2^2.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import torch
from torch import Tensor

from application.ports.output.model_trainer import ParameterState

logger = logging.getLogger(__name__)


def compute_parameter_distance(
    state_a: Mapping[str, Any],
    state_b: Mapping[str, Any],
) -> float:
    """Compute Frobenius/L2 parameter distance ||w_a - w_b||_2.

    Args:
        state_a: First parameter dictionary.
        state_b: Second parameter dictionary.

    Returns:
        Euclidean distance as float.
    """
    squared_diff_sum = 0.0
    for name, val_a in state_a.items():
        if name not in state_b:
            continue
        val_b = state_b[name]
        if (
            isinstance(val_a, Tensor)
            and isinstance(val_b, Tensor)
            and val_a.is_floating_point()
            and val_b.is_floating_point()
        ):
            diff = val_a.detach().cpu() - val_b.detach().cpu()
            squared_diff_sum += float(torch.sum(diff ** 2))
    return math.sqrt(squared_diff_sum)


def compute_drift_metrics(
    global_state: ParameterState,
    client_states: Mapping[str, ParameterState],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Calculate client drift and disagreement indicators for a round.

    Args:
        global_state: Parameters w_global from round start.
        client_states: Updated local parameters w_k from each client.
        weights: Client aggregation weights p_k (summing to 1).

    Returns:
        Dictionary of computed drift metrics.
    """
    if not client_states:
        return {}

    client_drifts: dict[str, float] = {}
    for cid, state_k in client_states.items():
        dist = compute_parameter_distance(state_k, global_state)
        client_drifts[cid] = dist

    cids = list(client_states.keys())
    k = len(cids)

    # 1. Individual and mean/weighted drift from global
    mean_drift = sum(client_drifts.values()) / k
    weighted_drift = sum(
        client_drifts[cid] * weights.get(cid, 1.0 / k) for cid in cids
    )
    max_drift = max(client_drifts.values())
    min_drift = min(client_drifts.values())

    # 2. Pairwise client disagreement: mean ||w_i - w_j||
    pairwise_distances: list[float] = []
    if k > 1:
        for cid_a, cid_b in combinations(cids, 2):
            p_dist = compute_parameter_distance(client_states[cid_a], client_states[cid_b])
            pairwise_distances.append(p_dist)
        mean_pairwise_disagreement = sum(pairwise_distances) / len(pairwise_distances)
    else:
        mean_pairwise_disagreement = 0.0

    # 3. Parameter variance around the mean parameter state
    # First compute unweighted mean state \bar{w}
    mean_state: dict[str, Tensor] = {}
    first_client = next(iter(client_states.values()))
    for key, tensor in first_client.items():
        if isinstance(tensor, Tensor) and tensor.is_floating_point():
            t_sum = torch.zeros_like(tensor, dtype=tensor.dtype, device="cpu")
            for c_state in client_states.values():
                t_sum += c_state[key].detach().cpu()
            mean_state[key] = t_sum / k

    # Compute variance around mean state
    variance_sum = 0.0
    for state_k in client_states.values():
        dist_to_mean = compute_parameter_distance(state_k, mean_state)
        variance_sum += dist_to_mean ** 2
    client_variance = variance_sum / k

    metrics: dict[str, float] = {
        "mean_drift": mean_drift,
        "weighted_drift": weighted_drift,
        "max_drift": max_drift,
        "min_drift": min_drift,
        "pairwise_disagreement": mean_pairwise_disagreement,
        "client_variance": client_variance,
    }

    # Add per-client drift keys
    for cid, d_val in client_drifts.items():
        metrics[f"drift_{cid}"] = d_val

    logger.debug(
        "Drift metrics: mean=%.6f, weighted=%.6f, disagreement=%.6f, var=%.6f",
        mean_drift,
        weighted_drift,
        mean_pairwise_disagreement,
        client_variance,
    )
    return metrics

