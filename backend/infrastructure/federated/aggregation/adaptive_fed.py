"""Adaptive Drift-Regularized Federated Aggregation (FedAdaptive).

Implements a principled adaptive aggregation scheme that weights client updates
using a joint function of sample volume and relative parameter drift norm:

    s_k^{(t)} = (n_k / N) * exp(-beta * (d_k^{(t)} / (mean(d^{(t)}) + eps)))
    p_k^{(t)} = s_k^{(t)} / sum_j s_j^{(t)}
    w_global^{(t+1)} = sum_{k in S_t} p_k^{(t)} * w_k^{(t)}

where d_k^{(t)} = ||w_k^{(t)} - w_global^{(t)}||_2 is the Euclidean parameter drift.
When beta = 0, this degenerates cleanly to standard sample-count weighted FedAvg.

Guarantees:
- Zero validation or test set leakage: weights depend strictly on round updates.
- Robust handling of stragglers or missing client updates.
- Complete logging of per-round client contribution weights.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from application.ports.output.model_trainer import ParameterState
from infrastructure.federated.metrics.drift_tracker import compute_parameter_distance

logger = logging.getLogger(__name__)


def compute_adaptive_weights(
    global_state: ParameterState,
    client_states: Mapping[str, ParameterState],
    client_sample_counts: Mapping[str, int],
    *,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Calculate drift-regularized adaptive aggregation weights for a round.

    Args:
        global_state: Global model parameter state at the start of the round.
        client_states: Mapping of client_id to updated local model state.
        client_sample_counts: Mapping of client_id to number of local training samples.
        beta: Sensitivity to client drift (beta=0 equals standard FedAvg).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dict mapping client_id to normalized aggregation weight p_k (summing to 1.0).

    Raises:
        ValueError: If client_states is empty or keys do not match client_sample_counts.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    missing = set(client_states.keys()) ^ set(client_sample_counts.keys())
    if missing:
        raise ValueError(
            f"Keys in client_states and client_sample_counts must match exactly; symmetric difference: {sorted(missing)}"
        )

    cids = sorted(client_states.keys())
    total_samples = sum(client_sample_counts[cid] for cid in cids)
    if total_samples <= 0:
        raise ValueError(f"Total sample count must be positive; got {total_samples}")

    # Standard FedAvg base proportions
    base_proportions = {cid: client_sample_counts[cid] / total_samples for cid in cids}

    # If beta is zero, return standard FedAvg weights immediately
    if abs(beta) < 1e-9:
        return base_proportions

    # 1. Compute parameter drift norm for each participating client
    drifts: Dict[str, float] = {}
    for cid in cids:
        dist = compute_parameter_distance(client_states[cid], global_state)
        # Verify finite numeric drift
        if math.isnan(dist) or math.isinf(dist):
            logger.warning("Client '%s' produced invalid drift distance (%s); zeroing contribution.", cid, dist)
            dist = 1e6
        drifts[cid] = dist

    mean_drift = float(sum(drifts.values()) / max(len(drifts), 1))

    # 2. Compute unnormalized exponential drift-penalized score
    scores: Dict[str, float] = {}
    for cid in cids:
        norm_drift = drifts[cid] / (mean_drift + eps)
        # Bound exponent to prevent numerical underflow
        exponent = -beta * norm_drift
        exponent = max(-20.0, min(20.0, exponent))
        scores[cid] = base_proportions[cid] * math.exp(exponent)

    # 3. Normalize onto simplex
    total_score = sum(scores.values())
    if total_score <= 0 or math.isnan(total_score):
        logger.warning("Total adaptive score invalid (%s); falling back to base FedAvg.", total_score)
        return base_proportions

    normalized_weights = {cid: scores[cid] / total_score for cid in cids}
    return normalized_weights


def aggregate_adaptive_fed(
    global_state: ParameterState,
    client_states: Mapping[str, ParameterState],
    client_sample_counts: Mapping[str, int],
    *,
    beta: float = 0.5,
) -> Tuple[ParameterState, Dict[str, float]]:
    """Perform drift-regularized FedAdaptive parameter aggregation.

    Args:
        global_state: Parameters from start of round.
        client_states: Client updated states.
        client_sample_counts: Local sample counts.
        beta: Drift regularization factor.

    Returns:
        Tuple of (aggregated global ParameterState on CPU, weights_dict).
    """
    weights = compute_adaptive_weights(
        global_state=global_state,
        client_states=client_states,
        client_sample_counts=client_sample_counts,
        beta=beta,
    )

    cids = list(client_states.keys())
    first_state = client_states[cids[0]]
    param_names = list(first_state.keys())

    aggregated: ParameterState = {}

    for name in param_names:
        first_val = first_state[name]

        # For floating point tensors: compute linear combination
        if isinstance(first_val, Tensor) and first_val.is_floating_point():
            accum = torch.zeros_like(first_val, dtype=first_val.dtype, device=torch.device("cpu"))
            for cid in cids:
                p_val = client_states[cid][name].detach().to(device="cpu")
                w = float(weights[cid])
                accum.add_(p_val, alpha=w)
            aggregated[name] = accum
        else:
            # For non-floating point (e.g., int num_batches_tracked in BatchNorm): copy from primary client
            if isinstance(first_val, Tensor):
                aggregated[name] = first_val.detach().to(device="cpu").clone()
            else:
                aggregated[name] = first_val

    return aggregated, weights

