r"""SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
    ICML 2020.

Mathematical formulation:
    Local update at step t:
        g_k(y) = \nabla f_k(y)
        \tilde{g}_k(y) = g_k(y) - c_k + c
        y \leftarrow y - \eta_L \tilde{g}_k(y)
    Control variate update (Option II):
        c_k^+ = c_k - c + \frac{1}{K_{steps} \eta_L} (x - y_k)
        \Delta c_k = c_k^+ - c_k = \frac{1}{K_{steps} \eta_L} (x - y_k) - c
    Server aggregation:
        x \leftarrow \sum_{k=1}^K p_k y_k
        c \leftarrow c + \sum_{k=1}^K p_k \Delta c_k
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from application.dto.training import TrainingConfig
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    EpochCallback,
    FitOutcome,
    ParameterState,
)
from application.ports.output.sequence_data import SequenceDataset
from infrastructure.models.pytorch.torch_training_engine import (
    PROGRESS_FRACTIONS,
    TorchTrainingEngine,
    _to_torch,
)

logger = logging.getLogger(__name__)


def init_control_variate(parameters: ParameterState) -> dict[str, Tensor]:
    """Initialize a zero control variate matching the shape of model parameters.

    Args:
        parameters: Model parameter state.

    Returns:
        Dictionary mapping parameter names to zero tensors on CPU.
    """
    control_variate: dict[str, Tensor] = {}
    for name, tensor in parameters.items():
        if isinstance(tensor, Tensor) and tensor.is_floating_point():
            control_variate[name] = torch.zeros_like(tensor, dtype=tensor.dtype, device="cpu")
    return control_variate


class SCAFFOLDTrainingEngine(TorchTrainingEngine):
    """Local training engine implementing the SCAFFOLD gradient-corrected objective.

    Maintains the client control variate c_k, applies the correction (g - c_k + c),
    and updates c_k after local steps complete.
    """

    def __init__(
        self,
        metrics: MetricsCalculatorPort,
        *,
        client_control_variate: dict[str, Tensor] | None = None,
        global_control_variate: dict[str, Tensor] | None = None,
    ) -> None:
        """Initialize SCAFFOLD engine.

        Args:
            metrics: Metrics calculator for per-epoch validation scoring.
            client_control_variate: Client local control variate c_k (CPU).
            global_control_variate: Server global control variate c (CPU).
        """
        super().__init__(metrics)
        self._c_k: dict[str, Tensor] = client_control_variate or {}
        self._c: dict[str, Tensor] = global_control_variate or {}
        self._initial_global_parameters: ParameterState | None = None
        self._total_steps_taken: int = 0
        self._last_delta_c: dict[str, Tensor] = {}

    @property
    def client_control_variate(self) -> dict[str, Tensor]:
        """Current client control variate c_k."""
        return self._c_k

    @property
    def global_control_variate(self) -> dict[str, Tensor]:
        """Current global control variate c."""
        return self._c

    @property
    def last_delta_c(self) -> dict[str, Tensor]:
        r"""Control variate delta \Delta c_k = c_k^+ - c_k from last training."""
        return self._last_delta_c

    def set_control_variates(
        self,
        client_cv: dict[str, Tensor],
        global_cv: dict[str, Tensor],
    ) -> None:
        """Update active control variates before a communication round."""
        self._c_k = {k: v.detach().cpu().clone() for k, v in client_cv.items()}
        self._c = {k: v.detach().cpu().clone() for k, v in global_cv.items()}

    def fit(
        self,
        config: TrainingConfig,
        train: SequenceDataset,
        validation: SequenceDataset,
        *,
        initial_parameters: ParameterState | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> FitOutcome:
        """Fit local client model using SCAFFOLD gradient correction."""
        self._initial_global_parameters = initial_parameters
        self._total_steps_taken = 0
        return super().fit(
            config,
            train,
            validation,
            initial_parameters=initial_parameters,
            on_epoch=on_epoch,
        )

    def _train_one_epoch(
        self,
        config: TrainingConfig,
        model: Any,
        train: SequenceDataset,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        r"""Execute one training epoch with gradient correction \tilde{g} = g - c_k + c."""
        if self._initial_global_parameters is None:
            self._initial_global_parameters = self._snapshot(model)

        device = next(model.parameters()).device
        model.train()

        # Cache control variates on target device for efficient step updates
        c_k_device: dict[str, Tensor] = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in self._c_k.items()
        }
        c_device: dict[str, Tensor] = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in self._c.items()
        }

        total_batches = max(1, -(-train.sequence_count // config.batch_size))
        report_every = max(1, total_batches // PROGRESS_FRACTIONS)
        loss_sum = 0.0
        seen = 0

        batches = train.batches(
            config.batch_size, shuffle=True, seed=config.random_seed + epoch
        )
        for index, batch in enumerate(batches, start=1):
            features, lengths, labels = _to_torch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, lengths)
            loss = criterion(logits, labels)

            loss.backward()

            # Apply SCAFFOLD gradient correction: \tilde{g} = g - c_k + c
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    c_k_val = c_k_device.get(name)
                    c_val = c_device.get(name)
                    if c_k_val is not None and c_val is not None:
                        param.grad.data.add_(c_val - c_k_val)

            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            self._total_steps_taken += 1

            count = len(batch)
            loss_sum += float(loss.detach()) * count
            seen += count
            if index % report_every == 0 or index == total_batches:
                logger.debug(
                    "  [SCAFFOLD] epoch %d: batch %d/%d, running loss %.6f",
                    epoch,
                    index,
                    total_batches,
                    loss_sum / seen,
                )

        return loss_sum / seen if seen else 0.0

    def compute_control_variate_update(
        self,
        final_model_parameters: ParameterState,
        learning_rate: float,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        r"""Calculate Option II control variate update after local training.

        c_k^+ = c_k - c + (1 / (K_{steps} * \eta_L)) * (x - y_k)
        \Delta c_k = c_k^+ - c_k = (1 / (K_{steps} * \eta_L)) * (x - y_k) - c

        Args:
            final_model_parameters: Final local model parameters y_k (CPU).
            learning_rate: Local optimizer learning rate \eta_L.

        Returns:
            Tuple of (new_c_k, delta_c_k).
        """
        if self._initial_global_parameters is None:
            raise RuntimeError("Initial global parameters not set; cannot compute SCAFFOLD update")

        total_steps = max(1, self._total_steps_taken)
        scale = 1.0 / (total_steps * learning_rate)

        new_c_k: dict[str, Tensor] = {}
        delta_c: dict[str, Tensor] = {}

        for name, initial_tensor in self._initial_global_parameters.items():
            if not isinstance(initial_tensor, Tensor) or not initial_tensor.is_floating_point():
                continue

            final_tensor = final_model_parameters.get(name)
            if final_tensor is None or not isinstance(final_tensor, Tensor):
                continue

            x = initial_tensor.detach().cpu()
            y = final_tensor.detach().cpu()
            old_c_k = self._c_k.get(name, torch.zeros_like(x))
            old_c = self._c.get(name, torch.zeros_like(x))

            # \Delta c_k = scale * (x - y) - c
            delta = scale * (x - y) - old_c
            # c_k^+ = old_c_k + delta = old_c_k - old_c + scale * (x - y)
            c_k_plus = old_c_k + delta

            new_c_k[name] = c_k_plus
            delta_c[name] = delta

        self._c_k = new_c_k
        self._last_delta_c = delta_c
        return new_c_k, delta_c


def aggregate_scaffold(
    client_states: Mapping[str, ParameterState],
    client_deltas: Mapping[str, dict[str, Tensor]],
    global_control_variate: dict[str, Tensor],
    weights: Mapping[str, float],
) -> tuple[ParameterState, dict[str, Tensor]]:
    r"""Perform sample-weighted SCAFFOLD parameter and control variate aggregation.

    x \leftarrow \sum_k p_k y_k
    c \leftarrow c + \sum_k p_k \Delta c_k

    Args:
        client_states: Mapping from client_id to local parameter state y_k.
        client_deltas: Mapping from client_id to control variate delta \Delta c_k.
        global_control_variate: Previous global control variate c.
        weights: Normalized client aggregation weights p_k.

    Returns:
        Tuple of (aggregated_parameters, updated_global_control_variate).
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    total_weight = sum(weights.values())
    normalized_weights = {cid: w / total_weight for cid, w in weights.items()}

    # 1. Parameter aggregation (standard weighted average)
    first_client = next(iter(client_states.values()))
    param_keys = list(first_client.keys())
    aggregated_params: dict[str, Tensor] = {}

    for key in param_keys:
        first_tensor = first_client[key]
        if not isinstance(first_tensor, Tensor):
            aggregated_params[key] = first_tensor
            continue

        if not first_tensor.is_floating_point():
            aggregated_params[key] = first_tensor.detach().cpu().clone()
            continue

        weighted_sum = torch.zeros_like(first_tensor, dtype=first_tensor.dtype, device="cpu")
        for cid, state in client_states.items():
            w = normalized_weights[cid]
            tensor = state[key]
            if isinstance(tensor, Tensor):
                weighted_sum += tensor.detach().to(device="cpu", dtype=first_tensor.dtype) * w
        aggregated_params[key] = weighted_sum

    # 2. Control variate aggregation: c_{new} = c + \sum_k p_k \Delta c_k
    updated_global_cv: dict[str, Tensor] = {}
    for key, c_tensor in global_control_variate.items():
        delta_sum = torch.zeros_like(c_tensor, dtype=c_tensor.dtype, device="cpu")
        for cid, deltas in client_deltas.items():
            w = normalized_weights[cid]
            if key in deltas:
                delta_sum += deltas[key].detach().to(device="cpu", dtype=c_tensor.dtype) * w
        updated_global_cv[key] = (c_tensor.detach().cpu() + delta_sum).clone()

    logger.debug(
        "SCAFFOLD aggregation completed across %d clients.",
        len(client_states),
    )
    return aggregated_params, updated_global_cv
