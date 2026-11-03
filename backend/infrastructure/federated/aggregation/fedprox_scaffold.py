r"""Hybrid FedProx-SCAFFOLD local training engine.

Combines FedProx proximal regularization with SCAFFOLD control-variate gradient correction:
    L = BCEWithLogitsLoss(logits, labels) + (mu / 2) * ||w - w_global||^2
    \tilde{g} = \nabla L - c_k + c

Strictly separated experimental implementation for research investigation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from application.dto.training import TrainingConfig
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from infrastructure.federated.aggregation.fedprox import compute_proximal_loss
from infrastructure.federated.aggregation.scaffold import SCAFFOLDTrainingEngine
from infrastructure.models.pytorch.torch_training_engine import (
    PROGRESS_FRACTIONS,
    _to_torch,
)

logger = logging.getLogger(__name__)


class FedProxSCAFFOLDTrainingEngine(SCAFFOLDTrainingEngine):
    """Local training engine combining FedProx proximal penalty with SCAFFOLD control variates."""

    def __init__(
        self,
        metrics: MetricsCalculatorPort,
        *,
        mu: float = 0.001,
        client_control_variate: dict[str, Tensor] | None = None,
        global_control_variate: dict[str, Tensor] | None = None,
    ) -> None:
        """Initialize hybrid engine.

        Args:
            metrics: Metrics calculator for per-epoch validation scoring.
            mu: Proximal regularization coefficient (mu >= 0).
            client_control_variate: Initial client control variate c_k.
            global_control_variate: Initial global control variate c.
        """
        super().__init__(
            metrics,
            client_control_variate=client_control_variate,
            global_control_variate=global_control_variate,
        )
        if mu < 0.0:
            raise ValueError(f"mu must be non-negative; got {mu}")
        self._mu = mu

    @property
    def mu(self) -> float:
        """Proximal coefficient."""
        return self._mu

    def _train_one_epoch(
        self,
        config: TrainingConfig,
        model: Any,
        train: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """Execute one training epoch with proximal penalty and control-variate correction."""
        if self._initial_global_parameters is None:
            self._initial_global_parameters = self._snapshot(model)

        device = next(model.parameters()).device
        model.train()

        c_k_device: dict[str, Tensor] = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in self._c_k.items()
        }
        c_device: dict[str, Tensor] = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in self._c.items()
        }

        cached_global = {}
        if self._initial_global_parameters is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._initial_global_parameters:
                    global_val = self._initial_global_parameters[name]
                    if isinstance(global_val, Tensor):
                        t = global_val.detach().to(device=device, dtype=param.dtype)
                    elif isinstance(global_val, np.ndarray):
                        t = torch.from_numpy(global_val).to(device=device, dtype=param.dtype)
                    else:
                        t = torch.as_tensor(global_val, device=device, dtype=param.dtype)
                    cached_global[name] = t

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
            bce_loss = criterion(logits, labels)

            prox_penalty = compute_proximal_loss(
                model, cached_global, self._mu, device
            )
            total_loss = bce_loss + prox_penalty

            total_loss.backward()

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
            loss_sum += float(total_loss.detach()) * count
            seen += count
            if index % report_every == 0 or index == total_batches:
                logger.debug(
                    "  [FedProx-SCAFFOLD mu=%.4f] epoch %d: batch %d/%d, running loss %.6f",
                    self._mu,
                    epoch,
                    index,
                    total_batches,
                    loss_sum / seen,
                )

        return loss_sum / seen if seen else 0.0

