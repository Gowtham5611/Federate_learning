"""FedProx local training engine and proximal regularizer.

Computes:
    L_FedProx = BCEWithLogitsLoss(logits, labels) + (mu / 2) * ||w - w_global||^2
where w is the local client model parameter tensor and w_global is the parameter
tensor received from the server at the start of the round.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
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



def compute_proximal_loss(
    model: nn.Module,
    global_parameters: Mapping[str, Any],
    mu: float,
    device: torch.device,
) -> Tensor:
    """Calculate the proximal regularization term (mu / 2) * ||w - w_global||^2.

    Args:
        model: Local PyTorch model with parameters w.
        global_parameters: Dictionary of global parameters w_global from round start.
        mu: Proximal coefficient (mu >= 0).
        device: Device where model parameters reside.

    Returns:
        Scalar Tensor containing the proximal penalty.
    """
    if mu <= 0.0:
        return torch.tensor(0.0, device=device)

    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_parameters:
            global_val = global_parameters[name]
            if isinstance(global_val, Tensor):
                target = global_val.detach().to(device=device, dtype=param.dtype)
            elif isinstance(global_val, np.ndarray):
                target = torch.from_numpy(global_val).to(device=device, dtype=param.dtype)
            else:
                target = torch.as_tensor(global_val, device=device, dtype=param.dtype)
            penalty = penalty + torch.sum((param - target) ** 2)

    return (mu / 2.0) * penalty


class FedProxTrainingEngine(TorchTrainingEngine):
    """Local training engine implementing the FedProx proximal objective.

    Extends TorchTrainingEngine without modifying Phase 4 behavior.
    """

    def __init__(self, metrics: MetricsCalculatorPort, mu: float = 0.01) -> None:
        """Initialize FedProx engine.

        Args:
            metrics: Metrics calculator for per-epoch validation scoring.
            mu: FedProx proximal regularization coefficient (default: 0.01).
        """
        super().__init__(metrics)
        if mu < 0.0:
            raise ValueError(f"mu must be non-negative; got {mu}")
        self._mu = mu
        self._global_parameters: ParameterState | None = None

    @property
    def mu(self) -> float:
        """Proximal coefficient."""
        return self._mu

    def fit(
        self,
        config: TrainingConfig,
        train: SequenceDataset,
        validation: SequenceDataset,
        *,
        initial_parameters: ParameterState | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> FitOutcome:
        """Fit local client model using BCEWithLogitsLoss + proximal penalty.

        Captures initial_parameters as w_global for the proximal objective.
        """
        self._global_parameters = initial_parameters
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
        """Execute one training epoch with proximal regularization added to loss."""
        # If no initial_parameters were provided, snapshot model at start of epoch 1
        if self._global_parameters is None:
            self._global_parameters = self._snapshot(model)

        device = next(model.parameters()).device
        if device.type == "cuda":
            for p in model.parameters():
                if not p.is_cuda:
                    raise RuntimeError("Strict CUDA check failed in FedProx: model parameter is not on CUDA device!")
        model.train()

        cached_global = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._global_parameters:
                global_val = self._global_parameters[name]
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
            if device.type == "cuda":
                if not features.is_cuda or not labels.is_cuda:
                    raise RuntimeError("Strict CUDA check failed in FedProx: batch tensors are not on CUDA!")
            optimizer.zero_grad(set_to_none=True)
            try:
                logits = model(features, lengths)
                bce_loss = criterion(logits, labels)

                prox_penalty = compute_proximal_loss(
                    model, cached_global, self._mu, device
                )
                total_loss = bce_loss + prox_penalty

                total_loss.backward()
            except torch.cuda.OutOfMemoryError as oom:
                logger.error(
                    "CUDA Out-of-Memory in FedProx at epoch %d, batch %d/%d (batch size %d). "
                    "Halting execution safely without CPU fallback or data truncation.",
                    epoch, index, total_batches, config.batch_size,
                )
                raise RuntimeError(
                    f"CUDA Out-of-Memory error in FedProx with batch size {config.batch_size}. "
                    "Reduce --batch-size and resume training."
                ) from oom
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            count = len(batch)
            loss_sum += float(total_loss.detach()) * count
            seen += count
            if index % report_every == 0 or index == total_batches:
                logger.debug(
                    "  [FedProx mu=%.4f] epoch %d: batch %d/%d, running loss %.6f "
                    "(bce: %.6f, prox: %.6f)",
                    self._mu,
                    epoch,
                    index,
                    total_batches,
                    loss_sum / seen,
                    float(bce_loss.detach()),
                    float(prox_penalty.detach()),
                )

        return loss_sum / seen if seen else 0.0

