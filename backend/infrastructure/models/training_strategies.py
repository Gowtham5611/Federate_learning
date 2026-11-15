"""Class Imbalance and Training Strategies Module (Phase E).

Implements rigorous, research-defensible loss functions and sampling strategies for highly imbalanced AML data:
- Standard BCEWithLogitsLoss
- Class-Weighted BCEWithLogitsLoss (pos_weight = N_neg / N_pos computed strictly on train split)
- Focal Loss (Lin et al., 2017) operating directly on logits with gamma focusing parameter
- Training-only weighted sampling strategy

Strict Constraints:
- No resampling or class-weight calculation on validation or test sets.
- Logits consistency: PyTorch models return unactivated logits; sigmoid is applied only for probability metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


class FocalLoss(nn.Module):
    """Numerically stable Focal Loss for binary classification operating on logits.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground-truth class.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss directly from raw logits and binary targets in {0, 1}."""
        # Ensure target shape matches logits
        targets = targets.view_as(logits).float()
        
        # Binary cross entropy with logits (per-sample loss)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Probabilities
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        
        # Alpha balancing factor
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TrainingStrategyFactory:
    """Factory for selecting and parameterizing loss functions and samplers from training split statistics."""

    @staticmethod
    def create_loss_function(
        strategy_name: str,
        train_targets: Optional[np.ndarray] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Construct loss module and return configuration metadata."""
        strategy = strategy_name.lower().strip()
        metadata: Dict[str, Any] = {
            "strategy": strategy,
            "gamma": gamma,
            "alpha": alpha,
        }

        if strategy in ("weighted_bce", "class_weighted"):
            if train_targets is None or len(train_targets) == 0:
                raise ValueError("train_targets required to calculate class weighting ratio.")
            n_pos = int((train_targets == 1).sum())
            n_neg = int((train_targets == 0).sum())
            pos_weight_val = float(n_neg / max(1, n_pos))
            pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            metadata["pos_weight"] = round(pos_weight_val, 4)
            metadata["n_pos"] = n_pos
            metadata["n_neg"] = n_neg
            return criterion, metadata

        elif strategy in ("focal", "focal_loss"):
            criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
            metadata["focal_alpha"] = alpha
            metadata["focal_gamma"] = gamma
            return criterion, metadata

        elif strategy in ("standard", "bce", "standard_bce"):
            criterion = nn.BCEWithLogitsLoss()
            metadata["pos_weight"] = 1.0
            return criterion, metadata

        else:
            raise ValueError(
                f"Unknown training strategy '{strategy_name}'. "
                f"Valid options: ['standard_bce', 'class_weighted', 'focal_loss']."
            )

    @staticmethod
    def create_weighted_sampler(train_targets: np.ndarray) -> WeightedRandomSampler:
        """Create a training-only WeightedRandomSampler to balance batches during SGD."""
        targets = np.asarray(train_targets, dtype=int)
        class_counts = np.bincount(targets, minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler

