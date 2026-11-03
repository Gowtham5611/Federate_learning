"""Loss functions for extreme class imbalance in transaction fraud detection.

Implements:
1. Standard BCEWithLogitsLoss (Baseline).
2. Positive-Class Weighted BCEWithLogitsLoss (pos_weight derived strictly from training data).
3. Numerically stable Binary Focal Loss with logits (Lin et al., 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BinaryFocalLossWithLogits(nn.Module):
    """Numerically stable Binary Focal Loss operating directly on logits.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1 - p).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute binary focal loss.
        
        Args:
            logits: Predicted logits of shape (N,) or (N, 1).
            targets: Binary ground truth labels {0.0, 1.0} of matching shape.
        """
        targets = targets.view_as(logits).float()
        
        # Standard BCE loss per element without reduction
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Compute p_t
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Alpha balancing factor
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        
        focal_loss = alpha_t * modulating_factor * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_loss_criterion(
    loss_type: str = "bce",
    *,
    pos_weight_value: float | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Build criterion for training with numerical stability.
    
    Args:
        loss_type: 'bce' (unweighted), 'weighted_bce', or 'focal'.
        pos_weight_value: Weight for positive class (used if loss_type == 'weighted_bce').
        alpha: Balancing factor for focal loss.
        gamma: Focusing parameter for focal loss.
        device: Device to place tensors on.
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "weighted_bce":
        if pos_weight_value is None:
            # Training split HI-Small: 3,044,917 neg / 2,299 pos = 1324.45
            pos_weight_value = 1324.4528
        pw_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    elif loss_type == "focal":
        return BinaryFocalLossWithLogits(alpha=alpha, gamma=gamma, reduction="mean")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

