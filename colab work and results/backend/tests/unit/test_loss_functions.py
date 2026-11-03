"""Unit tests for cost-sensitive loss functions."""

import pytest
import torch
import torch.nn as nn
from infrastructure.models.pytorch.loss_functions import (
    BinaryFocalLossWithLogits,
    build_loss_criterion,
)


def test_loss_criteria_construction():
    bce = build_loss_criterion("bce")
    assert isinstance(bce, nn.BCEWithLogitsLoss)

    wbce = build_loss_criterion("weighted_bce", pos_weight_value=1324.45)
    assert isinstance(wbce, nn.BCEWithLogitsLoss)
    assert wbce.pos_weight is not None

    focal = build_loss_criterion("focal", alpha=0.25, gamma=2.0)
    assert isinstance(focal, BinaryFocalLossWithLogits)


def test_focal_loss_numerical_stability():
    focal = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
    
    # Extreme logits
    logits = torch.tensor([-50.0, 0.0, 50.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0])
    
    loss = focal(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_weighted_bce_penalizes_false_negatives():
    unweighted = nn.BCEWithLogitsLoss()
    weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
    
    # Positive target with low logit (false negative)
    logit = torch.tensor([-5.0])
    target = torch.tensor([1.0])
    
    l_unweighted = unweighted(logit, target).item()
    l_weighted = weighted(logit, target).item()
    
    assert l_weighted > 10.0 * l_unweighted

