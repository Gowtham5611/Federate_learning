"""Unit tests for TCN, Transformer Encoder, and Residual MLP sequence classifiers."""

import pytest
import torch
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from infrastructure.models.pytorch.sequence_classifier import build_classifier


@pytest.mark.parametrize("arch", [ModelArchitecture.TCN, ModelArchitecture.TRANSFORMER, ModelArchitecture.RESIDUAL_MLP])
def test_advanced_classifier_forward_backward(arch: ModelArchitecture):
    """Test forward and backward pass, shapes, and gradients."""
    batch_size = 8
    seq_len = 10
    feat_dim = 57

    cfg = TrainingConfig(
        architecture=arch,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
    )
    model = build_classifier(cfg, feature_dimension=feat_dim)

    # Random features and lengths
    features = torch.randn(batch_size, seq_len, feat_dim)
    lengths = torch.tensor([10, 8, 5, 3, 2, 10, 7, 4], dtype=torch.int64)

    logits = model(features, lengths)
    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"

    # Verify backward pass
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"


@pytest.mark.parametrize("arch", [ModelArchitecture.TCN, ModelArchitecture.TRANSFORMER, ModelArchitecture.RESIDUAL_MLP])
def test_checkpoint_roundtrip(arch: ModelArchitecture, tmp_path):
    """Test checkpoint state_dict save and load."""
    feat_dim = 57
    cfg = TrainingConfig(architecture=arch, hidden_size=32, num_layers=2)
    model = build_classifier(cfg, feature_dimension=feat_dim)

    ckpt_path = tmp_path / f"{arch.value}_test.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model = build_classifier(cfg, feature_dimension=feat_dim)
    loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=False))

    # Verify weights identical
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)

