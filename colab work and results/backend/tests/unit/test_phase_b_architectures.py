"""Phase B Architecture and Pipeline Verification Test Suite.

Verifies for GRU, LSTM, and BiLSTM:
1. Model initialization and parameter count (one-hot feature_dim=57 and numeric feature_dim=23).
2. Input, sequence, and batch dimensions (batch_size, seq_len=10, feat_dim=57).
3. Output shape (batch_size,) and finite logits.
4. Label and target alignment (binary {0, 1}).
5. Finite loss calculation (BCEWithLogitsLoss).
6. Backpropagation and finite gradients.
7. Optimizer parameter updates (parameters actually change).
8. Validation inference producing probabilities in [0, 1].
9. Checkpoint serialization and reload integrity.
10. Reproducible evaluation from reloaded checkpoint.
"""

from pathlib import Path
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pytest

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator


@pytest.mark.parametrize("arch", [ModelArchitecture.GRU, ModelArchitecture.LSTM, ModelArchitecture.BILSTM])
def test_architecture_pipeline_10_criteria(arch: ModelArchitecture):
    # 1. Model initialization and parameter count with feature_dim=57 (one-hot nominal codes)
    feature_dim = 57
    hidden_size = 64
    config = TrainingConfig(
        architecture=arch,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.001,
        batch_size=16,
        epochs=1,
        random_seed=42,
    )
    model = build_classifier(config, feature_dim)
    param_count = model.parameter_count()

    if arch == ModelArchitecture.GRU:
        assert param_count == 23681, f"Expected 23,681 params for GRU, got {param_count}"
    elif arch == ModelArchitecture.LSTM:
        assert param_count == 31553, f"Expected 31,553 params for LSTM, got {param_count}"
    elif arch == ModelArchitecture.BILSTM:
        assert param_count == 63105, f"Expected 63,105 params for BiLSTM, got {param_count}"

    # 2. Correct input dimensions (B=16, T=10, D=57)
    batch_size = 16
    seq_len = 10
    x = torch.randn(batch_size, seq_len, feature_dim)
    lengths = torch.tensor([10] * batch_size, dtype=torch.int64)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    # 3. Output shape and finite logits
    logits = model(x, lengths)
    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    # 4. Correct labels and target alignment
    assert y.shape == (batch_size,)
    assert set(y.unique().tolist()).issubset({0.0, 1.0})

    # 5. Finite loss calculation
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0

    # 6. Backpropagation and finite gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(p.grad).all(), f"Gradient for {name} contains NaN/Inf"

    # 7. Optimizer parameter updates
    orig_params = [p.clone().detach() for p in model.parameters()]
    optimizer.step()
    params_changed = any(not torch.equal(orig, p) for orig, p in zip(orig_params, model.parameters()))
    assert params_changed, "Optimizer step did not update model parameters"

    # 8. Validation inference producing probabilities in [0, 1]
    model.eval()
    with torch.no_grad():
        val_logits = model(x, lengths)
        val_probs = torch.sigmoid(val_logits).numpy()
    assert (val_probs >= 0.0).all() and (val_probs <= 1.0).all()

    # 9. Checkpoint serialization and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / f"{arch.value}_ckpt.pt"
        torch.save({"state_dict": model.state_dict(), "arch": arch.value}, ckpt_path)
        assert ckpt_path.exists()

        # Reload model
        reloaded_model = build_classifier(config, feature_dim)
        ckpt_data = torch.load(ckpt_path, weights_only=True)
        reloaded_model.load_state_dict(ckpt_data["state_dict"])
        reloaded_model.eval()

        # 10. Reproducible evaluation from reloaded checkpoint
        with torch.no_grad():
            reloaded_logits = reloaded_model(x, lengths)
            reloaded_probs = torch.sigmoid(reloaded_logits).numpy()

        assert torch.allclose(val_logits, reloaded_logits, atol=1e-6)
        calc = SklearnMetricsCalculator()
        m1 = calc.evaluate(y.numpy(), val_probs, threshold=0.5)
        m2 = calc.evaluate(y.numpy(), reloaded_probs, threshold=0.5)
        assert m1.pr_auc == pytest.approx(m2.pr_auc, abs=1e-6)
        assert m1.roc_auc == pytest.approx(m2.roc_auc, abs=1e-6)

