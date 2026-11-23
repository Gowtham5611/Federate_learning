"""Unit and regression tests for Phase 12: Explainability & Attributions."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from application.dto.training import TrainingConfig
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.explainability.integrated_gradients import SequenceIntegratedGradients

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "phase12"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(65536):
            h.update(b)
    return h.hexdigest()

# 1. Checkpoint and model contract verification
def test_checkpoint_and_model_contract():
    hi_ckpt = BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt"
    li_ckpt = BASE_DIR / "results/LI_Medium_Results/checkpoints/config_5/global_best.pt"
    
    assert hi_ckpt.exists(), f"HI-Small checkpoint missing: {hi_ckpt}"
    assert li_ckpt.exists(), f"LI-Medium checkpoint missing: {li_ckpt}"
    
    assert sha256(hi_ckpt) == "dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4"
    assert sha256(li_ckpt) == "bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2"

# 2. Input/output tensor shapes
def test_input_output_tensor_shapes():
    cfg_lstm = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=64, num_layers=1)
    model_lstm = build_classifier(cfg_lstm, feature_dimension=57)
    
    x = torch.randn(8, 10, 57)
    lengths = torch.tensor([10, 9, 8, 7, 6, 5, 4, 1], dtype=torch.int64)
    out = model_lstm(x, lengths)
    
    assert out.shape == torch.Size([8])
    assert model_lstm.parameter_count() == 31553

    cfg_bilstm = TrainingConfig(architecture=ModelArchitecture.BILSTM, hidden_size=64, num_layers=1)
    model_bilstm = build_classifier(cfg_bilstm, feature_dimension=57)
    out_bi = model_bilstm(x, lengths)
    assert out_bi.shape == torch.Size([8])
    assert model_bilstm.parameter_count() == 63105

# 3. Probability semantics
def test_probability_semantics():
    cfg = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=64, num_layers=1)
    model = build_classifier(cfg, feature_dimension=57)
    
    x = torch.randn(4, 10, 57)
    lengths = torch.tensor([10, 7, 4, 2], dtype=torch.int64)
    logits = model(x, lengths)
    probs = torch.sigmoid(logits)
    
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert torch.isfinite(probs).all()

# 4. Prediction parity verification artifact
def test_prediction_parity_verification_artifact():
    csv_path = RESULTS_DIR / "phase12_model_verification.csv"
    assert csv_path.exists(), f"Missing parity CSV: {csv_path}"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    for _, row in df.iterrows():
        assert row["parity_status"] == "PASSED"
        assert row["max_abs_diff"] < 1e-5
        assert row["labels_matched"] == True

# 5. Deterministic sampling reproducibility
def test_deterministic_sampling_reproducibility():
    np.random.seed(42)
    s1 = np.random.choice(1000, size=50, replace=False)
    np.random.seed(42)
    s2 = np.random.choice(1000, size=50, replace=False)
    assert np.array_equal(s1, s2)

# 6. Split separation integrity
def test_split_separation_integrity():
    csv_path = RESULTS_DIR / "phase12_sampling_provenance.csv"
    assert csv_path.exists(), f"Missing provenance CSV: {csv_path}"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 600 # 300 HI-Small, 300 LI-Medium
    
    # Verify split assignments
    bg_df = df[df["role"] == "background_baseline"]
    assert len(bg_df) == 200
    assert (bg_df["split"] == "train").all()
    
    exp_df = df[df["role"].isin(["validation_negative", "validation_positive"])]
    assert len(exp_df) == 400
    assert (exp_df["split"] == "validation").all()
    
    # Zero test split leakage
    assert (df["split"] != "test").all()

# 7. Attribution output dimensions and feature mapping
def test_attribution_output_dimensions_and_feature_mapping():
    csv_path = RESULTS_DIR / "phase12_global_feature_importance.csv"
    assert csv_path.exists(), f"Missing global importance CSV: {csv_path}"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 114 # 57 features x 2 datasets
    
    for ds in ["HI-Small", "LI-Medium"]:
        sub = df[df["dataset"] == ds]
        assert len(sub) == 57
        assert set(sub["rank"]) == set(range(1, 58))
        assert set(sub["feature_group"]).issubset({
            "Transaction Attributes", "Payment Format", "Currency Rails", 
            "Temporal Dynamics", "Sender Causal History", "Receiver Causal History", "Other"
        })

# 8. Numerical finiteness and completeness residual
def test_numerical_finiteness_and_completeness():
    json_path = RESULTS_DIR / "phase12_explainer_validation.json"
    assert json_path.exists(), f"Missing validation JSON: {json_path}"
    
    with open(json_path) as f:
        data = json.load(f)
        
    for ds_name in ["HI-Small", "LI-Medium"]:
        m = data["models"][ds_name]
        assert m["finite_values_all"] is True
        assert m["mean_rel_completeness_residual"] < 0.001 # < 0.1% mean relative residual
        assert m["padding_leakage_status"] == "PASSED (Zero Leakage)"

# 9. Zero padding leakage guarantee on synthetic inputs
def test_zero_padding_leakage_guarantee():
    cfg = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=32, num_layers=1)
    model = build_classifier(cfg, feature_dimension=57)
    explainer = SequenceIntegratedGradients(model, target="logit")
    
    x = torch.randn(4, 10, 57)
    lengths = torch.tensor([5, 8, 3, 10], dtype=torch.int64)
    attrs, diag = explainer.attribute(x, lengths, steps=20)
    
    assert attrs.shape == torch.Size([4, 10, 57])
    for b in range(4):
        L = lengths[b].item()
        if L < 10:
            # Positions L..9 must have exactly 0 attribution
            pad_slice = attrs[b, L:, :]
            assert torch.all(pad_slice == 0.0)

# 10. Synthetic Integrated Gradients completeness axiom
def test_synthetic_integrated_gradients_completeness_axiom():
    cfg = TrainingConfig(architecture=ModelArchitecture.BILSTM, hidden_size=16, num_layers=1)
    model = build_classifier(cfg, feature_dimension=10)
    explainer = SequenceIntegratedGradients(model, target="logit")
    
    x = torch.randn(3, 8, 10)
    lengths = torch.tensor([8, 5, 2], dtype=torch.int64)
    attrs, diag = explainer.attribute(x, lengths, steps=50)
    
    assert diag["mean_rel_residual"] < 0.01 # < 1% error
    assert diag["max_rel_residual"] < 0.05

# 11. Artifact preservation of previous phases
def test_artifact_preservation():
    p7_sum = BASE_DIR / "results/phase7/phase7_summary.csv"
    p8_rep = BASE_DIR / "results/phase8/phase8_report.json"
    p9_11 = BASE_DIR / "results/phase9_11_audit/phase9_11_audit_report.md"
    
    assert p7_sum.exists()
    assert p8_rep.exists()
    assert p9_11.exists()

