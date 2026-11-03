"""Pilot run of Integrated Gradients for HI-Small and LI-Medium models."""

import sys
from pathlib import Path
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from application.dto.training import TrainingConfig
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.data.sequences.mmap_sequence_provider import MmapSequenceProvider
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.explainability.integrated_gradients import SequenceIntegratedGradients

def run_pilot_model(dataset_name: str, ckpt_path: Path, arch: ModelArchitecture, hidden_size: int, is_mmap: bool = False):
    print(f"\n=======================================================")
    print(f"PILOT RUN: {dataset_name} ({arch.value})")
    print(f"=======================================================")
    
    data_dir = Path(f"data/processed/ibm_aml/{dataset_name}")
    cfg = TrainingConfig(architecture=arch, batch_size=1024, hidden_size=hidden_size, num_layers=1)
    
    if is_mmap and (data_dir / "mmap").exists():
        provider = MmapSequenceProvider(data_dir)
    else:
        provider = ProcessedSequenceProvider(data_dir)
        
    t0 = time.time()
    ds_val = provider.load(SplitName.VALIDATION, cfg)
    print(f"Loaded validation split in {time.time()-t0:.2f}s: {ds_val.sequence_count} sequences")
    
    # Stratified sampling of 10 pilot examples: 5 negative (is_laundering=0), 5 positive (is_laundering=1)
    val_labels = ds_val.labels.astype(int)
    pos_indices = np.flatnonzero(val_labels == 1)
    neg_indices = np.flatnonzero(val_labels == 0)
    
    np.random.seed(42)
    selected_pos = np.random.choice(pos_indices, size=5, replace=False)
    selected_neg = np.random.choice(neg_indices, size=5, replace=False)
    pilot_indices = np.concatenate([selected_neg, selected_pos])
    
    print(f"Sampled 10 validation instances: 5 negative, 5 positive")
    
    # Extract sequences using _assemble
    target_rows = ds_val.target_rows[pilot_indices]
    batch = ds_val._assemble(target_rows)
    x_batch = torch.from_numpy(batch.features).float()
    l_batch = torch.from_numpy(batch.lengths).long()
    
    print(f"Batch shapes: features={x_batch.shape}, lengths={l_batch.shape}")
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_classifier(cfg, ds_val.feature_dimension)
    model.load_state_dict(ckpt["parameters"])
    model.eval()
    
    # Initialize explainer with target='logit'
    explainer = SequenceIntegratedGradients(model, target="logit")
    
    t_start = time.time()
    attributions, diag = explainer.attribute(x_batch, l_batch, steps=50)
    t_end = time.time()
    
    print(f"\n--- Pilot Diagnostic Summary ---")
    print(f"Attribution shape: {attributions.shape}")
    print(f"Computation time (10 samples, 50 steps): {t_end - t_start:.3f} seconds ({((t_end-t_start)/10)*1000:.1f} ms/sample)")
    print(f"Max absolute completeness residual: {diag['max_abs_residual']:.6e}")
    print(f"Mean absolute completeness residual: {diag['mean_abs_residual']:.6e}")
    print(f"Max relative completeness residual: {diag['max_rel_residual']:.6%}")
    print(f"Mean relative completeness residual: {diag['mean_rel_residual']:.6%}")
    
    # Verify padding positions have exact 0 attribution
    padding_mask = torch.arange(10).unsqueeze(0) >= l_batch.unsqueeze(1)
    padding_attrs = attributions[padding_mask]
    max_pad_attr = padding_attrs.abs().max().item() if padding_attrs.numel() > 0 else 0.0
    print(f"Max attribution in padding positions (must be exactly 0): {max_pad_attr:.8e}")
    
    # Show sample predictions and attributions
    print(f"\n--- Per-Sample Attribution vs Logit Delta Check ---")
    for i in range(10):
        cls_name = "Laundering" if i >= 5 else "Legitimate"
        idx = pilot_indices[i]
        length = l_batch[i].item()
        p_val = torch.sigmoid(torch.tensor(diag['f_input'][i])).item()
        p_base = torch.sigmoid(torch.tensor(diag['f_baseline'][i])).item()
        f_diff = diag['f_diff'][i]
        ig_sum = diag['attr_sum'][i]
        res = diag['abs_residuals'][i]
        rel_res = diag['rel_residuals'][i]
        print(f"[{i:02d}] Idx={idx:<7d} | {cls_name:<10s} | L={length:2d} | P(y=1)={p_val:.4f} | F(x)-F(x0)={f_diff:+.4f} | Sum(IG)={ig_sum:+.4f} | Res={res:.2e} ({rel_res:.4%})")

    return {
        "dataset": dataset_name,
        "runtime_seconds": t_end - t_start,
        "ms_per_sample": ((t_end-t_start)/10)*1000,
        "max_abs_residual": diag["max_abs_residual"],
        "mean_abs_residual": diag["mean_abs_residual"],
        "max_rel_residual": diag["max_rel_residual"],
        "padding_leakage": max_pad_attr
    }

if __name__ == "__main__":
    res_hi = run_pilot_model(
        dataset_name="HI-Small",
        ckpt_path=Path("results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt"),
        arch=ModelArchitecture.LSTM,
        hidden_size=64,
        is_mmap=False
    )
    
    res_li = run_pilot_model(
        dataset_name="LI-Medium",
        ckpt_path=Path("results/LI_Medium_Results/checkpoints/config_5/global_best.pt"),
        arch=ModelArchitecture.BILSTM,
        hidden_size=64,
        is_mmap=True
    )
    
    print("\n=======================================================")
    print("ALL PILOTS FINISHED SUCCESSFULLY")
    print("=======================================================")
