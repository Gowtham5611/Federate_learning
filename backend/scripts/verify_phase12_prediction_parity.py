"""Verify prediction parity for Phase 12 explanation candidate models."""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from application.dto.training import TrainingConfig
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.data.sequences.mmap_sequence_provider import MmapSequenceProvider
from infrastructure.models.pytorch.sequence_classifier import build_classifier

def verify_hi_small():
    print("--- Verifying HI-Small Prediction Parity ---")
    data_dir = Path("data/processed/ibm_aml/HI-Small")
    ckpt_path = Path("results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt")
    saved_npz_path = Path("results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz")
    
    saved_data = np.load(saved_npz_path)
    saved_probs = saved_data["probabilities"]
    saved_labels = saved_data["labels"]
    print(f"Loaded saved predictions: {len(saved_probs)} rows, min={saved_probs.min():.6e}, max={saved_probs.max():.6e}")
    
    # Load dataset using same config
    cfg = TrainingConfig(architecture=ModelArchitecture.LSTM, batch_size=8192, hidden_size=64, num_layers=1)
    provider = ProcessedSequenceProvider(data_dir)
    ds_val = provider.load(SplitName.VALIDATION, cfg)
    print(f"Loaded validation split: {ds_val.sequence_count} sequences, {ds_val.positive_count} positives")
    
    # Verify label parity
    val_labels = ds_val.labels.astype(int)
    label_matches = np.array_equal(val_labels, saved_labels)
    print(f"Label alignment exact match: {label_matches}")
    if not label_matches:
        raise ValueError("HI-Small validation labels do not match saved labels!")

    # Load model checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_classifier(cfg, ds_val.feature_dimension)
    model.load_state_dict(ckpt["parameters"])
    model.eval()

    # Test first 10,000 samples for fast bitwise/numerical check
    sample_size = 10000
    recomputed_probs = []
    current_count = 0
    with torch.no_grad():
        for batch in ds_val.batches(batch_size=2048, shuffle=False):
            bx = torch.from_numpy(batch.features)
            bl = torch.from_numpy(batch.lengths)
            logits = model(bx, bl).view(-1)
            p = torch.sigmoid(logits).numpy().astype(np.float32)
            recomputed_probs.append(p)
            current_count += len(p)
            if current_count >= sample_size:
                break
                
    recomputed_probs = np.concatenate(recomputed_probs)[:sample_size]
    reference_slice = saved_probs[:sample_size]
    
    abs_diff = np.abs(recomputed_probs - reference_slice)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"HI-Small Sample ({sample_size} sequences):")
    print(f"  Max Absolute Difference: {max_diff:.8e}")
    print(f"  Mean Absolute Difference: {mean_diff:.8e}")
    passed = max_diff < 1e-5
    print(f"  Parity Status: {'PASSED' if passed else 'FAILED'}")
    
    return {
        "dataset": "HI-Small",
        "model_name": "hi_small_lstm_fedavg_iid_seed42",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": "dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4",
        "architecture": "LSTM (unidirectional, hidden=64, layers=1)",
        "sample_size": sample_size,
        "total_validation_rows": len(saved_probs),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": 1e-5,
        "labels_matched": label_matches,
        "parity_status": "PASSED" if passed else "FAILED"
    }

def verify_li_medium():
    print("\n--- Verifying LI-Medium Prediction Parity ---")
    data_dir = Path("data/processed/ibm_aml/LI-Medium")
    ckpt_path = Path("results/LI_Medium_Results/checkpoints/config_5/global_best.pt")
    saved_npz_path = Path("results/phase7/LI_Medium/validation_predictions/iid_validation_predictions.npz")
    
    saved_data = np.load(saved_npz_path)
    saved_probs = saved_data["probabilities"]
    saved_labels = saved_data["labels"]
    print(f"Loaded saved predictions: {len(saved_probs)} rows, min={saved_probs.min():.6e}, max={saved_probs.max():.6e}")
    
    cfg = TrainingConfig(architecture=ModelArchitecture.BILSTM, batch_size=8192, hidden_size=64, num_layers=1)
    # Check if mmap or processed sequence provider
    if (data_dir / "mmap").exists():
        provider = MmapSequenceProvider(data_dir)
    else:
        provider = ProcessedSequenceProvider(data_dir)
        
    ds_val = provider.load(SplitName.VALIDATION, cfg)
    print(f"Loaded validation split: {ds_val.sequence_count} sequences, {ds_val.positive_count} positives")
    
    val_labels = ds_val.labels.astype(int)
    label_matches = np.array_equal(val_labels, saved_labels)
    print(f"Label alignment exact match: {label_matches}")
    if not label_matches:
        raise ValueError("LI-Medium validation labels do not match saved labels!")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_classifier(cfg, ds_val.feature_dimension)
    model.load_state_dict(ckpt["parameters"])
    model.eval()

    sample_size = 10000
    recomputed_probs = []
    current_count = 0
    with torch.no_grad():
        for batch in ds_val.batches(batch_size=2048, shuffle=False):
            bx = torch.from_numpy(batch.features)
            bl = torch.from_numpy(batch.lengths)
            logits = model(bx, bl).view(-1)
            p = torch.sigmoid(logits).numpy().astype(np.float32)
            recomputed_probs.append(p)
            current_count += len(p)
            if current_count >= sample_size:
                break
                
    recomputed_probs = np.concatenate(recomputed_probs)[:sample_size]
    reference_slice = saved_probs[:sample_size]
    
    abs_diff = np.abs(recomputed_probs - reference_slice)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"LI-Medium Sample ({sample_size} sequences):")
    print(f"  Max Absolute Difference: {max_diff:.8e}")
    print(f"  Mean Absolute Difference: {mean_diff:.8e}")
    passed = max_diff < 1e-5
    print(f"  Parity Status: {'PASSED' if passed else 'FAILED'}")
    
    return {
        "dataset": "LI-Medium",
        "model_name": "config_5_bilstm_fedavg_iid",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": "bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2",
        "architecture": "BiLSTM (bidirectional, hidden=64*2=128, layers=1)",
        "sample_size": sample_size,
        "total_validation_rows": len(saved_probs),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": 1e-5,
        "labels_matched": label_matches,
        "parity_status": "PASSED" if passed else "FAILED"
    }

if __name__ == "__main__":
    records = []
    res_hi = verify_hi_small()
    records.append(res_hi)
    
    res_li = verify_li_medium()
    records.append(res_li)
    
    out_csv = Path("results/phase12/phase12_model_verification.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved parity verification to {out_csv}")

