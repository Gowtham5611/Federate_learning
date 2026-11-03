"""Phase D Diagnostic Script: Reproduce and analyze FedProx mu=0.1 model collapse across loss variants.

Evaluates 3 experimental variants on real transaction sequences:
Variant A: Baseline BCEWithLogitsLoss (unweighted)
Variant B: Positive-Class Weighted BCEWithLogitsLoss (pos_weight = 1324.45 derived strictly from train split)
Variant C: Binary Focal Loss with logits (alpha=0.25, gamma=2.0)

Records per client and round:
- Task loss, proximal penalty, total loss
- Gradient norms before/after clipping
- Local parameter-update norms
- Distance from round-start global parameters (drift)
- Probability minimum, maximum, mean, std
- Final layer bias value
- Validation PR-AUC, ROC-AUC, Recall at 0.5, and Recall at frozen threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.data.partitioners.non_iid_partitioner import NonIIDPartitioner
from infrastructure.data.partitioners.client_sequence_provider import ClientSequenceProvider
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import compute_proximal_loss
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.loss_functions import build_loss_criterion
from infrastructure.models.pytorch.torch_training_engine import _to_torch


def run_fedprox_loss_diagnostic(
    loss_variant: str,
    processed_dir: Path,
    output_dir: Path,
    *,
    row_limit: int = 5000,
    rounds: int = 2,
    mu: float = 0.1,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    provider = ProcessedSequenceProvider(processed_dir)
    meta = provider._metadata
    metrics_calc = SklearnMetricsCalculator()
    
    # 1. Read subset of real training data
    train_path = meta.split_path(SplitName.TRAIN.value)
    train_cols = list(meta.columns_to_read())
    train_dtypes = provider._read_dtypes()
    train_df = pd.read_csv(train_path, usecols=train_cols, dtype=train_dtypes, nrows=row_limit)[train_cols]
    
    # Non-IID Partition
    partitioner = NonIIDPartitioner(seed=seed)
    client_frames, _ = partitioner.partition(
        train_df,
        target_column=meta.target_column,
        account_column=meta.sender_column,
    )
    
    # Build datasets
    config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=64,
        num_layers=1,
        learning_rate=0.001,
        batch_size=256,
        epochs=1,
        random_seed=seed,
    )
    client_provider = ClientSequenceProvider(meta)
    client_datasets = {
        cid: client_provider.build_client_dataset(cdf, config, client_id=cid, split=SplitName.TRAIN)
        for cid, cdf in client_frames.items()
    }
    
    # Load subset of validation data
    val_dataset = provider.load(SplitName.VALIDATION, config)
    # Subset validation dataset for diagnostic speed
    val_subset_len = min(10000, val_dataset.sequence_count)
    val_batches = list(val_dataset.batches(config.batch_size, shuffle=False))
    val_features_list, val_lengths_list, val_labels_list = [], [], []
    collected = 0
    for b in val_batches:
        f, l, y = _to_torch(b, torch.device("cpu"))
        val_features_list.append(f)
        val_lengths_list.append(l)
        val_labels_list.append(y)
        collected += len(b)
        if collected >= val_subset_len:
            break
            
    val_features = torch.cat(val_features_list, dim=0)[:val_subset_len]
    val_lengths = torch.cat(val_lengths_list, dim=0)[:val_subset_len]
    val_labels = torch.cat(val_labels_list, dim=0)[:val_subset_len]
    
    # Build criterion
    criterion = build_loss_criterion(loss_variant)
    
    feature_dim = client_datasets["client_1"].feature_dimension
    global_model = build_classifier(config, feature_dim)
    current_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    
    history = []
    
    for r in range(1, rounds + 1):
        round_start_state = {k: v.detach().clone() for k, v in current_state.items()}
        client_states = {}
        client_counts = {}
        client_diagnostics = {}
        
        for cid, c_dataset in client_datasets.items():
            local_model = build_classifier(config, feature_dim)
            local_model.load_state_dict(round_start_state)
            local_model.train()
            
            optimizer = torch.optim.Adam(local_model.parameters(), lr=config.learning_rate)
            
            task_losses, prox_losses, pre_clip_norms, post_clip_norms = [], [], [], []
            
            batches = list(c_dataset.batches(config.batch_size, shuffle=True, seed=seed + r))
            for b in batches:
                features, lengths, labels = _to_torch(b, torch.device("cpu"))
                optimizer.zero_grad()
                logits = local_model(features, lengths)
                task_loss = criterion(logits, labels)
                prox_loss = compute_proximal_loss(local_model, round_start_state, mu=mu, device=torch.device("cpu"))
                total_loss = task_loss + prox_loss
                total_loss.backward()
                
                # Measure gradient norms before clipping
                total_norm_sq = 0.0
                for p in local_model.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.data.norm(2).item() ** 2
                pre_clip = np.sqrt(total_norm_sq)
                pre_clip_norms.append(pre_clip)
                
                # Clip grad
                nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                
                post_norm_sq = 0.0
                for p in local_model.parameters():
                    if p.grad is not None:
                        post_norm_sq += p.grad.data.norm(2).item() ** 2
                post_clip = np.sqrt(post_norm_sq)
                post_clip_norms.append(post_clip)
                
                optimizer.step()
                task_losses.append(task_loss.item())
                prox_losses.append(prox_loss.item())
                
            local_state = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
            client_states[cid] = local_state
            client_counts[cid] = c_dataset.sequence_count
            
            # Distance from round start
            l2_dist_sq = sum(
                torch.sum((local_state[k] - round_start_state[k]) ** 2).item()
                for k in local_state if "num_batches_tracked" not in k
            )
            l2_dist = np.sqrt(l2_dist_sq)
            
            # Inspect final layer bias
            last_bias = local_state.get("head.bias") or local_state.get("fc.bias") or local_state.get("classifier.bias")
            bias_val = float(last_bias[0]) if last_bias is not None else float(list(local_state.values())[-1][0])
            
            client_diagnostics[cid] = {
                "mean_task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
                "mean_prox_penalty": float(np.mean(prox_losses)) if prox_losses else 0.0,
                "mean_pre_clip_norm": float(np.mean(pre_clip_norms)) if pre_clip_norms else 0.0,
                "mean_post_clip_norm": float(np.mean(post_clip_norms)) if post_clip_norms else 0.0,
                "l2_drift_from_global": float(l2_dist),
                "final_layer_bias": bias_val,
            }
            
        # Aggregation
        weights = compute_aggregation_weights(client_counts)
        current_state = aggregate_fedavg(client_states, weights)
        global_model.load_state_dict(current_state)
        global_model.eval()
        
        # Validation evaluation
        with torch.no_grad():
            val_logits = global_model(val_features, val_lengths)
            val_probs = torch.sigmoid(val_logits).numpy()
            
        y_true = val_labels.numpy()
        val_eval_05 = metrics_calc.evaluate(y_true, val_probs, threshold=0.5)
        frozen_thresh = metrics_calc.select_threshold(y_true, val_probs, split=SplitName.VALIDATION)
        val_eval_frozen = metrics_calc.evaluate(y_true, val_probs, threshold=frozen_thresh)
        
        round_summary = {
            "round": r,
            "loss_variant": loss_variant,
            "mu": mu,
            "clients": client_diagnostics,
            "val_pr_auc": float(val_eval_05.pr_auc),
            "val_roc_auc": float(val_eval_05.roc_auc),
            "recall_at_0_5": float(val_eval_05.recall),
            "f1_at_0_5": float(val_eval_05.f1),
            "frozen_threshold": float(frozen_thresh),
            "recall_at_frozen": float(val_eval_frozen.recall),
            "f1_at_frozen": float(val_eval_frozen.f1),
            "prob_stats": {
                "min": float(np.min(val_probs)),
                "max": float(np.max(val_probs)),
                "mean": float(np.mean(val_probs)),
                "std": float(np.std(val_probs)),
            }
        }
        history.append(round_summary)
        print(f"[{loss_variant.upper()} Round {r}] PR-AUC: {val_eval_05.pr_auc:.5f}, "
              f"ROC-AUC: {val_eval_05.roc_auc:.5f}, Rec@0.5: {val_eval_05.recall:.4f}, "
              f"Prob Mean: {np.mean(val_probs):.6f}, Thresh: {frozen_thresh:.4f}, Rec@Thresh: {val_eval_frozen.recall:.4f}")
        
    output_file = output_dir / f"fedprox_diagnostic_{loss_variant}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    return history


if __name__ == "__main__":
    p_dir = Path("data/processed/ibm_aml/HI-Small")
    out_dir = Path("artifacts/final_baseline/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Running Variant A: Baseline BCE (unweighted) ===")
    run_fedprox_loss_diagnostic("bce", p_dir, out_dir, row_limit=3000, rounds=2, mu=0.1)
    
    print("\n=== Running Variant B: Weighted BCE (pos_weight=1324.45) ===")
    run_fedprox_loss_diagnostic("weighted_bce", p_dir, out_dir, row_limit=3000, rounds=2, mu=0.1)
    
    print("\n=== Running Variant C: Binary Focal Loss (alpha=0.25, gamma=2.0) ===")
    run_fedprox_loss_diagnostic("focal", p_dir, out_dir, row_limit=3000, rounds=2, mu=0.1)
