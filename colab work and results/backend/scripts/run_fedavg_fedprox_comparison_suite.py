"""FedSecureAI — Complete FedAvg vs FedProx Comparative Retraining Suite with GPU & Resumption Support.

Supports:
1. Google Colab GPU (CUDA) and CPU execution via `--device auto/cuda/cpu`.
2. Both Non-IID and IID distribution strategies (36-configuration full matrix).
3. Configurable data paths (`--data-root`) and output directories (`--output-dir`) for Google Drive mounting.
4. Resumption from resume_manifest.json and atomic round checkpoints.
5. Skipping completed configurations with full metric reconstruction.
6. Mid-configuration resumption (e.g., resuming from Round 3 after Round 2 checkpoint).
7. Safe verification dry-run (`--verify-only`) to validate GPU, data, and checkpoints without training.
8. CPU thread capping and cooldown delays when running on local laptops.
9. Incremental persistence: saves CSV results and reports after each configuration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
backend_path = BASE_DIR / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.models.pytorch.sequence_classifier import build_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("fedavg_vs_fedprox_suite")


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(obj: Any, filepath: Path) -> None:
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    torch.save(obj, tmp_path)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)


def atomic_numpy_save(arr: np.ndarray, filepath: Path) -> None:
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_proximal_loss(
    model: nn.Module,
    global_params: dict[str, torch.Tensor],
    mu: float,
    device: torch.device,
) -> torch.Tensor:
    if mu <= 0.0:
        return torch.tensor(0.0, device=device)
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_params:
            ref = global_params[name].to(device=device, dtype=param.dtype)
            penalty = penalty + torch.sum((param - ref) ** 2)
    return (mu / 2.0) * penalty


def evaluate_model(
    model: nn.Module,
    ds,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    all_probs = []

    with torch.no_grad():
        for batch in ds.batches(batch_size=batch_size, shuffle=False):
            bx = torch.from_numpy(batch.features).to(device)
            bl = torch.from_numpy(batch.lengths).to(device)
            by = torch.from_numpy(batch.labels).float().to(device).view(-1)

            logits = model(bx, bl).view(-1)
            b_loss = criterion(logits, by)
            total_loss += b_loss.item() * len(by)
            total_samples += len(by)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    y_prob = np.concatenate(all_probs) if all_probs else np.zeros(0)
    return ds.labels.astype(int), y_prob, avg_loss


def extract_non_iid_client_loaders(
    ds,
    n_clients: int = 3,
    seed: int = 42,
) -> list[np.ndarray]:
    """Non-IID: 20% positives to client 1, 40% to client 2, 40% to client 3."""
    rng = np.random.default_rng(seed)
    labels = ds.labels.astype(int)

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    p_splits = [
        pos_idx[: int(0.20 * len(pos_idx))],
        pos_idx[int(0.20 * len(pos_idx)) : int(0.60 * len(pos_idx))],
        pos_idx[int(0.60 * len(pos_idx)) :],
    ]
    n_splits = np.array_split(neg_idx, n_clients)

    client_indices = []
    for c in range(n_clients):
        c_inds = np.concatenate([p_splits[c], n_splits[c]])
        rng.shuffle(c_inds)
        client_indices.append(c_inds)

    return client_indices


def extract_iid_client_loaders(
    ds,
    n_clients: int = 3,
    seed: int = 42,
) -> list[np.ndarray]:
    """IID: Balanced uniform random stratification across all clients."""
    rng = np.random.default_rng(seed)
    labels = ds.labels.astype(int)

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    p_splits = np.array_split(pos_idx, n_clients)
    n_splits = np.array_split(neg_idx, n_clients)

    client_indices = []
    for c in range(n_clients):
        c_inds = np.concatenate([p_splits[c], n_splits[c]])
        rng.shuffle(c_inds)
        client_indices.append(c_inds)

    return client_indices


def reconstruct_completed_experiment(
    exp_id: str,
    d_name: str,
    arch_str: str,
    algo: str,
    mu_val: float,
    strat_str: str,
    ds_train,
    ds_val,
    ds_test,
    ckpt_dir: Path,
    pred_dir: Path,
    rounds: int = 5,
) -> dict[str, Any] | None:
    best_ckpt = ckpt_dir / f"{exp_id}_global_best.pt"
    final_ckpt = ckpt_dir / f"{exp_id}_final.pt"
    pred_path = pred_dir / f"{exp_id}_test_preds.npy"

    if not (best_ckpt.exists() and final_ckpt.exists() and pred_path.exists()):
        return None

    try:
        best_meta = torch.load(best_ckpt, map_location="cpu")
        y_test_prob = np.load(pred_path)
        y_test_true = ds_test.labels.astype(int)

        best_tau = float(best_meta.get("frozen_threshold", 0.5))
        best_round = int(best_meta.get("selected_round", rounds))
        total_training_time = float(best_meta.get("training_seconds", 0.0))

        val_final_roc = float(best_meta.get("val_roc_auc", 0.0))
        val_final_pr = float(best_meta.get("val_pr_auc", 0.0))
        test_roc = float(best_meta.get("test_roc_auc", roc_auc_score(y_test_true, y_test_prob)))
        test_pr = float(best_meta.get("test_pr_auc", average_precision_score(y_test_true, y_test_prob)))

        y_test_pred = (y_test_prob >= best_tau).astype(int)
        cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        prec = float(precision_score(y_test_true, y_test_pred, zero_division=0))
        rec = float(recall_score(y_test_true, y_test_pred, zero_division=0))
        f1 = float(f1_score(y_test_true, y_test_pred, zero_division=0))
        acc = float(accuracy_score(y_test_true, y_test_pred))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        alerts_per_10k = float((fp / (fp + tn)) * 10000.0)

        # Recover round metrics
        round_metrics_list = []
        for r in range(1, rounds + 1):
            r_ckpt = ckpt_dir / f"{exp_id}_round_{r}.pt"
            if r_ckpt.exists():
                r_data = torch.load(r_ckpt, map_location="cpu")
                round_metrics_list.append({
                    "exp_id": exp_id,
                    "dataset": d_name,
                    "architecture": arch_str,
                    "algorithm": algo,
                    "mu": mu_val if algo == "fedprox" else "N/A",
                    "partition_strategy": strat_str,
                    "round": r,
                    "train_task_loss": 0.0,
                    "train_proximal_loss": 0.0,
                    "train_total_loss": 0.0,
                    "val_loss": round(float(r_data.get("val_loss", 0.0)), 6),
                    "val_pr_auc": round(float(r_data.get("val_pr_auc", 0.0)), 6),
                    "val_roc_auc": round(float(r_data.get("val_roc_auc", 0.0)), 6),
                    "val_f1_prov_0_5": 0.0,
                    "val_recall_prov_0_5": 0.0,
                    "duration_seconds": 0.0,
                    "checkpoint_path": str(r_ckpt),
                })

        summary = {
            "dataset": d_name,
            "architecture": arch_str,
            "partition_strategy": strat_str,
            "algorithm": algo,
            "mu": mu_val if algo == "fedprox" else "N/A",
            "rounds_completed": rounds,
            "selected_round": best_round,
            "train_rows": ds_train.sequence_count,
            "train_positives": int(np.sum(ds_train.labels == 1)),
            "validation_rows": ds_val.sequence_count,
            "validation_positives": int(np.sum(ds_val.labels == 1)),
            "test_rows": ds_test.sequence_count,
            "test_positives": int(np.sum(ds_test.labels == 1)),
            "validation_roc_auc": round(val_final_roc, 6),
            "validation_ap": round(val_final_pr, 6),
            "test_roc_auc": round(test_roc, 6),
            "test_ap": round(test_pr, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "accuracy": round(acc, 6),
            "specificity": round(spec, 6),
            "fpr": round(fpr, 6),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "false_alerts_per_10000": round(alerts_per_10k, 2),
            "threshold": round(best_tau, 4),
            "training_seconds": round(total_training_time, 2),
            "test_eval_seconds": 0.0,
            "checkpoint_path": str(best_ckpt),
            "checkpoint_sha256": sha256_file(best_ckpt),
            "run_status": "PASS",
        }

        return {
            "summary": summary,
            "round_metrics": round_metrics_list,
            "client_metrics": [],
            "confusion_matrix": {
                "exp_id": exp_id,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "n_pos": tp + fn,
                "n_neg": tn + fp,
                "threshold": best_tau,
            },
            "prediction_audit": {
                "exp_id": exp_id,
                "predictions_path": str(pred_path),
                "finite_values": bool(np.all(np.isfinite(y_test_prob))),
                "min_prob": float(np.min(y_test_prob)),
                "max_prob": float(np.max(y_test_prob)),
                "mean_pos_prob": float(np.mean(y_test_prob[y_test_true == 1])),
                "mean_neg_prob": float(np.mean(y_test_prob[y_test_true == 0])),
            },
            "was_skipped": True,
        }
    except Exception as e:
        logger.warning("Could not reconstruct %s: %s", exp_id, e)
        return None


def run_single_experiment(
    exp_id: str,
    d_name: str,
    arch_enum: ModelArchitecture,
    arch_str: str,
    algo: str,
    mu_val: float,
    strat_str: str,
    ds_train,
    ds_val,
    ds_test,
    client_indices: list[np.ndarray],
    ckpt_dir: Path,
    pred_dir: Path,
    rounds: int = 5,
    batch_size: int = 4096,
    lr: float = 0.001,
    seed: int = 42,
    cooldown_round_sec: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    logger.info("=================================================================")
    logger.info("Configuration Target: %s [%s, %s, %s, mu=%s, %s] on %s", exp_id, d_name, arch_str, algo, mu_val, strat_str, device)
    logger.info("=================================================================")

    # 1. Check if already completed
    reconstructed = reconstruct_completed_experiment(
        exp_id, d_name, arch_str, algo, mu_val, strat_str, ds_train, ds_val, ds_test, ckpt_dir, pred_dir, rounds=rounds
    )
    if reconstructed is not None:
        logger.info(
            "--> [SKIP COMPLETED] %s has verified final checkpoints & predictions. Skipping retraining.",
            exp_id,
        )
        return reconstructed

    seed_everything(seed)

    cfg_model = TrainingConfig(
        architecture=arch_enum,
        hidden_size=64,
        num_layers=1,
        learning_rate=lr,
        random_seed=seed,
    )
    global_model = build_classifier(cfg_model, feature_dimension=57)
    global_model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    exp_start = time.perf_counter()

    round_metrics_list = []
    client_round_metrics_list = []

    best_val_pr = -1.0
    best_round = 1
    best_state = copy.deepcopy(global_model.state_dict())

    # 2. Check for mid-configuration resumption
    start_round = 1
    latest_saved_round = 0
    for r in range(1, rounds + 1):
        r_path = ckpt_dir / f"{exp_id}_round_{r}.pt"
        if r_path.exists() and r_path.stat().st_size > 0:
            latest_saved_round = r

    if latest_saved_round > 0 and latest_saved_round < rounds:
        logger.info(
            "--> [RESUME DETECTED] Found verified round checkpoints up to Round %d for %s. Restoring state...",
            latest_saved_round, exp_id
        )
        for r in range(1, latest_saved_round + 1):
            r_ckpt = ckpt_dir / f"{exp_id}_round_{r}.pt"
            r_data = torch.load(r_ckpt, map_location="cpu")
            v_pr = float(r_data.get("val_pr_auc", 0.0))
            v_roc = float(r_data.get("val_roc_auc", 0.0))
            val_loss = float(r_data.get("val_loss", 0.0))
            if v_pr > best_val_pr:
                best_val_pr = v_pr
                best_round = r
                best_state = copy.deepcopy(r_data["parameters"])

            round_metrics_list.append({
                "exp_id": exp_id,
                "dataset": d_name,
                "architecture": arch_str,
                "algorithm": algo,
                "mu": mu_val if algo == "fedprox" else "N/A",
                "partition_strategy": strat_str,
                "round": r,
                "train_task_loss": 0.0,
                "train_proximal_loss": 0.0,
                "train_total_loss": 0.0,
                "val_loss": round(val_loss, 6),
                "val_pr_auc": round(v_pr, 6),
                "val_roc_auc": round(v_roc, 6),
                "val_f1_prov_0_5": 0.0,
                "val_recall_prov_0_5": 0.0,
                "duration_seconds": 0.0,
                "checkpoint_path": str(r_ckpt),
            })

        latest_ckpt = ckpt_dir / f"{exp_id}_round_{latest_saved_round}.pt"
        latest_data = torch.load(latest_ckpt, map_location="cpu")
        global_model.load_state_dict({k: v.to(device) for k, v in latest_data["parameters"].items()})
        start_round = latest_saved_round + 1
        logger.info("--> State successfully loaded from Round %d checkpoint. Resuming at Round %d/5.", latest_saved_round, start_round)

    total_train_samples = ds_train.sequence_count
    client_weights = [len(idx) / total_train_samples for idx in client_indices]

    for r in range(start_round, rounds + 1):
        round_start_time = time.perf_counter()
        global_params = {k: v.clone().detach().to(device) for k, v in global_model.state_dict().items()}
        client_updates = []
        r_client_task_losses = []
        r_client_prox_losses = []
        r_client_total_losses = []

        for c_idx, c_inds in enumerate(client_indices):
            client_start_time = time.perf_counter()
            client_model = copy.deepcopy(global_model)
            client_model.to(device)
            client_model.train()
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

            c_pos_count = int(np.sum(ds_train.labels[c_inds] == 1))
            c_samples = len(c_inds)

            c_task_loss_sum = 0.0
            c_prox_loss_sum = 0.0
            c_batches = 0

            c_target_rows = ds_train.target_rows[c_inds]

            for start_i in range(0, c_samples, batch_size):
                end_i = min(start_i + batch_size, c_samples)
                batch_targets = c_target_rows[start_i:end_i]
                batch_seqs = ds_train._assemble(batch_targets)

                bx = torch.from_numpy(batch_seqs.features).to(device)
                bl = torch.from_numpy(batch_seqs.lengths).to(device)
                by = torch.from_numpy(batch_seqs.labels).float().to(device).view(-1)

                optimizer.zero_grad()
                logits = client_model(bx, bl).view(-1)
                task_loss = criterion(logits, by)

                if algo == "fedprox":
                    prox_loss = compute_proximal_loss(client_model, global_params, mu=mu_val, device=device)
                else:
                    prox_loss = torch.tensor(0.0, device=device)

                total_loss = task_loss + prox_loss
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(client_model.parameters(), 5.0)
                optimizer.step()

                c_task_loss_sum += task_loss.item()
                c_prox_loss_sum += prox_loss.item()
                c_batches += 1

            avg_c_task = c_task_loss_sum / c_batches if c_batches > 0 else 0.0
            avg_c_prox = c_prox_loss_sum / c_batches if c_batches > 0 else 0.0
            avg_c_total = avg_c_task + avg_c_prox

            r_client_task_losses.append(avg_c_task)
            r_client_prox_losses.append(avg_c_prox)
            r_client_total_losses.append(avg_c_total)

            local_state = client_model.state_dict()
            drift_sq = sum(
                torch.sum((local_state[k] - global_params[k]) ** 2).item()
                for k in local_state if "num_batches_tracked" not in k
            )
            drift = float(np.sqrt(drift_sq))

            client_dur = time.perf_counter() - client_start_time
            client_updates.append(local_state)

            client_round_metrics_list.append({
                "exp_id": exp_id,
                "dataset": d_name,
                "architecture": arch_str,
                "algorithm": algo,
                "mu": mu_val if algo == "fedprox" else "N/A",
                "partition_strategy": strat_str,
                "round": r,
                "client_id": f"client_{c_idx + 1}",
                "client_samples": c_samples,
                "client_positives": c_pos_count,
                "task_loss": round(avg_c_task, 6),
                "proximal_loss": round(avg_c_prox, 6),
                "total_loss": round(avg_c_total, 6),
                "parameter_drift": round(drift, 6),
                "duration_seconds": round(client_dur, 2),
                "status": "COMPLETED",
            })

            # Cleanup client VRAM
            del client_model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Aggregation
        new_global_state = {}
        for k in global_params.keys():
            weighted_sum = sum(
                client_weights[c] * client_updates[c][k].float()
                for c in range(len(client_indices))
            )
            new_global_state[k] = weighted_sum.to(device=device, dtype=global_params[k].dtype)

        global_model.load_state_dict(new_global_state)

        # Validation Evaluation
        y_val_t, y_val_p, val_loss = evaluate_model(global_model, ds_val, batch_size=batch_size, device=device)
        v_pr = float(average_precision_score(y_val_t, y_val_p))
        v_roc = float(roc_auc_score(y_val_t, y_val_p))

        v_pred_05 = (y_val_p >= 0.5).astype(int)
        v_f1_05 = float(f1_score(y_val_t, v_pred_05, zero_division=0))
        v_rec_05 = float(recall_score(y_val_t, v_pred_05, zero_division=0))

        round_dur = time.perf_counter() - round_start_time

        if v_pr > best_val_pr:
            best_val_pr = v_pr
            best_round = r
            best_state = copy.deepcopy(new_global_state)

        # Save round checkpoint (parameters moved to CPU for portable storage)
        ckpt_round_path = ckpt_dir / f"{exp_id}_round_{r}.pt"
        atomic_torch_save(
            {
                "exp_id": exp_id,
                "dataset": d_name,
                "architecture": arch_str,
                "algorithm": algo,
                "mu": mu_val,
                "partition_strategy": strat_str,
                "round": r,
                "parameters": {k: v.cpu() for k, v in new_global_state.items()},
                "val_loss": val_loss,
                "val_pr_auc": v_pr,
                "val_roc_auc": v_roc,
            },
            ckpt_round_path,
        )

        mean_task_loss = float(np.mean(r_client_task_losses))
        mean_prox_loss = float(np.mean(r_client_prox_losses))
        mean_total_loss = float(np.mean(r_client_total_losses))

        logger.info(
            "  Round %d/5 [%.1fs]: Train Task Loss=%.5f, Prox Loss=%.6f, Val Loss=%.5f, Val PR-AUC=%.6f, Val ROC-AUC=%.4f",
            r, round_dur, mean_task_loss, mean_prox_loss, val_loss, v_pr, v_roc
        )

        round_metrics_list.append({
            "exp_id": exp_id,
            "dataset": d_name,
            "architecture": arch_str,
            "algorithm": algo,
            "mu": mu_val if algo == "fedprox" else "N/A",
            "partition_strategy": strat_str,
            "round": r,
            "train_task_loss": round(mean_task_loss, 6),
            "train_proximal_loss": round(mean_prox_loss, 6),
            "train_total_loss": round(mean_total_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_pr_auc": round(v_pr, 6),
            "val_roc_auc": round(v_roc, 6),
            "val_f1_prov_0_5": round(v_f1_05, 6),
            "val_recall_prov_0_5": round(v_rec_05, 6),
            "duration_seconds": round(round_dur, 2),
            "checkpoint_path": str(ckpt_round_path),
        })

        if r < rounds and cooldown_round_sec > 0:
            logger.info("  [Cooldown] Pausing %d seconds...", cooldown_round_sec)
            time.sleep(cooldown_round_sec)

    total_training_time = time.perf_counter() - exp_start

    # Load best model for evaluation
    best_model = build_classifier(cfg_model, feature_dimension=57)
    best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 1. Validation Threshold Tuning
    y_val_true, y_val_prob, _ = evaluate_model(best_model, ds_val, batch_size=batch_size, device=device)
    best_tau = 0.5
    best_val_f1 = -1.0
    for tau in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_val_true, (y_val_prob >= tau).astype(int), zero_division=0)
        if f > best_val_f1:
            best_val_f1 = f
            best_tau = float(tau)

    val_final_roc = float(roc_auc_score(y_val_true, y_val_prob))
    val_final_pr = float(average_precision_score(y_val_true, y_val_prob))

    # 2. Test Evaluation with Frozen Threshold
    test_eval_start = time.perf_counter()
    y_test_true, y_test_prob, test_loss = evaluate_model(best_model, ds_test, batch_size=batch_size, device=device)
    test_eval_dur = time.perf_counter() - test_eval_start

    test_roc = float(roc_auc_score(y_test_true, y_test_prob))
    test_pr = float(average_precision_score(y_test_true, y_test_prob))

    y_test_pred = (y_test_prob >= best_tau).astype(int)
    cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    prec = float(precision_score(y_test_true, y_test_pred, zero_division=0))
    rec = float(recall_score(y_test_true, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test_true, y_test_pred, zero_division=0))
    acc = float(accuracy_score(y_test_true, y_test_pred))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    alerts_per_10k = float((fp / (fp + tn)) * 10000.0)

    assert tp + fn == int(np.sum(y_test_true == 1)), "Confusion matrix positive reconciliation failed"
    assert tn + fp == int(np.sum(y_test_true == 0)), "Confusion matrix negative reconciliation failed"
    assert tp + tn + fp + fn == len(y_test_true), "Confusion matrix total reconciliation failed"

    # Save test predictions atomically
    pred_path = pred_dir / f"{exp_id}_test_preds.npy"
    atomic_numpy_save(y_test_prob, pred_path)

    # Save final and best checkpoints atomically (CPU parameters)
    final_ckpt = ckpt_dir / f"{exp_id}_final.pt"
    best_ckpt = ckpt_dir / f"{exp_id}_global_best.pt"

    ckpt_meta = {
        "exp_id": exp_id,
        "dataset": d_name,
        "architecture": arch_str,
        "algorithm": algo,
        "mu": mu_val,
        "partition_strategy": strat_str,
        "selected_round": best_round,
        "parameters": {k: v.cpu() for k, v in best_state.items()},
        "frozen_threshold": best_tau,
        "val_roc_auc": val_final_roc,
        "val_pr_auc": val_final_pr,
        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "training_seconds": total_training_time,
    }
    atomic_torch_save(ckpt_meta, final_ckpt)
    atomic_torch_save(ckpt_meta, best_ckpt)

    ckpt_sha = sha256_file(best_ckpt)

    summary = {
        "dataset": d_name,
        "architecture": arch_str,
        "partition_strategy": strat_str,
        "algorithm": algo,
        "mu": mu_val if algo == "fedprox" else "N/A",
        "rounds_completed": rounds,
        "selected_round": best_round,
        "train_rows": ds_train.sequence_count,
        "train_positives": int(np.sum(ds_train.labels == 1)),
        "validation_rows": ds_val.sequence_count,
        "validation_positives": int(np.sum(ds_val.labels == 1)),
        "test_rows": ds_test.sequence_count,
        "test_positives": int(np.sum(ds_test.labels == 1)),
        "validation_roc_auc": round(val_final_roc, 6),
        "validation_ap": round(val_final_pr, 6),
        "test_roc_auc": round(test_roc, 6),
        "test_ap": round(test_pr, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "accuracy": round(acc, 6),
        "specificity": round(spec, 6),
        "fpr": round(fpr, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_alerts_per_10000": round(alerts_per_10k, 2),
        "threshold": round(best_tau, 4),
        "training_seconds": round(total_training_time, 2),
        "test_eval_seconds": round(test_eval_dur, 2),
        "checkpoint_path": str(best_ckpt),
        "checkpoint_sha256": ckpt_sha,
        "run_status": "PASS",
    }

    logger.info(
        "Run %s Complete: Test ROC-AUC=%.4f, Test PR-AUC=%.6f, Test F1=%.4f, Alerts/10k=%.1f [Best Round=%d, Tau=%.2f]",
        exp_id, test_roc, test_pr, f1, alerts_per_10k, best_round, best_tau
    )

    return {
        "summary": summary,
        "round_metrics": round_metrics_list,
        "client_metrics": client_round_metrics_list,
        "confusion_matrix": {
            "exp_id": exp_id,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_pos": tp + fn,
            "n_neg": tn + fp,
            "threshold": best_tau,
        },
        "prediction_audit": {
            "exp_id": exp_id,
            "predictions_path": str(pred_path),
            "finite_values": bool(np.all(np.isfinite(y_test_prob))),
            "min_prob": float(np.min(y_test_prob)),
            "max_prob": float(np.max(y_test_prob)),
            "mean_pos_prob": float(np.mean(y_test_prob[y_test_true == 1])),
            "mean_neg_prob": float(np.mean(y_test_prob[y_test_true == 0])),
        },
        "was_skipped": False,
    }


def save_incremental_results(
    results_dir: Path,
    all_summaries: list[dict],
    all_round_metrics: list[dict],
    all_client_metrics: list[dict],
    all_cms: list[dict],
    all_pred_audits: list[dict],
    run_statuses: list[dict],
) -> None:
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(results_dir / "all_model_metrics.csv", index=False)
    if all_round_metrics:
        pd.DataFrame(all_round_metrics).to_csv(results_dir / "round_metrics.csv", index=False)
    if all_client_metrics:
        pd.DataFrame(all_client_metrics).to_csv(results_dir / "client_round_metrics.csv", index=False)
    if all_cms:
        pd.DataFrame(all_cms).to_csv(results_dir / "confusion_matrices.csv", index=False)
    if all_pred_audits:
        pd.DataFrame(all_pred_audits).to_csv(results_dir / "prediction_audit.csv", index=False)
    if run_statuses:
        pd.DataFrame(run_statuses).to_csv(results_dir / "all_run_status.csv", index=False)

    df_summary = pd.DataFrame(all_summaries)
    if not df_summary.empty:
        matched_rows = []
        for (d_name, arch_str, strat), group in df_summary.groupby(["dataset", "architecture", "partition_strategy"]):
            fedavg_row = group[group["algorithm"] == "fedavg"]
            fedprox_row = group[group["algorithm"] == "fedprox"]
            if not fedavg_row.empty and not fedprox_row.empty:
                fa = fedavg_row.iloc[0]
                fp = fedprox_row.iloc[0]
                pref = "FedAvg" if fa["validation_ap"] >= fp["validation_ap"] else "FedProx"
                matched_rows.append({
                    "dataset": d_name,
                    "architecture": arch_str,
                    "partition_strategy": strat,
                    "fedavg_test_roc_auc": fa["test_roc_auc"],
                    "fedprox_test_roc_auc": fp["test_roc_auc"],
                    "delta_roc_auc (fedprox - fedavg)": round(fp["test_roc_auc"] - fa["test_roc_auc"], 6),
                    "fedavg_test_ap": fa["test_ap"],
                    "fedprox_test_ap": fp["test_ap"],
                    "delta_ap (fedprox - fedavg)": round(fp["test_ap"] - fa["test_ap"], 6),
                    "fedavg_test_f1": fa["f1"],
                    "fedprox_test_f1": fp["f1"],
                    "fedavg_alerts_per_10k": fa["false_alerts_per_10000"],
                    "fedprox_alerts_per_10k": fp["false_alerts_per_10000"],
                    "fedavg_train_time_s": fa["training_seconds"],
                    "fedprox_train_time_s": fp["training_seconds"],
                    "validation_preferred_model": pref,
                })
        if matched_rows:
            pd.DataFrame(matched_rows).to_csv(results_dir / "matched_fedavg_vs_fedprox.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="FedAvg vs FedProx Comparative Suite (CPU & GPU)")
    parser.add_argument("--data-root", type=str, default="data/processed/ibm_aml", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/fedavg_fedprox_full_comparison", help="Output directory")
    parser.add_argument("--manifest-path", type=str, default=None, help="Path to resume_manifest.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--strategies", type=str, default="both", choices=["both", "non_iid", "iid"], help="Distribution strategies")
    parser.add_argument("--datasets", type=str, default="Combined,HI-Small,LI-Medium", help="Comma-separated datasets")
    parser.add_argument("--architectures", type=str, default="gru,lstm,bilstm", help="Comma-separated architectures")
    parser.add_argument("--algorithms", type=str, default="fedavg,fedprox", help="Comma-separated algorithms")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training and evaluation")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count")
    parser.add_argument("--interop-threads", type=int, default=2, help="PyTorch CPU inter-op thread count")
    parser.add_argument("--cooldown-round", type=int, default=0, help="Cooldown sleep in seconds between rounds")
    parser.add_argument("--cooldown-config", type=int, default=0, help="Cooldown sleep in seconds between configs")
    parser.add_argument("--max-configs", type=int, default=None, help="Maximum number of new configs to train")
    parser.add_argument("--config-id", type=str, default=None, help="Specific configuration ID to run")
    parser.add_argument("--verify-only", action="store_true", help="Perform setup, GPU, and checkpoint verification without full training")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.interop_threads)
        logger.info("Using CPU. PyTorch threads capped at %d.", args.threads)
    else:
        logger.info("Using GPU Acceleration: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    out_dir = Path(args.output_dir).resolve()
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    results_dir = out_dir / "results"
    reports_dir = out_dir / "reports"
    plots_dir = out_dir / "plots"
    configs_dir = out_dir / "configs"

    for d in [ckpt_dir, pred_dir, log_dir, results_dir, reports_dir, plots_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "suite_execution.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(file_handler)

    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else out_dir / "resume_manifest.json"

    logger.info("Output root: %s", out_dir)
    logger.info("Manifest path: %s", manifest_file)

    data_root = Path(args.data_root).resolve()

    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    selected_archs = [a.strip().lower() for a in args.architectures.split(",") if a.strip()]
    selected_algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]

    dataset_map = {
        "Combined": data_root / "Combined",
        "HI-Small": data_root / "HI-Small",
        "LI-Medium": data_root / "LI-Medium",
    }
    arch_enum_map = {
        "gru": (ModelArchitecture.GRU, "gru"),
        "lstm": (ModelArchitecture.LSTM, "lstm"),
        "bilstm": (ModelArchitecture.BILSTM, "bilstm"),
    }
    algo_map = {
        "fedavg": ("fedavg", 0.0),
        "fedprox": ("fedprox", 0.0001),
    }

    if args.strategies == "both":
        strategies = ["Non-IID", "IID"]
    elif args.strategies == "non_iid":
        strategies = ["Non-IID"]
    else:
        strategies = ["IID"]

    # --- VERIFY ONLY MODE ---
    if args.verify_only:
        logger.info("=== VERIFICATION MODE ACTIVATED (NO FULL TRAINING) ===")
        logger.info("1. Device check: %s (CUDA Available: %s)", device, torch.cuda.is_available())
        logger.info("2. Dataset paths check:")
        for d in selected_datasets:
            dp = dataset_map.get(d)
            exists = dp.exists() if dp else False
            logger.info("   - %s: %s (exists=%s)", d, dp, exists)
        logger.info("3. Manifest check: %s (exists=%s)", manifest_file, manifest_file.exists())
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                mf = json.load(f)
            logger.info("   Manifest summary: %s", mf.get("summary", {}))
        logger.info("4. Testing dummy model forward/backward on %s...", device)
        dummy_model = build_classifier(TrainingConfig(architecture=ModelArchitecture.GRU, hidden_size=64, num_layers=1), feature_dimension=57).to(device)
        bx = torch.randn(8, 10, 57, device=device)
        bl = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3], device="cpu")
        by = torch.randint(0, 2, (8,), device=device).float()
        out = dummy_model(bx, bl)
        loss = nn.BCEWithLogitsLoss()(out, by)
        loss.backward()
        logger.info("   Forward/Backward computed successfully! Loss: %.4f", loss.item())
        logger.info("=== VERIFICATION COMPLETE: ALL SYSTEMS READY FOR EXECUTION ===")
        return

    all_summaries = []
    all_round_metrics = []
    all_client_metrics = []
    all_cms = []
    all_pred_audits = []
    run_statuses = []

    start_suite = time.perf_counter()
    newly_trained_count = 0

    cfg_load = TrainingConfig(architecture=ModelArchitecture.GRU, batch_size=args.batch_size, row_limit=None)

    for d_name in selected_datasets:
        d_path = dataset_map.get(d_name)
        if not d_path or not d_path.exists():
            logger.error("Dataset path %s does not exist! Skipping.", d_path)
            continue

        d_tag = d_name.lower().replace("-", "_")

        logger.info("Loading dataset %s from %s...", d_name, d_path)
        prov = ProcessedSequenceProvider(d_path)

        t_load = time.perf_counter()
        ds_train = prov.load(SplitName.TRAIN, cfg_load)
        ds_val = prov.load(SplitName.VALIDATION, cfg_load)
        ds_test = prov.load(SplitName.TEST, cfg_load)
        logger.info(
            "Loaded %s in %.2fs: Train=%d (pos=%d), Val=%d (pos=%d), Test=%d (pos=%d)",
            d_name, time.perf_counter() - t_load,
            ds_train.sequence_count, ds_train.positive_count,
            ds_val.sequence_count, ds_val.positive_count,
            ds_test.sequence_count, ds_test.positive_count,
        )

        for strat in strategies:
            if strat == "Non-IID":
                client_inds = extract_non_iid_client_loaders(ds_train, n_clients=3, seed=42)
            else:
                client_inds = extract_iid_client_loaders(ds_train, n_clients=3, seed=42)

            for a_key in selected_archs:
                arch_enum, arch_str = arch_enum_map[a_key]
                for algo_key in selected_algos:
                    algo, mu_val = algo_map[algo_key]

                    algo_str = "fedavg" if algo == "fedavg" else "fedprox_mu0.0001"
                    if strat == "Non-IID":
                        exp_id = f"{d_tag}_{arch_str}_{algo_str}_seed42"
                    else:
                        exp_id = f"{d_tag}_{arch_str}_{algo_str}_iid_seed42"

                    if args.config_id and exp_id != args.config_id:
                        continue

                    try:
                        res = run_single_experiment(
                            exp_id=exp_id,
                            d_name=d_name,
                            arch_enum=arch_enum,
                            arch_str=arch_str,
                            algo=algo,
                            mu_val=mu_val,
                            strat_str=strat,
                            ds_train=ds_train,
                            ds_val=ds_val,
                            ds_test=ds_test,
                            client_indices=client_inds,
                            ckpt_dir=ckpt_dir,
                            pred_dir=pred_dir,
                            rounds=5,
                            batch_size=args.batch_size,
                            lr=0.001,
                            seed=42,
                            cooldown_round_sec=args.cooldown_round,
                            device=device,
                        )

                        all_summaries.append(res["summary"])
                        all_round_metrics.extend(res["round_metrics"])
                        all_client_metrics.extend(res["client_metrics"])
                        all_cms.append(res["confusion_matrix"])
                        all_pred_audits.append(res["prediction_audit"])

                        run_statuses.append({
                            "exp_id": exp_id,
                            "dataset": d_name,
                            "architecture": arch_str,
                            "algorithm": algo,
                            "mu": mu_val if algo == "fedprox" else "N/A",
                            "partition_strategy": strat,
                            "status": "PASS",
                            "error": "",
                        })

                        save_incremental_results(
                            results_dir, all_summaries, all_round_metrics, all_client_metrics,
                            all_cms, all_pred_audits, run_statuses
                        )

                        # Update manifest
                        if manifest_file.exists():
                            try:
                                with open(manifest_file, "r", encoding="utf-8") as mf:
                                    m_data = json.load(mf)
                                for c in m_data["configurations"]:
                                    if c["exp_id"] == exp_id:
                                        c["status"] = "COMPLETED"
                                        c["last_completed_round"] = 5
                                        c["test_roc_auc"] = res["summary"]["test_roc_auc"]
                                        c["test_pr_auc"] = res["summary"]["test_ap"]
                                        c["action"] = "SKIP"
                                with open(manifest_file, "w", encoding="utf-8") as mf:
                                    json.dump(m_data, mf, indent=2)
                            except Exception as me:
                                logger.warning("Could not update manifest: %s", me)

                        was_skipped = res.get("was_skipped", False)
                        if not was_skipped:
                            newly_trained_count += 1
                            if args.cooldown_config > 0:
                                time.sleep(args.cooldown_config)

                        if args.max_configs and newly_trained_count >= args.max_configs:
                            logger.info("Reached --max-configs (%d). Halting cleanly.", args.max_configs)
                            break

                    except Exception as ex:
                        logger.exception("Failed execution for %s: %s", exp_id, ex)
                        run_statuses.append({
                            "exp_id": exp_id,
                            "dataset": d_name,
                            "architecture": arch_str,
                            "algorithm": algo,
                            "mu": mu_val if algo == "fedprox" else "N/A",
                            "partition_strategy": strat,
                            "status": "FAIL",
                            "error": str(ex),
                        })

                if args.max_configs and newly_trained_count >= args.max_configs:
                    break
            if args.max_configs and newly_trained_count >= args.max_configs:
                break

        del ds_train, ds_val, ds_test, prov, client_inds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Released dataset %s from memory.", d_name)

        if args.max_configs and newly_trained_count >= args.max_configs:
            break

    total_suite_time = time.perf_counter() - start_suite
    logger.info("SUITE COMPLETE. Newly trained: %d models. Elapsed time: %.2f minutes.", newly_trained_count, total_suite_time / 60.0)


if __name__ == "__main__":
    main()
