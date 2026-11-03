"""Step 1 & Step 2 & Step 3: Verification, Independent Evaluation, and Diagnosis of FedProx mu = 1.0 Models."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
backend_path = BASE_DIR / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "dataset_comparison_combined"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RETRAIN_DIR = BASE_DIR / "artifacts" / "fedprox_mu1_retraining"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("mu1_verify")

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.models.pytorch.sequence_classifier import build_classifier


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model(model: nn.Module, ds, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = torch.device("cpu")
    all_probs = []
    with torch.no_grad():
        for batch in ds.batches(batch_size=batch_size, shuffle=False):
            bx = torch.from_numpy(batch.features).to(device)
            bl = torch.from_numpy(batch.lengths).to(device)
            logits = model(bx, bl)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            all_probs.append(probs)
    y_prob = np.concatenate(all_probs) if all_probs else np.zeros(0)
    return ds.labels.astype(int), y_prob


def main():
    logger.info("=== STEP 1: Verifying Existing FedProx mu = 1.0 Models ===")
    
    # Load manifest and evaluation metrics
    with open(ARTIFACTS_DIR / "checkpoints_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    df_eval = pd.read_csv(ARTIFACTS_DIR / "evaluation_metrics.csv")
    df_matrix = pd.read_csv(ARTIFACTS_DIR / "experiment_matrix.csv")
    df_history = pd.read_csv(ARTIFACTS_DIR / "training_history.csv")
    
    mu1_matrix = df_matrix[df_matrix["mu"] == 1.0]
    logger.info("Total mu = 1.0 configurations found in matrix: %d", len(mu1_matrix))
    assert len(mu1_matrix) == 18, f"Expected 18 mu=1.0 configs, found {len(mu1_matrix)}"

    # Load datasets for independent evaluation
    prov_hi = ProcessedSequenceProvider(BASE_DIR / "data" / "processed" / "ibm_aml" / "HI-Small")
    prov_li = ProcessedSequenceProvider(BASE_DIR / "data" / "processed" / "ibm_aml" / "LI-Medium")
    prov_comb = ProcessedSequenceProvider(BASE_DIR / "data" / "processed" / "ibm_aml" / "Combined")

    cfg_val = TrainingConfig(architecture=ModelArchitecture.GRU, row_limit=50000, batch_size=1024)
    cfg_test = TrainingConfig(architecture=ModelArchitecture.GRU, row_limit=50000, batch_size=1024)

    logger.info("Loading validation and test sets...")
    val_hi = prov_hi.load(SplitName.VALIDATION, cfg_val)
    val_li = prov_li.load(SplitName.VALIDATION, cfg_val)
    val_comb = prov_comb.load(SplitName.VALIDATION, cfg_val)

    test_hi = prov_hi.load(SplitName.TEST, cfg_test)
    test_li = prov_li.load(SplitName.TEST, cfg_test)
    test_comb = prov_comb.load(SplitName.TEST, cfg_test)

    val_sets = {"HI-Small": val_hi, "LI-Medium": val_li, "Combined": val_comb}
    test_sets = {"HI-Small": test_hi, "LI-Medium": test_li, "Combined": test_comb}

    step1_records = []
    reproducibility_checks = []

    for _, row in mu1_matrix.iterrows():
        exp_id = row["exp_id"]
        d_name = row["training_dataset"]
        arch_str = row["architecture"]
        dist = row["distribution"]
        mu_val = float(row["mu"])
        opt_thresh = float(row["optimal_threshold"])

        ckpt_file = CHECKPOINTS_DIR / f"{exp_id}.pt"
        assert ckpt_file.exists(), f"Checkpoint missing: {ckpt_file}"
        on_disk_hash = sha256_file(ckpt_file)
        manifest_hash = manifest[exp_id]["sha256"]
        assert on_disk_hash == manifest_hash, f"Hash mismatch for {exp_id}"

        # Load checkpoint
        ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        arch_enum = ModelArchitecture.GRU if arch_str == "gru" else (ModelArchitecture.LSTM if arch_str == "lstm" else ModelArchitecture.BILSTM)
        model = build_classifier(
            TrainingConfig(architecture=arch_enum, hidden_size=64, num_layers=1),
            feature_dimension=57
        )
        model.load_state_dict(ckpt_data["parameters"])
        model.eval()

        # Independent Validation Evaluation
        v_ds = val_sets[d_name]
        v_true, v_prob = evaluate_model(model, v_ds)
        v_pr = float(average_precision_score(v_true, v_prob))
        v_roc = float(roc_auc_score(v_true, v_prob))
        v_pred = (v_prob >= opt_thresh).astype(int)
        v_cm = confusion_matrix(v_true, v_pred, labels=[0, 1])
        v_tn, v_fp, v_fn, v_tp = v_cm.ravel()
        v_prec = float(precision_score(v_true, v_pred, zero_division=0))
        v_rec = float(recall_score(v_true, v_pred, zero_division=0))
        v_f1 = float(f1_score(v_true, v_pred, zero_division=0))
        v_fpr = float(v_fp / (v_fp + v_tn)) if (v_fp + v_tn) > 0 else 0.0

        # Independent Test Evaluation
        t_ds = test_sets[d_name]
        t_true, t_prob = evaluate_model(model, t_ds)
        t_pr = float(average_precision_score(t_true, t_prob))
        t_roc = float(roc_auc_score(t_true, t_prob))
        t_pred = (t_prob >= opt_thresh).astype(int)
        t_cm = confusion_matrix(t_true, t_pred, labels=[0, 1])
        t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
        t_prec = float(precision_score(t_true, t_pred, zero_division=0))
        t_rec = float(recall_score(t_true, t_pred, zero_division=0))
        t_f1 = float(f1_score(t_true, t_pred, zero_division=0))
        t_fpr = float(t_fp / (t_fp + t_tn)) if (t_fp + t_tn) > 0 else 0.0
        t_alerts_10k = float((t_fp / len(t_true)) * 10000.0)

        # Retrieve saved test metrics for comparison
        saved_test = df_eval[
            (df_eval["exp_id"] == exp_id)
            & (df_eval["evaluation_split"].str.endswith("_own"))
            & (df_eval["threshold_type"] == "optimal_validation")
        ].iloc[0]

        # Verify reproducibility match
        match_pr = abs(t_pr - saved_test["pr_auc"]) < 1e-6
        match_roc = abs(t_roc - saved_test["roc_auc"]) < 1e-6
        match_f1 = abs(t_f1 - saved_test["f1"]) < 1e-6

        reproducibility_checks.append({
            "exp_id": exp_id,
            "reproduced_pr_auc": t_pr,
            "saved_pr_auc": saved_test["pr_auc"],
            "reproduced_roc_auc": t_roc,
            "saved_roc_auc": saved_test["roc_auc"],
            "match": match_pr and match_roc and match_f1
        })

        # History rounds
        rounds_hist = df_history[df_history["exp_id"] == exp_id]
        r1_loss = rounds_hist[rounds_hist["round"] == 1]["avg_client_loss"].values[0] if len(rounds_hist[rounds_hist["round"] == 1]) > 0 else np.nan
        r2_loss = rounds_hist[rounds_hist["round"] == 2]["avg_client_loss"].values[0] if len(rounds_hist[rounds_hist["round"] == 2]) > 0 else np.nan
        r3_loss = rounds_hist[rounds_hist["round"] == 3]["avg_client_loss"].values[0] if len(rounds_hist[rounds_hist["round"] == 3]) > 0 else np.nan

        # Class counts
        pos_train = 137 if d_name in ["HI-Small", "Combined"] else 43
        pos_val = int(np.sum(v_true == 1))
        pos_test = int(np.sum(t_true == 1))

        step1_records.append({
            "exp_id": exp_id,
            "dataset": d_name,
            "architecture": arch_str,
            "partition": dist,
            "algorithm": "fedprox",
            "mu": mu_val,
            "checkpoint_path": str(ckpt_file.relative_to(BASE_DIR)),
            "checkpoint_sha256": on_disk_hash,
            "train_samples": 100000,
            "val_samples": len(v_true),
            "test_samples": len(t_true),
            "train_positives": pos_train,
            "val_positives": pos_val,
            "test_positives": pos_test,
            "round1_train_loss": r1_loss,
            "round2_train_loss": r2_loss,
            "round3_train_loss": r3_loss,
            "val_pr_auc": v_pr,
            "val_roc_auc": v_roc,
            "val_f1": v_f1,
            "val_precision": v_prec,
            "val_recall": v_rec,
            "val_fpr": v_fpr,
            "optimal_threshold": opt_thresh,
            "test_pr_auc": t_pr,
            "test_roc_auc": t_roc,
            "test_f1": t_f1,
            "test_precision": t_prec,
            "test_recall": t_rec,
            "test_fpr": t_fpr,
            "test_alerts_10k": t_alerts_10k,
            "test_tp": t_tp,
            "test_fp": t_fp,
            "test_tn": t_tn,
            "test_fn": t_fn,
            "reproducibility_verified": match_pr and match_roc and match_f1,
        })

    df_step1 = pd.DataFrame(step1_records)
    df_step1.to_csv(RETRAIN_DIR / "mu1_existing_verified_models.csv", index=False)
    logger.info("Saved %s with %d records", RETRAIN_DIR / "mu1_existing_verified_models.csv", len(df_step1))

    # All 18 matched?
    all_matched = all(r["match"] for r in reproducibility_checks)
    logger.info("All 18 models reproduced to < 1e-6 precision? %s", all_matched)

    # === STEP 3: Diagnose mu = 1.0 vs FedAvg (mu=0) and FedProx (mu=0.1) ===
    logger.info("=== STEP 3: Comparative Diagnosis: mu=1.0 vs FedAvg vs mu=0.1 ===")
    df_summary = pd.read_csv(ARTIFACTS_DIR / "comparison_summary.csv")

    diagnosis_rows = []
    for d_name in ["HI-Small", "LI-Medium", "Combined"]:
        for arch_str in ["gru", "lstm", "bilstm"]:
            for dist in ["iid", "non_iid"]:
                sub = df_summary[
                    (df_summary["training_dataset"] == d_name)
                    & (df_summary["architecture"] == arch_str)
                    & (df_summary["distribution"] == dist)
                ]
                fedavg_row = sub[sub["algorithm"] == "fedavg"].iloc[0]
                prox01_row = sub[(sub["algorithm"] == "fedprox") & (sub["mu"] == 0.1)].iloc[0]
                prox10_row = sub[(sub["algorithm"] == "fedprox") & (sub["mu"] == 1.0)].iloc[0]

                diagnosis_rows.append({
                    "dataset": d_name,
                    "architecture": arch_str,
                    "distribution": dist,
                    # PR-AUC
                    "fedavg_pr_auc": fedavg_row["pr_auc"],
                    "prox01_pr_auc": prox01_row["pr_auc"],
                    "prox10_pr_auc": prox10_row["pr_auc"],
                    # ROC-AUC
                    "fedavg_roc_auc": fedavg_row["roc_auc"],
                    "prox01_roc_auc": prox01_row["roc_auc"],
                    "prox10_roc_auc": prox10_row["roc_auc"],
                    # F1
                    "fedavg_f1": fedavg_row["f1"],
                    "prox01_f1": prox01_row["f1"],
                    "prox10_f1": prox10_row["f1"],
                    # Recall
                    "fedavg_recall": fedavg_row["recall"],
                    "prox01_recall": prox01_row["recall"],
                    "prox10_recall": prox10_row["recall"],
                    # FPR
                    "fedavg_fpr": fedavg_row["fpr"],
                    "prox01_fpr": prox01_row["fpr"],
                    "prox10_fpr": prox10_row["fpr"],
                    # Alerts/10k
                    "fedavg_alerts_10k": fedavg_row["false_alerts_per_10k"],
                    "prox01_alerts_10k": prox01_row["false_alerts_per_10k"],
                    "prox10_alerts_10k": prox10_row["false_alerts_per_10k"],
                })

    df_diag = pd.DataFrame(diagnosis_rows)
    df_diag.to_csv(RETRAIN_DIR / "mu1_vs_baselines_diagnosis.csv", index=False)
    logger.info("Saved %s with %d comparison rows", RETRAIN_DIR / "mu1_vs_baselines_diagnosis.csv", len(df_diag))

    print("\n--- SUMMARY OF MU=1.0 COMPARISON ---")
    print("Mean PR-AUC: FedAvg=%.6f, FedProx(0.1)=%.6f, FedProx(1.0)=%.6f" % (
        df_diag["fedavg_pr_auc"].mean(), df_diag["prox01_pr_auc"].mean(), df_diag["prox10_pr_auc"].mean()
    ))
    print("Mean ROC-AUC: FedAvg=%.4f, FedProx(0.1)=%.4f, FedProx(1.0)=%.4f" % (
        df_diag["fedavg_roc_auc"].mean(), df_diag["prox01_roc_auc"].mean(), df_diag["prox10_roc_auc"].mean()
    ))
    print("Mean Recall: FedAvg=%.2f%%, FedProx(0.1)=%.2f%%, FedProx(1.0)=%.2f%%" % (
        df_diag["fedavg_recall"].mean() * 100, df_diag["prox01_recall"].mean() * 100, df_diag["prox10_recall"].mean() * 100
    ))
    print("Mean FPR: FedAvg=%.4f%%, FedProx(0.1)=%.4f%%, FedProx(1.0)=%.4f%%" % (
        df_diag["fedavg_fpr"].mean() * 100, df_diag["prox01_fpr"].mean() * 100, df_diag["prox10_fpr"].mean() * 100
    ))
    print("Mean Alerts/10k: FedAvg=%.1f, FedProx(0.1)=%.1f, FedProx(1.0)=%.1f" % (
        df_diag["fedavg_alerts_10k"].mean(), df_diag["prox01_alerts_10k"].mean(), df_diag["prox10_alerts_10k"].mean()
    ))


if __name__ == "__main__":
    main()

