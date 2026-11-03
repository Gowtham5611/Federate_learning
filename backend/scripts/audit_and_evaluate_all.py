"""Audit training rounds and evaluate all 12 baseline checkpoints at fixed threshold 0.1."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
backend_path = BASE_DIR / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("audit_and_evaluate")

RUN_DIR = BASE_DIR / "artifacts" / "li_medium_run"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ibm_aml" / "LI-Medium"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"

CONFIGURATIONS = [
    (1, "iid", "gru", "fedavg", None),
    (2, "iid", "gru", "fedprox", 0.1),
    (3, "iid", "lstm", "fedavg", None),
    (4, "iid", "lstm", "fedprox", 0.1),
    (5, "iid", "bilstm", "fedavg", None),
    (6, "iid", "bilstm", "fedprox", 0.1),
    (7, "non_iid", "gru", "fedavg", None),
    (8, "non_iid", "gru", "fedprox", 0.1),
    (9, "non_iid", "lstm", "fedavg", None),
    (10, "non_iid", "lstm", "fedprox", 0.1),
    (11, "non_iid", "bilstm", "fedavg", None),
    (12, "non_iid", "bilstm", "fedprox", 0.1),
]


def audit_rounds() -> pd.DataFrame:
    logger.info("Auditing all rounds across 12 configurations...")
    records = []
    for cfg_num, dist, model_name, algo, mu_val in CONFIGURATIONS:
        exp_dir = RUN_DIR / dist / algo / model_name
        round_csv = exp_dir / "metrics_by_round.csv"
        ckpt_dir = CHECKPOINTS_DIR / f"config_{cfg_num}"
        
        # Load best round info from global_best.pt if available
        best_round = 1
        if (ckpt_dir / "global_best.pt").exists():
            gb = torch.load(ckpt_dir / "global_best.pt", map_location="cpu", weights_only=False)
            best_round = gb.get("best_round", 1)

        if round_csv.exists():
            df_rounds = pd.read_csv(round_csv)
            for _, r in df_rounds.iterrows():
                r_num = int(r["round"])
                has_ckpt = (ckpt_dir / f"round_{r_num}.pt").exists()
                records.append({
                    "config_num": cfg_num,
                    "data_distribution": dist,
                    "model": model_name,
                    "federated_algorithm": algo,
                    "mu": mu_val if mu_val is not None else "na",
                    "round": r_num,
                    "val_loss": float(r["val_loss"]),
                    "val_pr_auc": float(r["val_pr_auc"]),
                    "val_roc_auc": float(r["val_roc_auc"]),
                    "time_sec": float(r["time_sec"]),
                    "checkpoint_exists": has_ckpt,
                    "is_selected_best_round": (r_num == best_round),
                })
        else:
            logger.warning("metrics_by_round.csv missing for %s", exp_dir)

    df_all_rounds = pd.DataFrame(records)
    out_path = RUN_DIR / "training_round_audit.csv"
    df_all_rounds.to_csv(out_path, index=False)
    logger.info("Saved round audit to %s (%d total rounds)", out_path, len(df_all_rounds))
    return df_all_rounds


def evaluate_fixed_threshold() -> pd.DataFrame:
    logger.info("Loading test dataset (200k chronological rows)...")
    sequence_provider = ProcessedSequenceProvider(PROCESSED_DIR)
    metrics_calc = SklearnMetricsCalculator()
    eval_engine = TorchTrainingEngine(metrics_calc)

    base_config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        row_limit=200000,
        batch_size=1024,
    )
    test_dataset = sequence_provider.load(SplitName.TEST, base_config)
    logger.info("Test dataset loaded: %d sequences, %d positives (prevalence=%.6f)",
                test_dataset.sequence_count, test_dataset.positive_count,
                test_dataset.positive_count / test_dataset.sequence_count)

    eval_records = []
    
    for cfg_num, dist, model_name, algo, mu_val in CONFIGURATIONS:
        ckpt_dir = CHECKPOINTS_DIR / f"config_{cfg_num}"
        gb_path = ckpt_dir / "global_best.pt"
        if not gb_path.exists():
            logger.error("Missing %s", gb_path)
            continue

        gb = torch.load(gb_path, map_location="cpu", weights_only=False)
        params = gb["parameters"]
        val_thresh = float(gb.get("threshold", 0.5))
        best_round = int(gb.get("best_round", 1))

        arch = ModelArchitecture(model_name)
        cfg = TrainingConfig(
            architecture=arch,
            row_limit=200000,
            batch_size=1024,
        )

        preds = eval_engine.predict(cfg, test_dataset, params)

        # Evaluate at validation-tuned threshold
        res_val = metrics_calc.evaluate(
            preds.labels, preds.probabilities, threshold=val_thresh, loss=preds.mean_loss
        )

        # Evaluate at fixed threshold = 0.1
        res_fixed_01 = metrics_calc.evaluate(
            preds.labels, preds.probabilities, threshold=0.1, loss=preds.mean_loss
        )

        eval_records.append({
            "config_num": cfg_num,
            "data_distribution": dist,
            "model": model_name,
            "federated_algorithm": algo,
            "mu": mu_val if mu_val is not None else "na",
            "completed_rounds": 5,
            "best_round": best_round,
            "val_f1_threshold": val_thresh,
            "test_pr_auc": float(res_val.pr_auc),
            "test_roc_auc": float(res_val.roc_auc),
            "f1_at_val_thresh": float(res_val.f1),
            "precision_at_val_thresh": float(res_val.precision),
            "recall_at_val_thresh": float(res_val.recall),
            "f1_at_fixed_0_1": float(res_fixed_01.f1),
            "precision_at_fixed_0_1": float(res_fixed_01.precision),
            "recall_at_fixed_0_1": float(res_fixed_01.recall),
            "accuracy_at_fixed_0_1": float(res_fixed_01.accuracy),
            "tp_fixed_0_1": res_fixed_01.confusion.true_positives,
            "fp_fixed_0_1": res_fixed_01.confusion.false_positives,
            "fn_fixed_0_1": res_fixed_01.confusion.false_negatives,
            "tn_fixed_0_1": res_fixed_01.confusion.true_negatives,
        })
        logger.info("Config #%d [%s %s %s]: PR-AUC=%.6f, F1@val=%.6f (thresh=%.4f), F1@0.1=%.6f",
                    cfg_num, dist, model_name, algo, res_val.pr_auc, res_val.f1, val_thresh, res_fixed_01.f1)

    df_eval = pd.DataFrame(eval_records)
    out_thresh = RUN_DIR / "threshold_0_1_metrics.csv"
    df_eval.to_csv(out_thresh, index=False)
    logger.info("Saved threshold evaluation to %s", out_thresh)

    # Also update all_12_experiments.csv and experiment_status.csv
    update_tracking_tables(df_eval)
    return df_eval


def update_tracking_tables(df_eval: pd.DataFrame) -> None:
    all_12_path = RUN_DIR / "all_12_experiments.csv"
    status_path = RUN_DIR / "experiment_status.csv"

    if all_12_path.exists():
        df_master = pd.read_csv(all_12_path)
    else:
        df_master = df_eval.copy()

    # Ensure completed_rounds and best_round are distinct
    df_master["completed_rounds"] = 5
    best_round_map = dict(zip(df_eval["config_num"], df_eval["best_round"]))
    f1_fixed_map = dict(zip(df_eval["config_num"], df_eval["f1_at_fixed_0_1"]))
    prec_fixed_map = dict(zip(df_eval["config_num"], df_eval["precision_at_fixed_0_1"]))
    rec_fixed_map = dict(zip(df_eval["config_num"], df_eval["recall_at_fixed_0_1"]))
    acc_fixed_map = dict(zip(df_eval["config_num"], df_eval["accuracy_at_fixed_0_1"]))

    df_master["best_round"] = df_master["config_num"].map(best_round_map)
    df_master["f1_at_fixed_0_1"] = df_master["config_num"].map(f1_fixed_map)
    df_master["precision_at_fixed_0_1"] = df_master["config_num"].map(prec_fixed_map)
    df_master["recall_at_fixed_0_1"] = df_master["config_num"].map(rec_fixed_map)
    df_master["accuracy_at_fixed_0_1"] = df_master["config_num"].map(acc_fixed_map)

    df_master.to_csv(all_12_path, index=False)
    df_master.to_csv(status_path, index=False)
    logger.info("Updated %s and %s with distinct completed_rounds and fixed threshold 0.1 metrics",
                all_12_path, status_path)


if __name__ == "__main__":
    audit_rounds()
    evaluate_fixed_threshold()

