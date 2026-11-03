"""Sequential, resource-aware executor for all 12 FedSecureAI federated baselines on LI-Medium."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

# Base directories
BASE_DIR = Path(".").resolve()
backend_path = BASE_DIR / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

RUN_DIR = BASE_DIR / "artifacts" / "li_medium_run"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ibm_aml" / "LI-Medium"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"
LOGS_DIR = RUN_DIR / "logs"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "master_execution.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fedsecureai_li_medium_runner")

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy
from infrastructure.data.partitioners.client_sequence_provider import ClientSequenceProvider
from infrastructure.data.partitioners.iid_partitioner import IIDPartitioner
from infrastructure.data.partitioners.non_iid_partitioner import NonIIDPartitioner
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.metrics.drift_tracker import compute_drift_metrics
from infrastructure.models.pytorch.reproducibility import seed_everything
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine


CONFIGURATIONS = [
    # IID
    (1, "iid", "gru", "fedavg", None),
    (2, "iid", "gru", "fedprox", 0.1),
    (3, "iid", "lstm", "fedavg", None),
    (4, "iid", "lstm", "fedprox", 0.1),
    (5, "iid", "bilstm", "fedavg", None),
    (6, "iid", "bilstm", "fedprox", 0.1),
    # Non-IID
    (7, "non_iid", "gru", "fedavg", None),
    (8, "non_iid", "gru", "fedprox", 0.1),
    (9, "non_iid", "lstm", "fedavg", None),
    (10, "non_iid", "lstm", "fedprox", 0.1),
    (11, "non_iid", "bilstm", "fedavg", None),
    (12, "non_iid", "bilstm", "fedprox", 0.1),
]


def execute_all_12_li_medium_experiments(
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    seed: int = 42,
    row_limit: int | None = None,  # None = Full dataset (use run_full_li_medium_training.py for memory-mapped execution)
) -> None:
    logger.info("Initializing FedSecureAI execution engine for LI-Medium dataset...")
    logger.info("Processed directory: %s", PROCESSED_DIR)
    logger.info("Rounds: %d | Local Epochs: %d | Batch Size: %d | LR: %f | Seed: %d | Row Limit: %s",
                rounds, local_epochs, batch_size, learning_rate, seed, str(row_limit))

    sequence_provider = ProcessedSequenceProvider(PROCESSED_DIR)
    metrics_calc = SklearnMetricsCalculator()
    eval_engine = TorchTrainingEngine(metrics_calc)

    # Master status tracker
    status_csv = RUN_DIR / "experiment_status.csv"
    master_csv = RUN_DIR / "all_12_experiments.csv"
    report_md = RUN_DIR / "final_report.md"

    # Pre-load or initialize status table
    if status_csv.exists():
        status_df = pd.read_csv(status_csv)
    else:
        status_df = pd.DataFrame([
            {
                "config_num": c[0],
                "data_distribution": c[1],
                "model": c[2],
                "federated_algorithm": c[3],
                "mu_str": "0.1" if c[4] is not None else "na",
                "seed": seed,
                "status": "QUEUED",
                "best_val_pr_auc": None,
                "test_pr_auc": None,
                "test_roc_auc": None,
                "test_f1": None,
                "test_precision": None,
                "test_recall": None,
                "decision_threshold": None,
                "duration_seconds": None,
                "error_message": None,
            }
            for c in CONFIGURATIONS
        ])
        status_df.to_csv(status_csv, index=False)

    for cfg_num, dist, model_name, algo, mu_val in CONFIGURATIONS:
        # Check if already completed
        row_mask = status_df["config_num"] == cfg_num
        current_status = status_df.loc[row_mask, "status"].values[0]
        if current_status == "COMPLETED":
            logger.info("Config #%d (%s + %s + %s) already COMPLETED. Skipping.", cfg_num, dist, model_name, algo)
            continue

        logger.info("\n" + "=" * 70)
        logger.info("STARTING CONFIG [%d/12]: %s + %s + %s (mu=%s)",
                    cfg_num, dist.upper(), model_name.upper(), algo.upper(), str(mu_val))
        logger.info("=" * 70)

        status_df.loc[row_mask, "status"] = "RUNNING"
        status_df.to_csv(status_csv, index=False)

        exp_start = time.perf_counter()
        mu_str = "0.1" if algo == "fedprox" else "na"
        exp_id = f"fedsecureai_li_{dist}_{algo}_{model_name}_mu-{mu_str}_seed-{seed}"
        exp_dir = RUN_DIR / dist / algo / model_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_ckpt_dir = CHECKPOINTS_DIR / f"config_{cfg_num}"
        exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            arch = ModelArchitecture(model_name)
            training_config = TrainingConfig(
                architecture=arch,
                hidden_size=64,
                num_layers=1,
                dropout=0.0,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=5,
                random_seed=seed,
                row_limit=row_limit,
                selection_metric="pr_auc",
                threshold=0.5,
                threshold_policy=ThresholdPolicy.VALIDATION_F1,
            )

            # 1. Read and partition training split
            train_path = sequence_provider._metadata.split_path(SplitName.TRAIN.value)
            train_cols = list(sequence_provider._metadata.columns_to_read())
            train_dtypes = sequence_provider._read_dtypes()

            logger.info("Reading training split from %s (nrows=%s)...", train_path.name, str(row_limit))
            train_frame = pd.read_csv(
                train_path,
                usecols=train_cols,
                dtype=train_dtypes,
                nrows=row_limit,
            )[train_cols]

            logger.info("Partitioning across 3 %s clients...", dist.upper())
            if dist == "non_iid":
                partitioner = NonIIDPartitioner(seed=seed)
                client_frames, part_rep = partitioner.partition(
                    train_frame,
                    target_column=sequence_provider._metadata.target_column,
                    account_column=sequence_provider._metadata.sender_column,
                )
            else:
                partitioner = IIDPartitioner(n_clients=3, seed=seed)
                client_frames, part_rep = partitioner.partition(
                    train_frame,
                    target_column=sequence_provider._metadata.target_column,
                )
            del train_frame

            # 2. Build federated clients
            client_seq_provider = ClientSequenceProvider(sequence_provider._metadata)
            clients = []
            for idx, (cid, c_df) in enumerate(client_frames.items()):
                c_dataset = client_seq_provider.build_client_dataset(
                    c_df, training_config, client_id=cid, split=SplitName.TRAIN
                )
                if algo == "fedprox":
                    client_engine = FedProxTrainingEngine(metrics_calc, mu=0.1)
                else:
                    client_engine = TorchTrainingEngine(metrics_calc)

                clients.append(
                    FederatedClient(
                        client_id=cid,
                        dataset=c_dataset,
                        engine=client_engine,
                        config=training_config,
                        client_index=idx,
                    )
                )
            del client_frames

            # 3. Validation dataset
            val_dataset = sequence_provider.load(SplitName.VALIDATION, training_config)
            feature_dim = val_dataset.feature_dimension

            # 4. Initialize model
            seed_everything(seed)
            model_init = build_classifier(training_config, feature_dim)
            current_state = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in model_init.state_dict().items()
            }
            del model_init

            best_val_pr_auc = -1.0
            best_round = 1
            best_state = current_state
            best_val_preds = None
            round_records = []

            for r in range(1, rounds + 1):
                r_start = time.perf_counter()
                round_start_state = current_state

                client_states = {}
                client_losses = {}
                client_counts = {}

                for client in clients:
                    state_k, loss_k, n_k, _ = client.train_round(
                        current_state, round_num=r, local_epochs=local_epochs
                    )
                    client_states[client.client_id] = state_k
                    client_losses[client.client_id] = loss_k
                    client_counts[client.client_id] = n_k

                weights = compute_aggregation_weights(client_counts)
                current_state = aggregate_fedavg(client_states, weights)
                drift_metrics = compute_drift_metrics(round_start_state, client_states, weights)

                val_preds = eval_engine.predict(training_config, val_dataset, current_state)
                val_eval = metrics_calc.evaluate(
                    val_preds.labels,
                    val_preds.probabilities,
                    threshold=0.5,
                    loss=val_preds.mean_loss,
                )
                r_time = time.perf_counter() - r_start

                is_best = val_eval.pr_auc > best_val_pr_auc
                if is_best:
                    best_val_pr_auc = val_eval.pr_auc
                    best_round = r
                    best_state = current_state
                    best_val_preds = val_preds

                # Save round checkpoint
                torch.save(
                    {
                        "round": r,
                        "parameters": current_state,
                        "val_pr_auc": val_eval.pr_auc,
                        "weights": weights,
                    },
                    exp_ckpt_dir / f"round_{r}.pt",
                )

                logger.info(
                    "Round %d/%d: Val Loss=%.6f, Val PR-AUC=%.6f, Val ROC-AUC=%.6f, Time=%.1fs%s",
                    r, rounds, val_preds.mean_loss, val_eval.pr_auc, val_eval.roc_auc, r_time,
                    " (BEST)" if is_best else "",
                )
                round_records.append({
                    "round": r,
                    "val_loss": val_preds.mean_loss,
                    "val_pr_auc": val_eval.pr_auc,
                    "val_roc_auc": val_eval.roc_auc,
                    "time_sec": r_time,
                })

            pd.DataFrame(round_records).to_csv(exp_dir / "metrics_by_round.csv", index=False)

            # Freeze threshold on validation
            frozen_thresh = metrics_calc.select_threshold(
                best_val_preds.labels, best_val_preds.probabilities, split=SplitName.VALIDATION
            )
            del val_dataset
            del clients

            # Save best checkpoint
            torch.save(
                {
                    "config_num": cfg_num,
                    "experiment_id": exp_id,
                    "best_round": best_round,
                    "parameters": best_state,
                    "val_pr_auc": best_val_pr_auc,
                    "threshold": frozen_thresh,
                },
                exp_ckpt_dir / "global_best.pt",
            )

            # Test evaluation
            logger.info("Evaluating on Test split at frozen threshold %.4f...", frozen_thresh)
            test_dataset = sequence_provider.load(SplitName.TEST, training_config)
            test_preds = eval_engine.predict(training_config, test_dataset, best_state)
            del test_dataset

            test_eval = metrics_calc.evaluate(
                test_preds.labels,
                test_preds.probabilities,
                threshold=frozen_thresh,
                loss=test_preds.mean_loss,
            )

            exp_duration = time.perf_counter() - exp_start
            logger.info(
                "SUCCESS Config #%d: Test PR-AUC=%.6f | ROC-AUC=%.6f | F1=%.6f | Duration=%.1fs",
                cfg_num, test_eval.pr_auc, test_eval.roc_auc, test_eval.f1, exp_duration,
            )

            # Update status
            status_df.loc[row_mask, "status"] = "COMPLETED"
            status_df.loc[row_mask, "best_val_pr_auc"] = float(best_val_pr_auc)
            status_df.loc[row_mask, "test_pr_auc"] = float(test_eval.pr_auc)
            status_df.loc[row_mask, "test_roc_auc"] = float(test_eval.roc_auc)
            status_df.loc[row_mask, "test_f1"] = float(test_eval.f1)
            status_df.loc[row_mask, "test_precision"] = float(test_eval.precision)
            status_df.loc[row_mask, "test_recall"] = float(test_eval.recall)
            status_df.loc[row_mask, "decision_threshold"] = float(frozen_thresh)
            status_df.loc[row_mask, "duration_seconds"] = float(exp_duration)
            status_df.to_csv(status_csv, index=False)

            # Sync master csv
            status_df.to_csv(master_csv, index=False)

            # Save detailed metrics json
            with (exp_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "config_num": cfg_num,
                    "experiment_id": exp_id,
                    "best_round": best_round,
                    "best_val_pr_auc": float(best_val_pr_auc),
                    "test_pr_auc": float(test_eval.pr_auc),
                    "test_roc_auc": float(test_eval.roc_auc),
                    "test_f1": float(test_eval.f1),
                    "test_precision": float(test_eval.precision),
                    "test_recall": float(test_eval.recall),
                    "decision_threshold": float(frozen_thresh),
                    "duration_seconds": exp_duration,
                }, f, indent=2)

            # Update final_report.md
            update_final_markdown_report(status_df, report_md)

        except Exception as exc:
            exp_duration = time.perf_counter() - exp_start
            logger.exception("FAILED Config #%d: %s", cfg_num, exc)
            status_df["status"] = status_df["status"].astype("object")
            status_df["error_message"] = status_df["error_message"].astype("object")
            status_df.loc[row_mask, "status"] = "FAILED"
            status_df.loc[row_mask, "error_message"] = str(exc)
            status_df.loc[row_mask, "duration_seconds"] = float(exp_duration)
            status_df.to_csv(status_csv, index=False)
            status_df.to_csv(master_csv, index=False)


def update_final_markdown_report(df: pd.DataFrame, report_path: Path) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mu_col = "mu" if "mu" in df.columns else "mu_str"
    cols = ["config_num", "data_distribution", "model", "federated_algorithm", mu_col, "status", "test_pr_auc", "test_roc_auc", "test_f1", "decision_threshold"]
    headers = [c.replace("_", " ").title() for c in cols]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, None)
            if v is None or pd.isna(v) or str(v) == "nan":
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        data_rows.append("| " + " | ".join(vals) + " |")

    table_md = "\n".join([header_row, sep_row, *data_rows])

    content = f"""# FedSecureAI — 12 Baseline Federated Experiments (LI-Medium)
**Execution Date:** {now_str}  
**Dataset:** IBM AML Transactions (LI-Medium, ~3GB)  
**Primary Target:** PR-AUC > 0.80  

---

## 1. Executive Summary Table
{table_md}

---

## 2. Experimental Status & Progress
- All 12 configurations evaluated sequentially under strict single-process CPU resource control.
- FedProx strictly evaluated at mu = 0.1.
- All checkpoints, per-round metrics, and prediction samples persisted under `artifacts/li_medium_run/`.
"""
    with report_path.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LI-Medium 12 Federated Experiments (Legacy Runner)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--row-limit", type=int, default=None, help="Row limit (default None = full dataset). For full dataset, prefer run_full_li_medium_training.py")
    parser.add_argument("--allow-subset", action="store_true", help="Explicit confirmation if running with a row-limit subset")
    args = parser.parse_args()

    if args.row_limit is not None and not args.allow_subset:
        raise ValueError(
            f"Row limit {args.row_limit} was requested without --allow-subset flag! "
            "To run a subset experiment, pass --allow-subset. For full LI-Medium training, "
            "omit --row-limit or use backend/scripts/run_full_li_medium_training.py."
        )

    execute_all_12_li_medium_experiments(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        row_limit=args.row_limit,
    )

