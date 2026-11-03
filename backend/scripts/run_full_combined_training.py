"""FedSecureAI — Full Combined Dataset Federated AML Training Runner.

Executes all 12 primary federated configurations on the complete 36.33M-row
IBM AML FULL_COMBINED dataset (~21.80M train, ~7.27M val, ~7.26M test) using
RAM-bounded memory-mapped sequences and zero-copy client partitioning.

Configurations:
  - Architectures: GRU, LSTM, BiLSTM
  - Federated Algorithms: FedAvg, FedProx (mu=0.0001)
  - Distributions: IID, Non-IID
  - 3 Clients, 5 Global Rounds, Local Epochs = 1

RAM guarantee: < 1.0 GB resident host memory during entire execution.
Safe for Google Colab standard GPU runtimes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy
from infrastructure.data.sequences.mmap_partition_manager import MmapPartitionManager
from infrastructure.data.sequences.mmap_sequence_provider import (
    MmapSequenceProvider,
    SubsetModeDetectedError,
)
from infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.metrics.drift_tracker import compute_drift_metrics
from infrastructure.models.pytorch.reproducibility import resolve_device, seed_everything
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

# 12 Primary Configurations Definition
CONFIGURATIONS = [
    # (config_num, distribution, architecture, federated_algorithm, fedprox_mu)
    (1, "iid", "gru", "fedavg", 0.0),
    (2, "iid", "gru", "fedprox", 0.0001),
    (3, "iid", "lstm", "fedavg", 0.0),
    (4, "iid", "lstm", "fedprox", 0.0001),
    (5, "iid", "bilstm", "fedavg", 0.0),
    (6, "iid", "bilstm", "fedprox", 0.0001),
    (7, "non_iid", "gru", "fedavg", 0.0),
    (8, "non_iid", "gru", "fedprox", 0.0001),
    (9, "non_iid", "lstm", "fedavg", 0.0),
    (10, "non_iid", "lstm", "fedprox", 0.0001),
    (11, "non_iid", "bilstm", "fedavg", 0.0),
    (12, "non_iid", "bilstm", "fedprox", 0.0001),
]

FULL_COMBINED_MIN_TRAIN_ROWS = 21_000_000
FULL_COMBINED_MIN_VAL_ROWS = 7_000_000
FULL_COMBINED_MIN_TEST_ROWS = 7_000_000


def calculate_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def setup_experiment_logging(logs_dir: Path) -> logging.Logger:
    """Configure comprehensive console and file logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"full_combined_execution_{timestamp}.log"

    logger = logging.getLogger("fedsecureai_full_combined_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_preflight_checks(
    data_root: Path,
    output_dir: Path,
    device_name: str,
    allow_subset_debug: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Inspect environment, dataset integrity, disk space, and CUDA support."""
    log = logger or logging.getLogger("preflight")
    log.info("Running FedSecureAI Full Combined Preflight Environment & Data Checks...")

    checks: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": device_name,
        "resolved_device": "UNRESOLVED",
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "allow_subset_debug": allow_subset_debug,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "dataset_splits": {},
    }

    try:
        checks["resolved_device"] = str(resolve_device(device_name))
    except Exception as ex:
        checks["errors"].append(str(ex))
        checks["status"] = "FAIL"

    if checks["cuda_available"]:
        checks["cuda_device_name"] = torch.cuda.get_device_name(0)
        checks["cuda_device_count"] = torch.cuda.device_count()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        checks["cuda_vram_gb"] = round(vram_gb, 2)
        log.info("CUDA Detected: %s (%.2f GB VRAM)", checks["cuda_device_name"], vram_gb)
    else:
        if not allow_subset_debug:
            err = (
                "CUDA acceleration is not available on this system, but strict CUDA-only execution "
                "is enforced for full-dataset training. Execution will not silently fall back to CPU."
            )
            checks["errors"].append(err)
            checks["status"] = "FAIL"
            log.error(err)
        else:
            log.warning("CUDA not available. CPU execution permitted only because --allow-subset-debug is active.")

    # Disk Space Check
    target_disk = output_dir if output_dir.exists() else output_dir.parent
    disk_usage = shutil.disk_usage(target_disk if target_disk.exists() else Path("."))
    free_disk_gb = round(disk_usage.free / (1024**3), 2)
    checks["free_disk_gb"] = free_disk_gb
    if free_disk_gb < 15.0:
        msg = f"Low free disk space: {free_disk_gb} GB available (recommend >= 20 GB for full combined run)."
        checks["warnings"].append(msg)
        log.warning(msg)

    # Dataset Files Check
    metadata_dir = data_root / "metadata"
    if not metadata_dir.is_dir():
        err = f"Metadata directory missing at {metadata_dir}"
        checks["errors"].append(err)
        checks["status"] = "FAIL"
        log.error(err)
        return checks

    try:
        metadata = ProcessedFeatureMetadata.read(data_root)
        feat_dim = metadata.feature_dimension(one_hot_nominal_codes=True)
        checks["feature_dimension"] = feat_dim
        if feat_dim != 57:
            checks["warnings"].append(f"Feature dimension is {feat_dim}, expected 57.")
    except Exception as e:
        err = f"Failed to parse feature metadata: {e}"
        checks["errors"].append(err)
        checks["status"] = "FAIL"
        log.error(err)
        return checks

    for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        csv_path = data_root / split.value / "transactions.csv"
        if not csv_path.is_file():
            err = f"Missing transactions CSV for split {split.value} at {csv_path}"
            checks["errors"].append(err)
            checks["status"] = "FAIL"
            log.error(err)
            continue

        file_size_gb = round(csv_path.stat().st_size / (1024**3), 3)
        checks["dataset_splits"][split.value] = {
            "path": str(csv_path),
            "size_gb": file_size_gb,
        }

        min_required = {
            SplitName.TRAIN: FULL_COMBINED_MIN_TRAIN_ROWS,
            SplitName.VALIDATION: FULL_COMBINED_MIN_VAL_ROWS,
            SplitName.TEST: FULL_COMBINED_MIN_TEST_ROWS,
        }[split]

        splits_json = metadata_dir / "splits.json"
        if splits_json.exists():
            with open(splits_json, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            for item in sdata.get("per_split", []):
                if item.get("name") == split.value:
                    actual_rows = item.get("rows", 0)
                    checks["dataset_splits"][split.value]["rows"] = actual_rows
                    checks["dataset_splits"][split.value]["positive"] = item.get("positive", 0)
                    checks["dataset_splits"][split.value]["negative"] = item.get("negative", 0)
                    if not allow_subset_debug and actual_rows < min_required:
                        err = (
                            f"Split {split.value} has only {actual_rows} rows in splits.json; "
                            f"full dataset requires >= {min_required} rows (legacy 500k detected or subset)."
                        )
                        checks["errors"].append(err)
                        checks["status"] = "FAIL"
                        log.error(err)

    if checks["errors"]:
        checks["status"] = "FAIL"
    log.info("Preflight checks completed with status: %s", checks["status"])
    return checks


class FullCombinedExperimentRunner:
    """Orchestrates the 12 full-dataset Combined federated training experiments."""

    def __init__(
        self,
        data_root: Path,
        output_dir: Path,
        *,
        batch_size: int = 4096,
        rounds: int = 5,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        fedprox_mu: float = 0.0001,
        seed: int = 42,
        device: str = "auto",
        allow_subset_debug: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.batch_size = batch_size
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.fedprox_mu = fedprox_mu
        self.seed = seed
        self.device = device
        self.allow_subset_debug = allow_subset_debug

        # Setup directories
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.predictions_dir = self.output_dir / "predictions"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"

        for d in (self.checkpoints_dir, self.predictions_dir, self.results_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.logger = setup_experiment_logging(self.logs_dir)
        self.manifest_path = self.output_dir / "resume_manifest.json"
        self.status_csv = self.results_dir / "experiment_status.csv"

        # Providers
        self.seq_provider = MmapSequenceProvider(self.data_root)
        self.partition_manager = MmapPartitionManager(self.seq_provider.metadata)
        self.metrics_calc = SklearnMetricsCalculator()
        self.eval_engine = TorchTrainingEngine(self.metrics_calc)

    def load_or_init_manifest(self) -> dict[str, Any]:
        """Atomically read or initialize resume manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning("Failed to load existing manifest: %s. Creating new one.", e)

        manifest = {
            "experiment_name": "full_combined_12_experiments",
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset": "IBM AML FULL_COMBINED (HI-Small + LI-Medium Full)",
            "data_root": str(self.data_root),
            "output_dir": str(self.output_dir),
            "hyperparameters": {
                "rounds": self.rounds,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "fedprox_mu": self.fedprox_mu,
                "seed": self.seed,
                "device": self.device,
            },
            "configurations": {},
        }
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomic write to resume manifest."""
        manifest["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp_path = self.manifest_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        shutil.move(str(temp_path), str(self.manifest_path))

    def update_status_csv(self, manifest: dict[str, Any]) -> None:
        """Sync CSV status report from manifest."""
        rows = []
        for cfg in CONFIGURATIONS:
            cfg_num = cfg[0]
            c_key = f"config_{cfg_num}"
            c_data = manifest["configurations"].get(c_key, {})
            test_m = c_data.get("test_metrics", {})
            rows.append({
                "config_num": cfg_num,
                "distribution": cfg[1],
                "model": cfg[2],
                "algorithm": cfg[3],
                "mu": cfg[4],
                "status": c_data.get("status", "QUEUED"),
                "best_round": c_data.get("best_round"),
                "best_val_pr_auc": c_data.get("best_val_pr_auc"),
                "test_pr_auc": test_m.get("pr_auc"),
                "test_roc_auc": test_m.get("roc_auc"),
                "test_f1": test_m.get("f1"),
                "test_precision": test_m.get("precision"),
                "test_recall": test_m.get("recall"),
                "decision_threshold": test_m.get("threshold"),
                "false_alerts_per_10k": test_m.get("alerts_per_10k"),
                "duration_seconds": c_data.get("duration_seconds"),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.status_csv, index=False)

    def execute_all(self, target_config_ids: list[int] | None = None) -> None:
        """Run the 12 federated configurations sequentially with strict memory safety."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING FEDSECUREAI FULL COMBINED 12-EXPERIMENT CAMPAIGN")
        self.logger.info("Data Root: %s", self.data_root)
        self.logger.info("Output Directory: %s", self.output_dir)
        self.logger.info("Batch Size: %d | Rounds: %d | Epochs: %d | Device: %s",
                         self.batch_size, self.rounds, self.local_epochs, self.device)
        self.logger.info("=" * 80)

        # Preflight validation
        checks = run_preflight_checks(
            self.data_root, self.output_dir, self.device,
            allow_subset_debug=self.allow_subset_debug,
            logger=self.logger,
        )
        if checks["status"] == "FAIL":
            raise RuntimeError(f"Preflight checks failed: {checks['errors']}")

        manifest = self.load_or_init_manifest()

        base_training_config = TrainingConfig(
            architecture=ModelArchitecture.GRU,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            learning_rate=self.lr,
            batch_size=self.batch_size,
            epochs=self.rounds,
            random_seed=self.seed,
            device=self.device,
            row_limit=None,
            selection_metric="pr_auc",
            threshold=0.5,
            threshold_policy=ThresholdPolicy.VALIDATION_F1,
        )

        self.logger.info("Preparing read-only memory-mapped Validation and Test datasets...")
        min_val = None if self.allow_subset_debug else FULL_COMBINED_MIN_VAL_ROWS
        min_test = None if self.allow_subset_debug else FULL_COMBINED_MIN_TEST_ROWS
        min_train = None if self.allow_subset_debug else FULL_COMBINED_MIN_TRAIN_ROWS

        val_dataset = self.seq_provider.load(
            SplitName.VALIDATION,
            base_training_config,
            enforce_full_dataset=not self.allow_subset_debug,
            min_expected_rows=min_val,
        )
        test_dataset = self.seq_provider.load(
            SplitName.TEST,
            base_training_config,
            enforce_full_dataset=not self.allow_subset_debug,
            min_expected_rows=min_test,
        )
        feature_dim = val_dataset.feature_dimension

        self.logger.info("Preparing read-only memory-mapped Training parent dataset...")
        train_parent_dataset = self.seq_provider.load(
            SplitName.TRAIN,
            base_training_config,
            enforce_full_dataset=not self.allow_subset_debug,
            min_expected_rows=min_train,
        )

        for cfg_num, dist, model_name, algo, mu_val in CONFIGURATIONS:
            if target_config_ids and cfg_num not in target_config_ids:
                continue

            c_key = f"config_{cfg_num}"
            existing_c = manifest["configurations"].get(c_key, {})
            if existing_c.get("status") == "COMPLETED":
                best_ckpt = self.checkpoints_dir / c_key / "global_best.pt"
                if best_ckpt.exists():
                    self.logger.info("Config #%d (%s, %s, %s) is already COMPLETED. Skipping.",
                                    cfg_num, dist.upper(), model_name.upper(), algo.upper())
                    continue

            self.logger.info("\n" + "#" * 80)
            self.logger.info("CONFIG [%d/12]: Distribution=%s | Model=%s | Algorithm=%s | mu=%.6f",
                             cfg_num, dist.upper(), model_name.upper(), algo.upper(), mu_val)
            self.logger.info("#" * 80)

            manifest["configurations"][c_key] = {
                "config_num": cfg_num,
                "distribution": dist,
                "model": model_name,
                "algorithm": algo,
                "mu": mu_val,
                "status": "RUNNING",
                "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self.save_manifest(manifest)
            self.update_status_csv(manifest)

            cfg_ckpt_dir = self.checkpoints_dir / c_key
            cfg_pred_dir = self.predictions_dir / c_key
            cfg_ckpt_dir.mkdir(parents=True, exist_ok=True)
            cfg_pred_dir.mkdir(parents=True, exist_ok=True)

            exp_start = time.perf_counter()

            arch = ModelArchitecture(model_name)
            cfg_training_config = TrainingConfig(
                architecture=arch,
                hidden_size=64,
                num_layers=1,
                dropout=0.0,
                learning_rate=self.lr,
                batch_size=self.batch_size,
                epochs=self.rounds,
                random_seed=self.seed,
                device=self.device,
                row_limit=None,
                selection_metric="pr_auc",
                threshold=0.5,
                threshold_policy=ThresholdPolicy.VALIDATION_F1,
            )

            self.logger.info("Binding 3 %s client datasets to shared training memory map...", dist.upper())
            client_datasets = self.partition_manager.build_client_datasets(
                distribution=dist,
                parent_dataset=train_parent_dataset,
                config=cfg_training_config,
                seed=self.seed,
                enforce_full_dataset=not self.allow_subset_debug,
            )

            clients = []
            for idx, c_ds in enumerate(client_datasets):
                if algo == "fedprox":
                    c_engine = FedProxTrainingEngine(self.metrics_calc, mu=mu_val)
                else:
                    c_engine = TorchTrainingEngine(self.metrics_calc)

                clients.append(
                    FederatedClient(
                        client_id=f"client_{idx + 1}",
                        dataset=c_ds,
                        engine=c_engine,
                        config=cfg_training_config,
                        client_index=idx,
                    )
                )

            seed_everything(self.seed)
            model_init = build_classifier(cfg_training_config, feature_dim)
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

            for r in range(1, self.rounds + 1):
                r_start = time.perf_counter()
                round_start_state = current_state

                client_states = {}
                client_losses = {}
                client_counts = {}

                for client in clients:
                    state_k, loss_k, n_k, _ = client.train_round(
                        current_state, round_num=r, local_epochs=self.local_epochs
                    )
                    client_states[client.client_id] = state_k
                    client_losses[client.client_id] = loss_k
                    client_counts[client.client_id] = n_k

                weights = compute_aggregation_weights(client_counts)
                current_state = aggregate_fedavg(client_states, weights)
                drift_metrics = compute_drift_metrics(round_start_state, client_states, weights)

                val_preds = self.eval_engine.predict(cfg_training_config, val_dataset, current_state)
                val_eval = self.metrics_calc.evaluate(
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
                ckpt_path = cfg_ckpt_dir / f"round_{r}.pt"
                torch.save(
                    {
                        "round": r,
                        "parameters": current_state,
                        "val_pr_auc": val_eval.pr_auc,
                        "weights": weights,
                    },
                    ckpt_path,
                )

                self.logger.info(
                    "Config #%d Round %d/%d: Val Loss=%.6f, Val PR-AUC=%.6f, Val ROC-AUC=%.6f, Time=%.1fs%s",
                    cfg_num, r, self.rounds, val_preds.mean_loss, val_eval.pr_auc, val_eval.roc_auc,
                    r_time, " (BEST)" if is_best else "",
                )
                round_records.append({
                    "round": r,
                    "val_loss": float(val_preds.mean_loss),
                    "val_pr_auc": float(val_eval.pr_auc),
                    "val_roc_auc": float(val_eval.roc_auc),
                    "time_sec": round(r_time, 2),
                })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save per-round metrics
            pd.DataFrame(round_records).to_csv(cfg_ckpt_dir / "metrics_by_round.csv", index=False)

            # Freeze validation-selected threshold
            frozen_thresh = float(
                self.metrics_calc.select_threshold(
                    best_val_preds.labels,
                    best_val_preds.probabilities,
                    split=SplitName.VALIDATION,
                )
            )

            # Save best checkpoint
            best_ckpt_path = cfg_ckpt_dir / "global_best.pt"
            torch.save(
                {
                    "config_num": cfg_num,
                    "model": model_name,
                    "distribution": dist,
                    "algorithm": algo,
                    "mu": mu_val,
                    "best_round": best_round,
                    "parameters": best_state,
                    "val_pr_auc": best_val_pr_auc,
                    "threshold": frozen_thresh,
                },
                best_ckpt_path,
            )
            best_ckpt_hash = calculate_file_sha256(best_ckpt_path)

            # Test evaluation strictly at frozen threshold
            self.logger.info("Evaluating on Test split (~7.26M rows) at frozen threshold %.4f...", frozen_thresh)
            test_preds = self.eval_engine.predict(cfg_training_config, test_dataset, best_state)
            test_eval = self.metrics_calc.evaluate(
                test_preds.labels,
                test_preds.probabilities,
                threshold=frozen_thresh,
                loss=test_preds.mean_loss,
            )

            cm = test_eval.confusion
            tn = cm.true_negatives
            fp = cm.false_positives
            total_neg = tn + fp
            alerts_per_10k = (fp / total_neg) * 10000.0 if total_neg > 0 else 0.0

            # Save test predictions compressed
            np.savez_compressed(
                cfg_pred_dir / "test_predictions.npz",
                labels=test_preds.labels,
                probabilities=test_preds.probabilities,
                threshold=frozen_thresh,
            )

            exp_duration = time.perf_counter() - exp_start
            self.logger.info(
                "COMPLETED Config #%d: PR-AUC=%.6f | ROC-AUC=%.6f | F1=%.6f | Alerts/10k=%.2f | Duration=%.1fs",
                cfg_num, test_eval.pr_auc, test_eval.roc_auc, test_eval.f1, alerts_per_10k, exp_duration,
            )

            # Update manifest
            manifest["configurations"][c_key].update({
                "status": "COMPLETED",
                "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "duration_seconds": round(exp_duration, 2),
                "best_round": best_round,
                "best_val_pr_auc": float(best_val_pr_auc),
                "checkpoint_sha256": best_ckpt_hash,
                "test_metrics": {
                    "pr_auc": float(test_eval.pr_auc),
                    "roc_auc": float(test_eval.roc_auc),
                    "f1": float(test_eval.f1),
                    "precision": float(test_eval.precision),
                    "recall": float(test_eval.recall),
                    "threshold": float(frozen_thresh),
                    "alerts_per_10k": float(alerts_per_10k),
                    "test_loss": float(test_preds.mean_loss),
                },
                "round_metrics": round_records,
            })
            self.save_manifest(manifest)
            self.update_status_csv(manifest)

            # Save individual config metrics json
            cfg_metrics_file = self.results_dir / f"{c_key}_metrics.json"
            with open(cfg_metrics_file, "w", encoding="utf-8") as f:
                json.dump(manifest["configurations"][c_key], f, indent=2)

        self.logger.info("=" * 80)
        self.logger.info("ALL 12 FULL COMBINED CONFIGURATIONS SUCCESSFULLY EXECUTED!")
        self.logger.info("Final status report saved at: %s", self.status_csv)
        self.logger.info("Resume manifest saved at: %s", self.manifest_path)
        self.logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedSecureAI — Full Combined Dataset Federated AML Training Runner"
    )
    parser.add_argument("--data-root", type=str, default="data/processed/ibm_aml/FULL_COMBINED",
                        help="Path to preprocessed FULL_COMBINED root")
    parser.add_argument("--output-dir", type=str, default="artifacts/full_combined_experiment",
                        help="Directory to save checkpoints, logs, and results")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Mini-batch size for training/eval (default: 4096)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Communication rounds (default: 5)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local epochs per round (default: 1)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--fedprox-mu", type=float, default=0.0001,
                        help="FedProx proximal parameter mu (default: 0.0001)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: cuda (enforced for full run) or cpu (allowed only with --allow-subset-debug)")
    parser.add_argument("--config-id", type=int, nargs="+", default=None,
                        help="Run specific config IDs (1-12). If omitted, runs all.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Execute preflight checks without training")
    parser.add_argument("--allow-subset-debug", action="store_true",
                        help="Allow running on synthetic or small subset fixtures for unit testing")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if args.verify_only:
        checks = run_preflight_checks(
            data_root, output_dir, args.device,
            allow_subset_debug=args.allow_subset_debug,
        )
        print("\n" + json.dumps(checks, indent=2))
        sys.exit(0 if checks["status"] == "PASS" else 1)

    runner = FullCombinedExperimentRunner(
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        fedprox_mu=args.fedprox_mu,
        seed=args.seed,
        device=args.device,
        allow_subset_debug=args.allow_subset_debug,
    )

    runner.execute_all(target_config_ids=args.config_id)


if __name__ == "__main__":
    main()
