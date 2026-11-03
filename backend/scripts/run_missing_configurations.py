"""Execution runner for the four missing baseline configurations (4, 5, 6, 12).

Supports:
- Atomic checkpointing per round
- Resumption from existing round checkpoints
- Complete logging of client training, aggregation, and validation metrics
- Single-pass held-out test evaluation with frozen validation threshold
- Graceful termination and status recording
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add backend to path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from application.services.fedsecureai_run_service import FedSecureAIRunService


def run_configuration(
    config_num: int,
    base_runs_dir: Path,
    processed_dir: Path,
    *,
    rounds: int = 5,
    local_epochs: int = 1,
    row_limit: int | None = None,
    batch_size: int = 1024,
) -> None:
    config_map = {
        4: ("iid", "fedprox", "lstm"),
        5: ("iid", "fedavg", "bilstm"),
        6: ("iid", "fedprox", "bilstm"),
        12: ("non_iid", "fedprox", "bilstm"),
    }
    if config_num not in config_map:
        raise ValueError(f"Config {config_num} not in missing configurations (4, 5, 6, 12)")

    dist, algo, model = config_map[config_num]
    print(f"\n=======================================================")
    print(f"Launching Config {config_num}: {dist.upper()} {algo.upper()} {model.upper()}")
    print(f"  Rounds: {rounds}, Local Epochs: {local_epochs}, Batch Size: {batch_size}, Row Limit: {row_limit}")
    print(f"=======================================================")

    service = FedSecureAIRunService(
        base_runs_dir=base_runs_dir,
        processed_dir=processed_dir,
        rounds=rounds,
        local_epochs=local_epochs,
        row_limit=row_limit,
        batch_size=batch_size,
    )
    rec = service.run_single_experiment(config_num, dist, algo, model)
    print(f"\n[Result Config {config_num}] Status: {rec.status}")
    print(f"  Best Val PR-AUC: {rec.best_val_pr_auc}")
    print(f"  Test PR-AUC: {rec.test_pr_auc}")
    print(f"  Test ROC-AUC: {rec.test_roc_auc}")
    print(f"  Test F1: {rec.test_f1}")
    print(f"  Checkpoint: {rec.checkpoint_path}")
    if rec.error_message:
        print(f"  Error: {rec.error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedSecureAI missing configurations.")
    parser.add_argument("--config", type=int, required=True, choices=[4, 5, 6, 12], help="Config number to execute")
    parser.add_argument("--rounds", type=int, default=5, help="Number of communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--row-limit", type=int, default=None, help="Row limit (default: full dataset)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="experiments/results/fedsecureai_runs/20261016_continuation")
    args = parser.parse_args()

    run_configuration(
        args.config,
        Path(args.output_dir),
        Path("data/processed/ibm_aml/HI-Small"),
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        row_limit=args.row_limit,
        batch_size=args.batch_size,
    )

