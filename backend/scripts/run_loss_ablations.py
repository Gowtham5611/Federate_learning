"""Controlled pilot ablation comparing Unweighted BCE, Weighted BCE, and Focal Loss."""

import json
import logging
import sys
import time
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
from infrastructure.data.partitioners.client_sequence_provider import ClientSequenceProvider
from infrastructure.data.partitioners.iid_partitioner import IIDPartitioner
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.models.pytorch.loss_functions import build_loss_criterion
from infrastructure.models.pytorch.reproducibility import seed_everything
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("loss_ablation")

RUN_DIR = BASE_DIR / "artifacts" / "li_medium_run"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ibm_aml" / "LI-Medium"

# Mathematically grounded pos_weight for LI-Medium: 18,744,079 / 8,138 = 2303.277
LI_MEDIUM_POS_WEIGHT = 2303.2774


class CustomCriterionClient(FederatedClient):
    """FederatedClient that injects a custom loss criterion (Weighted BCE or Focal Loss)."""

    def __init__(self, *args, custom_criterion: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_criterion = custom_criterion

    def train_round(self, global_parameters, *, round_num: int, local_epochs: int = 1):
        started = time.perf_counter()
        seed = self._config.random_seed + (round_num * 1000) + self._client_index
        seed_everything(seed)
        device = torch.device("cpu")

        model = build_classifier(self._config, self._dataset.feature_dimension)
        if isinstance(self._engine, TorchTrainingEngine):
            self._engine._load_into(model, global_parameters, self._dataset.feature_dimension)
        model.to(device)

        if isinstance(self._engine, FedProxTrainingEngine):
            self._engine._global_parameters = global_parameters

        criterion = self._custom_criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)

        total_loss = 0.0
        for ep in range(1, local_epochs + 1):
            ep_loss = self._engine._train_one_epoch(
                self._config, model, self._dataset, criterion, optimizer, ep
            )
            total_loss += ep_loss

        avg_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
        new_state = {name: p.detach().to("cpu", copy=True) for name, p in model.state_dict().items()}
        return new_state, avg_loss, self._dataset.sequence_count, time.perf_counter() - started


def run_ablation(
    loss_type: str,
    algo: str = "fedavg",
    mu: float | None = None,
    rounds: int = 3,
    row_limit: int = 200000,
    seed: int = 42,
) -> dict:
    logger.info("Starting ablation: Loss=%s | Algo=%s (mu=%s) | Rounds=%d", loss_type, algo, str(mu), rounds)
    sequence_provider = ProcessedSequenceProvider(PROCESSED_DIR)
    metrics_calc = SklearnMetricsCalculator()
    eval_engine = TorchTrainingEngine(metrics_calc)

    training_config = TrainingConfig(
        architecture=ModelArchitecture.GRU,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.001,
        batch_size=1024,
        epochs=rounds,
        random_seed=seed,
        row_limit=row_limit,
    )

    # 1. Read and partition
    train_path = sequence_provider._metadata.split_path(SplitName.TRAIN.value)
    train_cols = list(sequence_provider._metadata.columns_to_read())
    train_dtypes = sequence_provider._read_dtypes()
    train_frame = pd.read_csv(train_path, usecols=train_cols, dtype=train_dtypes, nrows=row_limit)[train_cols]

    partitioner = IIDPartitioner(n_clients=3, seed=seed)
    client_frames, _ = partitioner.partition(train_frame, target_column="is_laundering")
    del train_frame

    # Criterion factory
    if loss_type == "bce":
        crit = build_loss_criterion("bce")
    elif loss_type == "weighted_bce":
        crit = build_loss_criterion("weighted_bce", pos_weight_value=LI_MEDIUM_POS_WEIGHT)
    elif loss_type == "focal":
        crit = build_loss_criterion("focal", alpha=0.25, gamma=2.0)
    else:
        raise ValueError(loss_type)

    # 2. Build clients
    client_seq_provider = ClientSequenceProvider(sequence_provider._metadata)
    clients = []
    for idx, (cid, c_df) in enumerate(client_frames.items()):
        c_dataset = client_seq_provider.build_client_dataset(
            c_df, training_config, client_id=cid, split=SplitName.TRAIN
        )
        if algo == "fedprox":
            engine = FedProxTrainingEngine(metrics_calc, mu=0.1)
        else:
            engine = TorchTrainingEngine(metrics_calc)

        clients.append(
            CustomCriterionClient(
                client_id=cid,
                dataset=c_dataset,
                engine=engine,
                config=training_config,
                client_index=idx,
                custom_criterion=crit,
            )
        )
    del client_frames

    # 3. Validation & Test datasets
    val_dataset = sequence_provider.load(SplitName.VALIDATION, training_config)
    test_dataset = sequence_provider.load(SplitName.TEST, training_config)

    # 4. Train
    seed_everything(seed)
    model_init = build_classifier(training_config, val_dataset.feature_dimension)
    current_state = {k: v.detach().to("cpu", copy=True) for k, v in model_init.state_dict().items()}
    del model_init

    best_val_pr_auc = -1.0
    best_state = current_state
    best_round = 1
    best_val_preds = None

    for r in range(1, rounds + 1):
        client_states = {}
        client_counts = {}
        for c in clients:
            s_k, _, n_k, _ = c.train_round(current_state, round_num=r, local_epochs=1)
            client_states[c.client_id] = s_k
            client_counts[c.client_id] = n_k

        weights = compute_aggregation_weights(client_counts)
        current_state = aggregate_fedavg(client_states, weights)

        val_preds = eval_engine.predict(training_config, val_dataset, current_state)
        val_eval = metrics_calc.evaluate(val_preds.labels, val_preds.probabilities, threshold=0.5, loss=val_preds.mean_loss)
        logger.info("Round %d/%d: Val Loss=%.6f, Val PR-AUC=%.6f, Val ROC-AUC=%.6f",
                    r, rounds, val_preds.mean_loss, val_eval.pr_auc, val_eval.roc_auc)

        if val_eval.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_eval.pr_auc
            best_round = r
            best_state = current_state
            best_val_preds = val_preds

    # Freeze threshold on validation
    frozen_thresh = metrics_calc.select_threshold(best_val_preds.labels, best_val_preds.probabilities, split=SplitName.VALIDATION)

    # Test evaluation
    test_preds = eval_engine.predict(training_config, test_dataset, best_state)
    test_eval_val_th = metrics_calc.evaluate(test_preds.labels, test_preds.probabilities, threshold=frozen_thresh, loss=test_preds.mean_loss)
    test_eval_fixed_01 = metrics_calc.evaluate(test_preds.labels, test_preds.probabilities, threshold=0.1, loss=test_preds.mean_loss)

    return {
        "loss_type": loss_type,
        "federated_algorithm": algo,
        "mu": mu if mu is not None else "na",
        "rounds": rounds,
        "best_round": best_round,
        "best_val_pr_auc": float(best_val_pr_auc),
        "test_pr_auc": float(test_eval_val_th.pr_auc),
        "test_roc_auc": float(test_eval_val_th.roc_auc),
        "val_threshold": float(frozen_thresh),
        "f1_at_val_thresh": float(test_eval_val_th.f1),
        "recall_at_val_thresh": float(test_eval_val_th.recall),
        "precision_at_val_thresh": float(test_eval_val_th.precision),
        "f1_at_fixed_0_1": float(test_eval_fixed_01.f1),
        "recall_at_fixed_0_1": float(test_eval_fixed_01.recall),
        "precision_at_fixed_0_1": float(test_eval_fixed_01.precision),
        "tp_fixed_0_1": test_eval_fixed_01.confusion.true_positives,
        "fp_fixed_0_1": test_eval_fixed_01.confusion.false_positives,
    }


def main():
    experiments = [
        ("bce", "fedavg", None),
        ("weighted_bce", "fedavg", None),
        ("focal", "fedavg", None),
        ("bce", "fedprox", 0.1),
        ("weighted_bce", "fedprox", 0.1),
        ("focal", "fedprox", 0.1),
    ]
    results = []
    for l_type, algo, mu_val in experiments:
        res = run_ablation(l_type, algo, mu_val, rounds=3, row_limit=200000)
        results.append(res)

    df_res = pd.DataFrame(results)
    out_csv = RUN_DIR / "loss_ablation_results.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info("Saved ablation results to %s:\n%s", out_csv, df_res.to_string())


if __name__ == "__main__":
    main()

