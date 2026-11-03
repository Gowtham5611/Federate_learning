"""Federated server orchestrating communication rounds and centralized validation.

Coordinates the federated learning protocol:
1. Global model initialization and broadcast to clients.
2. Local training execution on each client.
3. Collection of client parameter states and sample counts.
4. Federated aggregation (FedAvg sample-weighted aggregation).
5. Centralized validation on the global model after every round.
6. Checkpointing of round, best, and final models.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from application.dto.experiment import (
    ClassificationMetrics,
    ExperimentResult,
)
from application.dto.federated import (
    FederatedConfig,
    RoundRecord,
)
from application.ports.output.experiment_store import Checkpoint
from application.ports.output.federated_store import FederatedCheckpointStorePort
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    ParameterState,
    SplitPredictions,
    TrainingEnginePort,
)
from application.ports.output.sequence_data import SequenceDataset
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.split_name import SplitName
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy, TrainingMode
from infrastructure.federated.aggregation.fedavg import (
    aggregate_fedavg,
    compute_aggregation_weights,
)
from infrastructure.federated.aggregation.scaffold import (
    SCAFFOLDTrainingEngine,
    aggregate_scaffold,
    init_control_variate,
)
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.metrics.drift_tracker import compute_drift_metrics
from infrastructure.models.pytorch.reproducibility import (
    seed_everything,
)
from infrastructure.models.pytorch.sequence_classifier import build_classifier

logger = logging.getLogger(__name__)


class FederatedServer:
    """Orchestrates communication rounds, client coordination, and global evaluation."""

    def __init__(
        self,
        *,
        engine: TrainingEnginePort,
        metrics: MetricsCalculatorPort,
        checkpoint_store: FederatedCheckpointStorePort,
    ) -> None:
        """Initialize federated server.

        Args:
            engine: Model training and evaluation engine.
            metrics: Metrics calculator for centralized validation.
            checkpoint_store: Store for persisting global checkpoints.
        """
        self._engine = engine
        self._metrics = metrics
        self._checkpoint_store = checkpoint_store

    def initialize_global_model(
        self,
        config: FederatedConfig,
        feature_dimension: int,
    ) -> ParameterState:
        """Initialize the global model parameters with the configured random seed."""
        seed_everything(config.seed)
        model = build_classifier(config.training_config, feature_dimension)
        return {
            name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in model.state_dict().items()
        }

    def run_rounds(
        self,
        *,
        experiment_id: str,
        fed_config: FederatedConfig,
        clients: Sequence[FederatedClient],
        validation_dataset: SequenceDataset,
        initial_state: ParameterState,
    ) -> tuple[
        tuple[RoundRecord, ...],
        int,  # best_round
        ParameterState,  # best_state
        ParameterState,  # final_state
        tuple[ExperimentResult, ...],  # per-round validation results
        SplitPredictions,  # best validation predictions
    ]:
        """Execute all federated communication rounds.

        Returns:
            Tuple of:
            - Tuple of RoundRecord for each round
            - Index of best round
            - ParameterState of best global model
            - ParameterState of final global model
            - Tuple of ExperimentResult rows for each round's validation
            - SplitPredictions for the best global model on validation
        """
        current_state = initial_state
        best_state = initial_state
        best_round = 1
        best_score: float | None = None
        best_val_predictions: SplitPredictions | None = None

        round_records: list[RoundRecord] = []
        val_result_rows: list[ExperimentResult] = []

        cfg = fed_config.training_config
        algo = fed_config.algorithm.lower()
        is_scaffold = algo in {"scaffold", "fedprox_scaffold"}
        if algo == "fedprox":
            loss_type = "fedprox"
        elif algo == "scaffold":
            loss_type = "scaffold"
        elif algo == "fedprox_scaffold":
            loss_type = "fedprox_scaffold"
        else:
            loss_type = "bce_with_logits"

        global_control_variate = init_control_variate(initial_state) if is_scaffold else {}

        for r in range(1, fed_config.rounds + 1):
            round_start = time.perf_counter()
            round_start_state = current_state
            logger.info("=== Starting Federated Round %d/%d ===", r, fed_config.rounds)

            # 1. Local client training from current_state
            client_states: dict[str, ParameterState] = {}
            client_losses: dict[str, float] = {}
            client_counts: dict[str, int] = {}
            client_deltas: dict[str, dict[str, Any]] = {}

            for client in clients:
                if is_scaffold and isinstance(client.engine, SCAFFOLDTrainingEngine):
                    client.engine.set_control_variates(
                        client.engine.client_control_variate, global_control_variate
                    )

                state_k, loss_k, n_k, _ = client.train_round(
                    current_state,
                    round_num=r,
                    local_epochs=fed_config.local_epochs,
                )
                client_states[client.client_id] = state_k
                client_losses[client.client_id] = loss_k
                client_counts[client.client_id] = n_k
                if is_scaffold:
                    client_deltas[client.client_id] = client.control_variate_delta

            # 2. Server aggregation
            weights = compute_aggregation_weights(client_counts)
            if is_scaffold:
                current_state, global_control_variate = aggregate_scaffold(
                    client_states, client_deltas, global_control_variate, weights
                )
            else:
                current_state = aggregate_fedavg(client_states, weights)

            # 3. Quantitative client drift tracking
            drift_metrics = compute_drift_metrics(round_start_state, client_states, weights)

            # 4. Centralized validation of the new global model
            val_preds = self._engine.predict(cfg, validation_dataset, current_state)
            val_metrics = self._metrics.evaluate(
                val_preds.labels,
                val_preds.probabilities,
                threshold=cfg.threshold,
                loss=val_preds.mean_loss,
            )

            round_duration = time.perf_counter() - round_start

            # Check if this round improved validation score
            score = _selection_value(cfg.selection_metric, val_metrics, val_preds.mean_loss)
            improved = best_score is None or _is_better(cfg.selection_metric, score, best_score)

            if improved:
                best_score = score
                best_round = r
                best_state = current_state
                best_val_predictions = val_preds

            # 4. Save global checkpoint for round
            ckpt = Checkpoint(
                experiment_id=experiment_id,
                kind=CheckpointKind.GLOBAL_ROUND,
                architecture=cfg.architecture,
                epoch=r,
                parameters=current_state,
                config=cfg,
                feature_dimension=validation_dataset.feature_dimension,
                threshold=cfg.threshold,
                validation_loss=val_preds.mean_loss,
                random_seed=fed_config.seed,
                validation_metrics=val_metrics.to_dict(),
            )
            ckpt_path = self._checkpoint_store.save_round(ckpt, round_num=r)

            if improved:
                best_ckpt = Checkpoint(
                    experiment_id=experiment_id,
                    kind=CheckpointKind.GLOBAL_BEST,
                    architecture=cfg.architecture,
                    epoch=r,
                    parameters=best_state,
                    config=cfg,
                    feature_dimension=validation_dataset.feature_dimension,
                    threshold=cfg.threshold,
                    validation_loss=val_preds.mean_loss,
                    random_seed=fed_config.seed,
                    validation_metrics=val_metrics.to_dict(),
                )
                self._checkpoint_store.save_best(best_ckpt)

            record = RoundRecord(
                round=r,
                aggregation_weights=weights,
                client_train_losses=client_losses,
                client_sample_counts=client_counts,
                validation_loss=val_preds.mean_loss,
                validation_metrics=val_metrics,
                round_seconds=round_duration,
                checkpoint_path=str(ckpt_path),
                selected=improved,
                drift_metrics=drift_metrics,
            )
            round_records.append(record)

            # Record round validation result row
            val_row = ExperimentResult(
                experiment_id=experiment_id,
                architecture=cfg.architecture,
                training_mode=TrainingMode.FEDERATED,
                split=SplitName.VALIDATION,
                metrics=val_metrics,
                threshold_policy=ThresholdPolicy.FIXED,
                seed=fed_config.seed,
                loss_type=loss_type,
                training_time_seconds=round_duration,
                evaluation_time_seconds=val_preds.seconds,
                sequence_count=len(val_preds),
                hyperparameters=fed_config.to_dict(),
                data_distribution=fed_config.distribution,
                federated_algorithm=fed_config.algorithm,
                client_id=None,
                federated_round=r,
                mu=fed_config.mu,
            )
            val_result_rows.append(val_row)

            logger.info(
                "Round %d/%d complete: val_loss=%.6f, val_pr_auc=%.6f, "
                "val_recall=%.6f, mean_drift=%.6f, time=%.1fs%s",
                r,
                fed_config.rounds,
                val_preds.mean_loss,
                val_metrics.pr_auc,
                val_metrics.recall,
                drift_metrics.get("mean_drift", 0.0),
                round_duration,
                " (best global model so far)" if improved else "",
            )

        # Mark selected round in records
        final_records = tuple(
            RoundRecord(
                round=rec.round,
                aggregation_weights=rec.aggregation_weights,
                client_train_losses=rec.client_train_losses,
                client_sample_counts=rec.client_sample_counts,
                validation_loss=rec.validation_loss,
                validation_metrics=rec.validation_metrics,
                round_seconds=rec.round_seconds,
                checkpoint_path=rec.checkpoint_path,
                selected=(rec.round == best_round),
                drift_metrics=rec.drift_metrics,
            )
            for rec in round_records
        )

        # Save final checkpoint
        final_ckpt = Checkpoint(
            experiment_id=experiment_id,
            kind=CheckpointKind.FINAL,
            architecture=cfg.architecture,
            epoch=fed_config.rounds,
            parameters=current_state,
            config=cfg,
            feature_dimension=validation_dataset.feature_dimension,
            threshold=cfg.threshold,
            validation_loss=round_records[-1].validation_loss,
            random_seed=fed_config.seed,
            validation_metrics=round_records[-1].validation_metrics.to_dict(),
        )
        self._checkpoint_store.save_final(final_ckpt)

        if best_val_predictions is None:
            best_val_predictions = self._engine.predict(cfg, validation_dataset, best_state)

        return (
            final_records,
            best_round,
            best_state,
            current_state,
            tuple(val_result_rows),
            best_val_predictions,
        )


def _selection_value(
    metric: SelectionMetric, measured: ClassificationMetrics, loss: float
) -> float:
    """Extract metric value for checkpoint selection."""
    if metric is SelectionMetric.LOSS:
        return loss
    if metric is SelectionMetric.RECALL:
        return measured.recall
    if metric is SelectionMetric.F1:
        return measured.f1
    return measured.pr_auc


def _is_better(metric: SelectionMetric, candidate: float, incumbent: float) -> bool:
    """Check if candidate score outperforms incumbent."""
    if metric.higher_is_better:
        return candidate > incumbent
    return candidate < incumbent
