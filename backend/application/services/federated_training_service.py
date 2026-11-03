"""Federated training application service.

Coordinates the end-to-end execution of a federated experiment:
1. Partitions the training split into 3 deterministic stratified IID clients.
2. Initializes client sequence datasets and client instances.
3. Coordinates communication rounds via FederatedServer.
4. Performs centralized validation and threshold selection on validation.
5. Scores the final best global model on test (never used for training/selection).
6. Persists reports, checkpoints, and comparison results.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from application.dto.experiment import (
    ExperimentResult,
)
from application.dto.federated import (
    FederatedConfig,
    FederatedExperimentReport,
    IIDPartitionReport,
    NonIIDPartitionReport,
)
from application.ports.output.experiment_store import ExperimentResultStorePort
from application.ports.output.federated_store import FederatedCheckpointStorePort
from application.ports.output.metrics_calculator import MetricsCalculatorPort
from application.ports.output.model_trainer import (
    SplitPredictions,
    TrainingEnginePort,
)
from domain.enums.split_name import SplitName
from domain.enums.training_mode import ThresholdPolicy, TrainingMode
from domain.exceptions.training import FeatureDimensionMismatchError
from infrastructure.data.partitioners.client_sequence_provider import (
    ClientSequenceProvider,
)
from infrastructure.data.partitioners.iid_partitioner import IIDPartitioner
from infrastructure.data.partitioners.non_iid_partitioner import NonIIDPartitioner
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.federated.aggregation.fedprox import FedProxTrainingEngine
from infrastructure.federated.aggregation.fedprox_scaffold import FedProxSCAFFOLDTrainingEngine
from infrastructure.federated.aggregation.scaffold import SCAFFOLDTrainingEngine
from infrastructure.federated.clients.federated_client import FederatedClient
from infrastructure.federated.server.federated_server import FederatedServer
from infrastructure.models.pytorch.reproducibility import (
    describe_runtime,
    request_deterministic_kernels,
    resolve_device,
)
from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

logger = logging.getLogger(__name__)


class FederatedTrainingService:
    """Orchestrates an end-to-end federated learning experiment."""

    def __init__(
        self,
        *,
        sequence_provider: ProcessedSequenceProvider,
        metrics: MetricsCalculatorPort,
        checkpoint_store: FederatedCheckpointStorePort,
        result_store: ExperimentResultStorePort,
        training_engine: TrainingEnginePort,
    ) -> None:
        """Initialize service.

        Args:
            sequence_provider: Provider for raw/processed splits and metadata.
            metrics: Metrics calculator for validation scoring and thresholding.
            checkpoint_store: Store for saving global models.
            result_store: Store for appending result rows and writing reports.
            training_engine: Engine for model scoring and evaluation.
        """
        self._sequences = sequence_provider
        self._metrics = metrics
        self._checkpoints = checkpoint_store
        self._results = result_store
        self._engine = training_engine

    def run(self, fed_config: FederatedConfig) -> FederatedExperimentReport:
        """Execute a full federated experiment."""
        started = time.perf_counter()
        config = fed_config.training_config
        experiment_id = self._generate_experiment_id(fed_config)

        device = resolve_device(config.device)
        deterministic = request_deterministic_kernels()
        runtime = describe_runtime(device, deterministic=deterministic)

        logger.info(
            "Starting Federated Experiment %s | Model=%s | Dist=%s | Algo=%s | "
            "Clients=%d | Rounds=%d | LocalEpochs=%d | Seed=%d | Mu=%s",
            experiment_id,
            config.architecture.display_name,
            fed_config.distribution,
            fed_config.algorithm,
            fed_config.n_clients,
            fed_config.rounds,
            fed_config.local_epochs,
            fed_config.seed,
            fed_config.mu,
        )

        prep_started = time.perf_counter()

        # 1. Read the training split as a DataFrame to partition
        train_path = self._sequences._metadata.split_path(SplitName.TRAIN.value)
        train_cols = list(self._sequences._metadata.columns_to_read())
        train_dtypes = self._sequences._read_dtypes()

        train_frame = pd.read_csv(
            train_path,
            usecols=train_cols,
            dtype=train_dtypes,
            nrows=config.row_limit,
        )[train_cols]

        # 2. Partition training data across clients
        partition_report: IIDPartitionReport | NonIIDPartitionReport
        dist_norm = fed_config.distribution.lower().replace("-", "_")
        if dist_norm == "non_iid":
            non_iid_partitioner = NonIIDPartitioner(seed=fed_config.seed)
            client_frames, partition_report = non_iid_partitioner.partition(
                train_frame,
                target_column=self._sequences._metadata.target_column,
                account_column=self._sequences._metadata.sender_column,
            )
        else:
            iid_partitioner = IIDPartitioner(
                n_clients=fed_config.n_clients,
                seed=fed_config.seed,
            )
            client_frames, partition_report = iid_partitioner.partition(
                train_frame,
                target_column=self._sequences._metadata.target_column,
            )
        del train_frame

        # 3. Build client sequence datasets and client instances
        client_seq_provider = ClientSequenceProvider(self._sequences._metadata)
        clients: list[FederatedClient] = []

        for idx, (cid, c_df) in enumerate(client_frames.items()):
            c_dataset = client_seq_provider.build_client_dataset(
                c_df, config, client_id=cid, split=SplitName.TRAIN
            )
            # Choose local engine: FedProx, SCAFFOLD, FedProx-SCAFFOLD, or FedAvg (standard)
            algo = fed_config.algorithm.lower()
            if algo == "fedprox":
                local_engine: TrainingEnginePort = FedProxTrainingEngine(
                    self._metrics, mu=fed_config.mu if fed_config.mu is not None else 0.01
                )
            elif algo == "scaffold":
                local_engine = SCAFFOLDTrainingEngine(self._metrics)
            elif algo == "fedprox_scaffold":
                local_engine = FedProxSCAFFOLDTrainingEngine(
                    self._metrics, mu=fed_config.mu if fed_config.mu is not None else 0.001
                )
            else:
                local_engine = TorchTrainingEngine(self._metrics)

            clients.append(
                FederatedClient(
                    client_id=cid,
                    dataset=c_dataset,
                    engine=local_engine,
                    config=config,
                    client_index=idx,
                )
            )

        # 4. Load centralized validation dataset
        validation = self._sequences.load(SplitName.VALIDATION, config)
        prep_seconds = time.perf_counter() - prep_started

        feature_dim = validation.feature_dimension
        for c in clients:
            if c._dataset.feature_dimension != feature_dim:
                raise FeatureDimensionMismatchError(
                    feature_dim,
                    c._dataset.feature_dimension,
                    f"client {c.client_id} dataset",
                )

        # 5. Initialize Server and run rounds
        server = FederatedServer(
            engine=self._engine,
            metrics=self._metrics,
            checkpoint_store=self._checkpoints,
        )
        initial_state = server.initialize_global_model(fed_config, feature_dim)

        training_start = time.perf_counter()
        (
            round_records,
            best_round,
            best_state,
            final_state,
            val_result_rows,
            best_val_preds,
        ) = server.run_rounds(
            experiment_id=experiment_id,
            fed_config=fed_config,
            clients=clients,
            validation_dataset=validation,
            initial_state=initial_state,
        )
        training_seconds = time.perf_counter() - training_start

        # Clean up client datasets before evaluating test
        del clients
        del client_frames

        # 6. Freeze threshold on validation using best global model
        frozen_threshold, threshold_selected_on = self._freeze_threshold(
            config, best_val_preds
        )
        logger.info(
            "Experiment %s froze threshold %.6f (%s) from best round %d",
            experiment_id,
            frozen_threshold,
            f"fitted on {threshold_selected_on}" if threshold_selected_on else "configured",
            best_round,
        )

        del validation

        # 7. Final evaluation on TEST using best global model
        test = self._sequences.load(SplitName.TEST, config)
        test_preds = self._engine.predict(config, test, best_state)
        del test

        algo = fed_config.algorithm.lower()
        if algo == "fedprox":
            loss_type = "fedprox"
        elif algo == "scaffold":
            loss_type = "scaffold"
        elif algo == "fedprox_scaffold":
            loss_type = "fedprox_scaffold"
        else:
            loss_type = "bce_with_logits"

        # Final evaluation rows
        final_rows = [
            *val_result_rows,
            # Validation at frozen threshold (if learned)
            *(
                [
                    self._create_result_row(
                        experiment_id=experiment_id,
                        fed_config=fed_config,
                        split=SplitName.VALIDATION,
                        predictions=best_val_preds,
                        threshold=frozen_threshold,
                        policy=config.threshold_policy,
                        round_num=best_round,
                        loss_type=loss_type,
                        training_time=training_seconds,
                    )
                ]
                if config.threshold_policy.is_learned
                else []
            ),
            # Test at fixed threshold
            self._create_result_row(
                experiment_id=experiment_id,
                fed_config=fed_config,
                split=SplitName.TEST,
                predictions=test_preds,
                threshold=config.threshold,
                policy=ThresholdPolicy.FIXED,
                round_num=best_round,
                loss_type=loss_type,
                training_time=training_seconds,
            ),
            # Test at frozen threshold (if learned)
            *(
                [
                    self._create_result_row(
                        experiment_id=experiment_id,
                        fed_config=fed_config,
                        split=SplitName.TEST,
                        predictions=test_preds,
                        threshold=frozen_threshold,
                        policy=config.threshold_policy,
                        round_num=best_round,
                        loss_type=loss_type,
                        training_time=training_seconds,
                    )
                ]
                if config.threshold_policy.is_learned
                else []
            ),
        ]

        total_seconds = time.perf_counter() - started
        param_count = self._engine.count_parameters(config, feature_dim)

        report = FederatedExperimentReport(
            experiment_id=experiment_id,
            config=fed_config,
            partition_report=partition_report,
            rounds=round_records,
            results=tuple(final_rows),
            selected_round=best_round,
            frozen_threshold=frozen_threshold,
            threshold_selected_on=threshold_selected_on,
            device=str(device),
            parameter_count=param_count,
            training_seconds=training_seconds,
            data_preparation_seconds=prep_seconds,
            total_seconds=total_seconds,
            torch_version=str(runtime.get("framework_version", "unknown")),
            deterministic=deterministic,
        )

        # 8. Persist results
        self._results.save_report(report)  # type: ignore[arg-type]
        self._results.append_rows(report)  # type: ignore[arg-type]

        logger.info(
            "Federated experiment %s complete in %.1fs. Selected round %d.",
            experiment_id,
            total_seconds,
            best_round,
        )

        return report

    def _freeze_threshold(
        self, config: Any, predictions: SplitPredictions
    ) -> tuple[float, str | None]:
        """Determine decision threshold to apply to test."""
        if not config.threshold_policy.is_learned:
            return config.threshold, None
        chosen = self._metrics.select_threshold(
            predictions.labels,
            predictions.probabilities,
            split=SplitName.VALIDATION,
        )
        return chosen, SplitName.VALIDATION.value

    def _create_result_row(
        self,
        *,
        experiment_id: str,
        fed_config: FederatedConfig,
        split: SplitName,
        predictions: SplitPredictions,
        threshold: float,
        policy: ThresholdPolicy,
        round_num: int,
        loss_type: str,
        training_time: float,
    ) -> ExperimentResult:
        """Create a single flat ExperimentResult row."""
        measured = self._metrics.evaluate(
            predictions.labels,
            predictions.probabilities,
            threshold=threshold,
            loss=predictions.mean_loss,
        )
        return ExperimentResult(
            experiment_id=experiment_id,
            architecture=fed_config.training_config.architecture,
            training_mode=TrainingMode.FEDERATED,
            split=split,
            metrics=measured,
            threshold_policy=policy,
            seed=fed_config.seed,
            loss_type=loss_type,
            training_time_seconds=training_time,
            evaluation_time_seconds=predictions.seconds,
            sequence_count=len(predictions),
            hyperparameters=fed_config.to_dict(),
            data_distribution=fed_config.distribution,
            federated_algorithm=fed_config.algorithm,
            client_id=None,
            federated_round=round_num,
            mu=fed_config.mu,
        )

    @staticmethod
    def _generate_experiment_id(fed_config: FederatedConfig) -> str:
        """Construct filesystem-safe unique experiment ID."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        arch = fed_config.training_config.architecture.value
        dist = fed_config.distribution.replace("-", "_")
        algo = fed_config.algorithm
        seed = fed_config.seed
        return f"federated_{dist}_{algo}_{arch}_seed{seed}_{stamp}"

