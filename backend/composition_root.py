"""FedSecureAI entry point and composition root.

This is the only module that knows which concrete adapters back which ports.
Everything below it depends on abstractions, which is what allows the dataset
manager to be tested with no network, no credentials and no Kaggle package.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from application.services.analysis_service import AnalysisService
    from application.services.calibration_service import CalibrationService

from application.dto.calibration import CalibrationConfig
from application.dto.ensemble import EnsembleConfig
from application.dto.federated import FederatedConfig
from application.dto.training import TrainingConfig
from application.services.dataset_manager import DatasetManager
from application.services.preprocessing_service import PreprocessingService
from domain.enums.model_architecture import ModelArchitecture
from domain.exceptions.base import FedSecureAIError
from infrastructure.data.downloaders.kaggle_downloader import KaggleDownloader
from infrastructure.data.loaders.csv_chunk_loader import CsvChunkLoader
from infrastructure.data.preprocessors.aml_preprocessor import (
    RAW_DTYPES,
    AmlPreprocessor,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    build_ibm_aml_transaction_schema,
)
from infrastructure.data.sequences.processed_sequence_provider import (
    ProcessedSequenceProvider,
)
from infrastructure.data.sources.ibm_aml_data_source import IBMAMLDataSource
from infrastructure.data.sources.yaml_dataset_registry import YamlDatasetRegistry
from infrastructure.data.splitters.chronological_splitter import ChronologicalSplitter
from infrastructure.data.writers.csv_interim_storage import CsvInterimStorage
from infrastructure.data.writers.csv_processed_data_writer import CsvProcessedDataWriter
from infrastructure.persistence.csv_experiment_result_store import (
    CsvExperimentResultStore,
)
from infrastructure.persistence.ensemble_result_store import EnsembleResultStore
from infrastructure.utils.config_loader import ConfigurationError, load_config
from infrastructure.utils.env import load_env_file
from infrastructure.utils.logging_config import configure_logging
from infrastructure.utils.paths import PROJECT_ROOT, resolve

if TYPE_CHECKING:
    from application.services.centralized_training_service import CentralizedTrainingService
    from application.services.ensemble_service import EnsembleService
    from application.services.federated_training_service import FederatedTrainingService
from interfaces.cli.app import run as run_cli

DEFAULT_CHUNK_SIZE = 200_000
"""Fallback rows-per-read when configuration does not say."""


def build_dataset_manager(config: dict[str, Any]) -> DatasetManager:
    """Compose the dataset manager from its concrete adapters.

    Neither adapter performs I/O or requires credentials at construction time,
    so this is safe to call regardless of whether Kaggle is reachable or
    configured.
    """
    registry_path = config.get("paths", {}).get(
        "dataset_registry", "datasets/dataset_registry.yaml"
    )
    dataset_settings = config.get("dataset") or {}
    return DatasetManager(
        registry=YamlDatasetRegistry(resolve(str(registry_path))),
        downloader=KaggleDownloader(),
        project_root=PROJECT_ROOT,
        dataset_id=dataset_settings.get("active_dataset_id"),
    )


def build_preprocessing_service(
    config: dict[str, Any],
    manager: DatasetManager,
    *,
    chunk_size: int | None = None,
    keep_interim: bool = False,
) -> PreprocessingService:
    """Compose the preprocessing pipeline from its concrete adapters.

    The source is the same ``DatasetManager``-backed one the rest of the
    application uses, so preprocessing reads whatever the registry declares and
    cannot reach a different file. Nothing here can download: the source has no
    downloader, by design.

    Args:
        config: Loaded application configuration.
        manager: The dataset lifecycle service, used only to locate the file.
        chunk_size: Overrides ``preprocessing.chunk_size`` when given.
        keep_interim: Overrides ``preprocessing.keep_interim`` when true.

    Returns:
        A service ready to run. Constructing it creates the output directories
        but reads no data.
    """
    settings = config.get("preprocessing") or {}
    paths = config.get("paths") or {}
    split_settings = settings.get("split") or {}
    subdirectory = str(settings.get("output_subdirectory", "ibm_aml/HI-Small"))

    loader = CsvChunkLoader()
    raw_root = resolve(str(paths.get("data_raw", "data/raw")))
    interim_root = resolve(str(paths.get("data_interim", "data/interim")))
    processed_root = resolve(str(paths.get("data_processed", "data/processed")))

    return PreprocessingService(
        source=IBMAMLDataSource(manager),
        loader=loader,
        interim=CsvInterimStorage(
            interim_root / subdirectory, loader, partition_dtypes=RAW_DTYPES
        ),
        writer=CsvProcessedDataWriter(
            processed_root / subdirectory, raw_directory=raw_root
        ),
        preprocessor=AmlPreprocessor(),
        splitter=ChronologicalSplitter(
            train_fraction=float(split_settings.get("train_fraction", 0.60)),
            validation_fraction=float(split_settings.get("validation_fraction", 0.20)),
        ),
        schema=build_ibm_aml_transaction_schema(),
        chunk_size=int(chunk_size or settings.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        keep_interim=keep_interim or bool(settings.get("keep_interim", False)),
    )


def build_training_session(
    config: dict[str, Any], *, overrides: Mapping[str, object] | None = None
) -> tuple[CentralizedTrainingService, TrainingConfig]:
    """Compose the centralized training service and its base configuration.

    Two things are returned rather than one because they are decided in different
    places: the service is the set of adapters, and the configuration is the
    experiment. Keeping them separate is what lets ``--model all`` run three
    experiments through one service with one configuration varied in one field.

    Precedence is configuration file, then command line — so a value the caller did
    not supply keeps its configured default rather than being reset.

    Args:
        config: Loaded application configuration.
        overrides: Command-line values, keyed as in :class:`TrainingConfig` plus
            ``processed_directory``, ``checkpoint_directory`` and
            ``results_directory``.

    Returns:
        The service, ready to run, and the configuration to run it with.

    Raises:
        ProcessedMetadataError: The preprocessing artefacts are missing, which is
            what happens when ``train`` is run before ``preprocess``.
    """
    supplied: dict[str, object] = dict(overrides or {})
    settings = config.get("training") or {}
    paths = config.get("paths") or {}
    preprocessing = config.get("preprocessing") or {}
    reproducibility = config.get("reproducibility") or {}
    subdirectory = str(preprocessing.get("output_subdirectory", "ibm_aml/HI-Small"))

    processed_root = _resolved_directory(
        supplied.pop("processed_directory", None),
        resolve(str(paths.get("data_processed", "data/processed"))) / subdirectory,
    )
    checkpoint_root = _resolved_directory(
        supplied.pop("checkpoint_directory", None),
        resolve(str(paths.get("model_checkpoints", "models/checkpoints"))),
    )
    results_root = _resolved_directory(
        supplied.pop("results_directory", None),
        resolve(str(paths.get("experiment_results", "experiments/results"))),
    )

    # The global seed is the default so that one configured value governs every
    # stochastic component in the project; `--random-seed` still overrides it.
    merged: dict[str, Any] = {"random_seed": reproducibility.get("random_seed", 42)}
    merged.update({key: value for key, value in settings.items() if value is not None})
    merged.update(supplied)
    merged.setdefault("architecture", ModelArchitecture.GRU.value)
    training_config = TrainingConfig.from_dict(merged)

    from application.services.centralized_training_service import CentralizedTrainingService
    from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
    from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine
    from infrastructure.persistence.torch_checkpoint_store import TorchCheckpointStore

    metrics = SklearnMetricsCalculator()
    service = CentralizedTrainingService(
        sequences=ProcessedSequenceProvider(processed_root),
        engine=TorchTrainingEngine(metrics),
        metrics=metrics,
        checkpoints=TorchCheckpointStore(checkpoint_root),
        results=CsvExperimentResultStore.for_centralized(results_root),
    )
    return service, training_config


def build_federated_session(
    config: dict[str, Any], *, overrides: Mapping[str, object] | None = None
) -> tuple[FederatedTrainingService, FederatedConfig]:
    """Compose the federated training service and its base configuration.

    Args:
        config: Loaded application configuration.
        overrides: Command-line overrides.

    Returns:
        The federated training service and base FederatedConfig.
    """
    supplied: dict[str, object] = dict(overrides or {})
    settings = config.get("training") or {}
    fed_settings = config.get("federated") or {}
    paths = config.get("paths") or {}
    preprocessing = config.get("preprocessing") or {}
    reproducibility = config.get("reproducibility") or {}
    subdirectory = str(preprocessing.get("output_subdirectory", "ibm_aml/HI-Small"))

    processed_root = _resolved_directory(
        supplied.pop("processed_directory", None),
        resolve(str(paths.get("data_processed", "data/processed"))) / subdirectory,
    )
    checkpoint_root = _resolved_directory(
        supplied.pop("checkpoint_directory", None),
        resolve(str(paths.get("model_checkpoints", "models/checkpoints"))),
    )
    results_root = _resolved_directory(
        supplied.pop("results_directory", None),
        resolve(str(paths.get("experiment_results", "experiments/results"))),
    )

    algorithm = str(
        supplied.pop("algorithm", fed_settings.get("algorithm", "fedavg"))
    ).lower()
    distribution = str(
        supplied.pop("distribution", fed_settings.get("distribution", "iid"))
    ).lower()
    n_clients = int(str(supplied.pop("n_clients", fed_settings.get("n_clients", 3))))
    rounds = int(str(supplied.pop("rounds", fed_settings.get("rounds", 5))))
    local_epochs = int(str(supplied.pop("local_epochs", fed_settings.get("local_epochs", 1))))
    seed = int(
        str(
            supplied.pop(
                "seed",
                fed_settings.get("seed", reproducibility.get("random_seed", 42)),
            )
        )
    )
    mu_raw = supplied.pop("mu", fed_settings.get("mu"))
    mu: float | None = (
        (float(str(mu_raw)) if mu_raw is not None else 0.01)
        if algorithm in {"fedprox", "fedprox_scaffold"}
        else None
    )

    merged: dict[str, Any] = {"random_seed": seed}
    merged.update({key: value for key, value in settings.items() if value is not None})
    merged.update(supplied)
    merged.setdefault("architecture", ModelArchitecture.GRU.value)
    training_config = TrainingConfig.from_dict(merged)

    fed_config = FederatedConfig(
        training_config=training_config,
        n_clients=n_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        distribution=distribution,
        algorithm=algorithm,
        mu=mu,
        seed=seed,
    )

    dist_folder = "non_iid" if distribution in {"non_iid", "non-iid"} else "iid"
    from application.services.federated_training_service import FederatedTrainingService
    from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
    from infrastructure.federated.server.federated_checkpoint_store import (
        FederatedCheckpointStore,
    )
    from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

    metrics = SklearnMetricsCalculator()
    service = FederatedTrainingService(
        sequence_provider=ProcessedSequenceProvider(processed_root),
        metrics=metrics,
        checkpoint_store=FederatedCheckpointStore(
            checkpoint_root / "federated" / dist_folder, algorithm=algorithm
        ),
        result_store=CsvExperimentResultStore.for_federated(
            results_root, distribution=distribution, algorithm=algorithm
        ),
        training_engine=TorchTrainingEngine(metrics),
    )
    return service, fed_config


def build_ensemble_session(
    config: dict[str, Any],
    *,
    overrides: Mapping[str, object] | None = None,
) -> tuple[EnsembleService, EnsembleConfig]:
    """Assemble an ensemble service and configuration from base settings and overrides."""
    supplied: dict[str, Any] = dict(overrides or {})
    paths = config.get("paths") or {}
    preprocessing = config.get("preprocessing") or {}
    subdirectory = str(preprocessing.get("output_subdirectory", "ibm_aml/HI-Small"))

    processed_root = _resolved_directory(
        supplied.pop("processed_directory", None),
        resolve(str(paths.get("data_processed", "data/processed"))) / subdirectory,
    )
    checkpoint_root = _resolved_directory(
        supplied.pop("checkpoint_directory", None),
        resolve(str(paths.get("model_checkpoints", "models/checkpoints"))),
    )
    results_root = _resolved_directory(
        supplied.pop("results_directory", None),
        resolve(str(paths.get("experiment_results", "experiments/results"))),
    )

    ensemble_cfg_raw = config.get("ensemble", {})
    alpha_step_raw = supplied.pop("alpha_step", ensemble_cfg_raw.get("alpha_step", 0.05))
    alpha_step = float(alpha_step_raw)

    seed_raw = supplied.pop("seed", ensemble_cfg_raw.get("seed", 42))
    seed = int(seed_raw)

    batch_size_raw = supplied.pop("batch_size", ensemble_cfg_raw.get("batch_size", 1024))
    batch_size = int(batch_size_raw)

    row_limit_raw = supplied.pop("row_limit", ensemble_cfg_raw.get("row_limit"))
    row_limit = int(row_limit_raw) if row_limit_raw is not None else None

    device = supplied.pop("device", ensemble_cfg_raw.get("device"))
    device_str = str(device) if device is not None else None

    ensemble_config = EnsembleConfig(
        alpha_step=alpha_step,
        seed=seed,
        batch_size=batch_size,
        row_limit=row_limit,
        device=device_str,
    )

    from application.services.ensemble_service import EnsembleService
    from application.services.model_selector import ModelSelector
    from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
    from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine

    metrics = SklearnMetricsCalculator()
    model_selector = ModelSelector(results_root=results_root, checkpoints_root=checkpoint_root)
    result_store = EnsembleResultStore(root=results_root)
    sequence_provider = ProcessedSequenceProvider(processed_root)
    training_engine = TorchTrainingEngine(metrics)

    service = EnsembleService(
        model_selector=model_selector,
        sequence_provider=sequence_provider,
        metrics=metrics,
        checkpoint_root=checkpoint_root,
        result_store=result_store,
        training_engine=training_engine,
    )
    return service, ensemble_config


def build_analysis_session(
    config: dict[str, Any],
    *,
    overrides: Mapping[str, object],
) -> tuple[AnalysisService, str | None]:
    """Compose the Phase 8 performance analysis service and dependencies."""
    supplied = dict(overrides)

    paths_cfg = config.get("paths", {})
    results_root_default = resolve(paths_cfg.get("results_directory", "experiments/results"))
    results_root = _resolved_directory(
        supplied.pop("results_directory", None), results_root_default
    )

    output_root_default = results_root / "analysis"
    output_root = _resolved_directory(
        supplied.pop("output_directory", None), output_root_default
    )

    analysis_id_raw = supplied.pop("analysis_id", None)
    analysis_id = str(analysis_id_raw) if analysis_id_raw is not None else None
    from application.services.analysis_service import AnalysisService
    from infrastructure.persistence.analysis_result_store import AnalysisResultStore

    store = AnalysisResultStore(root=output_root)
    service = AnalysisService(results_root=results_root, analysis_store=store)

    return service, analysis_id


def build_calibration_session(
    config: dict[str, Any],
    *,
    overrides: Mapping[str, object],
) -> tuple[CalibrationService, CalibrationConfig]:
    """Compose the Phase 10 Probability Calibration service and configuration."""
    supplied = dict(overrides)

    paths_cfg = config.get("paths", {})
    results_root_default = resolve(paths_cfg.get("results_directory", "experiments/results"))
    results_root = _resolved_directory(
        supplied.pop("results_directory", None), results_root_default
    )

    checkpoint_root_default = resolve(paths_cfg.get("checkpoint_directory", "models/checkpoints"))
    checkpoint_root = _resolved_directory(
        supplied.pop("checkpoint_directory", None), checkpoint_root_default
    )

    processed_root_default = resolve(
        paths_cfg.get("processed_data_dir", "data/processed/ibm_aml/HI-Small")
    )
    processed_root = _resolved_directory(
        supplied.pop("processed_directory", None), processed_root_default
    )

    calibration_cfg_raw = config.get("calibration", {})
    seed_raw = supplied.pop("seed", calibration_cfg_raw.get("seed", 42))
    seed_val = int(str(seed_raw))

    batch_size_raw = supplied.pop("batch_size", calibration_cfg_raw.get("batch_size", 1024))
    batch_size_val = int(str(batch_size_raw))

    raw_device = supplied.pop("device", calibration_cfg_raw.get("device"))
    device_val = str(raw_device) if raw_device is not None else None

    row_limit_raw = supplied.pop("row_limit", calibration_cfg_raw.get("row_limit"))
    row_limit_val = int(str(row_limit_raw)) if row_limit_raw is not None else None

    n_bins_ece_raw = supplied.pop("n_bins_ece", calibration_cfg_raw.get("n_bins_ece", 15))
    n_bins_ece_val = int(str(n_bins_ece_raw))

    sel_metric_raw = supplied.pop(
        "selection_metric", calibration_cfg_raw.get("selection_metric", "log_loss")
    )
    selection_metric_val = str(sel_metric_raw)

    calibration_config = CalibrationConfig(
        seed=seed_val,
        batch_size=batch_size_val,
        device=device_val,
        row_limit=row_limit_val,
        n_bins_ece=n_bins_ece_val,
        selection_metric=selection_metric_val,
    )

    from application.services.calibration_service import CalibrationService
    from application.services.model_selector import ModelSelector
    from infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
    from infrastructure.models.pytorch.torch_training_engine import TorchTrainingEngine
    from infrastructure.persistence.calibration_result_store import CalibrationResultStore

    metrics = SklearnMetricsCalculator()
    model_selector = ModelSelector(results_root=results_root, checkpoints_root=checkpoint_root)
    result_store = CalibrationResultStore(root=results_root)
    sequence_provider = ProcessedSequenceProvider(processed_root)
    training_engine = TorchTrainingEngine(metrics)

    service = CalibrationService(
        model_selector=model_selector,
        sequence_provider=sequence_provider,
        metrics=metrics,
        checkpoint_root=checkpoint_root,
        result_store=result_store,
        training_engine=training_engine,
    )
    return service, calibration_config


def _resolved_directory(supplied: object, default: Path) -> Path:
    """Resolve a directory override against the project root, or take the default."""
    if supplied is None:
        return default
    return resolve(str(supplied))


def main(argv: Sequence[str] | None = None) -> int:
    """Configure the application and dispatch to the CLI.

    Returns:
        Process exit code: 0 on success, 1 on a configuration or dataset error.
    """
    # Before configuration, so that `${VAR}` placeholders in the YAML and the
    # Kaggle credential provider both see whatever `.env` defines. Absent file:
    # no effect. No value is logged.
    load_env_file()

    try:
        config = load_config("base")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(config)

    try:
        manager = build_dataset_manager(config)

        def preprocessing_factory(
            *, chunk_size: int | None, keep_interim: bool
        ) -> PreprocessingService:
            """Build the service only if the preprocess command actually runs."""
            return build_preprocessing_service(
                config, manager, chunk_size=chunk_size, keep_interim=keep_interim
            )

        def training_factory(
            *, overrides: Mapping[str, object]
        ) -> tuple[CentralizedTrainingService, TrainingConfig]:
            """Build the training session only if the train command actually runs."""
            return build_training_session(config, overrides=overrides)

        def federated_factory(
            *, overrides: Mapping[str, object]
        ) -> tuple[FederatedTrainingService, FederatedConfig]:
            """Build the federated training session on demand."""
            return build_federated_session(config, overrides=overrides)

        def ensemble_factory(
            *, overrides: Mapping[str, object]
        ) -> tuple[EnsembleService, EnsembleConfig]:
            """Build the ensemble session on demand."""
            return build_ensemble_session(config, overrides=overrides)

        def analysis_factory(
            *, overrides: Mapping[str, object]
        ) -> tuple[AnalysisService, str | None]:
            """Build the analysis session on demand."""
            return build_analysis_session(config, overrides=overrides)

        def calibration_factory(
            *, overrides: Mapping[str, object]
        ) -> tuple[CalibrationService, CalibrationConfig]:
            """Build the calibration session on demand."""
            return build_calibration_session(config, overrides=overrides)

        return run_cli(
            argv,
            manager,
            preprocessing_factory,
            training_factory,
            federated_factory=federated_factory,
            ensemble_factory=ensemble_factory,
            analysis_factory=analysis_factory,
            calibration_factory=calibration_factory,
        )
    except FedSecureAIError as exc:
        # Deliberate, explained failures are reported without a traceback; the
        # message is written to be actionable on its own.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
