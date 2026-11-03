"""Command-line argument parsing and dispatch.

Dependencies are injected: :func:`run` receives an already-constructed
:class:`~application.services.dataset_manager.DatasetManager` rather than
building one. The composition root is ``composition_root.py``, keeping this
module free of knowledge about which adapters are in use and testable without
them.

The preprocessing service arrives as a *factory* rather than an instance,
because constructing it creates output directories. Building it eagerly would
mean ``dataset-info`` had filesystem side effects.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from application.services.dataset_manager import DatasetManager
from domain.enums.model_architecture import ORDERED_ARCHITECTURES, ModelArchitecture
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy
from interfaces.cli.commands.analysis_commands import (
    AnalysisSessionFactory,
    analysis_command,
)
from interfaces.cli.commands.calibration_commands import (
    CalibrationSessionFactory,
    calibration_command,
)
from interfaces.cli.commands.dataset_commands import (
    dataset_download_command,
    dataset_info_command,
)
from interfaces.cli.commands.ensemble_commands import (
    EnsembleSessionFactory,
    ensemble_command,
)
from interfaces.cli.commands.federated_commands import (
    FederatedSessionFactory,
    federated_command,
)
from interfaces.cli.commands.preprocess_commands import (
    PreprocessingServiceFactory,
    preprocess_command,
)
from interfaces.cli.commands.train_commands import (
    TrainingSessionFactory,
    train_command,
)

PROGRAM = "fedsecureai"

DESCRIPTION = (
    "FedSecureAI - Privacy-Preserving Federated Deep Learning for "
    "Anti-Money Laundering Detection"
)

ALL_MODELS = "all"
"""``--model`` value that trains every architecture from one configuration."""

MODEL_CHOICES = (*(a.value for a in ORDERED_ARCHITECTURES), ALL_MODELS)

_TRAIN_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("sequence_length", "sequence_length"),
    ("sequence_stride", "sequence_stride"),
    ("hidden_size", "hidden_size"),
    ("num_layers", "num_layers"),
    ("dropout", "dropout"),
    ("batch_size", "batch_size"),
    ("learning_rate", "learning_rate"),
    ("optimizer", "optimizer"),
    ("epochs", "epochs"),
    ("random_seed", "random_seed"),
    ("early_stopping_patience", "early_stopping_patience"),
    ("selection_metric", "selection_metric"),
    ("threshold", "threshold"),
    ("threshold_policy", "threshold_policy"),
    ("row_limit", "row_limit"),
    ("num_workers", "num_workers"),
    ("device", "device"),
    ("checkpoint_directory", "checkpoint_directory"),
    ("results_directory", "results_directory"),
    ("processed_directory", "processed_directory"),
)
"""Parsed attribute names paired with the override keys they supply.

Only values the user actually gave are forwarded, so an unset flag leaves the
configured default in place rather than overwriting it with ``None``.
"""


_FEDERATED_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("distribution", "distribution"),
    ("algorithm", "algorithm"),
    ("mu", "mu"),
    ("rounds", "rounds"),
    ("local_epochs", "local_epochs"),
    ("seed", "seed"),
    *_TRAIN_OVERRIDES,
)

_ENSEMBLE_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("alpha_step", "alpha_step"),
    ("seed", "seed"),
    ("batch_size", "batch_size"),
    ("row_limit", "row_limit"),
    ("device", "device"),
    ("processed_directory", "processed_directory"),
    ("checkpoint_directory", "checkpoint_directory"),
    ("results_directory", "results_directory"),
)

_ANALYSIS_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("results_directory", "results_directory"),
    ("output_directory", "output_directory"),
    ("data_directory", "data_directory"),
    ("analysis_id", "analysis_id"),
)

_CALIBRATION_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("seed", "seed"),
    ("batch_size", "batch_size"),
    ("row_limit", "row_limit"),
    ("device", "device"),
    ("n_bins_ece", "n_bins_ece"),
    ("selection_metric", "selection_metric"),
    ("processed_directory", "processed_directory"),
    ("checkpoint_directory", "checkpoint_directory"),
    ("results_directory", "results_directory"),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=DESCRIPTION,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )

    # Dataset information command
    subparsers.add_parser(
        "dataset-info",
        help="Show the configured dataset and its local status (never downloads)",
        description=(
            "Show the configured dataset and whether its files are present "
            "locally. This command never downloads data."
        ),
    )

    # Dataset download command
    download = subparsers.add_parser(
        "dataset-download",
        help=(
            "Download the configured dataset "
            "(the only command that transfers data; credentials are never printed)"
        ),
        description=(
            "Download the configured dataset from its remote provider. "
            "This is the only command that transfers data, and running it "
            "is the explicit request to do so. Requires KAGGLE_USERNAME and "
            "KAGGLE_KEY in the environment or in a local .env file. "
            "Kaggle credentials are never printed or logged. "
            "An existing local copy is left alone unless --force is given."
        ),
    )

    download.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download even if the expected files are already "
            "present locally"
        ),
    )

    # Preprocessing command
    preprocess = subparsers.add_parser(
        "preprocess",
        help=(
            "Preprocess the local dataset into chronological train/validation/"
            "test splits (never downloads)"
        ),
        description=(
            "Read the already-downloaded dataset in bounded chunks and write "
            "train, validation and test splits plus the metadata needed to "
            "reproduce them. The split is chronological, never random, and "
            "every learned statistic is fitted on the training rows alone. "
            "This command never downloads data: if the dataset is absent it "
            "says so and stops. The raw file is opened read-only and its "
            "SHA-256 is checked before and after the run. "
            "This phase produces data only; no model is trained."
        ),
    )

    preprocess.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="ROWS",
        help=(
            "Rows held in memory per read, overriding the configured value. "
            "Bounds peak memory independently of the file's size"
        ),
    )

    preprocess.add_argument(
        "--keep-interim",
        action="store_true",
        help=(
            "Keep the intermediate working files instead of deleting them. "
            "They are a second copy of the dataset on disk; nothing "
            "downstream should read them"
        ),
    )

    _add_train_parser(subparsers)
    _add_federated_parser(subparsers)
    _add_ensemble_parser(subparsers)
    _add_analysis_parser(subparsers)
    _add_calibration_parser(subparsers)

    return parser



def _add_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Define the ``train`` subcommand.

    Split out because it carries every hyperparameter the specification requires to
    be configurable, and folding two dozen options into :func:`build_parser` would
    bury the other commands.

    Deliberately absent: ``--iid``, ``--non-iid``, ``--client``, ``--fedavg``,
    ``--fedprox`` and ``--mu``. Passing any of them is an error, which is the
    correct outcome while the behaviour they name does not exist.
    """
    train = subparsers.add_parser(
        "train",
        help=(
            "Train centralized GRU/LSTM/BiLSTM baselines on the preprocessed "
            "splits (never downloads, never preprocesses)"
        ),
        description=(
            "Train one or more centralized sequence classifiers on the "
            "train/validation/test splits that 'preprocess' wrote. Sequences are "
            "per-account windows in chronological order ending at the transaction "
            "being classified, built from one split's rows only. The loss is "
            "unweighted BCEWithLogitsLoss, so the natural class imbalance is "
            "preserved. The best checkpoint is chosen on validation and the "
            "decision threshold is fitted on validation and then frozen; the test "
            "split is not read until both are settled. "
            "IID partitioning, typology-based Non-IID partitioning, FedAvg, "
            "FedProx and federated aggregation are deferred to later phases and "
            "have no options here."
        ),
    )

    train.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help=(
            f"Architecture to train, or '{ALL_MODELS}' to train every architecture "
            f"from one configuration so the baselines stay comparable"
        ),
    )

    sequences = train.add_argument_group("sequence construction")
    sequences.add_argument(
        "--sequence-length",
        type=int,
        metavar="N",
        help="Transactions per sequence, including the one being classified",
    )
    sequences.add_argument(
        "--sequence-stride",
        type=int,
        metavar="N",
        help=(
            "Emit a sequence every N transactions. Above 1 this discards labelled "
            "transactions and changes the effective class balance"
        ),
    )
    sequences.add_argument(
        "--no-one-hot-nominal-codes",
        dest="one_hot_nominal_codes",
        action="store_false",
        default=None,
        help=(
            "Feed the three category-code columns as integers instead of expanding "
            "them. Preprocessing recorded these codes as carrying no ordering, so "
            "this asserts an ordering the data does not have"
        ),
    )

    architecture = train.add_argument_group("architecture")
    architecture.add_argument(
        "--hidden-size", type=int, metavar="N", help="Hidden units per direction"
    )
    architecture.add_argument(
        "--num-layers", type=int, metavar="N", help="Stacked recurrent layers"
    )
    architecture.add_argument(
        "--dropout",
        type=float,
        metavar="P",
        help="Dropout between recurrent layers; ignored when --num-layers is 1",
    )

    optimisation = train.add_argument_group("optimisation")
    optimisation.add_argument(
        "--epochs", type=int, metavar="N", help="Maximum passes over the training split"
    )
    optimisation.add_argument(
        "--batch-size", type=int, metavar="N", help="Sequences per optimisation step"
    )
    optimisation.add_argument(
        "--learning-rate", type=float, metavar="LR", help="Optimiser step size"
    )
    optimisation.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd", "rmsprop"),
        help="Optimiser to use",
    )
    optimisation.add_argument(
        "--gradient-clip-norm",
        type=float,
        metavar="NORM",
        help="Global gradient-norm cap; pass 0 to disable clipping",
    )
    optimisation.add_argument(
        "--early-stopping-patience",
        type=int,
        metavar="N",
        help=(
            "Epochs without validation improvement before stopping. 0 trains the "
            "full budget"
        ),
    )
    optimisation.add_argument(
        "--selection-metric",
        choices=tuple(metric.value for metric in SelectionMetric),
        help="Validation quantity that chooses the best checkpoint",
    )
    optimisation.add_argument(
        "--random-seed",
        type=int,
        metavar="SEED",
        help="Seed for Python, NumPy and PyTorch",
    )

    decision = train.add_argument_group("decision threshold")
    decision.add_argument(
        "--threshold",
        type=float,
        metavar="P",
        help="Fixed decision threshold, and the baseline the fitted one is compared to",
    )
    decision.add_argument(
        "--threshold-policy",
        choices=tuple(policy.value for policy in ThresholdPolicy),
        help=(
            "How the threshold applied to test is arrived at. Fitting happens on "
            "validation only, never on test"
        ),
    )

    runtime = train.add_argument_group("runtime and output")
    runtime.add_argument(
        "--row-limit",
        type=int,
        metavar="ROWS",
        help=(
            "Read at most this many rows per split. Marks the run as reduced in "
            "the log and the report, so its numbers cannot be mistaken for a full "
            "run's"
        ),
    )
    runtime.add_argument(
        "--num-workers", type=int, metavar="N", help="Data-loading worker processes"
    )
    runtime.add_argument(
        "--device", metavar="DEVICE", help="Torch device, e.g. cpu or cuda"
    )
    runtime.add_argument(
        "--processed-directory",
        metavar="DIR",
        help="Directory holding the preprocessed splits, overriding the configured one",
    )
    runtime.add_argument(
        "--checkpoint-directory",
        metavar="DIR",
        help="Where best.pt and final.pt are written, overriding the configured one",
    )
    runtime.add_argument(
        "--results-directory",
        metavar="DIR",
        help="Where reports and the results table are written",
    )


def _add_federated_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Define the ``federated`` subcommand for IID federated learning."""
    fed = subparsers.add_parser(
        "federated",
        help=(
            "Train federated GRU/LSTM/BiLSTM models on deterministic "
            "3-client IID partitions (never downloads, never preprocesses)"
        ),
        description=(
            "Train one or more federated recurrent sequence classifiers using "
            "deterministic 3-client IID partitioning with centralized validation. "
            "FedAvg performs sample-weighted parameter averaging; FedProx adds a "
            "proximal penalty to the local loss. The decision threshold is frozen on "
            "centralized validation and evaluated on test once federated rounds finish."
        ),
    )

    fed.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help=(
            f"Architecture to train, or '{ALL_MODELS}' to train every architecture"
        ),
    )

    fed_group = fed.add_argument_group("federated configuration")
    fed_group.add_argument(
        "--distribution",
        choices=("iid", "non-iid", "non_iid"),
        default="iid",
        help="Partitioning distribution across clients ('iid' or 'non-iid')",
    )
    fed_group.add_argument(
        "--algorithm",
        required=True,
        choices=("fedavg", "fedprox", "scaffold", "fedprox_scaffold"),
        help="Federated algorithm: 'fedavg', 'fedprox', 'scaffold', or 'fedprox_scaffold'",
    )
    fed_group.add_argument(
        "--mu",
        type=float,
        default=None,
        metavar="MU",
        help="FedProx proximal coefficient (e.g. 0.01); ignored/null for FedAvg",
    )
    fed_group.add_argument(
        "--rounds",
        type=int,
        metavar="N",
        help="Number of federated communication rounds (default: 5)",
    )
    fed_group.add_argument(
        "--local-epochs",
        type=int,
        metavar="N",
        help="Local training epochs per round on each client (default: 1)",
    )
    fed_group.add_argument(
        "--seed",
        type=int,
        metavar="SEED",
        help="Random seed for client partitioning and round orchestration",
    )

    sequences = fed.add_argument_group("sequence construction")
    sequences.add_argument(
        "--sequence-length",
        type=int,
        metavar="N",
        help="Transactions per sequence, including the one being classified",
    )
    sequences.add_argument(
        "--sequence-stride",
        type=int,
        metavar="N",
        help="Emit a sequence every N transactions",
    )
    sequences.add_argument(
        "--no-one-hot-nominal-codes",
        dest="one_hot_nominal_codes",
        action="store_false",
        default=None,
        help="Feed categorical columns as integers instead of one-hot indicator vectors",
    )

    architecture = fed.add_argument_group("architecture")
    architecture.add_argument(
        "--hidden-size", type=int, metavar="N", help="Hidden units per direction"
    )
    architecture.add_argument(
        "--num-layers", type=int, metavar="N", help="Stacked recurrent layers"
    )
    architecture.add_argument(
        "--dropout",
        type=float,
        metavar="P",
        help="Dropout between recurrent layers",
    )

    optimisation = fed.add_argument_group("optimisation")
    optimisation.add_argument(
        "--batch-size", type=int, metavar="N", help="Sequences per optimization step"
    )
    optimisation.add_argument(
        "--learning-rate", type=float, metavar="LR", help="Optimizer step size"
    )
    optimisation.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd", "rmsprop"),
        help="Optimizer name",
    )
    optimisation.add_argument(
        "--gradient-clip-norm",
        type=float,
        metavar="NORM",
        help="Global gradient-norm cap; 0 disables clipping",
    )
    optimisation.add_argument(
        "--selection-metric",
        choices=tuple(metric.value for metric in SelectionMetric),
        help="Validation quantity that picks the best checkpoint",
    )

    decision = fed.add_argument_group("decision threshold")
    decision.add_argument(
        "--threshold",
        type=float,
        metavar="P",
        help="Fixed decision threshold",
    )
    decision.add_argument(
        "--threshold-policy",
        choices=tuple(policy.value for policy in ThresholdPolicy),
        help="Threshold selection policy applied to validation predictions",
    )

    runtime = fed.add_argument_group("runtime and output")
    runtime.add_argument(
        "--row-limit",
        type=int,
        metavar="ROWS",
        help="Read at most this many rows per split (for smoke tests)",
    )
    runtime.add_argument(
        "--num-workers", type=int, metavar="N", help="Data-loading worker processes"
    )
    runtime.add_argument(
        "--device", metavar="DEVICE", help="Torch device, e.g. cpu or cuda"
    )
    runtime.add_argument(
        "--processed-directory",
        metavar="DIR",
        help="Directory holding the preprocessed splits",
    )
    runtime.add_argument(
        "--checkpoint-directory",
        metavar="DIR",
        help="Where global checkpoints are written",
    )
    runtime.add_argument(
        "--results-directory",
        metavar="DIR",
        help="Where reports and results table are written",
    )


def _add_ensemble_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Define the ``ensemble`` subcommand for Phase 7 IID + Non-IID ensemble."""
    ens = subparsers.add_parser(
        "ensemble",
        help="Run Phase 7 IID + Non-IID ensemble modeling",
        description=(
            "Combine prediction probabilities from the best Phase 5 IID model and "
            "the best Phase 6 Non-IID model using validation-tuned alpha and "
            "frozen decision threshold, with single-pass test evaluation."
        ),
    )
    ens.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        metavar="STEP",
        help="Grid step size for alpha search between 0.0 and 1.0 (default: 0.05)",
    )
    ens.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for ensemble orchestration and reproducibility (default: 42)",
    )
    ens.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="Batch size for model inference passes (default: 1024)",
    )
    ens.add_argument(
        "--row-limit",
        type=int,
        default=None,
        metavar="ROWS",
        help="Read at most this many rows per split (for smoke tests)",
    )
    ens.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device, e.g. cpu or cuda",
    )
    ens.add_argument(
        "--processed-directory",
        default=None,
        metavar="DIR",
        help="Directory holding the preprocessed splits",
    )
    ens.add_argument(
        "--checkpoint-directory",
        default=None,
        metavar="DIR",
        help="Directory where model checkpoints are located",
    )
    ens.add_argument(
        "--results-directory",
        default=None,
        metavar="DIR",
        help="Directory where experiment results are written",
    )


def _add_analysis_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Define the ``analyze`` subcommand for Phase 8."""
    analysis = subparsers.add_parser(
        "analyze",
        help=(
            "Run Phase 8 performance analysis, cross-paradigm comparisons, "
            "ablation study, and generate 10 publication figures"
        ),
        description=(
            "Scientifically analyze existing experimental artifacts across "
            "Centralized (Phase 4), IID Federated (Phase 5), Non-IID Federated (Phase 6), "
            "and Ensemble (Phase 7). Performs zero retraining."
        ),
    )
    analysis.add_argument(
        "--results-directory",
        default=None,
        metavar="DIR",
        help="Directory where experiment results are located",
    )
    analysis.add_argument(
        "--output-directory",
        default=None,
        metavar="DIR",
        help="Directory where Phase 8 analysis artifacts will be written",
    )
    analysis.add_argument(
        "--data-directory",
        default=None,
        metavar="DIR",
        help="Directory where processed data partition reports are located",
    )
    analysis.add_argument(
        "--analysis-id",
        default=None,
        metavar="ID",
        help="Unique identifier for the analysis run (default: auto timestamped)",
    )


def _add_calibration_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Define the ``calibrate`` subcommand for Phase 10 Probability Calibration."""
    cal = subparsers.add_parser(
        "calibrate",
        help="Run Phase 10 probability calibration on the official reference AML model",
        description=(
            "Fit post-processing calibration methods (Temperature Scaling, Platt Scaling, "
            "Isotonic Regression) on validation predictions and evaluate reliability "
            "improvements on untouched test data with zero neural network retraining."
        ),
    )
    cal.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for calibration orchestration and reproducibility (default: 42)",
    )
    cal.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="Batch size for model inference passes (default: 1024)",
    )
    cal.add_argument(
        "--row-limit",
        type=int,
        default=None,
        metavar="ROWS",
        help="Read at most this many rows per split (for smoke tests)",
    )
    cal.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device, e.g. cpu or cuda",
    )
    cal.add_argument(
        "--n-bins-ece",
        type=int,
        default=15,
        metavar="BINS",
        help="Number of bins for Expected Calibration Error (default: 15)",
    )
    cal.add_argument(
        "--selection-metric",
        choices=("log_loss", "ece", "brier_score"),
        default="log_loss",
        help="Predefined validation selection metric for choosing calibrator (default: log_loss)",
    )
    cal.add_argument(
        "--processed-directory",
        default=None,
        metavar="DIR",
        help="Directory holding the preprocessed splits",
    )
    cal.add_argument(
        "--checkpoint-directory",
        default=None,
        metavar="DIR",
        help="Where model checkpoints are located",
    )
    cal.add_argument(
        "--results-directory",
        default=None,
        metavar="DIR",
        help="Where experiment results and calibration artifacts are written",
    )


def run(
    argv: Sequence[str] | None,
    dataset_manager: DatasetManager,
    preprocessing_factory: PreprocessingServiceFactory | None = None,
    training_factory: TrainingSessionFactory | None = None,
    federated_factory: FederatedSessionFactory | None = None,
    ensemble_factory: EnsembleSessionFactory | None = None,
    analysis_factory: AnalysisSessionFactory | None = None,
    calibration_factory: CalibrationSessionFactory | None = None,
) -> int:
    """Parse ``argv`` and dispatch to the matching command.

    Args:
        argv: Argument list excluding program name.
        dataset_manager: Injected dataset service.
        preprocessing_factory: Builds preprocessing service on demand.
        training_factory: Builds centralized training service on demand.
        federated_factory: Builds federated training service on demand.
        ensemble_factory: Builds ensemble service on demand.
        analysis_factory: Builds analysis service and dependencies on demand.
        calibration_factory: Builds calibration service on demand.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset-info":
        return dataset_info_command(dataset_manager)

    if args.command == "dataset-download":
        return dataset_download_command(
            dataset_manager,
            force=bool(args.force),
        )

    if args.command == "preprocess":
        if preprocessing_factory is None:
            parser.error("preprocessing is not available in this configuration")
        return preprocess_command(
            preprocessing_factory,
            chunk_size=args.chunk_size,
            keep_interim=bool(args.keep_interim),
        )

    if args.command == "train":
        if training_factory is None:
            parser.error("training is not available in this configuration")
        return train_command(
            training_factory,
            architectures=_selected_architectures(args.model),
            overrides=_training_overrides(args),
        )

    if args.command == "federated":
        if federated_factory is None:
            parser.error("federated training is not available in this configuration")
        return federated_command(
            federated_factory,
            architectures=_selected_architectures(args.model),
            overrides=_federated_overrides(args),
        )

    if args.command == "ensemble":
        if ensemble_factory is None:
            parser.error("ensemble modeling is not available in this configuration")
        return ensemble_command(
            ensemble_factory,
            overrides=_ensemble_overrides(args),
        )

    if args.command == "analyze":
        if analysis_factory is None:
            parser.error("analysis is not available in this configuration")
        return analysis_command(
            analysis_factory,
            overrides=_analysis_overrides(args),
        )

    if args.command == "calibrate":
        if calibration_factory is None:
            parser.error("calibration is not available in this configuration")
        return calibration_command(
            calibration_factory,
            overrides=_calibration_overrides(args),
        )

    parser.print_help()
    return 0


def _selected_architectures(model: str) -> tuple[ModelArchitecture, ...]:
    """Resolve ``--model`` into the architectures to train, in a fixed order."""
    if model == ALL_MODELS:
        return ORDERED_ARCHITECTURES
    return (ModelArchitecture(model),)


def _training_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the training options the user actually supplied."""
    overrides: dict[str, object] = {}
    for attribute, key in _TRAIN_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value

    if args.one_hot_nominal_codes is not None:
        overrides["one_hot_nominal_codes"] = bool(args.one_hot_nominal_codes)
    if args.gradient_clip_norm is not None:
        overrides["gradient_clip_norm"] = args.gradient_clip_norm or None
    return overrides


def _federated_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the federated options the user actually supplied."""
    overrides: dict[str, object] = {}
    for attribute, key in _FEDERATED_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value

    if getattr(args, "one_hot_nominal_codes", None) is not None:
        overrides["one_hot_nominal_codes"] = bool(args.one_hot_nominal_codes)
    if getattr(args, "gradient_clip_norm", None) is not None:
        overrides["gradient_clip_norm"] = args.gradient_clip_norm or None
    return overrides


def _ensemble_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the ensemble options the user actually supplied."""
    overrides: dict[str, object] = {}
    for attribute, key in _ENSEMBLE_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _analysis_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the analysis options the user actually supplied."""
    overrides: dict[str, object] = {}
    for attribute, key in _ANALYSIS_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _calibration_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the calibration options the user actually supplied."""
    overrides: dict[str, object] = {}
    for attribute, key in _CALIBRATION_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    return overrides

