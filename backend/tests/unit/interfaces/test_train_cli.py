"""Tests for the training CLI.

Nothing is trained here. ``train_command`` takes a *factory*, and these tests pass
one that returns a recording double, which makes the CLI's own claims checkable
separately from the training it triggers:

- **``--model all`` runs one configuration three times.** The double records the
  configurations it was handed, so "no architecture was tuned separately" is an
  assertion about observed objects rather than a promise in a docstring.
- **Unset flags stay unset.** An override dictionary containing ``None`` would
  overwrite a configured default with nothing; the collected overrides are
  inspected directly.
- **The federated flags do not exist.** ``--iid``, ``--non-iid``, ``--client``,
  ``--fedavg``, ``--fedprox`` and ``--mu`` must each be a parse error while the
  behaviour they name is unimplemented.
- **The printed output does not read like a later phase.** No winner is declared
  and the deferral notice is in the tool's own output.

Every report the double returns is hand-built with invented numbers. They exercise
the formatter and describe no real run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from application.dto.experiment import (
    ClassificationMetrics,
    ConfusionCounts,
    DataProvenance,
    ExperimentReport,
    ExperimentResult,
    SequenceStrategy,
)
from application.dto.training import EpochRecord, TrainingConfig
from application.services.centralized_training_service import CentralizedTrainingService
from application.services.dataset_manager import DatasetManager
from domain.enums.model_architecture import ORDERED_ARCHITECTURES, ModelArchitecture
from domain.enums.split_name import SplitName
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy, TrainingMode
from domain.exceptions.training import EmptySequenceDatasetError
from interfaces.cli.app import (
    ALL_MODELS,
    MODEL_CHOICES,
    _training_overrides,
    build_parser,
    run,
)
from interfaces.cli.commands.train_commands import DEFERRED_NOTICE, train_command
from tests.fixtures.dataset_registry_doubles import (
    InMemoryDatasetRegistry,
    make_descriptor,
)
from tests.fixtures.downloader_doubles import ExplodingDownloader

pytestmark = pytest.mark.unit

FEDERATED_FLAGS = ("--iid", "--non-iid", "--client", "--fedavg", "--fedprox", "--mu")


def _metrics(*, recall: float = 0.5, false_negatives: int = 7) -> ClassificationMetrics:
    """Invented metrics. They describe no real model."""
    return ClassificationMetrics(
        precision=0.25,
        recall=recall,
        f1=1 / 3,
        pr_auc=0.2,
        roc_auc=0.8,
        accuracy=0.999,
        confusion=ConfusionCounts(
            true_positives=7,
            true_negatives=900,
            false_positives=21,
            false_negatives=false_negatives,
        ),
        threshold=0.04,
        positive_rate=0.0015,
        loss=0.01,
    )


def _report(
    architecture: ModelArchitecture = ModelArchitecture.GRU,
    *,
    config: TrainingConfig | None = None,
    early_stopped: bool = False,
    deterministic: bool = True,
    threshold_selected_on: str | None = SplitName.VALIDATION.value,
    policy: ThresholdPolicy = ThresholdPolicy.VALIDATION_F1,
    recall: float = 0.5,
) -> ExperimentReport:
    """A synthetic report with invented numbers, for the formatter only."""
    resolved = config or TrainingConfig(architecture=architecture, threshold_policy=policy)
    return ExperimentReport(
        experiment_id=f"{architecture.value}_centralized_seed42",
        config=resolved,
        provenance=DataProvenance(
            processed_directory="/synthetic/processed",
            preprocessing_version="test-0.0.0",
            feature_columns=("amount_log", "same_bank"),
            identifier_columns=("timestamp", "from_account", "to_account"),
            target_column="is_laundering",
            feature_dimension=5,
            rows_per_split={"train": 1200, "validation": 400, "test": 400},
            positives_per_split={"train": 12, "validation": 4, "test": 4},
        ),
        sequences=SequenceStrategy(
            grouping_column="from_account",
            sequence_length=10,
            stride=1,
            target_assignment="newest transaction in the window",
            padding="right-padded with zeros",
            boundary_policy="no sequence crosses a split boundary",
            ordering="row order, verified non-decreasing by timestamp",
        ),
        epochs=(
            EpochRecord(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.45,
                validation_metrics={"pr_auc": 0.1, "recall": 0.25},
                seconds=30.0,
                selected=False,
            ),
            EpochRecord(
                epoch=2,
                train_loss=0.4,
                validation_loss=0.42,
                validation_metrics={"pr_auc": 0.2, "recall": recall},
                seconds=31.0,
                selected=True,
            ),
        ),
        results=(
            ExperimentResult(
                experiment_id=f"{architecture.value}_centralized_seed42",
                architecture=architecture,
                training_mode=TrainingMode.CENTRALIZED,
                split=SplitName.VALIDATION,
                metrics=_metrics(recall=recall),
                threshold_policy=policy,
                seed=42,
                loss_type="bce_with_logits",
                training_time_seconds=61.0,
                evaluation_time_seconds=2.0,
                sequence_count=400,
            ),
            ExperimentResult(
                experiment_id=f"{architecture.value}_centralized_seed42",
                architecture=architecture,
                training_mode=TrainingMode.CENTRALIZED,
                split=SplitName.TEST,
                metrics=_metrics(recall=recall),
                threshold_policy=policy,
                seed=42,
                loss_type="bce_with_logits",
                training_time_seconds=61.0,
                evaluation_time_seconds=2.0,
                sequence_count=400,
            ),
        ),
        selected_epoch=2,
        frozen_threshold=0.04,
        threshold_selected_on=threshold_selected_on,
        device="cpu",
        parameter_count=13_505,
        training_seconds=61.0,
        data_preparation_seconds=4.0,
        total_seconds=67.0,
        early_stopped=early_stopped,
        torch_version="0.0.0-test",
        deterministic=deterministic,
    )


class _RecordingFactory:
    """Records the overrides and configurations it saw; trains nothing."""

    def __init__(self, *, base: TrainingConfig | None = None) -> None:
        self.overrides: list[Mapping[str, object]] = []
        self.configs: list[TrainingConfig] = []
        self._base = base or TrainingConfig(architecture=ModelArchitecture.GRU)

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[CentralizedTrainingService, TrainingConfig]:
        self.overrides.append(dict(overrides))
        return cast("CentralizedTrainingService", self), self._base

    def run(self, config: TrainingConfig) -> ExperimentReport:
        self.configs.append(config)
        return _report(config.architecture, config=config)


class _ExplodingFactory:
    """Fails if the command builds a service it should not have built."""

    def __call__(
        self, *, overrides: Mapping[str, object]
    ) -> tuple[CentralizedTrainingService, TrainingConfig]:
        raise AssertionError(f"no service should be built here (overrides={overrides})")


def _manager(tmp_path: Path) -> DatasetManager:
    """A manager with no network-capable collaborators."""
    return DatasetManager(
        registry=InMemoryDatasetRegistry((make_descriptor(),)),
        downloader=ExplodingDownloader(),
        project_root=tmp_path,
    )


def _parse(*argv: str) -> argparse.Namespace:
    """Parse a ``train`` invocation."""
    return build_parser().parse_args(["train", *argv])


# --- Model selection ---------------------------------------------------------


def test_train_is_a_registered_subcommand() -> None:
    assert build_parser().parse_args(["train", "--model", "gru"]).command == "train"


def test_the_model_is_required() -> None:
    # No default architecture: a run must say which baseline it is.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_each_architecture_can_be_named_on_its_own(
    architecture: ModelArchitecture,
) -> None:
    assert architecture.value in MODEL_CHOICES
    assert _parse("--model", architecture.value) is not None


def test_all_is_a_model_choice() -> None:
    assert ALL_MODELS in MODEL_CHOICES
    assert set(MODEL_CHOICES) == {a.value for a in ORDERED_ARCHITECTURES} | {ALL_MODELS}


def test_an_unknown_architecture_is_rejected_before_anything_runs() -> None:
    with pytest.raises(SystemExit):
        _parse("--model", "transformer")


def test_all_trains_every_architecture_from_one_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The comparability claim, as an assertion: three runs, one config, only the
    # architecture differing.
    factory = _RecordingFactory()

    exit_code = run(["train", "--model", ALL_MODELS], _manager(Path()), None, factory)
    capsys.readouterr()

    assert exit_code == 0
    assert [config.architecture for config in factory.configs] == list(
        ORDERED_ARCHITECTURES
    )
    baseline = factory.configs[0].to_dict()
    for config in factory.configs[1:]:
        assert config.to_dict() | {"architecture": baseline["architecture"]} == baseline


def test_one_named_architecture_trains_only_that_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = _RecordingFactory()

    run(["train", "--model", "bilstm"], _manager(Path()), None, factory)
    capsys.readouterr()

    assert [config.architecture for config in factory.configs] == [
        ModelArchitecture.BILSTM
    ]


def test_the_architectures_are_trained_in_a_fixed_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # So two `--model all` runs produce the same table row order.
    first, second = _RecordingFactory(), _RecordingFactory()

    run(["train", "--model", ALL_MODELS], _manager(Path()), None, first)
    run(["train", "--model", ALL_MODELS], _manager(Path()), None, second)
    capsys.readouterr()

    assert [c.architecture for c in first.configs] == [
        c.architecture for c in second.configs
    ]


# --- Federated flags are absent ----------------------------------------------


@pytest.mark.parametrize("flag", FEDERATED_FLAGS)
def test_a_federated_flag_is_a_parse_error(flag: str) -> None:
    # Accepting one now would imply behaviour that does not exist. Rejecting it is
    # the honest outcome.
    with pytest.raises(SystemExit):
        _parse("--model", "gru", flag)


@pytest.mark.parametrize("flag", FEDERATED_FLAGS)
def test_a_federated_flag_with_a_value_is_also_a_parse_error(flag: str) -> None:
    with pytest.raises(SystemExit):
        _parse("--model", "gru", flag, "0.1")


@pytest.mark.parametrize("flag", FEDERATED_FLAGS)
def test_no_federated_flag_appears_in_the_help(flag: str) -> None:
    assert flag not in build_parser().format_help()


def test_the_train_help_does_not_advertise_federated_options() -> None:
    help_text = build_parser().format_help()

    for word in ("fedavg", "fedprox", "aggregation", "--client"):
        assert word not in help_text.lower()


# --- Overrides ---------------------------------------------------------------


def test_no_flags_means_no_overrides() -> None:
    # An override of None would overwrite a configured default with nothing.
    assert _training_overrides(_parse("--model", "gru")) == {}


def test_only_the_flags_that_were_given_are_forwarded() -> None:
    overrides = _training_overrides(
        _parse("--model", "gru", "--epochs", "3", "--hidden-size", "16")
    )

    assert overrides == {"epochs": 3, "hidden_size": 16}


@pytest.mark.parametrize(
    ("flag", "value", "key", "expected"),
    [
        ("--sequence-length", "7", "sequence_length", 7),
        ("--sequence-stride", "2", "sequence_stride", 2),
        ("--hidden-size", "32", "hidden_size", 32),
        ("--num-layers", "2", "num_layers", 2),
        ("--dropout", "0.2", "dropout", 0.2),
        ("--batch-size", "64", "batch_size", 64),
        ("--learning-rate", "0.01", "learning_rate", 0.01),
        ("--optimizer", "sgd", "optimizer", "sgd"),
        ("--epochs", "8", "epochs", 8),
        ("--random-seed", "7", "random_seed", 7),
        ("--early-stopping-patience", "1", "early_stopping_patience", 1),
        ("--selection-metric", "recall", "selection_metric", "recall"),
        ("--threshold", "0.03", "threshold", 0.03),
        ("--threshold-policy", "fixed", "threshold_policy", "fixed"),
        ("--row-limit", "5000", "row_limit", 5000),
        ("--num-workers", "2", "num_workers", 2),
        ("--device", "cpu", "device", "cpu"),
        ("--processed-directory", "/tmp/p", "processed_directory", "/tmp/p"),
        ("--checkpoint-directory", "/tmp/c", "checkpoint_directory", "/tmp/c"),
        ("--results-directory", "/tmp/r", "results_directory", "/tmp/r"),
    ],
)
def test_every_configurable_hyperparameter_has_a_flag(
    flag: str, value: str, key: str, expected: object
) -> None:
    # The specification requires each of these to be configurable, so each is
    # checked for arriving under the key the configuration reads.
    overrides = _training_overrides(_parse("--model", "gru", flag, value))

    assert overrides[key] == expected


def test_zero_gradient_clipping_is_forwarded_as_no_clipping() -> None:
    # 0.0 would zero every gradient; the configuration expresses "off" as None.
    overrides = _training_overrides(
        _parse("--model", "gru", "--gradient-clip-norm", "0")
    )

    assert overrides == {"gradient_clip_norm": None}


def test_a_positive_gradient_clip_is_forwarded_as_given() -> None:
    overrides = _training_overrides(
        _parse("--model", "gru", "--gradient-clip-norm", "2.5")
    )

    assert overrides == {"gradient_clip_norm": 2.5}


def test_the_one_hot_expansion_is_on_unless_it_is_switched_off() -> None:
    assert _training_overrides(_parse("--model", "gru")) == {}
    assert _training_overrides(
        _parse("--model", "gru", "--no-one-hot-nominal-codes")
    ) == {"one_hot_nominal_codes": False}


def test_the_overrides_reach_the_factory_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = _RecordingFactory()

    run(
        ["train", "--model", "gru", "--epochs", "2", "--row-limit", "100"],
        _manager(Path()),
        None,
        factory,
    )
    capsys.readouterr()

    assert factory.overrides == [{"epochs": 2, "row_limit": 100}]


def test_the_factory_is_asked_once_no_matter_how_many_architectures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One configuration is built and reused, rather than three built independently.
    factory = _RecordingFactory()

    run(["train", "--model", ALL_MODELS], _manager(Path()), None, factory)
    capsys.readouterr()

    assert len(factory.overrides) == 1
    assert len(factory.configs) == 3


@pytest.mark.parametrize("flag", ["--epochs", "--hidden-size", "--row-limit"])
def test_a_non_numeric_value_is_rejected_before_anything_runs(flag: str) -> None:
    with pytest.raises(SystemExit):
        _parse("--model", "gru", flag, "lots")


def test_an_unknown_optimizer_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        _parse("--model", "gru", "--optimizer", "lbfgs")


@pytest.mark.parametrize("metric", list(SelectionMetric))
def test_every_selection_metric_is_offered(metric: SelectionMetric) -> None:
    overrides = _training_overrides(
        _parse("--model", "gru", "--selection-metric", metric.value)
    )

    assert overrides["selection_metric"] == metric.value


@pytest.mark.parametrize("policy", list(ThresholdPolicy))
def test_every_threshold_policy_is_offered(policy: ThresholdPolicy) -> None:
    overrides = _training_overrides(
        _parse("--model", "gru", "--threshold-policy", policy.value)
    )

    assert overrides["threshold_policy"] == policy.value


# --- Dispatch ----------------------------------------------------------------


def test_a_successful_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = train_command(
        _RecordingFactory(),
        architectures=(ModelArchitecture.GRU,),
        overrides={},
    )
    capsys.readouterr()

    assert exit_code == 0


def test_no_service_is_built_for_the_other_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Building one reads metadata from disk and creates directories.
    run(["dataset-info"], _manager(tmp_path), None, _ExplodingFactory())
    capsys.readouterr()


def test_training_is_refused_when_no_factory_was_wired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        run(["train", "--model", "gru"], _manager(tmp_path))

    assert raised.value.code == 2
    assert "training is not available" in capsys.readouterr().err


def test_a_missing_split_propagates_rather_than_being_reported_as_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _EmptySplit:
        def __call__(
            self, *, overrides: Mapping[str, object]  # noqa: ARG002
        ) -> tuple[CentralizedTrainingService, TrainingConfig]:
            return (
                cast("CentralizedTrainingService", self),
                TrainingConfig(architecture=ModelArchitecture.GRU),
            )

        def run(self, config: TrainingConfig) -> ExperimentReport:  # noqa: ARG002
            raise EmptySequenceDatasetError(SplitName.TRAIN, 0)

    with pytest.raises(EmptySequenceDatasetError):
        train_command(
            _EmptySplit(), architectures=(ModelArchitecture.GRU,), overrides={}
        )

    capsys.readouterr()


# --- What the output says ----------------------------------------------------


def test_the_banner_states_the_loss_and_that_it_is_unweighted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "BCEWithLogitsLoss" in out
    assert "unweighted" in out
    assert "natural imbalance preserved" in out


def test_the_banner_states_when_the_test_split_is_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "not read until training has finished" in out
    assert "frozen" in out


def test_false_negatives_are_labelled_as_missed_laundering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The primary error of this project, named rather than left as an acronym.
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    assert "laundering transactions the model let through" in capsys.readouterr().out


def test_accuracy_is_flagged_as_uninformative_here(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "flagging nothing scores" in out


def test_the_frozen_threshold_and_the_split_it_came_from_are_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "fitted on validation" in out
    assert "then frozen" in out


def test_a_configured_threshold_is_reported_as_not_fitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Under a fixed policy nothing was fitted, and saying "fitted on None" would be
    # worse than saying so.
    class _Fixed(_RecordingFactory):
        def run(self, config: TrainingConfig) -> ExperimentReport:
            self.configs.append(config)
            return _report(
                config.architecture,
                config=config,
                threshold_selected_on=None,
                policy=ThresholdPolicy.FIXED,
            )

    train_command(_Fixed(), architectures=(ModelArchitecture.GRU,), overrides={})

    assert "not fitted" in capsys.readouterr().out


def test_the_selected_epoch_is_marked_in_the_curve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "selected checkpoint" in out
    assert "Selected epoch: 2" in out


def test_early_stopping_is_stated_when_it_happened(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Stopped(_RecordingFactory):
        def run(self, config: TrainingConfig) -> ExperimentReport:
            self.configs.append(config)
            return _report(config.architecture, config=config, early_stopped=True)

    train_command(_Stopped(), architectures=(ModelArchitecture.GRU,), overrides={})

    assert "early stopped" in capsys.readouterr().out


def test_determinism_is_reported_honestly_when_it_is_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Requested is not the same as achieved, and the output must not claim it.
    class _NonDeterministic(_RecordingFactory):
        def run(self, config: TrainingConfig) -> ExperimentReport:
            self.configs.append(config)
            return _report(config.architecture, config=config, deterministic=False)

    train_command(
        _NonDeterministic(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    assert "NOT AVAILABLE" in capsys.readouterr().out


def test_the_identifier_columns_are_reported_as_excluded_from_the_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    assert "Input width:    5" in out
    assert "processed feature column(s)" in out


def test_every_split_is_reported_with_its_positive_share(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )

    out = capsys.readouterr().out
    for split in ("train", "validation", "test"):
        assert split in out
    assert "positive(s)" in out


# --- The comparison ----------------------------------------------------------


def test_the_comparison_appears_only_when_more_than_one_model_ran(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=(ModelArchitecture.GRU,), overrides={}
    )
    single = capsys.readouterr().out

    train_command(
        _RecordingFactory(), architectures=ORDERED_ARCHITECTURES, overrides={}
    )
    several = capsys.readouterr().out

    assert "Test-split comparison" not in single
    assert "Test-split comparison" in several


def test_the_comparison_lists_every_architecture_that_ran(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=ORDERED_ARCHITECTURES, overrides={}
    )

    out = capsys.readouterr().out
    for architecture in ORDERED_ARCHITECTURES:
        assert architecture.value in out


def test_the_comparison_declares_no_winner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Single-seed measurements cannot separate a real difference from seed variance,
    # and the output says so instead of ranking.
    train_command(
        _RecordingFactory(), architectures=ORDERED_ARCHITECTURES, overrides={}
    )

    out = capsys.readouterr().out
    assert "No architecture is declared best" in out
    assert "single-seed measurements" in out
    for claim in ("is best", "outperforms", "winner", "recommended model"):
        assert claim not in out.lower()


def test_the_comparison_carries_the_deferral_notice_verbatim(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The scope boundary is in the tool's own output, not only in the docs.
    train_command(
        _RecordingFactory(), architectures=ORDERED_ARCHITECTURES, overrides={}
    )

    printed = capsys.readouterr().out
    for sentence in DEFERRED_NOTICE.split(". "):
        assert sentence.strip(". ") in " ".join(printed.split())


def test_the_notice_names_every_deferred_mechanism() -> None:
    for name in (
        "IID partitioning",
        "typology-based",
        "Non-IID",
        "FedAvg",
        "FedProx",
        "federated aggregation",
        "deferred to later phases",
    ):
        assert name in DEFERRED_NOTICE


def test_the_output_never_mentions_a_credential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_command(
        _RecordingFactory(), architectures=ORDERED_ARCHITECTURES, overrides={}
    )

    out = capsys.readouterr().out.lower()
    for word in ("kaggle", "password", "token", "api_key", "secret"):
        assert word not in out
