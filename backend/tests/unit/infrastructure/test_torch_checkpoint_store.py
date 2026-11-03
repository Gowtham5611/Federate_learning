"""Tests for :mod:`infrastructure.persistence.torch_checkpoint_store`.

Three properties, each of which would be invisible if it broke:

1. **Two files, never one.** ``best.pt`` and ``final.pt`` are separate paths, so
   writing the final checkpoint cannot overwrite the best one. Under early stopping
   they are different models, and a report quoting the best epoch's validation
   score while the disk held the last epoch's parameters would look correct.
2. **A round trip is exact.** Parameters come back tensor-for-tensor and the
   configuration comes back as a configuration, because a checkpoint that cannot be
   reloaded into the run that produced it is a file, not a result.
3. **Loading never executes code.** ``weights_only=True`` is only usable because
   the payload holds tensors and primitives, so a payload that quietly gained an
   object would break loading rather than pass silently.

Everything is written under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from application.dto.training import TrainingConfig
from application.ports.output.experiment_store import Checkpoint
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.training_mode import SelectionMetric, ThresholdPolicy
from domain.exceptions.training import (
    CheckpointArchitectureMismatchError,
    CheckpointNotFoundError,
)
from infrastructure.persistence.torch_checkpoint_store import (
    CHECKPOINT_SUFFIX,
    FORMAT_VERSION,
    TorchCheckpointStore,
)

pytestmark = pytest.mark.unit

EXPERIMENT = "gru_centralized_seed42"


def make_config(**overrides: object) -> TrainingConfig:
    """A configuration worth round-tripping: nothing left at its default."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "sequence_length": 7,
        "sequence_stride": 2,
        "hidden_size": 8,
        "num_layers": 2,
        "dropout": 0.25,
        "batch_size": 16,
        "learning_rate": 0.003,
        "optimizer": "adamw",
        "epochs": 4,
        "random_seed": 11,
        "early_stopping_patience": 1,
        "selection_metric": SelectionMetric.RECALL,
        "threshold": 0.04,
        "threshold_policy": ThresholdPolicy.FIXED,
        "one_hot_nominal_codes": False,
        "gradient_clip_norm": 2.5,
        "row_limit": 500,
        "num_workers": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def make_checkpoint(
    kind: CheckpointKind = CheckpointKind.BEST,
    *,
    experiment_id: str = EXPERIMENT,
    architecture: ModelArchitecture = ModelArchitecture.GRU,
    epoch: int = 2,
    fill: float = 1.0,
    threshold: float = 0.04,
    optimizer_state: dict[str, object] | None = None,
) -> Checkpoint:
    """A checkpoint whose parameters are recognisable by their fill value."""
    return Checkpoint(
        experiment_id=experiment_id,
        kind=kind,
        architecture=architecture,
        epoch=epoch,
        parameters={
            "encoder.weight_ih_l0": torch.full((12, 4), fill),
            "head.bias": torch.tensor([fill]),
        },
        config=make_config(architecture=architecture),
        feature_dimension=4,
        threshold=threshold,
        validation_loss=0.125,
        random_seed=11,
        optimizer_state=optimizer_state,
        validation_metrics={"recall": 0.5, "pr_auc": 0.25},
    )


@pytest.fixture
def store(tmp_path: Path) -> TorchCheckpointStore:
    """A store over a temporary root."""
    return TorchCheckpointStore(tmp_path / "checkpoints")


# --- Two files, never one ----------------------------------------------------


def test_best_and_final_go_to_different_paths(store: TorchCheckpointStore) -> None:
    best = store.path_for(EXPERIMENT, CheckpointKind.BEST)
    final = store.path_for(EXPERIMENT, CheckpointKind.FINAL)

    assert best != final
    assert best.name == f"{CheckpointKind.BEST.value}{CHECKPOINT_SUFFIX}"
    assert final.name == f"{CheckpointKind.FINAL.value}{CHECKPOINT_SUFFIX}"


def test_saving_the_final_checkpoint_leaves_the_best_one_alone(
    store: TorchCheckpointStore,
) -> None:
    # The failure this store exists to prevent: under early stopping the two are
    # different models, and one filename would silently keep only the later one.
    store.save(make_checkpoint(CheckpointKind.BEST, epoch=2, fill=1.0))
    store.save(make_checkpoint(CheckpointKind.FINAL, epoch=5, fill=9.0))

    best = store.load(EXPERIMENT, CheckpointKind.BEST)
    final = store.load(EXPERIMENT, CheckpointKind.FINAL)

    assert best.epoch == 2
    assert final.epoch == 5
    assert torch.equal(
        best.parameters["head.bias"], torch.tensor([1.0])
    )
    assert torch.equal(
        final.parameters["head.bias"], torch.tensor([9.0])
    )


def test_both_files_exist_on_disk_side_by_side(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint(CheckpointKind.BEST))
    store.save(make_checkpoint(CheckpointKind.FINAL))

    written = sorted(
        path.name for path in (store.root / EXPERIMENT).iterdir()
    )
    assert written == ["best.pt", "final.pt"]


def test_each_experiment_gets_its_own_directory(
    store: TorchCheckpointStore,
) -> None:
    # Two architectures from one `--model all` run must not collide.
    store.save(make_checkpoint(experiment_id="gru_run"))
    store.save(
        make_checkpoint(
            experiment_id="lstm_run", architecture=ModelArchitecture.LSTM
        )
    )

    assert (store.root / "gru_run" / "best.pt").is_file()
    assert (store.root / "lstm_run" / "best.pt").is_file()


def test_the_save_path_is_returned_and_is_where_the_file_landed(
    store: TorchCheckpointStore,
) -> None:
    path = store.save(make_checkpoint())

    assert path == store.path_for(EXPERIMENT, CheckpointKind.BEST)
    assert path.is_file()


def test_the_experiment_directory_is_created_on_demand(
    store: TorchCheckpointStore,
) -> None:
    assert not store.root.exists()

    store.save(make_checkpoint())

    assert (store.root / EXPERIMENT).is_dir()


def test_the_root_is_resolved_to_an_absolute_path(tmp_path: Path) -> None:
    store = TorchCheckpointStore(tmp_path / "nested" / ".." / "checkpoints")

    assert store.root.is_absolute()
    assert ".." not in store.root.parts


def test_resaving_the_same_kind_replaces_it(store: TorchCheckpointStore) -> None:
    # Later epochs legitimately overwrite the same kind; what must not happen is
    # one kind overwriting the other.
    store.save(make_checkpoint(CheckpointKind.BEST, epoch=1, fill=1.0))
    store.save(make_checkpoint(CheckpointKind.BEST, epoch=3, fill=3.0))

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)
    assert reloaded.epoch == 3


# --- An exact round trip -----------------------------------------------------


def test_parameters_come_back_tensor_for_tensor(
    store: TorchCheckpointStore,
) -> None:
    original = make_checkpoint()
    store.save(original)

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)

    assert set(reloaded.parameters) == set(original.parameters)
    for name, tensor in original.parameters.items():
        assert torch.equal(reloaded.parameters[name], tensor), name


def test_the_configuration_comes_back_as_a_configuration(
    store: TorchCheckpointStore,
) -> None:
    # Rebuilt from a plain dictionary rather than unpickled, and equal field for
    # field: a stored model is only reusable if its configuration is recoverable.
    original = make_checkpoint()
    store.save(original)

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)

    assert isinstance(reloaded.config, TrainingConfig)
    assert reloaded.config == original.config


def test_the_frozen_threshold_survives_the_round_trip(
    store: TorchCheckpointStore,
) -> None:
    # The threshold is part of the model's decision rule. A checkpoint without it
    # could be reloaded and would score differently.
    store.save(make_checkpoint(threshold=0.0137))

    assert store.load(EXPERIMENT, CheckpointKind.BEST).threshold == pytest.approx(
        0.0137
    )


def test_the_scalars_that_identify_a_run_survive(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint())

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)

    assert reloaded.experiment_id == EXPERIMENT
    assert reloaded.kind is CheckpointKind.BEST
    assert reloaded.architecture is ModelArchitecture.GRU
    assert reloaded.epoch == 2
    assert reloaded.feature_dimension == 4
    assert reloaded.random_seed == 11
    assert reloaded.validation_loss == pytest.approx(0.125)


def test_the_validation_metrics_survive_as_floats(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint())

    metrics = store.load(EXPERIMENT, CheckpointKind.BEST).validation_metrics

    assert metrics == {"recall": 0.5, "pr_auc": 0.25}
    assert all(isinstance(value, float) for value in metrics.values())


def test_an_absent_optimizer_state_stays_absent(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint(optimizer_state=None))

    assert store.load(EXPERIMENT, CheckpointKind.BEST).optimizer_state is None


def test_an_optimizer_state_of_tensors_survives(
    store: TorchCheckpointStore,
) -> None:
    store.save(
        make_checkpoint(optimizer_state={"step": torch.tensor(3.0)})
    )

    state = store.load(EXPERIMENT, CheckpointKind.BEST).optimizer_state
    assert state is not None
    assert torch.equal(state["step"], torch.tensor(3.0))


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_round_trips(
    store: TorchCheckpointStore, architecture: ModelArchitecture
) -> None:
    store.save(make_checkpoint(architecture=architecture))

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)

    assert reloaded.architecture is architecture
    assert reloaded.config.architecture is architecture


# --- Loading is reading data, not executing it -------------------------------


def test_the_payload_loads_under_weights_only(
    store: TorchCheckpointStore,
) -> None:
    # The check behind the store's safety claim: if the payload ever gained a
    # pickled object, this raises rather than passing quietly.
    path = store.save(make_checkpoint())

    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["format_version"] == FORMAT_VERSION
    assert payload["architecture"] == ModelArchitecture.GRU.value


def test_the_stored_configuration_is_a_plain_dictionary(
    store: TorchCheckpointStore,
) -> None:
    path = store.save(make_checkpoint())

    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert isinstance(payload["config"], dict)
    assert payload["config"]["architecture"] == ModelArchitecture.GRU.value


def test_a_checkpoint_loads_onto_the_cpu(store: TorchCheckpointStore) -> None:
    # Written from wherever training ran, read back on this machine.
    store.save(make_checkpoint())

    reloaded = store.load(EXPERIMENT, CheckpointKind.BEST)

    for name, tensor in reloaded.parameters.items():
        assert tensor.device.type == "cpu", name


# --- Refusals ----------------------------------------------------------------


def test_a_missing_checkpoint_names_the_path_it_looked_at(
    store: TorchCheckpointStore,
) -> None:
    with pytest.raises(CheckpointNotFoundError) as caught:
        store.load("never_trained", CheckpointKind.BEST)

    assert "best.pt" in str(caught.value)
    assert "never_trained" in str(caught.value)


def test_asking_for_the_wrong_kind_is_a_miss_not_a_fallback(
    store: TorchCheckpointStore,
) -> None:
    # Returning the best checkpoint when the final one was asked for would report
    # the wrong parameters under the right name.
    store.save(make_checkpoint(CheckpointKind.BEST))

    with pytest.raises(CheckpointNotFoundError):
        store.load(EXPERIMENT, CheckpointKind.FINAL)


def test_loading_into_a_different_architecture_is_refused(
    store: TorchCheckpointStore,
) -> None:
    # Parameters from different architectures are never interchangeable in this
    # project, and a partial key match would be worse than a failure.
    store.save(make_checkpoint(architecture=ModelArchitecture.GRU))

    with pytest.raises(CheckpointArchitectureMismatchError) as caught:
        store.load(
            EXPERIMENT,
            CheckpointKind.BEST,
            expected_architecture=ModelArchitecture.LSTM,
        )

    message = str(caught.value)
    assert "gru" in message
    assert "lstm" in message


def test_a_matching_architecture_check_passes(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint(architecture=ModelArchitecture.BILSTM))

    reloaded = store.load(
        EXPERIMENT,
        CheckpointKind.BEST,
        expected_architecture=ModelArchitecture.BILSTM,
    )

    assert reloaded.architecture is ModelArchitecture.BILSTM


def test_no_architecture_check_is_made_when_none_is_asked_for(
    store: TorchCheckpointStore,
) -> None:
    store.save(make_checkpoint(architecture=ModelArchitecture.LSTM))

    assert store.load(EXPERIMENT, CheckpointKind.BEST).architecture is (
        ModelArchitecture.LSTM
    )
