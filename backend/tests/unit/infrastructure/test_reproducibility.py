"""Tests for :mod:`infrastructure.models.pytorch.reproducibility`.

The module makes two claims that must not be conflated, so they are tested apart:

1. **Seeding works.** One seed fixes Python, NumPy and PyTorch, and the same seed
   gives the same draws.
2. **Determinism is requested, not guaranteed.** The request returns a boolean and
   that boolean is what gets recorded. These tests assert the *shape* of that
   answer — a bool, echoed into the runtime description — and never assert that it
   is ``True``, because on a CPU-only build it legitimately may not be.

``request_deterministic_kernels`` mutates process-wide torch state, so the tests
that call it restore what they changed.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from infrastructure.models.pytorch.reproducibility import (
    CUBLAS_DETERMINISTIC_VALUE,
    CUBLAS_WORKSPACE_VARIABLE,
    describe_runtime,
    request_deterministic_kernels,
    resolve_device,
    seed_everything,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def restored_torch_state() -> Iterator[None]:
    """Put the process-wide determinism switches back afterwards."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


# --- Seeding -----------------------------------------------------------------


def test_one_seed_reaches_all_three_generators() -> None:
    seed_everything(1234)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    seed_everything(1234)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_different_seeds_give_different_draws() -> None:
    seed_everything(1)
    first = float(torch.rand(1))
    seed_everything(2)
    second = float(torch.rand(1))

    assert first != second


def test_seeding_fixes_model_initialisation() -> None:
    # The property that makes an experiment repeatable: the same seed builds the
    # same untrained model.
    seed_everything(7)
    first = torch.nn.GRU(3, 4).state_dict()["weight_ih_l0"].clone()
    seed_everything(7)
    second = torch.nn.GRU(3, 4).state_dict()["weight_ih_l0"].clone()

    assert torch.equal(first, second)


def test_seeding_accepts_zero() -> None:
    # Zero is a legitimate seed, and a truthiness check somewhere would skip it.
    seed_everything(0)
    first = float(torch.rand(1))
    seed_everything(0)

    assert float(torch.rand(1)) == first


# --- Requesting determinism --------------------------------------------------


@pytest.mark.usefixtures("restored_torch_state")
def test_the_request_answers_with_a_boolean() -> None:
    # Never asserted to be True: on this CPU-only build the honest answer may be
    # either, and the value is recorded rather than relied upon.
    assert isinstance(request_deterministic_kernels(), bool)


@pytest.mark.usefixtures("restored_torch_state")
def test_the_request_sets_the_cublas_workspace_variable() -> None:
    previous = os.environ.get(CUBLAS_WORKSPACE_VARIABLE)
    try:
        request_deterministic_kernels()
        assert os.environ[CUBLAS_WORKSPACE_VARIABLE] == (
            previous if previous is not None else CUBLAS_DETERMINISTIC_VALUE
        )
    finally:
        if previous is None:
            os.environ.pop(CUBLAS_WORKSPACE_VARIABLE, None)
        else:
            os.environ[CUBLAS_WORKSPACE_VARIABLE] = previous


@pytest.mark.usefixtures("restored_torch_state")
def test_an_existing_workspace_setting_is_left_alone() -> None:
    # setdefault, not assignment: a value the user exported deliberately is theirs.
    previous = os.environ.get(CUBLAS_WORKSPACE_VARIABLE)
    os.environ[CUBLAS_WORKSPACE_VARIABLE] = ":16:8"
    try:
        request_deterministic_kernels()
        assert os.environ[CUBLAS_WORKSPACE_VARIABLE] == ":16:8"
    finally:
        if previous is None:
            os.environ.pop(CUBLAS_WORKSPACE_VARIABLE, None)
        else:
            os.environ[CUBLAS_WORKSPACE_VARIABLE] = previous


@pytest.mark.usefixtures("restored_torch_state")
def test_the_request_turns_off_cudnn_benchmarking() -> None:
    # Benchmarking picks whichever kernel is fastest on the day, which is the
    # opposite of reproducible.
    request_deterministic_kernels()

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.usefixtures("restored_torch_state")
def test_the_request_is_warn_only_so_a_run_is_never_aborted_for_it() -> None:
    # Recurrent kernels have no deterministic implementation on every backend.
    # Warning is right; raising would fail a run over a property nothing depends
    # on, and would push results towards not being produced at all.
    if request_deterministic_kernels():
        assert torch.is_deterministic_algorithms_warn_only_enabled()


@pytest.mark.usefixtures("restored_torch_state")
def test_asking_twice_is_harmless() -> None:
    assert request_deterministic_kernels() == request_deterministic_kernels()


# --- Device selection --------------------------------------------------------


def test_an_explicit_device_is_honoured() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_automatic_selection_falls_back_to_the_cpu() -> None:
    # No GPU on this machine, so the honest answer is the CPU rather than a
    # device string that would fail at the first tensor copy.
    device = resolve_device(None)

    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_an_empty_string_means_automatic() -> None:
    assert resolve_device("") == resolve_device(None)


def test_an_unusable_device_is_not_silently_replaced() -> None:
    # An explicit request is taken at face value; the failure surfaces as torch's
    # own error rather than as a quiet downgrade the results would not record.
    with pytest.raises((RuntimeError, ValueError)):
        resolve_device("not-a-device")


# --- The recorded runtime ----------------------------------------------------


def test_the_runtime_description_names_the_framework_and_its_version() -> None:
    runtime = describe_runtime(torch.device("cpu"), deterministic=False)

    assert runtime["framework"] == "torch"
    assert runtime["framework_version"] == torch.__version__


def test_the_runtime_description_records_the_determinism_answer() -> None:
    enabled = describe_runtime(torch.device("cpu"), deterministic=True)
    disabled = describe_runtime(torch.device("cpu"), deterministic=False)

    assert enabled["deterministic_kernels_requested"] == "True"
    assert disabled["deterministic_kernels_requested"] == "False"


def test_the_runtime_description_records_the_device_and_the_thread_count() -> None:
    # A training time is meaningless without them: the same model on the same data
    # takes a different time at a different thread count.
    runtime = describe_runtime(torch.device("cpu"), deterministic=False)

    assert runtime["device"] == "cpu"
    assert runtime["threads"] == str(torch.get_num_threads())
    assert runtime["cuda_available"] == str(torch.cuda.is_available())


def test_every_recorded_value_is_a_string() -> None:
    # The values go into a CSV column and a JSON report, so they are strings here
    # rather than at each call site.
    runtime = describe_runtime(torch.device("cpu"), deterministic=False)

    assert all(isinstance(value, str) for value in runtime.values())


def test_the_description_is_a_fresh_mapping_each_time() -> None:
    first = describe_runtime(torch.device("cpu"), deterministic=False)
    first["device"] = "tampered"
    second = describe_runtime(torch.device("cpu"), deterministic=False)

    assert second["device"] == "cpu"
