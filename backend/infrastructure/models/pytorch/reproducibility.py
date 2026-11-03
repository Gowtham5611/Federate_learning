"""Seeding, device selection, and an honest answer about determinism.

Seeding is easy to do and easy to overstate. Setting the three seeds makes a run
repeatable in the sense that matters day to day — the same configuration produces
the same model on the same machine — but it does not make results bitwise
reproducible across hardware, thread counts or library versions. Recurrent kernels
in particular reduce over time steps in an order that can depend on the backend.

So this module does two separate things and reports them separately: it seeds
everything it can, and it *asks* for deterministic kernels and records whether the
request succeeded. A ``False`` there is written into the experiment record rather
than quietly dropped, because "reproducible" is a claim, and a claim that turns out
to be untrue is worse than a limitation that was stated.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

CUBLAS_WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISTIC_VALUE = ":4096:8"
"""Setting cuBLAS requires for deterministic GEMMs.

Irrelevant on the CPU-only machine this phase runs on, and set anyway so that a
later GPU run does not silently lose determinism it thought it had.
"""


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch from one value.

    Args:
        seed: The seed. Recorded in every result row and checkpoint, so a run can
            be repeated rather than merely described.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU on this machine
        torch.cuda.manual_seed_all(seed)


def request_deterministic_kernels() -> bool:
    """Ask PyTorch for deterministic algorithms and report whether it agreed.

    Returns:
        ``True`` if deterministic kernels were enabled. ``False`` is a normal
        outcome — some operations have no deterministic implementation — and is
        reported rather than treated as an error, because the alternative is a run
        that fails for a property nothing here depends on.
    """
    os.environ.setdefault(CUBLAS_WORKSPACE_VARIABLE, CUBLAS_DETERMINISTIC_VALUE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError) as error:
        logger.warning(
            "Deterministic kernels are unavailable in this PyTorch build (%s). "
            "Runs remain seeded but are not guaranteed bitwise reproducible.",
            error,
        )
        return False
    return True


class CUDANotAvailableError(RuntimeError):
    """Raised when CUDA is required or requested, but is unavailable on the host."""


def resolve_device(requested: str | None = None, *, require_cuda: bool = False) -> torch.device:
    """Pick the device to train on.

    Args:
        requested: An explicit torch device string ('cuda', 'cpu', 'cuda:0', None).
        require_cuda: If True, strictly raises CUDANotAvailableError if CUDA is unavailable.

    Raises:
        CUDANotAvailableError: If CUDA is requested or required, but torch.cuda.is_available() is False.
    """
    if requested and requested.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise CUDANotAvailableError(
                f"CUDA was explicitly requested ('{requested}'), but torch.cuda.is_available() is False. "
                "Strict CUDA enforcement prohibits falling back to CPU."
            )
        return torch.device(requested)

    if requested and requested.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_cuda:
            raise CUDANotAvailableError(
                "Automatic device resolution required CUDA, but torch.cuda.is_available() is False. "
                "Full training requires CUDA GPU acceleration."
            )
        return torch.device("cpu")

    if requested and requested.lower() != "auto":
        if require_cuda and requested.lower() == "cpu":
            raise CUDANotAvailableError("Strict CUDA execution is required, but device='cpu' was provided.")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise CUDANotAvailableError(
            "CUDA is required for production training, but torch.cuda.is_available() is False."
        )

    return torch.device("cpu")


def describe_runtime(device: torch.device, *, deterministic: bool) -> dict[str, str]:
    """Facts about the environment a result was produced in.

    Recorded alongside the metrics because a training time means nothing without
    them, and because determinism guarantees are version- and backend-specific.
    """
    return {
        "framework": "torch",
        "framework_version": torch.__version__,
        "device": str(device),
        "cuda_available": str(torch.cuda.is_available()),
        "threads": str(torch.get_num_threads()),
        "deterministic_kernels_requested": str(deterministic),
    }
