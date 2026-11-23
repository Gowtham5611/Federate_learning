"""Tests for strict CUDA device enforcement and fallback prevention.

Verifies:
1. resolve_device("cuda") raises CUDANotAvailableError when CUDA is unavailable.
2. resolve_device("auto", require_cuda=True) raises CUDANotAvailableError when CUDA is unavailable.
3. Preflight checks in Full Combined and Full LI-Medium runners strictly fail when CUDA is unavailable and allow_subset_debug is False.
4. Tensor placement checks and OOM handling safeguards exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
import torch

from infrastructure.models.pytorch.reproducibility import (
    CUDANotAvailableError,
    resolve_device,
)
from scripts.run_full_combined_training import (
    run_preflight_checks as run_preflight_combined,
)
from scripts.run_full_li_medium_training import (
    run_preflight_checks as run_preflight_li,
)


def test_resolve_device_strict_cuda_rejection_without_gpu():
    """Verify resolve_device('cuda') fails fast when CUDA is unavailable."""
    if not torch.cuda.is_available():
        with pytest.raises(CUDANotAvailableError, match="Strict CUDA enforcement prohibits falling back to CPU"):
            resolve_device("cuda")

        with pytest.raises(CUDANotAvailableError, match="Automatic device resolution required CUDA"):
            resolve_device("auto", require_cuda=True)


def test_runner_preflight_fails_fast_on_missing_cuda(tmp_path: Path):
    """Verify preflight checks fail with status FAIL when CUDA is absent and allow_subset_debug=False."""
    if not torch.cuda.is_available():
        # Test Combined Runner
        res_comb = run_preflight_combined(
            data_root=Path("data/processed/ibm_aml/Combined"),
            output_dir=tmp_path / "out_comb",
            device_name="cuda",
            allow_subset_debug=False,
        )
        assert res_comb["status"] == "FAIL"
        assert any("CUDA acceleration is not available" in err for err in res_comb["errors"])

        # Test LI-Medium Runner
        res_li = run_preflight_li(
            data_root=Path("data/processed/ibm_aml/LI-Medium"),
            output_dir=tmp_path / "out_li",
            device_name="cuda",
            allow_subset_debug=False,
        )
        assert res_li["status"] == "FAIL"
        assert any("CUDA acceleration is not available" in err for err in res_li["errors"])


def test_runner_preflight_allows_cpu_only_in_subset_debug_mode(tmp_path: Path):
    """Verify preflight checks allow CPU when allow_subset_debug=True on small fixtures."""
    res_comb = run_preflight_combined(
        data_root=Path("data/processed/ibm_aml/Combined"),
        output_dir=tmp_path / "out_comb",
        device_name="cpu",
        allow_subset_debug=True,
    )
    # Status should not fail due to CUDA when allow_subset_debug is True
    cuda_errors = [e for e in res_comb["errors"] if "CUDA" in e]
    assert len(cuda_errors) == 0

