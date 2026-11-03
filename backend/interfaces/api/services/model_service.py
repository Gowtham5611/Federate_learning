"""Model management and inference service for FedSecureAI API.

Loads and verifies authoritative federated AML models, manages lifecycle,
and serves bounded, thread-safe sequence inference.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from domain.enums.model_architecture import ModelArchitecture
from application.dto.training import TrainingConfig
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from interfaces.api.schemas.common import (
    FEATURE_DIMENSION,
    FROZEN_THRESHOLDS,
    MAX_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    SUPPORTED_MODELS_BY_DATASET,
    SequenceItem,
)
from interfaces.api.schemas.predict import BatchPredictItemResult

logger = logging.getLogger(__name__)

# Base directory for resolving relative paths
BASE_DIR = Path(__file__).resolve().parents[4]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "HI-Small": {
        "model_name": "hi_small_lstm_fedavg_iid_seed42",
        "architecture": ModelArchitecture.LSTM,
        "hidden_size": 64,
        "num_layers": 1,
        "feature_dimension": 57,
        "checkpoint_path": BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt",
        "expected_sha256": "dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4",
        "frozen_threshold": 0.3377837836742401,
    },
    "LI-Medium": {
        "model_name": "config_5_bilstm_fedavg_iid",
        "architecture": ModelArchitecture.BILSTM,
        "hidden_size": 64,
        "num_layers": 1,
        "feature_dimension": 57,
        "checkpoint_path": BASE_DIR / "results/LI_Medium_Results/checkpoints/config_5/global_best.pt",
        "expected_sha256": "bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2",
        "frozen_threshold": 0.2384829968214035,
    },
}


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityError(Exception):
    """Raised when a checkpoint fails hash verification or model contract checks."""
    pass


class ModelService:
    """Singleton service for loading, verifying, and executing FedSecureAI models."""

    _instance: Optional[ModelService] = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._models: Dict[str, nn.Module] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._loading_status: Dict[str, str] = {ds: "unloaded" for ds in MODEL_REGISTRY}
        self._inference_lock = threading.Lock()
        self._initialized = True

    def load_all_models(self) -> None:
        """Load and verify all models declared in MODEL_REGISTRY."""
        with self._inference_lock:
            for dataset_name in MODEL_REGISTRY:
                self._load_single_model(dataset_name)

    @staticmethod
    def _resolve_checkpoint_path(dataset: str, default_path: Path) -> Path:
        """Resolve model checkpoint path from environment variables or fallback to default."""
        env_key = f"FEDSECUREAI_{dataset.upper().replace('-', '_')}_CHECKPOINT"
        override = os.getenv(env_key)
        if override:
            return Path(override)
        ckpt_dir = os.getenv("FEDSECUREAI_CHECKPOINT_DIR")
        if ckpt_dir:
            candidate = Path(ckpt_dir) / default_path.name
            if candidate.exists():
                return candidate
            candidate_nested = Path(ckpt_dir) / dataset / default_path.name
            if candidate_nested.exists():
                return candidate_nested
        return default_path

    def _load_single_model(self, dataset: str) -> None:
        """Load and authenticate one model checkpoint."""
        spec = MODEL_REGISTRY.get(dataset)
        if not spec:
            raise ValueError(f"Unknown dataset '{dataset}'. Must be one of {list(MODEL_REGISTRY.keys())}.")

        ckpt_path: Path = self._resolve_checkpoint_path(dataset, spec["checkpoint_path"])
        expected_hash: str = spec["expected_sha256"]
        model_name: str = spec["model_name"]
        arch: ModelArchitecture = spec["architecture"]

        if not ckpt_path.exists():
            self._loading_status[dataset] = "missing_checkpoint"
            raise ModelIntegrityError(f"Checkpoint for {dataset} ({model_name}) not found at {ckpt_path}")

        # Hash verification
        actual_hash = calculate_sha256(ckpt_path)
        if actual_hash != expected_hash:
            self._loading_status[dataset] = "hash_mismatch"
            raise ModelIntegrityError(
                f"Checkpoint SHA-256 mismatch for {dataset} ({model_name})! "
                f"Expected: {expected_hash}, Actual: {actual_hash}"
            )

        # Build PyTorch model
        cfg = TrainingConfig(
            architecture=arch,
            hidden_size=spec["hidden_size"],
            num_layers=spec["num_layers"],
        )
        model = build_classifier(cfg, spec["feature_dimension"])

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        params_key = "parameters" if "parameters" in ckpt else "model_state_dict"
        model.load_state_dict(ckpt[params_key])
        model.eval()

        # Warmup forward pass
        dummy_x = torch.zeros(1, MAX_SEQUENCE_LENGTH, spec["feature_dimension"])
        dummy_l = torch.tensor([MAX_SEQUENCE_LENGTH], dtype=torch.int64)
        with torch.no_grad():
            _ = model(dummy_x, dummy_l)

        self._models[dataset] = model
        self._metadata[dataset] = {
            "model_name": model_name,
            "architecture": arch.value,
            "parameter_count": model.parameter_count(),
            "sequence_length": MAX_SEQUENCE_LENGTH,
            "feature_dimension": spec["feature_dimension"],
            "frozen_threshold": spec["frozen_threshold"],
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": actual_hash,
            "checkpoint_verified": True,
        }
        self._loading_status[dataset] = "loaded"
        logger.info("Successfully loaded and verified %s model: %s", dataset, model_name)

    def is_ready(self) -> bool:
        """Check whether all declared models are loaded and ready."""
        return all(status == "loaded" for status in self._loading_status.values())

    def get_loaded_models(self) -> List[str]:
        """Return list of loaded model names."""
        return [meta["model_name"] for meta in self._metadata.values()]

    def get_model_metadata(self, dataset: str) -> Dict[str, Any]:
        """Return verified metadata for a dataset model."""
        if dataset not in self._metadata:
            raise KeyError(f"Metadata not found for dataset '{dataset}'. Model status: {self._loading_status.get(dataset, 'unknown')}")
        return dict(self._metadata[dataset])

    def get_model(self, dataset: str) -> nn.Module:
        """Get the PyTorch model for explainability or inspection."""
        if dataset not in self._models:
            raise KeyError(f"Model for '{dataset}' is not loaded.")
        return self._models[dataset]

    def predict_sequence(
        self, dataset: str, sequence: SequenceItem
    ) -> Tuple[float, float, str, float]:
        """Predict laundering probability for a single transaction sequence.

        Returns:
            (predicted_probability, decision_threshold, decision_label, latency_ms)
        """
        if dataset not in self._models:
            raise ModelIntegrityError(f"Model for '{dataset}' is not loaded or unavailable.")

        model = self._models[dataset]
        threshold = MODEL_REGISTRY[dataset]["frozen_threshold"]

        # Prepare (1, 10, 57) tensor
        feat_arr = np.zeros((1, MAX_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        seq_len = sequence.length or len(sequence.features)
        
        for t_idx, step in enumerate(sequence.features):
            feat_arr[0, t_idx, :] = np.array(step, dtype=np.float32)

        x_tensor = torch.from_numpy(feat_arr)
        l_tensor = torch.tensor([seq_len], dtype=torch.int64)

        t0 = time.perf_counter()
        with torch.no_grad():
            logit = model(x_tensor, l_tensor).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
        latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = "FLAGGED_LAUNDERING" if prob >= threshold else "CLEARED_LEGITIMATE"
        return prob, threshold, decision, latency_ms

    def predict_batch(
        self, dataset: str, items: List[SequenceItem]
    ) -> Tuple[List[BatchPredictItemResult], float]:
        """Predict laundering probabilities for a bounded batch of sequences (max 64).

        Returns:
            (results, total_batch_time_ms)
        """
        if dataset not in self._models:
            raise ModelIntegrityError(f"Model for '{dataset}' is not loaded.")

        if len(items) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(items)} exceeds maximum allowed limit of {MAX_BATCH_SIZE}.")

        model = self._models[dataset]
        threshold = MODEL_REGISTRY[dataset]["frozen_threshold"]

        batch_size = len(items)
        feat_arr = np.zeros((batch_size, MAX_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        lengths = []
        item_indices = []
        results: List[BatchPredictItemResult] = [None] * batch_size  # type: ignore

        # Prepare tensors with error isolation per item
        for idx, item in enumerate(items):
            try:
                seq_len = item.length or len(item.features)
                for t_idx, step in enumerate(item.features):
                    feat_arr[idx, t_idx, :] = np.array(step, dtype=np.float32)
                lengths.append(seq_len)
                item_indices.append(idx)
            except Exception as e:
                results[idx] = BatchPredictItemResult(
                    index=idx,
                    sample_id=item.sample_id,
                    success=False,
                    error=f"Item preparation error: {str(e)}",
                )

        t0 = time.perf_counter()
        if item_indices:
            valid_x = torch.from_numpy(feat_arr[item_indices])
            valid_l = torch.tensor(lengths, dtype=torch.int64)
            with torch.no_grad():
                logits = model(valid_x, valid_l).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()

            if isinstance(probs, float):
                probs = [probs]

            for pos, orig_idx in enumerate(item_indices):
                p = float(probs[pos])
                dec = "FLAGGED_LAUNDERING" if p >= threshold else "CLEARED_LEGITIMATE"
                results[orig_idx] = BatchPredictItemResult(
                    index=orig_idx,
                    sample_id=items[orig_idx].sample_id,
                    success=True,
                    predicted_probability=p,
                    decision_threshold=threshold,
                    decision=dec,
                )
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0

        return results, batch_latency_ms

