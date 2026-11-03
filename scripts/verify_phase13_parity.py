"""Verify inference parity of Phase 13 ModelService against Phase 7 validation predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np

# Ensure backend is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "backend"))

from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from interfaces.api.services.model_service import ModelService, MODEL_REGISTRY
from interfaces.api.schemas.common import SequenceItem


def verify_dataset_parity(
    dataset_name: str,
    arch: ModelArchitecture,
    hidden_size: int,
    npz_path: Path,
    num_samples: int = 500,
) -> dict:
    data_dir = BASE_DIR / f"data/processed/ibm_aml/{dataset_name}"
    provider = ProcessedSequenceProvider(data_dir)
    cfg = TrainingConfig(
        architecture=arch,
        batch_size=64,
        hidden_size=hidden_size,
        num_layers=1,
        row_limit=num_samples,
    )
    ds = provider.load(SplitName.VALIDATION, cfg)
    batch = ds._assemble(ds.target_rows[:num_samples])

    saved_data = np.load(npz_path)
    saved_probs = saved_data["probabilities"][:num_samples]
    saved_model_name = str(saved_data.get("model_name", "unknown"))

    model_service = ModelService()
    api_probs = []
    diffs = []

    for i in range(num_samples):
        feat = batch.features[i].tolist()
        length = int(batch.lengths[i])
        seq_item = SequenceItem(features=feat, length=length)
        prob, threshold, decision, latency = model_service.predict_sequence(
            dataset=dataset_name, sequence=seq_item
        )
        ref_prob = float(saved_probs[i])
        diff = abs(prob - ref_prob)
        api_probs.append(prob)
        diffs.append(diff)

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    passed = max_diff < 1e-5

    meta = model_service.get_model_metadata(dataset_name)
    spec = MODEL_REGISTRY[dataset_name]

    return {
        "dataset": dataset_name,
        "model_name": meta["model_name"],
        "architecture": meta["architecture"],
        "parameter_count": meta["parameter_count"],
        "checkpoint_path": str(spec["checkpoint_path"].relative_to(BASE_DIR)),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "expected_sha256": spec["expected_sha256"],
        "frozen_threshold": meta["frozen_threshold"],
        "num_samples_evaluated": num_samples,
        "max_absolute_error": max_diff,
        "mean_absolute_error": mean_diff,
        "tolerance": 1e-5,
        "parity_passed": passed,
    }


def main():
    print("Loading all models into ModelService...")
    ms = ModelService()
    ms.load_all_models()

    results = {}

    print("Verifying HI-Small parity...")
    hi_npz = BASE_DIR / "results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz"
    results["HI-Small"] = verify_dataset_parity(
        dataset_name="HI-Small",
        arch=ModelArchitecture.LSTM,
        hidden_size=64,
        npz_path=hi_npz,
        num_samples=500,
    )

    print("Verifying LI-Medium parity...")
    li_npz = BASE_DIR / "results/phase7/LI_Medium/validation_predictions/iid_validation_predictions.npz"
    results["LI-Medium"] = verify_dataset_parity(
        dataset_name="LI-Medium",
        arch=ModelArchitecture.BILSTM,
        hidden_size=64,
        npz_path=li_npz,
        num_samples=500,
    )

    out_file = BASE_DIR / "results/phase13/phase13_model_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Parity verification written to {out_file}")
    print(f"HI-Small: max_error={results['HI-Small']['max_absolute_error']:.8e}, passed={results['HI-Small']['parity_passed']}")
    print(f"LI-Medium: max_error={results['LI-Medium']['max_absolute_error']:.8e}, passed={results['LI-Medium']['parity_passed']}")


if __name__ == "__main__":
    main()
