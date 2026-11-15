"""Verify Phase 13.1 numerical parity against Phase 7 validation predictions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "backend"))

from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from interfaces.api.services.model_service import ModelService, MODEL_REGISTRY, calculate_sha256
from interfaces.api.schemas.common import SequenceItem


def evaluate_parity_for_dataset(
    dataset_name: str,
    arch: ModelArchitecture,
    hidden_size: int,
    npz_path: Path,
    num_samples: int = 1000,
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

    spec = MODEL_REGISTRY[dataset_name]
    threshold = spec["frozen_threshold"]
    ckpt_path = spec["checkpoint_path"]
    actual_hash = calculate_sha256(ckpt_path)

    model_service = ModelService()
    diffs = []
    decision_matches = 0

    for i in range(num_samples):
        feat = batch.features[i].tolist()
        length = int(batch.lengths[i])
        seq_item = SequenceItem(features=feat, length=length)

        api_prob, api_thresh, api_dec, _ = model_service.predict_sequence(
            dataset=dataset_name, sequence=seq_item
        )
        ref_prob = float(saved_probs[i])
        ref_dec = "FLAGGED_LAUNDERING" if ref_prob >= threshold else "CLEARED_LEGITIMATE"

        diff = abs(api_prob - ref_prob)
        diffs.append(diff)

        if api_dec == ref_dec:
            decision_matches += 1

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    p99_diff = float(np.percentile(diffs, 99))
    decision_match_pct = (decision_matches / num_samples) * 100.0
    passed = max_diff < 1e-5 and decision_match_pct == 100.0

    return {
        "dataset": dataset_name,
        "model_name": spec["model_name"],
        "architecture": spec["architecture"].value,
        "parameter_count": model_service.get_model_metadata(dataset_name)["parameter_count"],
        "checkpoint_path": str(ckpt_path.relative_to(BASE_DIR)),
        "checkpoint_sha256_actual": actual_hash,
        "checkpoint_sha256_expected": spec["expected_sha256"],
        "frozen_threshold": threshold,
        "num_samples_evaluated": num_samples,
        "comparison_method": "Exact float32 inference vs saved Phase 7 NPZ probabilities",
        "max_absolute_error": max_diff,
        "mean_absolute_error": mean_diff,
        "p99_absolute_error": p99_diff,
        "tolerance": 1e-5,
        "threshold_decision_matches": decision_matches,
        "threshold_decision_match_pct": decision_match_pct,
        "parity_passed": passed,
    }


def main():
    print("Loading models into ModelService...")
    ms = ModelService()
    ms.load_all_models()

    report = {
        "audit_phase": "Phase 13.1",
        "evaluation_timestamp": "2026-10-27T07:00:00Z",
        "tolerance": 1e-5,
        "datasets": {},
    }

    print("Evaluating HI-Small (1000 samples)...")
    hi_npz = BASE_DIR / "results/phase7/HI_Small/validation_predictions/iid_validation_predictions.npz"
    report["datasets"]["HI-Small"] = evaluate_parity_for_dataset(
        dataset_name="HI-Small",
        arch=ModelArchitecture.LSTM,
        hidden_size=64,
        npz_path=hi_npz,
        num_samples=1000,
    )

    print("Evaluating LI-Medium (1000 samples)...")
    li_npz = BASE_DIR / "results/phase7/LI_Medium/validation_predictions/iid_validation_predictions.npz"
    report["datasets"]["LI-Medium"] = evaluate_parity_for_dataset(
        dataset_name="LI-Medium",
        arch=ModelArchitecture.BILSTM,
        hidden_size=64,
        npz_path=li_npz,
        num_samples=1000,
    )

    out_file = BASE_DIR / "results/phase13_1/phase13_1_parity_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Parity report written to {out_file}")
    for ds, res in report["datasets"].items():
        print(f"{ds}: Max Error = {res['max_absolute_error']:.8e}, Decision Match = {res['threshold_decision_match_pct']}%, Passed = {res['parity_passed']}")


if __name__ == "__main__":
    main()

