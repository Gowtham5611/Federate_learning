"""Retrieval utility for querying FedSecureAI experiment results.

Allows searching saved results by:
- date
- distribution
- algorithm
- model
- mu
- seed
- experiment_id

Does NOT start training; strictly reads saved metadata and metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def retrieve_experiments(
    runs_root: Path,
    *,
    date: str | None = None,
    distribution: str | None = None,
    algorithm: str | None = None,
    model: str | None = None,
    mu: str | None = None,
    seed: int | None = None,
    experiment_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search saved runs across manifests and summaries without initiating training."""
    matches = []

    # Find all all_12_experiments.csv or experiment_manifest.json under runs_root
    manifest_paths = list(runs_root.rglob("experiment_manifest.json"))
    if not manifest_paths:
        # Fallback to master CSVs
        csv_paths = list(runs_root.rglob("all_12_experiments.csv"))
        for cp in csv_paths:
            df = pd.read_csv(cp)
            for _, row in df.iterrows():
                rec = row.to_dict()
                matches.append(rec)
    else:
        for mp in manifest_paths:
            try:
                with mp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                exps = data.get("experiments") or data.get("configurations") or []
                matches.extend(exps)
            except Exception:
                pass

    filtered = []
    for exp in matches:
        e_id = exp.get("experiment_id") or ""
        e_dist = exp.get("data_distribution") or ""
        e_algo = exp.get("federated_algorithm") or ""
        e_model = exp.get("model") or ""
        e_mu = str(exp.get("mu_str") or exp.get("mu") or "")
        e_seed = exp.get("seed")

        if experiment_id and experiment_id.lower() not in e_id.lower():
            continue
        if date and date not in e_id:
            continue
        if distribution and distribution.lower() != e_dist.lower():
            continue
        if algorithm and algorithm.lower() != e_algo.lower():
            continue
        if model and model.lower() != e_model.lower():
            continue
        if mu and str(mu).lower() != e_mu.lower():
            continue
        if seed is not None and e_seed != seed:
            continue

        filtered.append(exp)

    return filtered


def main() -> int:
    """CLI entrypoint for retrieval utility."""
    parser = argparse.ArgumentParser(
        description="Retrieve FedSecureAI experiment results and metrics without training."
    )
    parser.add_argument(
        "--runs-dir",
        default="experiments/results/fedsecureai_runs",
        help="Root directory holding dated federated runs.",
    )
    parser.add_argument("--date", help="Date filter (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--distribution", choices=["iid", "non_iid"], help="Data distribution")
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"], help="Federated algorithm")
    parser.add_argument("--model", choices=["gru", "lstm", "bilstm"], help="Model architecture")
    parser.add_argument("--mu", help="Proximal coefficient (e.g. 0.1 or na)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--experiment-id", help="Exact or substring experiment ID")
    parser.add_argument("--json", action="store_true", help="Output in raw JSON format")

    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    results = retrieve_experiments(
        runs_dir,
        date=args.date,
        distribution=args.distribution,
        algorithm=args.algorithm,
        model=args.model,
        mu=args.mu,
        seed=args.seed,
        experiment_id=args.experiment_id,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\nFound {len(results)} matching experiment(s):")
    rule = "=" * 80
    print(rule)
    for r in results:
        e_id = r.get("experiment_id", "N/A")
        dist = r.get("data_distribution", "N/A")
        algo = r.get("federated_algorithm", "N/A")
        mdl = r.get("model", "N/A")
        status = r.get("status", "N/A")
        pr_auc = r.get("test_pr_auc")
        roc_auc = r.get("test_roc_auc")
        f1 = r.get("test_f1")
        pr_str = f"{pr_auc:.6f}" if pr_auc is not None else "N/A"
        roc_str = f"{roc_auc:.6f}" if roc_auc is not None else "N/A"
        f1_str = f"{f1:.6f}" if f1 is not None else "N/A"
        ckpt = r.get("checkpoint_path", "N/A")

        print(f"Experiment ID: {e_id}")
        print(f"  Configuration: Dist={dist} | Algo={algo} | Model={mdl} | Status={status}")
        print(f"  Test Metrics:  PR-AUC={pr_str} | ROC-AUC={roc_str} | F1={f1_str}")
        print(f"  Checkpoint:    {ckpt}")
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
