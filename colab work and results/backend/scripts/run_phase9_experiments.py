#!/usr/bin/env python
"""Runner script for Phase 9: FedProx Optimization & SCAFFOLD Experimentation.

Usage:
    python backend/scripts/run_phase9_experiments.py [--rounds 5] [--skip-hybrid]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure backend is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from application.services.phase9_optimization_service import (  # noqa: E402
    Phase9OptimizationService,
)


def main() -> int:
    """Execute complete Phase 9 experimental suite."""
    parser = argparse.ArgumentParser(
        description="Phase 9: FedProx Tuning, SCAFFOLD Experiment, & Client Drift Analysis"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated communication rounds per experiment (default: 5)",
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip the optional FedProx-SCAFFOLD hybrid experiment",
    )
    parser.add_argument(
        "--fedprox-only",
        action="store_true",
        help="Run only the FedProx mu ablation sweeps (do not run SCAFFOLD)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("Phase9Runner")

    start_time = time.time()
    logger.info("Initializing Phase 9 Optimization Service...")
    service = Phase9OptimizationService()

    if args.fedprox_only:
        logger.info(
            "Executing FedProx mu ablation ONLY (rounds=%d)...",
            args.rounds,
        )
        res = service.execute_fedprox_ablation_only(rounds=args.rounds)
        elapsed = time.time() - start_time
        logger.info(
            "FedProx ablation completed in %.2f seconds (%.2f minutes).",
            elapsed,
            elapsed / 60,
        )
        print("\n" + "=" * 80)
        print("PHASE 9 FEDPROX ABLATION EXECUTION COMPLETE")
        print("=" * 80)
        print(f"FedProx Ablation CSV: {res['mu_ablation_csv']}")
        print(f"Best FedProx Summary CSV: {res['best_fedprox_csv']}")
        print(f"FedAvg vs FedProx CSV: {res['fedavg_vs_fedprox_csv']}")
        print("=" * 80 + "\n")
        return 0

    logger.info(
        "Beginning execution of Phase 9 suite with rounds=%d, run_hybrid=%s...",
        args.rounds,
        not args.skip_hybrid,
    )

    synthesis = service.execute_phase9_suite(
        rounds=args.rounds,
        run_hybrid=not args.skip_hybrid,
    )

    elapsed = time.time() - start_time
    logger.info(
        "Phase 9 execution completed in %.2f seconds (%.2f minutes).",
        elapsed,
        elapsed / 60,
    )

    print("\n" + "=" * 80)
    print("PHASE 9 EXPERIMENTAL SUITE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Metadata JSON: {synthesis['metadata_path']}")
    print(f"Markdown Report: {synthesis['report_path']}")
    print(f"Visualizations: {len(synthesis['plots'])} figures generated")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
