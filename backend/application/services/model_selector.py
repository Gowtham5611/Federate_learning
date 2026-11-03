"""Automated discovery and validation-driven model selection for Phase 7 Ensemble.

Scans Phase 5 (IID) and Phase 6 (Non-IID) experiment results, filters completed full runs,
and identifies the top-performing model from each distribution based strictly on centralized
validation PR-AUC. Test splits are NEVER inspected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from application.dto.ensemble import ModelSelectionRecord
from domain.enums.checkpoint_kind import CheckpointKind
from domain.enums.model_architecture import ModelArchitecture
from domain.exceptions.training import CheckpointNotFoundError

logger = logging.getLogger(__name__)


class ModelSelector:
    """Discovers and selects the optimal IID and Non-IID models on validation."""

    def __init__(self, results_root: Path, checkpoints_root: Path) -> None:
        """Initialize selector with base result and checkpoint directories.

        Args:
            results_root: Directory holding experiments/results.
            checkpoints_root: Directory holding models/checkpoints.
        """
        self._results_root = results_root.resolve()
        self._checkpoints_root = checkpoints_root.resolve()

    def scan_iid_candidates(self) -> list[ModelSelectionRecord]:
        """Scan Phase 5 IID results for completed candidate experiments.

        Returns:
            List of ModelSelectionRecord for all valid full IID experiments.
        """
        candidates: list[ModelSelectionRecord] = []
        iid_dir = self._results_root / "federated" / "iid"

        for algo in ["fedavg", "fedprox"]:
            table_path = iid_dir / algo / f"iid_{algo}_results.csv"
            if not table_path.is_file():
                logger.warning("IID results table not found: %s", table_path)
                continue

            df = pd.read_csv(table_path)
            for exp_id in df["experiment_id"].dropna().unique():
                exp_df = df[df["experiment_id"] == exp_id]
                val_df = exp_df[exp_df["split"] == "validation"]
                if val_df.empty:
                    continue

                max_round = int(val_df["round"].max())
                # Exclude reduced smoke tests (only 5-round full runs are eligible)
                if max_round < 5:
                    logger.debug(
                        "Skipping reduced IID run %s (max round %d < 5)",
                        exp_id,
                        max_round,
                    )
                    continue

                r_max = val_df[val_df["round"] == max_round].iloc[0]
                model_name = str(r_max["model"]).lower()
                val_pr_auc = float(r_max["pr_auc"])

                # Verify checkpoint exists
                ckpt_path = (
                    self._checkpoints_root
                    / "federated"
                    / "iid"
                    / algo
                    / model_name
                    / exp_id
                    / f"{CheckpointKind.GLOBAL_BEST.value}.pt"
                )
                if not ckpt_path.is_file():
                    logger.warning(
                        "Checkpoint missing for IID experiment %s at %s",
                        exp_id,
                        ckpt_path,
                    )
                    continue

                candidates.append(
                    ModelSelectionRecord(
                        distribution="iid",
                        algorithm=algo,
                        model=model_name,
                        experiment_id=exp_id,
                        checkpoint_kind=CheckpointKind.GLOBAL_BEST.value,
                        checkpoint_path=str(ckpt_path),
                        selection_metric="pr_auc",
                        validation_score=val_pr_auc,
                    )
                )

        return candidates

    def scan_non_iid_candidates(self) -> list[ModelSelectionRecord]:
        """Scan Phase 6 Non-IID results for completed candidate experiments.

        Returns:
            List of ModelSelectionRecord for all valid Non-IID experiments.
        """
        candidates: list[ModelSelectionRecord] = []
        non_iid_dir = self._results_root / "federated" / "non_iid"

        architectures = [
            ModelArchitecture.GRU.value,
            ModelArchitecture.LSTM.value,
            ModelArchitecture.BILSTM.value,
        ]
        for algo in ["fedavg", "fedprox"]:
            for arch in architectures:
                arch_dir = non_iid_dir / algo / arch
                if not arch_dir.is_dir():
                    continue

                for exp_dir in arch_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue

                    csv_path = exp_dir / "results.csv"
                    report_path = exp_dir / "report.json"
                    if not (csv_path.is_file() and report_path.is_file()):
                        continue

                    with report_path.open(encoding="utf-8") as f:
                        rep_data = json.load(f)

                    selected_round = int(rep_data.get("selected_round", 5))
                    df = pd.read_csv(csv_path)
                    val_rows = df[
                        (df["round"] == selected_round) & (df["split"] == "validation")
                    ]
                    if val_rows.empty:
                        continue

                    val_pr_auc = float(val_rows.iloc[0]["pr_auc"])
                    exp_id = exp_dir.name

                    ckpt_path = (
                        self._checkpoints_root
                        / "federated"
                        / "non_iid"
                        / algo
                        / arch
                        / exp_id
                        / f"{CheckpointKind.GLOBAL_BEST.value}.pt"
                    )
                    if not ckpt_path.is_file():
                        logger.warning(
                            "Checkpoint missing for Non-IID experiment %s at %s",
                            exp_id,
                            ckpt_path,
                        )
                        continue

                    candidates.append(
                        ModelSelectionRecord(
                            distribution="non_iid",
                            algorithm=algo,
                            model=arch,
                            experiment_id=exp_id,
                            checkpoint_kind=CheckpointKind.GLOBAL_BEST.value,
                            checkpoint_path=str(ckpt_path),
                            selection_metric="pr_auc",
                            validation_score=val_pr_auc,
                        )
                    )

        return candidates

    def select_best(self, candidates: list[ModelSelectionRecord]) -> ModelSelectionRecord:
        """Deterministically select the model with the highest validation PR-AUC score.

        Args:
            candidates: List of eligible model candidates.

        Returns:
            The highest-scoring ModelSelectionRecord.
        """
        if not candidates:
            raise CheckpointNotFoundError("No eligible candidates found for model selection.")

        # Sort by validation score descending; tie-break deterministically by experiment_id
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.validation_score, c.experiment_id),
            reverse=True,
        )
        return sorted_candidates[0]

    def select_pair(
        self,
    ) -> tuple[
        ModelSelectionRecord,
        ModelSelectionRecord,
        list[ModelSelectionRecord],
        list[ModelSelectionRecord],
    ]:
        """Discover all candidates and select the optimal IID and Non-IID pair.

        Returns:
            Tuple of (best_iid, best_non_iid, all_iid_candidates, all_non_iid_candidates).
        """
        iid_candidates = self.scan_iid_candidates()
        non_iid_candidates = self.scan_non_iid_candidates()

        if not iid_candidates:
            raise CheckpointNotFoundError("No valid Phase 5 IID candidate checkpoints found.")
        if not non_iid_candidates:
            raise CheckpointNotFoundError("No valid Phase 6 Non-IID candidate checkpoints found.")

        best_iid = self.select_best(iid_candidates)
        best_non_iid = self.select_best(non_iid_candidates)

        logger.info(
            "Selected Best IID Model: %s (%s, %s, Val PR-AUC=%.6f)",
            best_iid.experiment_id,
            best_iid.algorithm.upper(),
            best_iid.model.upper(),
            best_iid.validation_score,
        )
        logger.info(
            "Selected Best Non-IID Model: %s (%s, %s, Val PR-AUC=%.6f)",
            best_non_iid.experiment_id,
            best_non_iid.algorithm.upper(),
            best_non_iid.model.upper(),
            best_non_iid.validation_score,
        )

        return best_iid, best_non_iid, iid_candidates, non_iid_candidates

