"""Persistence store for Phase 9 federated optimization artifacts.

Manages clean directory hierarchy under:
    experiments/results/phase9_federated_optimization/
    ├── fedprox/
    │   ├── mu_ablation/
    │   ├── best_model/
    │   └── reports/
    ├── scaffold/
    │   ├── gru/
    │   ├── lstm/
    │   ├── bilstm/
    │   └── reports/
    ├── comparisons/
    ├── convergence/
    ├── client_drift/
    ├── visualizations/
    ├── metadata.json
    └── report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class Phase9ResultStore:
    """Manages Phase 9 CSV tables, JSON reports, metadata, and Markdown summary."""

    def __init__(
        self,
        root: Path | str = "experiments/results/phase9_federated_optimization",
    ) -> None:
        """Initialize store and paths."""
        self._root = Path(root).resolve()
        self._fedprox_dir = self._root / "fedprox"
        self._mu_ablation_dir = self._fedprox_dir / "mu_ablation"
        self._best_model_dir = self._fedprox_dir / "best_model"
        self._fedprox_reports_dir = self._fedprox_dir / "reports"

        self._scaffold_dir = self._root / "scaffold"
        self._scaffold_reports_dir = self._scaffold_dir / "reports"

        self._comparisons_dir = self._root / "comparisons"
        self._convergence_dir = self._root / "convergence"
        self._drift_dir = self._root / "client_drift"
        self._viz_dir = self._root / "visualizations"

        self.ensure_directories()

    @property
    def root(self) -> Path:
        """Root directory."""
        return self._root

    @property
    def visualizations_dir(self) -> Path:
        """Visualizations directory."""
        return self._viz_dir

    def ensure_directories(self) -> None:
        """Create directory hierarchy."""
        for p in (
            self._root,
            self._fedprox_dir,
            self._mu_ablation_dir,
            self._best_model_dir,
            self._fedprox_reports_dir,
            self._scaffold_dir,
            self._scaffold_dir / "gru",
            self._scaffold_dir / "lstm",
            self._scaffold_dir / "bilstm",
            self._scaffold_reports_dir,
            self._comparisons_dir,
            self._convergence_dir,
            self._drift_dir,
            self._viz_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, target_path: Path) -> Path:
        """Save DataFrame to CSV."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        logger.info("Saved CSV table (%d rows) to %s", len(df), target_path)
        return target_path

    def save_mu_ablation_table(self, df: pd.DataFrame) -> Path:
        """Save FedProx mu ablation table."""
        return self.save_dataframe(df, self._mu_ablation_dir / "fedprox_mu_ablation.csv")

    def save_best_fedprox_summary(self, df: pd.DataFrame) -> Path:
        """Save best FedProx model summary."""
        return self.save_dataframe(df, self._best_model_dir / "best_fedprox_summary.csv")

    def save_scaffold_results(self, df: pd.DataFrame) -> Path:
        """Save SCAFFOLD results table."""
        return self.save_dataframe(df, self._scaffold_reports_dir / "scaffold_results.csv")

    def save_three_way_comparison(self, df: pd.DataFrame) -> Path:
        """Save FedAvg vs FedProx vs SCAFFOLD three-way comparison table."""
        return self.save_dataframe(df, self._comparisons_dir / "three_way_comparison.csv")

    def save_fedavg_vs_fedprox(self, df: pd.DataFrame) -> Path:
        """Save FedAvg vs Best FedProx comparison table."""
        return self.save_dataframe(df, self._comparisons_dir / "fedavg_vs_fedprox.csv")

    def save_convergence_trajectory(self, df: pd.DataFrame) -> Path:
        """Save convergence trajectory across rounds."""
        return self.save_dataframe(df, self._convergence_dir / "convergence_trajectory.csv")

    def save_client_drift_analysis(self, df: pd.DataFrame) -> Path:
        """Save client drift analysis table."""
        return self.save_dataframe(df, self._drift_dir / "client_drift_analysis.csv")

    def save_json(self, data: dict[str, Any], path: Path) -> Path:
        """Save JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved JSON file to %s", path)
        return path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        """Save root metadata JSON."""
        return self.save_json(metadata, self._root / "metadata.json")

    def save_report_markdown(self, markdown: str) -> Path:
        """Save Phase 9 comprehensive report Markdown."""
        path = self._root / "report.md"
        path.write_text(markdown, encoding="utf-8")
        logger.info("Saved Phase 9 report Markdown to %s", path)
        return path
