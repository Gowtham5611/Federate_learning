"""Experiment result store: one JSON report per run, separated flat CSV results tables.

The report is the complete record of a single experiment — configuration, data
provenance, sequence strategy, training curve, every result row — and is written
once per run to its own file under ``reports/``.

The CSV tables are the cross-experiment tables, cleanly separated by category:
- ``centralized/centralized_results.csv`` for centralized Phase 4 experiments.
- ``federated/iid/fedavg/iid_fedavg_results.csv`` for IID FedAvg experiments.
- ``federated/iid/fedprox/iid_fedprox_results.csv`` for IID FedProx experiments.

Because the header is fixed at the first append, a run whose rows have a different
shape is refused with :class:`HeaderMismatchError`. Similarly, appending rows of
the wrong category to a category-specific store is refused with
:class:`CategoryMismatchError`.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from application.dto.experiment import ExperimentReport
from application.ports.output.experiment_store import ExperimentResultStorePort

logger = logging.getLogger(__name__)

REPORTS_DIRECTORY = "reports"
RESULTS_TABLE = "experiment_results.csv"
CENTRALIZED_RESULTS_TABLE = Path("centralized") / "centralized_results.csv"
IID_FEDAVG_RESULTS_TABLE = Path("federated") / "iid" / "fedavg" / "iid_fedavg_results.csv"
IID_FEDPROX_RESULTS_TABLE = Path("federated") / "iid" / "fedprox" / "iid_fedprox_results.csv"
NON_IID_FEDAVG_RESULTS_TABLE = (
    Path("federated") / "non_iid" / "fedavg" / "non_iid_fedavg_results.csv"
)
NON_IID_FEDPROX_RESULTS_TABLE = (
    Path("federated") / "non_iid" / "fedprox" / "non_iid_fedprox_results.csv"
)

JSON_COLUMNS: frozenset[str] = frozenset({"hyperparameters"})
"""Columns held as a JSON string inside one CSV cell.

The alternative — one column per hyperparameter — would make the header change
whenever a configuration field is added, which is exactly what the header check
below exists to prevent.
"""


class HeaderMismatchError(ValueError):
    """The results table on disk has a different header than the rows being added.

    Raised instead of appending, because a CSV with two different headers is a file
    where every column after the divergence means something other than what the
    header says. Recovering the earlier rows from such a file requires guessing.
    """

    def __init__(self, path: Path, existing: list[str], incoming: list[str]) -> None:
        """Record both headers and the file they disagree about."""
        self.path = path
        self.existing = existing
        self.incoming = incoming
        added = [name for name in incoming if name not in existing]
        removed = [name for name in existing if name not in incoming]
        super().__init__(
            f"Refusing to append to '{path}': the existing table has "
            f"{len(existing)} column(s) and these rows have {len(incoming)}. "
            f"New: {added or 'none'}. Missing: {removed or 'none'}. Move or rename "
            f"the existing table to start a new one; appending would misalign every "
            f"row that follows."
        )


class CategoryMismatchError(ValueError):
    """The result rows belong to a different experiment category than the target store."""


def resolve_result_table_path(root: Path, report: ExperimentReport) -> Path:
    """Determine the categorized results CSV path for an experiment report.

    Args:
        root: Base experiment results root directory.
        report: Experiment report to inspect.

    Returns:
        The target path for the experiment category.
    """
    if report.results:
        first = report.results[0]
        mode = str(getattr(first.training_mode, "value", first.training_mode)).lower()
        dist = str(first.data_distribution).lower()
        algo = str(first.federated_algorithm).lower()
    else:
        cfg = report.config
        algo = str(getattr(cfg, "algorithm", "none")).lower()
        fed_algos = {"fedavg", "fedprox", "scaffold", "fedprox_scaffold"}
        mode = "federated" if algo in fed_algos else "centralized"

    if mode == "centralized":
        return root / CENTRALIZED_RESULTS_TABLE

    if mode == "federated":
        dist_norm = dist.replace("-", "_")
        if dist_norm == "iid" and algo == "fedavg":
            return root / IID_FEDAVG_RESULTS_TABLE
        if dist_norm == "iid" and algo == "fedprox":
            return root / IID_FEDPROX_RESULTS_TABLE
        if dist_norm == "non_iid":
            arch = "unknown"
            if report.results:
                arch = str(
                    getattr(
                        report.results[0].architecture,
                        "value",
                        report.results[0].architecture,
                    )
                ).lower()
            elif hasattr(report, "config"):
                cfg = report.config
                t_cfg = getattr(cfg, "training_config", cfg)
                arch = str(
                    getattr(
                        getattr(t_cfg, "architecture", None),
                        "value",
                        getattr(t_cfg, "architecture", "unknown"),
                    )
                ).lower()
            return (
                root
                / "federated"
                / "non_iid"
                / algo
                / arch
                / report.experiment_id
                / "results.csv"
            )
        return root / "federated" / dist_norm / algo / f"{dist_norm}_{algo}_results.csv"

    return root / CENTRALIZED_RESULTS_TABLE


class CsvExperimentResultStore(ExperimentResultStorePort):
    """Writes JSON reports and appends to flat CSV results tables separated by category."""

    def __init__(
        self,
        root: Path,
        *,
        table_path: Path | str | None = None,
        category: str | None = None,
    ) -> None:
        """Prepare the results tree.

        Args:
            root: Directory to write ``reports/`` and the results tables under.
            table_path: Explicit table path (relative to root or absolute). When
                omitted, the store routes rows automatically based on report metadata.
            category: Expected category for validation (e.g. 'centralized',
                'iid_fedavg', 'iid_fedprox', 'non_iid_fedavg', 'non_iid_fedprox').
        """
        self._root = root.resolve()
        if table_path is not None:
            tp = Path(table_path)
            self._explicit_table_path: Path | None = (
                tp.resolve() if tp.is_absolute() else (self._root / tp).resolve()
            )
        else:
            self._explicit_table_path = None
        self._category = category
        self._last_table_path: Path | None = None

    @classmethod
    def for_centralized(cls, root: Path) -> CsvExperimentResultStore:
        """Create a result store dedicated to centralized experiments."""
        return cls(
            root,
            table_path=CENTRALIZED_RESULTS_TABLE,
            category="centralized",
        )

    @classmethod
    def for_federated(
        cls,
        root: Path,
        *,
        distribution: str = "iid",
        algorithm: str = "fedavg",
    ) -> CsvExperimentResultStore:
        """Create a result store dedicated to a specific federated configuration."""
        dist = distribution.lower().replace("-", "_")
        algo = algorithm.lower()
        if dist == "iid" and algo == "fedavg":
            rel_path = IID_FEDAVG_RESULTS_TABLE
            return cls(
                root,
                table_path=rel_path,
                category="iid_fedavg",
            )
        if dist == "iid" and algo == "fedprox":
            rel_path = IID_FEDPROX_RESULTS_TABLE
            return cls(
                root,
                table_path=rel_path,
                category="iid_fedprox",
            )
        if dist == "non_iid":
            return cls(
                root,
                table_path=None,
                category=f"non_iid_{algo}",
            )
        rel_path = Path("federated") / dist / algo / f"{dist}_{algo}_results.csv"
        return cls(
            root,
            table_path=rel_path,
            category=f"{dist}_{algo}",
        )

    @property
    def root(self) -> Path:
        """Directory results are written under."""
        return self._root

    @property
    def table_path(self) -> Path:
        """The cross-experiment results table."""
        if self._explicit_table_path is not None:
            return self._explicit_table_path
        if self._last_table_path is not None:
            return self._last_table_path
        if self._category == "non_iid_fedavg":
            return self._root / "federated" / "non_iid" / "fedavg"
        if self._category == "non_iid_fedprox":
            return self._root / "federated" / "non_iid" / "fedprox"
        return self._root / RESULTS_TABLE

    def report_path(self, experiment_id: str) -> Path:
        """Where one experiment's full report lives."""
        return self._root / REPORTS_DIRECTORY / f"{experiment_id}.json"

    def save_report(self, report: ExperimentReport) -> Path:
        """Write one experiment's full report as JSON."""
        dist = ""
        algo = ""
        arch = ""
        if hasattr(report, "results") and report.results:
            first = report.results[0]
            dist = str(getattr(first, "data_distribution", "")).lower().replace("-", "_")
            algo = str(getattr(first, "federated_algorithm", "")).lower()
            arch = str(
                getattr(first.architecture, "value", first.architecture)
            ).lower()
        elif hasattr(report, "config"):
            cfg = report.config
            dist = str(getattr(cfg, "distribution", "")).lower().replace("-", "_")
            algo = str(getattr(cfg, "algorithm", "")).lower()
            t_cfg = getattr(cfg, "training_config", cfg)
            arch = str(
                getattr(
                    getattr(t_cfg, "architecture", None),
                    "value",
                    getattr(t_cfg, "architecture", "unknown"),
                )
            ).lower()

        content = (
            json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        )

        if dist == "non_iid" and algo in {"fedavg", "fedprox", "scaffold", "fedprox_scaffold"}:
            exp_dir = (
                self._root
                / "federated"
                / "non_iid"
                / algo
                / arch
                / report.experiment_id
            )
            exp_dir.mkdir(parents=True, exist_ok=True)
            report_file = exp_dir / "report.json"
            meta_file = exp_dir / "metadata.json"
            report_file.write_text(content, encoding="utf-8")
            meta_file.write_text(content, encoding="utf-8")
            logger.info("Wrote Non-IID experiment report to %s", report_file)
            global_path = self.report_path(report.experiment_id)
            global_path.parent.mkdir(parents=True, exist_ok=True)
            global_path.write_text(content, encoding="utf-8")
            return report_file

        path = self.report_path(report.experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote the experiment report for '%s' to %s", report.experiment_id, path)
        return path

    def append_rows(self, report: ExperimentReport) -> Path:
        """Append the report's result rows to the cross-experiment table.

        Raises:
            CategoryMismatchError: The report does not match the store's category.
            HeaderMismatchError: The table on disk has a different header.
        """
        rows = [_flatten(result.to_dict()) for result in report.results]
        if self._explicit_table_path is not None:
            self._validate_category_compatibility(report)
            path = self._explicit_table_path
        else:
            path = resolve_result_table_path(self._root, report)

        path.parent.mkdir(parents=True, exist_ok=True)
        self._last_table_path = path

        if not rows:
            return path

        header = list(rows[0])
        existing = _read_header(path)
        if existing is not None and existing != header:
            raise HeaderMismatchError(path, existing, header)

        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if existing is None:
                writer.writeheader()
            writer.writerows(rows)

        logger.info("Appended %d result row(s) to %s", len(rows), path)
        return path

    def _validate_category_compatibility(self, report: ExperimentReport) -> None:
        """Verify that incoming report matches the store's category constraint."""
        if not self._category or not report.results:
            return

        first = report.results[0]
        mode = str(getattr(first.training_mode, "value", first.training_mode)).lower()
        dist = str(first.data_distribution).lower().replace("-", "_")
        algo = str(first.federated_algorithm).lower()

        if self._category == "centralized" and mode != "centralized":
            raise CategoryMismatchError(
                f"Refusing to append federated results ({report.experiment_id}) "
                f"to centralized results table '{self._explicit_table_path}'."
            )
        if self._category == "iid_fedavg" and (
            mode != "federated" or dist != "iid" or algo != "fedavg"
        ):
            raise CategoryMismatchError(
                f"Refusing to append non-fedavg results ({report.experiment_id}: "
                f"mode={mode}, dist={dist}, algo={algo}) to IID FedAvg table "
                f"'{self._explicit_table_path}'."
            )
        if self._category == "iid_fedprox" and (
            mode != "federated" or dist != "iid" or algo != "fedprox"
        ):
            raise CategoryMismatchError(
                f"Refusing to append non-fedprox results ({report.experiment_id}: "
                f"mode={mode}, dist={dist}, algo={algo}) to IID FedProx table "
                f"'{self._explicit_table_path}'."
            )
        if self._category == "non_iid_fedavg" and (
            mode != "federated" or dist != "non_iid" or algo != "fedavg"
        ):
            raise CategoryMismatchError(
                f"Refusing to append mismatched results ({report.experiment_id}: "
                f"mode={mode}, dist={dist}, algo={algo}) to Non-IID FedAvg table "
                f"'{self._explicit_table_path}'."
            )
        if self._category == "non_iid_fedprox" and (
            mode != "federated" or dist != "non_iid" or algo != "fedprox"
        ):
            raise CategoryMismatchError(
                f"Refusing to append mismatched results ({report.experiment_id}: "
                f"mode={mode}, dist={dist}, algo={algo}) to Non-IID FedProx table "
                f"'{self._explicit_table_path}'."
            )

    def load_rows(self, *, category: str | None = None) -> list[dict[str, Any]]:
        """Read the results table back, oldest row first.

        Args:
            category: Optional category to load ('centralized', 'iid_fedavg',
                'iid_fedprox', 'non_iid_fedavg', 'non_iid_fedprox'). If omitted,
                loads from the store's current table.
        """
        if category == "centralized":
            path = self._root / CENTRALIZED_RESULTS_TABLE
        elif category in {"iid_fedavg", "iid/fedavg"}:
            path = self._root / IID_FEDAVG_RESULTS_TABLE
        elif category in {"iid_fedprox", "iid/fedprox"}:
            path = self._root / IID_FEDPROX_RESULTS_TABLE
        elif category in {"non_iid_fedavg", "non_iid/fedavg"}:
            base = self._root / "federated" / "non_iid" / "fedavg"
            all_rows: list[dict[str, Any]] = []
            if base.is_dir():
                for p in sorted(base.glob("*/*/results.csv")):
                    with p.open("r", encoding="utf-8", newline="") as handle:
                        all_rows.extend([_unflatten(row) for row in csv.DictReader(handle)])
            return all_rows
        elif category in {"non_iid_fedprox", "non_iid/fedprox"}:
            base = self._root / "federated" / "non_iid" / "fedprox"
            all_rows_prox: list[dict[str, Any]] = []
            if base.is_dir():
                for p in sorted(base.glob("*/*/results.csv")):
                    with p.open("r", encoding="utf-8", newline="") as handle:
                        all_rows_prox.extend([_unflatten(row) for row in csv.DictReader(handle)])
            return all_rows_prox
        elif category is not None:
            path = self._root / category
        else:
            path = self.table_path

        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [_unflatten(row) for row in csv.DictReader(handle)]


def _read_header(path: Path) -> list[str] | None:
    """Return the existing table's header, or ``None`` if there is no table yet."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            return row
    return None


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    """Encode the nested cells so the row fits one CSV line."""
    return {
        name: json.dumps(value, sort_keys=True) if name in JSON_COLUMNS else value
        for name, value in row.items()
    }


def _unflatten(row: dict[str, Any]) -> dict[str, Any]:
    """Decode the JSON-encoded cells of one read row."""
    decoded = dict(row)
    for name in JSON_COLUMNS & decoded.keys():
        raw = decoded[name]
        try:
            decoded[name] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.warning(
                "Column '%s' did not hold valid JSON; returning it as text.", name
            )
    return decoded
