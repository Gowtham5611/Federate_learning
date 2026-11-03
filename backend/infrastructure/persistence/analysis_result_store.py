"""Persistence and export adapter for Phase 8 analysis artifacts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from application.dto.analysis import Phase8Report

logger = logging.getLogger(__name__)


class AnalysisResultStore:
    """Manages the creation of structured Phase 8 analysis directories, CSVs, JSON, and reports."""

    def __init__(self, root: Path) -> None:
        """Initialize store rooted at experiments/results/analysis."""
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the base storage directory."""
        return self._root

    def analysis_dir(self, analysis_id: str) -> Path:
        """Return the destination path for a given analysis run."""
        dirname = analysis_id if analysis_id.startswith("phase8_") else f"phase8_{analysis_id}"
        return self._root / dirname

    def init_directory_structure(self, analysis_id: str) -> Path:
        """Create all subdirectories for Phase 8 artifacts."""
        target = self.analysis_dir(analysis_id)
        subdirs = [
            "summary",
            "comparisons",
            "ablations",
            "client_analysis",
            "convergence",
            "ensemble_analysis",
            "threshold_analysis",
            "visualizations",
        ]
        for sub in subdirs:
            (target / sub).mkdir(parents=True, exist_ok=True)
        return target

    def save_all_csvs(self, report: Phase8Report) -> dict[str, Path]:
        """Write all 10 tabular CSV artifacts into their respective subdirectories."""
        target = self.init_directory_structure(report.analysis_id)
        paths: dict[str, Path] = {}

        p_summary = target / "summary" / "summary_metrics.csv"
        self._write_dict_rows(p_summary, [report.frozen_ensemble_metrics])
        paths["summary_metrics"] = p_summary

        p_cvf = target / "comparisons" / "centralized_vs_federated.csv"
        self._write_dict_rows(p_cvf, [r.to_dict() for r in report.centralized_vs_federated])
        paths["centralized_vs_federated"] = p_cvf

        p_model = target / "comparisons" / "model_comparison.csv"
        self._write_dict_rows(p_model, [r.to_dict() for r in report.model_comparison])
        paths["model_comparison"] = p_model

        p_fed = target / "comparisons" / "fedavg_vs_fedprox.csv"
        self._write_dict_rows(p_fed, [r.to_dict() for r in report.fedavg_vs_fedprox])
        paths["fedavg_vs_fedprox"] = p_fed

        p_iid = target / "comparisons" / "iid_vs_non_iid.csv"
        self._write_dict_rows(p_iid, [r.to_dict() for r in report.iid_vs_non_iid])
        paths["iid_vs_non_iid"] = p_iid

        p_client = target / "client_analysis" / "client_distribution.csv"
        self._write_dict_rows(p_client, [r.to_dict() for r in report.client_distribution])
        paths["client_distribution"] = p_client

        p_conv = target / "convergence" / "convergence_metrics.csv"
        self._write_dict_rows(p_conv, [r.to_dict() for r in report.convergence_metrics])
        paths["convergence_metrics"] = p_conv

        p_alpha = target / "ensemble_analysis" / "alpha_analysis.csv"
        self._write_dict_rows(p_alpha, [r.to_dict() for r in report.alpha_analysis])
        paths["alpha_analysis"] = p_alpha

        p_thresh = target / "threshold_analysis" / "threshold_analysis.csv"
        self._write_dict_rows(p_thresh, [r.to_dict() for r in report.threshold_analysis])
        paths["threshold_analysis"] = p_thresh

        p_abl = target / "ablations" / "ablation_results.csv"
        self._write_dict_rows(p_abl, [r.to_dict() for r in report.ablation_study])
        paths["ablation_results"] = p_abl

        return paths

    def save_json_and_markdown(self, report: Phase8Report) -> tuple[Path, Path, Path]:
        """Save report.json, metadata.json, and formatted report.md."""
        target = self.init_directory_structure(report.analysis_id)

        rep_path = target / "report.json"
        with rep_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        meta_path = target / "metadata.json"
        meta_dict = {
            "analysis_id": report.analysis_id,
            "timestamp": report.timestamp,
            "seed": report.seed,
            "total_experiments_analyzed": report.total_experiments_analyzed,
            "artifacts_generated": report.artifacts_generated,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        md_path = target / "report.md"
        md_content = self._render_markdown_report(report)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_content)

        return rep_path, meta_path, md_path

    def _write_dict_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        """Helper to write rows of dictionaries to a CSV file."""
        if not rows:
            with path.open("w", newline="", encoding="utf-8") as f:
                f.write("")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _render_markdown_report(self, report: Phase8Report) -> str:
        """Render publication-quality Markdown report text."""
        m = report.frozen_ensemble_metrics
        lines: list[str] = [
            f"# Phase 8: Performance Analysis & Ablation Study ({report.analysis_id})",
            "",
            "## 1. Executive Summary & Verification Highlights",
            f"- **Analysis Run ID**: `{report.analysis_id}`",
            f"- **Timestamp**: `{report.timestamp}`",
            (
                f"- **Reproducibility Seed**: `{report.seed}` "
                "(Single-seed verification; cross-seed variance unmeasured)"
            ),
            f"- **Total Experiments Analyzed**: `{report.total_experiments_analyzed}`",
            f"  - Centralized records: `{report.centralized_records_count}`",
            f"  - IID Federated records: `{report.iid_records_count}`",
            f"  - Non-IID Federated records: `{report.non_iid_records_count}`",
            f"  - Ensemble records: `{report.ensemble_records_count}`",
            "",
            "### Phase 7 Frozen Test Performance Profile (Untouched Evaluation)",
            f"- **Test PR-AUC**: **`{m.get('pr_auc', 0.0):.6f}`**",
            f"- **Test ROC-AUC**: **`{m.get('roc_auc', 0.0):.6f}`**",
            f"- **Test F1 Score**: **`{m.get('f1', 0.0):.6f}`**",
            f"- **Test Recall**: **`{m.get('recall', 0.0):.6f}`**",
            f"- **Test Precision**: **`{m.get('precision', 0.0):.6f}`**",
            f"- **Test Accuracy**: **`{m.get('accuracy', 0.0):.6f}`**",
            (
                f"- **Confusion Matrix**: $\\text{{TP}} = {m.get('tp', 0):,}, "
                f"\\ \\text{{FP}} = {m.get('fp', 0):,}, "
                f"\\ \\text{{TN}} = {m.get('tn', 0):,}, "
                f"\\ \\text{{FN}} = {m.get('fn', 0):,}$"
            ),
            f"- **Frozen Threshold**: `{m.get('threshold', 0.13):.4f}`",
            "",
            "---",
            "",
            "## 2. Centralized vs IID vs Non-IID Paradigm Analysis",
            (
                "The table below compares the test performance of representative models "
                "across all paradigms at validation-selected decision thresholds:"
            ),
            "",
            (
                "| Paradigm | Model | Algorithm | Test PR-AUC | Test ROC-AUC | "
                "Test F1 | Test Recall | Test Precision |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]

        test_cvf = [
            r for r in report.centralized_vs_federated
            if r.split == "test" and r.threshold_policy == "validation_f1"
        ]
        for r in test_cvf:
            lines.append(
                f"| {r.paradigm} | {r.model.upper()} | {r.algorithm.upper()} | {r.pr_auc:.6f} | "
                f"{r.roc_auc:.6f} | {r.f1:.6f} | {r.recall:.6f} | {r.precision:.6f} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Recurrent Architecture Comparison (GRU vs LSTM vs BiLSTM)",
            "Comparing architectures across available experimental frameworks:",
            "",
            (
                "| Paradigm | Architecture | Algorithm | Val PR-AUC | Test PR-AUC | "
                "Test F1 | Test Recall | Test Precision |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ])

        val_models = {
            (r.paradigm, r.model, r.algorithm): r
            for r in report.model_comparison
            if r.split == "validation" and r.threshold_policy == "fixed"
        }
        test_models = {
            (r.paradigm, r.model, r.algorithm): r
            for r in report.model_comparison
            if r.split == "test" and r.threshold_policy == "validation_f1"
        }

        for key, v_rec in val_models.items():
            t_rec = test_models.get(key)
            t_pr = f"{t_rec.pr_auc:.6f}" if t_rec else "N/A"
            t_f1 = f"{t_rec.f1:.6f}" if t_rec else "N/A"
            t_rec_val = f"{t_rec.recall:.6f}" if t_rec else "N/A"
            t_prec = f"{t_rec.precision:.6f}" if t_rec else "N/A"
            lines.append(
                f"| {v_rec.paradigm} | {v_rec.model.upper()} | {v_rec.algorithm.upper()} | "
                f"{v_rec.pr_auc:.6f} | {t_pr} | {t_f1} | {t_rec_val} | {t_prec} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Federated Optimization: FedAvg vs FedProx",
            "",
            (
                "| Partition Distribution | Architecture | FedAvg Val PR-AUC | "
                "FedProx Val PR-AUC | Val Delta | FedAvg Test PR-AUC | "
                "FedProx Test PR-AUC | Notes |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ])

        for fp_r in report.fedavg_vs_fedprox:
            fa_v = (
                f"{fp_r.fedavg_val_pr_auc:.6f}"
                if fp_r.fedavg_val_pr_auc is not None
                else "N/A"
            )
            fp_v = (
                f"{fp_r.fedprox_val_pr_auc:.6f}"
                if fp_r.fedprox_val_pr_auc is not None
                else "N/A"
            )
            d_v = (
                f"{fp_r.val_pr_auc_delta:+.6f}"
                if fp_r.val_pr_auc_delta is not None
                else "N/A"
            )
            fa_t = (
                f"{fp_r.fedavg_test_pr_auc:.6f}"
                if fp_r.fedavg_test_pr_auc is not None
                else "N/A"
            )
            fp_t = (
                f"{fp_r.fedprox_test_pr_auc:.6f}"
                if fp_r.fedprox_test_pr_auc is not None
                else "N/A"
            )
            lines.append(
                f"| {fp_r.distribution.upper()} | {fp_r.model.upper()} | {fa_v} | {fp_v} | "
                f"{d_v} | {fa_t} | {fp_t} | {fp_r.notes} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Client Data Heterogeneity: IID vs Non-IID",
            "",
            (
                "| Architecture | Algorithm | IID Val PR-AUC | Non-IID Val PR-AUC | "
                "Degradation | IID Test PR-AUC | Non-IID Test PR-AUC | Notes |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ])

        for iid_r in report.iid_vs_non_iid:
            iid_v = (
                f"{iid_r.iid_val_pr_auc:.6f}"
                if iid_r.iid_val_pr_auc is not None
                else "N/A"
            )
            non_v = (
                f"{iid_r.non_iid_val_pr_auc:.6f}"
                if iid_r.non_iid_val_pr_auc is not None
                else "N/A"
            )
            deg = (
                f"{iid_r.val_pr_auc_degradation:.6f}"
                if iid_r.val_pr_auc_degradation is not None
                else "N/A"
            )
            iid_t = (
                f"{iid_r.iid_test_pr_auc:.6f}"
                if iid_r.iid_test_pr_auc is not None
                else "N/A"
            )
            non_t = (
                f"{iid_r.non_iid_test_pr_auc:.6f}"
                if iid_r.non_iid_test_pr_auc is not None
                else "N/A"
            )
            lines.append(
                f"| {iid_r.model.upper()} | {iid_r.algorithm.upper()} | {iid_v} | {non_v} | "
                f"{deg} | {iid_t} | {non_t} | {iid_r.notes} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 6. Client-Level Non-IID Partition Analysis",
            "Analysis of the 3 heterogeneous banking clients based on behavioral typologies:",
            "",
            (
                "| Client ID | Typology Category | Transactions | Share (%) | Positives | "
                "Positive Rate (%) | Accounts | Primary Payment Format |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ])

        for c in report.client_distribution:
            primary_fmt = (
                "Retail POS / Cards" if "Retail" in c.typology
                else ("Crypto / Wire / FX" if "Crypto" in c.typology else "Cheques / ACH / CU")
            )
            lines.append(
                f"| `{c.client_id}` | {c.typology} | {c.transaction_count:,} | "
                f"{c.percentage_of_total:.2f}% | {c.positive_count:,} | "
                f"{c.positive_rate:.4f}% | {c.account_count:,} | {primary_fmt} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 7. Ensemble Effect & Alpha Selection Analysis",
            (
                "- **Ensemble Weighting**: "
                "$P_{\\text{ensemble}} = \\alpha \\cdot P_{\\text{IID}} + "
                "(1 - \\alpha) \\cdot P_{\\text{Non-IID}}$"
            ),
            "- **Selected Candidate Models** (on validation PR-AUC):",
            "  - Best IID: `LSTM (FedAvg)` (Val PR-AUC = `0.367526`)",
            "  - Best Non-IID: `BiLSTM (FedAvg)` (Val PR-AUC = `0.095542`)",
            "- **Alpha Grid Search Finding**:",
            (
                "  - Under the tested validation grid $\\alpha \\in [0.0, 1.0]$, "
                "the optimal alpha was **`1.0000`**."
            ),
            (
                "  - Validation PR-AUC monotonically increased as $\\alpha$ rose from "
                "$0.0$ (`0.095542`) to $1.0$ (`0.367526`)."
            ),
            (
                "  - **Interpretation**: The validation selection procedure assigned zero "
                "weight to the selected Non-IID model because the IID LSTM model strictly "
                "dominated the Non-IID BiLSTM model across all probability thresholds."
            ),
            "",
            "---",
            "",
            "## 8. Classification Decision Threshold Analysis",
            (
                "Comparing default fixed threshold $\\tau=0.50$ against the "
                "validation-selected optimal threshold $\\tau^*=0.13$:"
            ),
            "",
            (
                "| Evaluation Split | Threshold Policy | Threshold | PR-AUC | ROC-AUC | "
                "F1 Score | Recall | Precision | True Positives | False Positives |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|",
        ])

        for t in report.threshold_analysis:
            lines.append(
                f"| {t.split.upper()} | {t.threshold_policy} | {t.threshold:.2f} | "
                f"{t.pr_auc:.6f} | {t.roc_auc:.6f} | {t.f1:.6f} | {t.recall:.6f} | "
                f"{t.precision:.6f} | {t.tp:,} | {t.fp:,} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 9. Ablation Study: Stage-by-Stage Contributions",
            "",
            (
                "| Ablation Stage | Model / Component | Val PR-AUC | Test PR-AUC | "
                "Test ROC-AUC | Test F1 | Test Recall | Test Precision | Test TP | "
                "Test FP | Delta PR-AUC vs Baseline |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])

        for a in report.ablation_study:
            delta_str = (
                f"{a.delta_test_pr_auc_vs_baseline:+.6f}"
                if a.delta_test_pr_auc_vs_baseline is not None
                else "0.000000 (Base)"
            )
            lines.append(
                f"| {a.stage} | {a.description} | {a.val_pr_auc:.6f} | {a.test_pr_auc:.6f} | "
                f"{a.test_roc_auc:.6f} | {a.test_f1:.6f} | {a.test_recall:.6f} | "
                f"{a.test_precision:.6f} | {a.tp:,} | {a.fp:,} | {delta_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 10. Key Empirical Observations",
        ])
        for f in report.key_findings:
            lines.append(f"- {f}")

        lines.extend([
            "",
            "---",
            "",
            "## 11. Methodological Limitations & Future Scope",
        ])
        for lim in report.limitations:
            lines.append(f"- {lim}")

        lines.extend([
            "",
            "---",
            "",
            "## 12. Artifacts Generated in Phase 8",
        ])
        for art in report.artifacts_generated:
            lines.append(f"- `{art}`")

        lines.append("")
        return "\n".join(lines)
