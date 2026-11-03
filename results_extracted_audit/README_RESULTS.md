# FedSecureAI — Comprehensive Colab Results Extraction & Forensic Audit

This directory contains the consolidated, audited, and independently verified experimental artifacts extracted from both primary Google Colab results directories in the workspace:

## Source Folders Audited
1. **Source Folder 1:** `Colab Results/` (`c:\Users\Acer\OneDrive\Desktop\Federate_learning\Colab Results`)
   - **Scope:** Completed full-scale 31.25M-row LI-Medium experiments (Configs 1–12).
   - **Key Contents:** Checkpoints for Configs 1–12, test prediction arrays (`test_predictions.npz`, 6.25M elements each), execution logs, manifests, and backup zip `li_medium_backup_20261020_125045.zip`.
2. **Source Folder 2:** `colab work and results/` (`c:\Users\Acer\OneDrive\Desktop\Federate_learning\colab work and results`)
   - **Scope:** Complete 5.08M-row HI-Small experiments (Configs 1–12, both Non-IID and IID on GPU), client local histories, calibration runs, and earlier recovery backups.
   - **Key Contents:** 84 checkpoints, 12 prediction arrays (`.npy`), client training histories, and calibration curves.

---

## Audit Verdict Summary
- **HI-Small Suite:** **PASS** (12/12 configurations complete and verified)
- **LI-Medium Suite:** **PASS** (12/12 configurations complete and verified)
- **Overall Project Verdict:** **PASS WITH WARNINGS** (HI-Small and LI-Medium are ready for ensemble modeling; full Combined 36.33M dataset training remains pending).

---

## Directory Inventory of Generated Artifacts
- [`results_dashboard.html`](results_dashboard.html): Self-contained interactive dashboard for VS Code / Browser.
- [`all_experiment_results.csv`](all_experiment_results.csv): Master table of all experiment runs with full hyperparameters and metrics.
- [`HI_Small_results.csv`](HI_Small_results.csv): Dedicated table for all 12 HI-Small configurations.
- [`LI_Medium_results.csv`](LI_Medium_results.csv): Dedicated table for all 12 LI-Medium configurations (including full and historical runs).
- [`dataset_comparison.csv`](dataset_comparison.csv): Direct side-by-side comparison across matching configurations.
- [`client_level_results.csv`](client_level_results.csv): Federated client training metrics and local class distributions.
- [`round_level_results.csv`](round_level_results.csv): Per-round validation metrics, loss curves, and durations.
- [`calibration_results.csv`](calibration_results.csv): Probability calibration metrics (Brier score, ECE, MCE, log loss).
- [`prediction_audit.csv`](prediction_audit.csv): Dimensional and statistical audit of all prediction arrays.
- [`discrepancies.csv`](discrepancies.csv): Documented discrepancies, root causes, and resolutions.
- [`configuration_coverage.csv`](configuration_coverage.csv): 12-configuration coverage status matrix.
- [`file_inventory.csv`](file_inventory.csv): Full workspace inventory of all result files.
- [`verification_report.md`](verification_report.md): In-depth forensic audit report.
- [`extract_results.py`](extract_results.py): Reproducible extraction and auditing script.
- [`run_extraction.ps1`](run_extraction.ps1): PowerShell automation runner.
