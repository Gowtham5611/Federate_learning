# FedSecureAI — Unified Experiment Results Repository

This repository contains the verified, reproducible, and forensically audited experimental artifacts for the **FedSecureAI: Federated Learning for Anti-Money Laundering** research suite.

## Directory Structure

```text
results/
├── HI_Small_Results/          # Complete verified results for IBM AML HI-Small (5.08M rows)
│   ├── archives/              # Original ZIP archives of the experiments
│   ├── checkpoints/           # Model checkpoints (.pt) organized by configuration
│   ├── metrics/               # Evaluation metrics and CSV summaries
│   ├── logs/                  # Training and execution logs
│   ├── manifests/             # Run manifests and status tracking
│   ├── configuration_reports/ # Detailed per-model configuration sheets
│   ├── verification/          # Recalculated metrics on full test split
│   └── README.md
│
├── LI_Medium_Results/         # Complete verified results for IBM AML LI-Medium (31.25M rows)
│   ├── archives/              # Original Colab ZIP backup archives
│   ├── checkpoints/           # Checkpoints for Configs 1-12 (IID and Non-IID)
│   ├── metrics/               # Official metric JSON files
│   ├── logs/                  # Execution logs from Google Colab Tesla T4 runs
│   ├── manifests/             # Colab execution manifests
│   ├── configuration_reports/ # Detailed per-configuration reports
│   ├── verification/          # Recalculated metrics across all 6.25M test rows
│   └── README.md
│
├── comparison/                # Cross-dataset comparative benchmarks
│   ├── HI_Small_vs_LI_Medium.csv
│   ├── HI_Small_vs_LI_Medium.md
│   └── comparison_methodology.md
│
├── audit/                     # Verification and forensic audit outputs
│   ├── file_inventory.csv     # Complete inventory with SHA-256 hashes
│   ├── integrity_checksums.csv# Checksums of all preserved checkpoints and predictions
│   ├── verification_summary.json
│   ├── verification_report.md
│   ├── missing_artifacts.csv
│   └── inconsistent_metrics.csv
│
└── README.md
```
