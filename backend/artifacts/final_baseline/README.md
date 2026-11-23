# FedSecureAI — Baseline Federated AML Campaign & Client-Level Audit

## Directory Overview: `artifacts/final_baseline/`

This dedicated results directory houses the complete baseline experiments, client-level audit records, global metrics, comparison tables, checkpoints, confusion matrices, and the Ensemble Energy Readiness Gate assessment for FedSecureAI on the IBM AML HI-Small transaction dataset (~5 million transactions, ~3.05M sequences).

### Master Files:
- `README.md`: Index and inventory guide.
- `experiment_manifest.json`: JSON manifest indexing all 12 baseline configurations and statuses.
- `dataset_audit.json`: Complete dataset inventory (Train: 3,047,216, Val: 1,015,565, Test: 1,015,564; 23 features; 1:1325 positive class ratio).
- `configuration_inventory.csv`: Parametric specification of all 12 experimental setups (IID/Non-IID x GRU/LSTM/BiLSTM x FedAvg/FedProx mu=0.1).
- `all_12_experiments.csv`: Master comparison table across all 12 configurations.
- `experiment_status.csv`: Detailed execution tracking table.

### Client-Level Evidence (3 Clients per experiment):
- `client_data_profiles.csv`: Exact client sample counts, positive prevalence, account counts, and typology mappings.
- `client_local_training_history.csv`: Per-client per-round training loss, gradient/update norms, proximal penalties, and durations.
- `client_local_evaluation.csv`: Client-specific validation PR-AUC, ROC-AUC, F1, precision, and recall.
- `client_update_statistics.csv`: Local weight drift ($||w_k - w_{global}||_2$) and update norms.
- `aggregation_history.csv`: Per-round sample-weighted aggregation details ($p_k = n_k / N$).

### Global & Comparison Tables:
- `global_round_metrics.csv`: Round-by-round global model validation metrics.
- `global_final_test_metrics.csv`: Single-pass test evaluations at both validation-frozen thresholds and fixed 0.5 threshold.
- `iid_comparison.csv`: IID configuration benchmark.
- `non_iid_comparison.csv`: Typology Non-IID configuration benchmark.
- `fedavg_vs_fedprox.csv`: Pivot table comparing FedAvg vs. FedProx ($\mu = 0.1$).
- `gru_lstm_bilstm_comparison.csv`: Architectural comparison table.
- `client_comparison.csv`: Cross-client comparative evaluation.
- `final_report.md`: Complete executive synthesis, theoretical failure analysis of FedProx $\mu=0.1$, and Ensemble Energy Readiness decision.

### Subdirectories:
- `checkpoints/`: Saved PyTorch model checkpoints (`global_best.pt`).
- `confusion_matrices/`: Confusion matrix JSONs with true positives, false positives, false negatives, true negatives.
- `plots/`: Comparison charts (`pr_auc_comparison.png`).
- `predictions/`: Sample test prediction CSVs with true labels and predicted probabilities.
- `logs/`: Execution logs.
