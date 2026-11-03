# Phase 8: Performance Analysis & Ablation Study (analysis_seed42_20261003T165402)

## 1. Executive Summary & Verification Highlights
- **Analysis Run ID**: `analysis_seed42_20261003T165402`
- **Timestamp**: `20261003T165402`
- **Reproducibility Seed**: `42` (Single-seed verification; cross-seed variance unmeasured)
- **Total Experiments Analyzed**: `18`
  - Centralized records: `24`
  - IID Federated records: `34`
  - Non-IID Federated records: `48`
  - Ensemble records: `4`

### Phase 7 Frozen Test Performance Profile (Untouched Evaluation)
- **Test PR-AUC**: **`0.543749`**
- **Test ROC-AUC**: **`0.976867`**
- **Test F1 Score**: **`0.571813`**
- **Test Recall**: **`0.473011`**
- **Test Precision**: **`0.722789`**
- **Test Accuracy**: **`0.998747`**
- **Confusion Matrix**: $\text{TP} = 850, \ \text{FP} = 326, \ \text{TN} = 1,013,441, \ \text{FN} = 947$
- **Frozen Threshold**: `0.1300`

---

## 2. Centralized vs IID vs Non-IID Paradigm Analysis
The table below compares the test performance of representative models across all paradigms at validation-selected decision thresholds:

| Paradigm | Model | Algorithm | Test PR-AUC | Test ROC-AUC | Test F1 | Test Recall | Test Precision |
|---|---|---|---|---|---|---|---|
| Centralized | GRU | CENTRALIZED | 0.288690 | 0.932799 | 0.355507 | 0.272120 | 0.512579 |
| Centralized | LSTM | CENTRALIZED | 0.477052 | 0.935404 | 0.552941 | 0.444630 | 0.731016 |
| Centralized | BILSTM | CENTRALIZED | 0.469810 | 0.924533 | 0.553605 | 0.412354 | 0.842045 |
| IID Federated | GRU | FEDAVG | 0.541314 | 0.978003 | 0.569111 | 0.449082 | 0.776708 |
| IID Federated | LSTM | FEDAVG | 0.543749 | 0.976867 | 0.572194 | 0.475237 | 0.718855 |
| IID Federated | GRU | FEDPROX | 0.002664 | 0.660889 | 0.004696 | 0.294380 | 0.002367 |
| Non-IID Federated | BILSTM | FEDAVG | 0.243366 | 0.930312 | 0.269776 | 0.164162 | 0.756410 |
| Non-IID Federated | GRU | FEDAVG | 0.130297 | 0.932816 | 0.154586 | 0.125209 | 0.201975 |
| Non-IID Federated | LSTM | FEDAVG | 0.216237 | 0.943128 | 0.257143 | 0.175292 | 0.482389 |
| Non-IID Federated | BILSTM | FEDPROX | 0.002394 | 0.598299 | 0.004548 | 0.299944 | 0.002292 |
| Non-IID Federated | GRU | FEDPROX | 0.002686 | 0.648656 | 0.006667 | 0.089594 | 0.003463 |
| Non-IID Federated | LSTM | FEDPROX | 0.002233 | 0.562063 | 0.004055 | 0.289928 | 0.002042 |
| Ensemble | ENSEMBLE_LSTM_BILSTM | FEDAVG+FEDAVG | 0.543749 | 0.976867 | 0.571813 | 0.473011 | 0.722789 |

---

## 3. Recurrent Architecture Comparison (GRU vs LSTM vs BiLSTM)
Comparing architectures across available experimental frameworks:

| Paradigm | Architecture | Algorithm | Val PR-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision |
|---|---|---|---|---|---|---|---|
| Centralized | GRU | CENTRALIZED | 0.162334 | 0.288690 | 0.355507 | 0.272120 | 0.512579 |
| Centralized | LSTM | CENTRALIZED | 0.333317 | 0.477052 | 0.552941 | 0.444630 | 0.731016 |
| Centralized | BILSTM | CENTRALIZED | 0.327268 | 0.469810 | 0.553605 | 0.412354 | 0.842045 |
| IID Federated | GRU | FEDAVG | 0.353277 | 0.541314 | 0.569111 | 0.449082 | 0.776708 |
| IID Federated | LSTM | FEDAVG | 0.367526 | 0.543749 | 0.572194 | 0.475237 | 0.718855 |
| IID Federated | GRU | FEDPROX | 0.001825 | 0.002664 | 0.004696 | 0.294380 | 0.002367 |
| Non-IID Federated | BILSTM | FEDAVG | 0.095542 | 0.243366 | 0.269776 | 0.164162 | 0.756410 |
| Non-IID Federated | GRU | FEDAVG | 0.036931 | 0.130297 | 0.154586 | 0.125209 | 0.201975 |
| Non-IID Federated | LSTM | FEDAVG | 0.065717 | 0.216237 | 0.257143 | 0.175292 | 0.482389 |
| Non-IID Federated | BILSTM | FEDPROX | 0.001726 | 0.002394 | 0.004548 | 0.299944 | 0.002292 |
| Non-IID Federated | GRU | FEDPROX | 0.001999 | 0.002686 | 0.006667 | 0.089594 | 0.003463 |
| Non-IID Federated | LSTM | FEDPROX | 0.001560 | 0.002233 | 0.004055 | 0.289928 | 0.002042 |

---

## 4. Federated Optimization: FedAvg vs FedProx

| Partition Distribution | Architecture | FedAvg Val PR-AUC | FedProx Val PR-AUC | Val Delta | FedAvg Test PR-AUC | FedProx Test PR-AUC | Notes |
|---|---|---|---|---|---|---|---|
| IID | GRU | 0.353277 | 0.001825 | -0.351452 | 0.541314 | 0.002664 | FedProx showed lower validation and test PR-AUC under IID GRU with mu=0.01 |
| IID | LSTM | 0.367526 | N/A | N/A | 0.543749 | N/A | IID FedProx LSTM artifact not present in historical repository |
| NON_IID | GRU | 0.036931 | 0.001999 | -0.034932 | 0.130297 | 0.002686 | FedProx (mu=0.01) underperformed FedAvg on Non-IID GRU |
| NON_IID | LSTM | 0.065717 | 0.001560 | -0.064158 | 0.216237 | 0.002233 | FedProx (mu=0.01) underperformed FedAvg on Non-IID LSTM |
| NON_IID | BILSTM | 0.095542 | 0.001726 | -0.093816 | 0.243366 | 0.002394 | FedProx (mu=0.01) underperformed FedAvg on Non-IID BILSTM |

---

## 5. Client Data Heterogeneity: IID vs Non-IID

| Architecture | Algorithm | IID Val PR-AUC | Non-IID Val PR-AUC | Degradation | IID Test PR-AUC | Non-IID Test PR-AUC | Notes |
|---|---|---|---|---|---|---|---|
| GRU | FEDAVG | 0.353277 | 0.036931 | -0.316346 | 0.541314 | 0.130297 | Typology client partitioning caused large degradation in GRU PR-AUC |
| LSTM | FEDAVG | 0.367526 | 0.065717 | -0.301808 | 0.543749 | 0.216237 | LSTM suffered -82.1% relative drop in validation PR-AUC under Non-IID partition |
| BILSTM | FEDAVG | N/A | 0.095542 | N/A | N/A | 0.243366 | IID BiLSTM full run was not executed; BiLSTM was top Non-IID candidate |
| GRU | FEDPROX | 0.001825 | 0.001999 | 0.000174 | 0.002664 | 0.002686 | FedProx GRU underperformed FedAvg in both IID and Non-IID environments |

---

## 6. Client-Level Non-IID Partition Analysis
Analysis of the 3 heterogeneous banking clients based on behavioral typologies:

| Client ID | Typology Category | Transactions | Share (%) | Positives | Positive Rate (%) | Accounts | Primary Payment Format |
|---|---|---|---|---|---|---|---|
| `client_1` | Retail Giant | 539,973 | 17.72% | 359 | 0.0665% | 44,345 | Retail POS / Cards |
| `client_2` | Offshore / Crypto Exchange | 981,684 | 32.22% | 965 | 0.0983% | 49,299 | Crypto / Wire / FX |
| `client_3` | Regional Credit Union | 1,525,559 | 50.06% | 975 | 0.0639% | 399,156 | Cheques / ACH / CU |

---

## 7. Ensemble Effect & Alpha Selection Analysis
- **Ensemble Weighting**: $P_{\text{ensemble}} = \alpha \cdot P_{\text{IID}} + (1 - \alpha) \cdot P_{\text{Non-IID}}$
- **Selected Candidate Models** (on validation PR-AUC):
  - Best IID: `LSTM (FedAvg)` (Val PR-AUC = `0.367526`)
  - Best Non-IID: `BiLSTM (FedAvg)` (Val PR-AUC = `0.095542`)
- **Alpha Grid Search Finding**:
  - Under the tested validation grid $\alpha \in [0.0, 1.0]$, the optimal alpha was **`1.0000`**.
  - Validation PR-AUC monotonically increased as $\alpha$ rose from $0.0$ (`0.095542`) to $1.0$ (`0.367526`).
  - **Interpretation**: The validation selection procedure assigned zero weight to the selected Non-IID model because the IID LSTM model strictly dominated the Non-IID BiLSTM model across all probability thresholds.

---

## 8. Classification Decision Threshold Analysis
Comparing default fixed threshold $\tau=0.50$ against the validation-selected optimal threshold $\tau^*=0.13$:

| Evaluation Split | Threshold Policy | Threshold | PR-AUC | ROC-AUC | F1 Score | Recall | Precision | True Positives | False Positives |
|---|---|---|---|---|---|---|---|---|---|
| VALIDATION | fixed | 0.50 | 0.367526 | 0.970486 | 0.382829 | 0.251619 | 0.800000 | 272 | 68 |
| VALIDATION | validation_f1 | 0.13 | 0.367526 | 0.970486 | 0.455026 | 0.358002 | 0.624194 | 387 | 233 |
| TEST | fixed | 0.50 | 0.543749 | 0.976867 | 0.509726 | 0.357262 | 0.889197 | 642 | 80 |
| TEST | validation_f1 | 0.13 | 0.543749 | 0.976867 | 0.571813 | 0.473011 | 0.722789 | 850 | 326 |

---

## 9. Ablation Study: Stage-by-Stage Contributions

| Ablation Stage | Model / Component | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Test Recall | Test Precision | Test TP | Test FP | Delta PR-AUC vs Baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage A | Centralized Baseline (LSTM, no data privacy partitioning) | 0.333317 | 0.477052 | 0.935404 | 0.552941 | 0.444630 | 0.731016 | 799 | 294 | +0.000000 |
| Stage B | IID Federated Learning (FedAvg LSTM across 3 homogeneous clients) | 0.367526 | 0.543749 | 0.976867 | 0.572194 | 0.475237 | 0.718855 | 854 | 334 | +0.066696 |
| Stage C | Non-IID Federated Learning (FedAvg BiLSTM across 3 heterogeneous clients) | 0.095542 | 0.243366 | 0.930312 | 0.269776 | 0.164162 | 0.756410 | 295 | 95 | -0.233686 |
| Stage D | Best IID Model Evaluated Alone (LSTM + FedAvg) | 0.367526 | 0.543749 | 0.976867 | 0.572194 | 0.475237 | 0.718855 | 854 | 334 | +0.066696 |
| Stage E | Best Non-IID Model Evaluated Alone (BiLSTM + FedAvg) | 0.095542 | 0.243366 | 0.930312 | 0.269776 | 0.164162 | 0.756410 | 295 | 95 | -0.233686 |
| Stage F | Final Ensemble (P_ens = 1.0 * P_IID + 0.0 * P_NonIID at tau=0.13) | 0.367526 | 0.543749 | 0.976867 | 0.571813 | 0.473011 | 0.722789 | 850 | 326 | +0.066696 |

---

## 10. Key Empirical Observations
- Data Heterogeneity Bottleneck: Transitioning from IID to Non-IID partitions caused substantial performance degradation across all architectures (e.g. FedAvg LSTM Val PR-AUC dropped from 0.3675 to 0.0657, an -82.1% relative drop).
- FedProx Behavior: Under both IID and Non-IID distributions with mu=0.01, FedProx did not improve convergence or accuracy over sample-weighted FedAvg, and produced lower validation PR-AUC across all tested models.
- Recurrent Architecture Hierarchy: In centralized training, LSTM (Test PR-AUC 0.4771) and BiLSTM (0.4698) substantially outperformed GRU (0.2887). In Non-IID federated training, BiLSTM achieved the highest validation PR-AUC (0.0955) among Non-IID candidates.
- Ensemble Optimization Outcome: In the linear grid search P_ens = alpha * P_IID + (1 - alpha) * P_NonIID, validation PR-AUC monotonically increased from alpha=0.0 (0.0955) to alpha=1.0 (0.3675). The validation-driven procedure selected alpha=1.0, assigning zero weight to the Non-IID candidate because the IID model strictly dominated.
- Decision Threshold Impact: Calibrating the decision threshold to validation F1 (tau* = 0.13) improved test Recall from 0.3573 (at tau=0.50) to 0.4730 and test F1 from 0.5097 to 0.5718, successfully capturing 850 of 1,797 laundering sequences with 72.3% precision.

---

## 11. Methodological Limitations & Future Scope
- Single-Seed Verification: All models were evaluated under seed=42; variance across random weight initializations and client subsamplings was not estimated.
- Absence of IID BiLSTM Full Run: An IID BiLSTM 5-round run was not present in the historical repository, preventing direct BiLSTM IID-vs-Non-IID comparison.
- Static Proximal Coefficient: FedProx was evaluated only at fixed mu=0.01; potential sensitivity across alternative values was outside project scope.
- Typology Granularity: Non-IID partitions were assigned based on payment format categories across 3 clients; real-world bank networks exhibit larger scale and dynamic temporal shifts.

---

## 12. Artifacts Generated in Phase 8
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\01_centralized_vs_federated_prauc.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\02_model_architecture_comparison.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\03_fedavg_vs_fedprox_comparison.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\04_iid_vs_non_iid_comparison.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\05_federated_convergence_curves.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\06_ensemble_alpha_vs_prauc.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\07_precision_recall_curves.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\08_final_confusion_matrix.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\09_client_distribution_breakdown.png`
- `analysis\phase8_analysis_seed42_20261003T165402\visualizations\10_final_metric_summary.png`
