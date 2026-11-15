# FedSecureAI — Comprehensive Experiment Results Audit & Verification Report

**Auditor:** Senior Machine Learning Engineer, Federated Learning Researcher & MLOps Auditor  
**Date:** 2026-10-21  
**Project:** FedSecureAI — Federated Learning for Anti-Money Laundering Detection  
**Workspace:** `c:\Users\Acer\OneDrive\Desktop\Federate_learning`  

---

## 1. Executive Summary & Verdicts

| Scope | Final Verdict | Justification |
| :--- | :---: | :--- |
| **HI-Small Experiment Suite** | **PASS** | All 12 configurations (IID & Non-IID, FedAvg & FedProx across GRU, LSTM, BiLSTM) are complete with 84 valid checkpoints, 12 prediction arrays, and exact metric recalculation matches down to the 6th decimal place. |
| **LI-Medium Experiment Suite** | **PASS** | All 12 full-scale configurations trained on the complete 31.25M-row dataset in Colab (Configs 1-12) possess valid checkpoints, full 6.25M-element prediction arrays, and 100% verified metrics. |
| **Entire FedSecureAI Project** | **PASS WITH WARNINGS** | Both independent dataset suites (HI-Small and LI-Medium) are 100% verified and ready for ensemble modeling. The WARNING designation signifies that the Combined dataset (36.33M rows) has not yet been executed in Colab and remains pending. |

---

## 2. Dataset Validation & Provenance

Every processed transaction row was reconciled against the raw source CSVs with **0 dropped rows**:

| Dataset | Split | Observed Rows | Expected Rows | Legitimate (0) | Laundering (1) | Positive Rate | Temporal Window |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **HI-Small** | Train | 3,047,216 | 3,047,216 | 3,044,917 | 2,299 | 0.0754% | 2022/09/01 00:00 - 2022/09/06 13:36 |
| **HI-Small** | Validation | 1,015,565 | 1,015,565 | 1,014,484 | 1,081 | 0.1064% | 2022/09/06 13:37 - 2022/09/08 16:12 |
| **HI-Small** | Test | 1,015,564 | 1,015,564 | 1,013,767 | 1,797 | 0.1769% | 2022/09/08 16:13 - 2022/09/18 16:18 |
| **HI-Small** | **Total** | **5,078,345** | **5,078,345** | **5,073,168** | **5,177** | **0.1019%** | **Full Range Preserved** |
| **LI-Medium** | Train | 18,752,217 | 18,752,217 | 18,744,079 | 8,138 | 0.0434% | 2022/09/01 00:00 - 2022/09/09 20:39 |
| **LI-Medium** | Validation | 6,250,154 | 6,250,154 | 6,246,034 | 4,120 | 0.0659% | 2022/09/09 20:40 - 2022/09/14 05:39 |
| **LI-Medium** | Test | 6,249,112 | 6,249,112 | 6,245,329 | 3,783 | 0.0605% | 2022/09/14 05:40 - 2022/09/27 14:58 |
| **LI-Medium** | **Total** | **31,251,483** | **31,251,483** | **31,235,442** | **16,041** | **0.0513%** | **Full Range Preserved** |

### Chronology and Leakage Verification
- **Strict Chronological Splitting**: Verified via `splits.json` boundaries. Timestamps are disjoint across Train, Validation, and Test.
- **Zero Temporal Leakage**: Validation start strictly follows Train end; Test start strictly follows Validation end.
- **Feature Pipeline Consistency**: Exact same standard scaling, amount log transform, and cyclical trigonometric features applied across all splits.

---

## 3. Configuration Audit Matrix (All 24 Configurations)

| Dataset | Config ID | Distribution | Architecture | Algorithm | Rounds | Clients | Checkpoint Status | Recalculated ROC-AUC | Reported ROC-AUC | Recalculated PR-AUC | Reported PR-AUC | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HI-Small | `hi_small_gru_fedavg_iid_seed42` | IID | GRU | FedAvg | 5 | 3 | VERIFIED | 0.954757 | 0.954757 | 0.394998 | 0.394998 | PASS |
| HI-Small | `hi_small_gru_fedprox_mu0.0001_iid_seed42` | IID | GRU | FedProx | 5 | 3 | VERIFIED | 0.935743 | 0.935743 | 0.249625 | 0.249625 | PASS |
| HI-Small | `hi_small_lstm_fedavg_iid_seed42` | IID | LSTM | FedAvg | 5 | 3 | VERIFIED | 0.955481 | 0.955481 | 0.475977 | 0.475977 | PASS |
| HI-Small | `hi_small_lstm_fedprox_mu0.0001_iid_seed42` | IID | LSTM | FedProx | 5 | 3 | VERIFIED | 0.910570 | 0.910570 | 0.282432 | 0.282432 | PASS |
| HI-Small | `hi_small_bilstm_fedavg_iid_seed42` | IID | BiLSTM | FedAvg | 5 | 3 | VERIFIED | 0.940446 | 0.940446 | 0.407818 | 0.407818 | PASS |
| HI-Small | `hi_small_bilstm_fedprox_mu0.0001_iid_seed42` | IID | BiLSTM | FedProx | 5 | 3 | VERIFIED | 0.904582 | 0.904582 | 0.297752 | 0.297752 | PASS |
| HI-Small | `hi_small_gru_fedavg_seed42` | Non-IID | GRU | FedAvg | 5 | 3 | VERIFIED | 0.950844 | 0.950844 | 0.344372 | 0.344372 | PASS |
| HI-Small | `hi_small_gru_fedprox_mu0.0001_seed42` | Non-IID | GRU | FedProx | 5 | 3 | VERIFIED | 0.930447 | 0.930447 | 0.227585 | 0.227585 | PASS |
| HI-Small | `hi_small_lstm_fedavg_seed42` | Non-IID | LSTM | FedAvg | 5 | 3 | VERIFIED | 0.951587 | 0.951587 | 0.450499 | 0.450499 | PASS |
| HI-Small | `hi_small_lstm_fedprox_mu0.0001_seed42` | Non-IID | LSTM | FedProx | 5 | 3 | VERIFIED | 0.907933 | 0.907933 | 0.279249 | 0.279249 | PASS |
| HI-Small | `hi_small_bilstm_fedavg_seed42` | Non-IID | BiLSTM | FedAvg | 5 | 3 | VERIFIED | 0.933841 | 0.933841 | 0.390451 | 0.390451 | PASS |
| HI-Small | `hi_small_bilstm_fedprox_mu0.0001_seed42` | Non-IID | BiLSTM | FedProx | 5 | 3 | VERIFIED | 0.898636 | 0.898636 | 0.281627 | 0.281627 | PASS |
| LI-Medium | `config_1` | IID | GRU | FedAvg | 5 | 3 | VERIFIED | 0.969705 | 0.969705 | 0.253598 | 0.253598 | PASS |
| LI-Medium | `config_2` | IID | GRU | FedProx | 5 | 3 | VERIFIED | 0.952056 | 0.952056 | 0.088746 | 0.088746 | PASS |
| LI-Medium | `config_3` | IID | LSTM | FedAvg | 5 | 3 | VERIFIED | 0.968630 | 0.968630 | 0.245072 | 0.245072 | PASS |
| LI-Medium | `config_4` | IID | LSTM | FedProx | 5 | 3 | VERIFIED | 0.949487 | 0.949487 | 0.089483 | 0.089483 | PASS |
| LI-Medium | `config_5` | IID | BiLSTM | FedAvg | 5 | 3 | VERIFIED | 0.969517 | 0.969517 | 0.253398 | 0.253398 | PASS |
| LI-Medium | `config_6` | IID | BiLSTM | FedProx | 5 | 3 | VERIFIED | 0.951358 | 0.951358 | 0.087103 | 0.087103 | PASS |
| LI-Medium | `config_7` | Non-IID | GRU | FedAvg | 5 | 3 | VERIFIED | 0.872551 | 0.872551 | 0.007560 | 0.007560 | PASS |
| LI-Medium | `config_8` | Non-IID | GRU | FedProx | 5 | 3 | VERIFIED | 0.837306 | 0.837306 | 0.005106 | 0.005106 | PASS |
| LI-Medium | `config_9` | Non-IID | LSTM | FedAvg | 5 | 3 | VERIFIED | 0.923644 | 0.923644 | 0.045785 | 0.045785 | PASS |
| LI-Medium | `config_10` | Non-IID | LSTM | FedProx | 5 | 3 | VERIFIED | 0.854946 | 0.854946 | 0.005388 | 0.005388 | PASS |
| LI-Medium | `config_11` | Non-IID | BiLSTM | FedAvg | 5 | 3 | VERIFIED | 0.895235 | 0.895235 | 0.023957 | 0.023957 | PASS |
| LI-Medium | `config_12` | Non-IID | BiLSTM | FedProx | 5 | 3 | VERIFIED | 0.870974 | 0.870974 | 0.007985 | 0.007985 | PASS |

---

## 4. Metrics Validation & Confusion Matrix Details

Every model was independently evaluated against its held-out test split using the exact frozen threshold selected on the validation split:

### HI-Small Confusion Matrix Audit (Test Split: 1,015,564 samples)
- `hi_small_lstm_fedavg_iid_seed42`: TP=588, FP=64, FN=1209, TN=1,013,703 | **Precision=90.18%**, **Recall=32.72%**, **F1=0.4802**, **Alerts/10k=6.42**
- `hi_small_lstm_fedavg_seed42`: TP=645, FP=123, FN=1152, TN=1,013,644 | **Precision=83.98%**, **Recall=35.89%**, **F1=0.5029**, **Alerts/10k=7.56**
- `hi_small_gru_fedavg_iid_seed42`: TP=786, FP=2039, FN=1011, TN=1,011,728 | **Precision=27.82%**, **Recall=43.74%**, **F1=0.3401**, **Alerts/10k=27.82**

### LI-Medium Confusion Matrix Audit (Test Split: 6,249,112 samples)
- `config_1` (GRU IID FedAvg): TP=804, FP=488, FN=2979, TN=6,244,841 | **Precision=62.23%**, **Recall=21.25%**, **F1=0.3168**, **Alerts/10k=2.07**
- `config_5` (BiLSTM IID FedAvg): TP=817, FP=558, FN=2966, TN=6,244,771 | **Precision=59.42%**, **Recall=21.60%**, **F1=0.3168**, **Alerts/10k=2.20**
- `config_9` (LSTM Non-IID FedAvg): TP=244, FP=2461, FN=3539, TN=6,242,868 | **Precision=9.02%**, **Recall=6.45%**, **F1=0.0752**, **Alerts/10k=4.33**

---

## 5. Scientific Validity & Algorithmic Insights

1. **Threshold Selection Integrity**:
   - The threshold was frozen strictly on the validation split prior to test inference. No test-set tuning occurred.
2. **Implementation of PR-AUC**:
   - Both original runs and this independent audit use **Average Precision (AP)** via `sklearn.metrics.average_precision_score`, avoiding linear trapezoidal overestimation in extreme imbalance.
3. **Root Cause of Non-IID Degradation in LI-Medium**:
   - In Non-IID partitioning, financial transactions are partitioned by account/bank clusters. With a 0.05% positive rate, local client batches often have 0 or 1 positive sample, causing gradient sparsity and divergence. FedProx ($\mu=0.0001$) further restricted local adaptation, leading to worse performance than FedAvg.

---

## 6. Final Recommendations for Ensemble Phase

1. **Ensemble Eligibility**:
   - **HI-Small**: 100% of the 12 models are eligible for ensemble blending. Recommended primary weights: `hi_small_lstm_fedavg_iid_seed42` (0.40) + `hi_small_lstm_fedavg_seed42` (0.35) + `hi_small_bilstm_fedavg_iid_seed42` (0.25).
   - **LI-Medium**: The 6 IID models (`config_1`, `config_3`, `config_5`) and top Non-IID models (`config_9`, `config_11`) are eligible for ensemble experiments.
2. **Pending Actions**:
   - The Combined dataset (36.33M rows) has not yet been executed in Colab. If full combined results are desired, run `colab_full_combined_and_li_medium_runner.ipynb` using `FedSecureAI_FULL_COMBINED_AND_LI_MEDIUM_COLAB_READY_v2.zip`.
