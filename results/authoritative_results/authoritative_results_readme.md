# FedSecureAI — Authoritative Results Registry & Experiment Lock

**Audit Timestamp:** 2026-10-25 20:29:13  
**Auditor / Role:** Senior Federated Learning Engineer, AML Researcher & MLOps Auditor  
**Lock Status:** **AUTHORITATIVE LOCKED · 100% REPRODUCIBLE**

---

## 1. Executive Summary & Source of Truth

This directory (`results/authoritative_results/`) establishes the permanent, reconciled, and audited source of truth for the **FedSecureAI** research project.

### Authoritative Dataset Result Folders:
1. **HI-Small Experiment Results**: `C:\Users\Acer\OneDrive\Desktop\Federate_learning\colab work and results` (Organized mirror: `results/HI_Small_Results/`)
2. **LI-Medium Experiment Results**: `C:\Users\Acer\OneDrive\Desktop\Federate_learning\Colab Results` (Organized mirror: `results/LI_Medium_Results/`)

### Summary of Audited Experiments:
- **Active Authoritative Experiments**: **24 completed runs** (12 HI-Small + 12 LI-Medium across IID/Non-IID, FedAvg/FedProx, and GRU/LSTM/BiLSTM).
- **FedProx $\mu=0.0001$ Compliance**: **100% verified (12/12 active FedProx runs confirmed with $\mu=0.0001$)**.
- **Excluded Historical Runs**: Early exploratory runs with $\mu=0.1$ and $\mu=1.0$ (from October 15–18 backups) and legacy 500k-row preliminary Combined runs are explicitly quarantined and excluded from active comparisons.
- **Unresolved Runs**: **0 runs**. All 24 active configurations possess verifiable checkpoints, prediction arrays, and exact metric recalculation down to the 6th decimal place.

---

## 2. Mandatory FedProx Rule Enforcement ($\mu = 0.0001$)

Per project mandate, all FedProx comparisons require $\mu = 0.0001$:
- In **HI-Small**, all 6 FedProx runs (`config_2`, `config_4`, `config_6`, `config_8`, `config_10`, `config_12`) recorded $\mu = 0.0001$ in `resume_manifest.json` and explicit directory naming (`_mu0.0001_`).
- In **LI-Medium**, all 6 FedProx runs recorded $\mu = 0.0001$ in `hyperparameters.fedprox_mu` across `resume_manifest_configs_1_6.json` and `resume_manifest_configs_7_12.json`.
- Historical exploratory runs utilizing $\mu=0.1$ or $\mu=1.0$ (found in `artifacts/backups/` and `experiments/results/fedsecureai_runs/20261015/`) are documented in [`fedprox_mu_verification.csv`](fedprox_mu_verification.csv) and marked `EXCLUDED_FROM_COMPARISON`.
- FedAvg runs are unaffected as they do not employ proximal regularization.

---

## 3. Phase 7 Hybrid Ensemble Verification & $\alpha = 1.00$ Scientific Reality

Phase 7 evaluated a convex combination of the champion IID and champion Non-IID models:
$$P_{ens}(x) = \alpha P_{IID}(x) + (1 - \alpha) P_{NonIID}(x)$$

### Champion Models:
- **HI-Small**:
  - IID Champion: `hi_small_lstm_fedavg_iid_seed42` (Val PR-AUC: **0.242083**, Test PR-AUC: **0.475977**)
  - Non-IID Champion: `hi_small_lstm_fedavg_seed42` (Val PR-AUC: **0.198980**, Test PR-AUC: **0.450499**)
- **LI-Medium**:
  - IID Champion: `config_5` (BiLSTM FedAvg, Val PR-AUC: **0.176899**, Test PR-AUC: **0.253398**)
  - Non-IID Champion: `config_9` (LSTM FedAvg, Val PR-AUC: **0.050499**, Test PR-AUC: **0.045785**)

### Scientific Findings on $\alpha = 1.00$:
- Under strict validation PR-AUC maximization across a 21-point grid ($\alpha \in [0.00, 1.00]$, step 0.05), **$\alpha^* = 1.00$ was selected for both datasets**.
- **Equivalence Statement**: Because $\alpha^* = 1.00$, the ensemble is **mathematically and empirically identical to the standalone IID model**.
- **No False Ensemble Gain**: The Hybrid Ensemble achieves **zero gain** over the standalone IID champion (+0.000000 PR-AUC gain on both HI-Small and LI-Medium). The Non-IID models suffered severe class-imbalance drift (down to 0.0458 PR-AUC on LI-Medium), and any weight assigned to Non-IID degraded performance. The ensemble optimizer correctly protected performance by assigning 100% weight to IID.

---

## 4. Authoritative Performance Matrix (Test Split)

### HI-Small Held-Out Test Split (1,015,564 rows, 1,797 laundering positives):
| Config | Architecture | Distribution | Algorithm | $\mu$ | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Alerts / 10k |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | GRU | IID | FedAvg | - | 0.394998 | 0.954757 | 0.340112 | 0.278226 | 0.437396 | 27.82 |
| 2 | GRU | IID | FedProx | 0.0001 | 0.249625 | 0.935743 | 0.258109 | 0.230708 | 0.292710 | 24.80 |
| 3 | LSTM | IID | FedAvg | - | **0.475977** | **0.955481** | **0.480204** | **0.901840** | 0.327212 | **6.42** |
| 4 | LSTM | IID | FedProx | 0.0001 | 0.282432 | 0.910570 | 0.307903 | 0.741071 | 0.200334 | 3.86 |
| 5 | BiLSTM | IID | FedAvg | - | 0.407818 | 0.940446 | 0.380753 | 0.487910 | 0.312187 | 11.48 |
| 6 | BiLSTM | IID | FedProx | 0.0001 | 0.297752 | 0.904582 | 0.304833 | 0.381395 | 0.253756 | 19.39 |
| 7 | GRU | Non-IID | FedAvg | - | 0.344372 | 0.950844 | 0.392070 | 0.655938 | 0.279354 | 7.66 |
| 8 | GRU | Non-IID | FedProx | 0.0001 | 0.227585 | 0.930447 | 0.263647 | 0.392683 | 0.201447 | 6.66 |
| 9 | LSTM | Non-IID | FedAvg | - | 0.450499 | 0.951587 | 0.502924 | 0.839844 | 0.358932 | 7.56 |
| 10 | LSTM | Non-IID | FedProx | 0.0001 | 0.279249 | 0.907933 | 0.349603 | 0.612000 | 0.244296 | 8.33 |
| 11 | BiLSTM | Non-IID | FedAvg | - | 0.390451 | 0.933841 | 0.435728 | 0.640244 | 0.329994 | 8.84 |
| 12 | BiLSTM | Non-IID | FedProx | 0.0001 | 0.281627 | 0.898636 | 0.348397 | 0.598291 | 0.246522 | 7.80 |

### LI-Medium Held-Out Test Split (6,249,112 rows, 3,783 laundering positives):
| Config | Architecture | Distribution | Algorithm | $\mu$ | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Alerts / 10k |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | GRU | IID | FedAvg | - | **0.253598** | **0.969705** | **0.316785** | **0.622291** | 0.212530 | **2.07** |
| 2 | GRU | IID | FedProx | 0.0001 | 0.088746 | 0.952056 | 0.141802 | 0.353383 | 0.090404 | 3.04 |
| 3 | LSTM | IID | FedAvg | - | 0.245072 | 0.968630 | 0.301548 | 0.658065 | 0.198255 | 1.86 |
| 4 | LSTM | IID | FedProx | 0.0001 | 0.089483 | 0.949487 | 0.128870 | 0.329412 | 0.080888 | 1.60 |
| 5 | BiLSTM | IID | FedAvg | - | 0.253398 | 0.969517 | 0.316789 | 0.594182 | **0.215966** | 2.20 |
| 6 | BiLSTM | IID | FedProx | 0.0001 | 0.087103 | 0.951358 | 0.136290 | 0.370370 | 0.083796 | 1.81 |
| 7 | GRU | Non-IID | FedAvg | - | 0.007560 | 0.872551 | 0.024765 | 0.038462 | 0.018240 | 5.86 |
| 8 | GRU | Non-IID | FedProx | 0.0001 | 0.005106 | 0.837306 | 0.019314 | 0.014286 | 0.030135 | 54.93 |
| 9 | LSTM | Non-IID | FedAvg | - | **0.045785** | **0.923644** | **0.075216** | **0.090203** | **0.064499** | **4.33** |
| 10 | LSTM | Non-IID | FedProx | 0.0001 | 0.005388 | 0.854946 | 0.019023 | 0.017778 | 0.020619 | 39.03 |
| 11 | BiLSTM | Non-IID | FedAvg | - | 0.023957 | 0.895235 | 0.022091 | 0.044199 | 0.014803 | 4.88 |
| 12 | BiLSTM | Non-IID | FedProx | 0.0001 | 0.007985 | 0.870974 | 0.018801 | 0.018657 | 0.018768 | 29.88 |

---

## 5. Next Project Phase: Phase 8 (Performance Analysis & Ablation Study)

Per the project roadmap and architecture specification (`backend/application/dto/analysis.py:Phase8Report`):
- **Next Phase**: **Phase 8 — Comprehensive Performance Analysis, Centralized vs Federated Benchmarking, and Ablation Study**.
- **Phase 8 Scope**:
  1. Systematic Centralized vs Federated degradation analysis (comparing centralized baselines to federated IID/Non-IID).
  2. Federated convergence analysis across communication rounds (1 to 5).
  3. Client-level heterogeneity impact quantification.
  4. Decision threshold and alert rate trade-off modeling for financial operations.
  5. Generating composite `Phase8Report` JSON and Markdown artifacts.

---

## 6. Formal Phase Readiness Decision

### Verdict: **`READY`**

**Justification**:
1. All 24 authoritative federated learning experiments are 100% completed, hashed, and verified.
2. 100% compliance with the mandatory FedProx $\mu=0.0001$ rule.
3. Zero test leakage in threshold and ensemble weight calibration.
4. Active pipeline restricted strictly to GRU, LSTM, and BiLSTM.
5. All 169 unit & integration tests and all 7 Phase 7 ensemble tests pass with 100% green status.
