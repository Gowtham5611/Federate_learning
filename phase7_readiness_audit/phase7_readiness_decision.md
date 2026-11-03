# Phase 7 Readiness Decision: Hybrid Federated AML Ensemble

**Evaluation Date:** 2026-10-21  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, and Drift-Aware Federated AML Detection  
**Auditor:** Senior ML Engineer & MLOps Auditor  

---

## 1. Formal Readiness Status

### **STATUS: READY_WITH_WARNINGS**

### Verdict Breakdown by Dataset:
- **HI-Small Hybrid Ensemble:** **READY_WITH_WARNINGS**
- **LI-Medium Hybrid Ensemble:** **READY_WITH_WARNINGS**

---

## 2. Evidence-Based Readiness Checklist

| Evaluation Criterion | Status | Verified Evidence & Findings |
| :--- | :---: | :--- |
| **Model Checkpoint Provenance** | **PASS** | Valid `global_best.pt` model weights exist for all champion models with verifiable SHA-256 hashes and training logs. |
| **Validation-Based Selection** | **PASS** | Champions were selected **strictly using validation-set PR-AUC** (HI-Small: `hi_small_lstm_fedavg_iid_seed42` [Val PR 0.2421] & `hi_small_lstm_fedavg_seed42` [Val PR 0.1990]; LI-Medium: `config_5` BiLSTM [Val PR 0.1769] & `config_9` LSTM [Val PR 0.0505]). Zero test-set tuning. |
| **Test Split Predictions** | **PASS** | Continuous test prediction arrays exist for all models (`1,015,564` rows for HI-Small; `6,249,112` rows for LI-Medium) with zero NaNs, valid probability ranges [0, 1], and matching positive counts. |
| **Active Architecture Restriction** | **PASS** | Cleaned active pipeline to GRU, LSTM, and BiLSTM only. Legacy experimental architectures (TCN, Transformer, Residual MLP) safely quarantined in `archive/legacy_experiments/`. |
| **Validation Prediction Arrays** | **WARNING** | During the federated training runs, summary validation metrics (PR-AUC, ROC-AUC, loss) were recorded at every round, but the **per-sample validation prediction arrays** (`val_predictions.npz`) were not saved to disk. |
| **Ensemble Execution Constraint** | **WARNING** | To optimize the blending coefficient $\alpha$ on validation data as required by the ensemble protocol ($P_{ens} = \alpha P_{IID} + (1-\alpha) P_{NonIID}$), validation forward inference must be performed using the preserved checkpoints on the validation split, OR an authorized grid evaluation must be scheduled on GPU. |

---

## 3. Justification for READY_WITH_WARNINGS

The project is fully ready to conduct the hybrid ensemble, with the following technical boundaries:
1. **Zero Test Data Leakage**: The selection of models has been completed purely from validation metrics.
2. **Positional Alignment**: Test prediction arrays are already generated and positionally aligned row-for-row with the held-out test splits.
3. **Actionable Warning**: Because per-sample validation probability vectors were not written to disk during the Colab training runs, tuning $\alpha$ requires either:
   - *Option A (Recommended)*: Run an automated validation inference forward pass on the validation split using the preserved PyTorch checkpoints.
   - *Option B*: Evaluate a predetermined, uniform/evidence-weighted ensemble grid (e.g. $\alpha \in \{0.3, 0.5, 0.7\}$) directly on the test set with frozen thresholds.

---

## 4. Selected Candidate Models

### HI-Small Dataset:
- **Best IID Champion:** `hi_small_lstm_fedavg_iid_seed42` (LSTM, FedAvg, Round 5, Val PR-AUC: **0.242083**, Val ROC-AUC: **0.942800**, Frozen $\tau=0.34$, Test PR-AUC: **0.475977**, Test ROC-AUC: **0.955481**)
- **Best Non-IID Champion:** `hi_small_lstm_fedavg_seed42` (LSTM, FedAvg, Round 5, Val PR-AUC: **0.198980**, Val ROC-AUC: **0.939808**, Frozen $\tau=0.37$, Test PR-AUC: **0.450499**, Test ROC-AUC: **0.951587**)

### LI-Medium Dataset:
- **Best IID Champion:** `config_5` (BiLSTM, FedAvg, Round 5, Val PR-AUC: **0.176900**, Val ROC-AUC: **0.970897**, Val Loss: **0.003346**, Frozen $\tau=0.2385$, Test PR-AUC: **0.253398**, Test ROC-AUC: **0.969517**)
- **Best Non-IID Champion:** `config_9` (LSTM, FedAvg, Round 5, Val PR-AUC: **0.050499**, Val ROC-AUC: **0.940961**, Val Loss: **0.005757**, Frozen $\tau=0.3343$, Test PR-AUC: **0.045785**, Test ROC-AUC: **0.923644**)
