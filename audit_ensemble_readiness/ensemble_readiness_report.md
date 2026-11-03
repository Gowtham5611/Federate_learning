# FedSecureAI — Ensemble Readiness Verification Report

**Date:** 2026-10-19 18:45:50  
**Objective:** Evaluate suitability of saved models for ensemble learning (averaging, weighted blending, stacking, rank fusion).

---

## 1. Ensemble Eligibility by Dataset

### A. HI-Small Candidate Models (100% Eligible)
All 12 configurations on HI-Small were trained on the full 3.05M training transactions and evaluated on the exact same held-out test split (1,015,564 transactions, 1,797 laundering positives).

| Model ID | Architecture | Strategy | Algorithm | Test PR-AUC | Test ROC-AUC | Ensemble Role |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `hi_small_lstm_fedavg_iid_seed42` | LSTM | IID | FedAvg | **0.475977** | **0.955481** | Primary Base Learner (Best Single) |
| `hi_small_lstm_fedavg_seed42` | LSTM | Non-IID | FedAvg | 0.450499 | 0.951587 | Core Recurrent Model |
| `hi_small_bilstm_fedavg_iid_seed42` | BiLSTM | IID | FedAvg | 0.407818 | 0.940446 | Bidirectional Temporal Diversity |
| `hi_small_gru_fedavg_iid_seed42` | GRU | IID | FedAvg | 0.394998 | 0.954757 | Fast Gated Temporal Diversity |
| `hi_small_bilstm_fedavg_seed42` | BiLSTM | Non-IID | FedAvg | 0.390451 | 0.933841 | Non-IID Typology Complement |
| `hi_small_gru_fedavg_seed42` | GRU | Non-IID | FedAvg | 0.344372 | 0.950844 | Non-IID Gated Complement |
| `hi_small_bilstm_fedprox_mu0.0001_iid_seed42` | BiLSTM | IID | FedProx | 0.297752 | 0.904582 | Proximal Regularized Complement |
| `hi_small_lstm_fedprox_mu0.0001_iid_seed42` | LSTM | IID | FedProx | 0.282432 | 0.910570 | Proximal Regularized Complement |
| `hi_small_bilstm_fedprox_mu0.0001_seed42` | BiLSTM | Non-IID | FedProx | 0.281627 | 0.898636 | Proximal Non-IID Complement |
| `hi_small_lstm_fedprox_mu0.0001_seed42` | LSTM | Non-IID | FedProx | 0.279249 | 0.907933 | Proximal Non-IID Complement |
| `hi_small_gru_fedprox_mu0.0001_iid_seed42` | GRU | IID | FedProx | 0.249625 | 0.935743 | Proximal Gated Complement |
| `hi_small_gru_fedprox_mu0.0001_seed42` | GRU | Non-IID | FedProx | 0.227585 | 0.930447 | Proximal Gated Complement |

**Readiness Verdict for HI-Small:** **READY.**  
All predictions are aligned positionally across rows `0 .. 1,015,563`. Zero retraining required.

---

### B. Combined Candidate Models (100% Eligible Independently)
All 12 configurations on Combined were trained on the full 300,000 training transactions and evaluated on the exact same held-out test split (100,000 transactions, 94 laundering positives).

- Test predictions exist for all 12 models (shape `(100,000,)`).
- Best single model: `combined_gru_fedavg_seed42` (PR-AUC = 0.001208, ROC-AUC = 0.465593).
- **Readiness Verdict for Combined:** **READY independently** (must never be mixed with HI-Small predictions).

---

### C. Full LI-Medium Candidate Models (NOT Eligible Yet)
- Full dataset training (18.75M train, 6.25M test) has not been run in Colab yet.
- Historical models from `li_medium_subset_comparison` were trained on only 200,000 rows and cannot serve as the final ensemble candidate.
- **Readiness Verdict for Full LI-Medium:** **NOT READY.** Awaiting execution of `colab_full_li_medium_runner.ipynb`.

---

## 2. Recommended Next Engineering Actions

1. **Option A (Immediate Ensemble Development on HI-Small)**:
   - Begin building the ensemble pipeline (Soft Voting, Rank Averaging, Stacking Logistic Meta-Model) using the **12 verified HI-Small models**.
   - With top base learners achieving PR-AUC 0.4760 and ROC-AUC 0.9555, ensemble combinations across GRU, LSTM, and BiLSTM under IID and Non-IID are primed to reach PR-AUC > 0.50+.

2. **Option B (Train Full LI-Medium in Google Colab First)**:
   - Upload `FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip` to Google Drive.
   - Run `colab_full_li_medium_runner.ipynb` on GPU.
   - Once all 12 full LI-Medium runs complete, bring back `artifacts/full_li_medium_experiment/` and build the full-dataset LI-Medium ensemble.
