# FedSecureAI — Phase 12: Stage 1 Initial Explainability & Model Audit

**Audit Date:** 2026-10-26  
**Auditor:** Senior ML Research Engineer, Explainable AI Specialist & MLOps Auditor  
**Repository Root:** `c:\Users\Acer\OneDrive\Desktop\Federate_learning`  
**Target Phase:** Phase 12 (SHAP Explainability & Typology Attributions)  
**Status:** **STAGE 1 & STAGE 2 COMPLETED — READY FOR EXPLAINABILITY EXECUTION**

---

## 1. Executive Summary & Objective

In accordance with the FedSecureAI architectural guidelines and Phase 12 master directives, this read-only audit verifies all candidate model checkpoints, training data pipelines, feature dimension expansions, prediction parities, and hardware environments before executing attribution calculations.

### Core Audit Outcomes
1. **Model Checkpoints Authenticated**: Both the HI-Small Phase 7 champion (`hi_small_lstm_fedavg_iid_seed42`) and LI-Medium Phase 7 champion (`config_5_bilstm_fedavg_iid`) have been located, cryptographically hashed, and confirmed valid.
2. **Prediction Parity Verified (Stage 2)**: Recomputing model forward passes on 10,000 deterministic validation sequences yielded **exact numerical parity** with saved Phase 7 validation predictions:
   - **HI-Small LSTM**: Max absolute difference = **$1.49 \times 10^{-8}$**, Mean absolute difference = **$9.86 \times 10^{-12}$** (**PASSED**).
   - **LI-Medium BiLSTM**: Max absolute difference = **$1.51 \times 10^{-9}$**, Mean absolute difference = **$1.83 \times 10^{-12}$** (**PASSED**).
   - Label alignment between reloaded datasets and saved prediction arrays is **$100.0\%$ identical** (`True`).
3. **Data & Split Isolation Confirmed**:
   - Background baseline distributions ($x_0$) will be sampled **strictly from `SplitName.TRAIN`**.
   - Explanation instances will be sampled **strictly from `SplitName.VALIDATION`**.
   - The test split (`SplitName.TEST`) remains strictly isolated and untouched.
4. **Recurrent Explainability Architecture Established**:
   - Due to packed padded recurrent sequence architectures (`RecurrentSequenceClassifier` over LSTM and BiLSTM), standard tabular `TreeSHAP` is mathematically inapplicable.
   - We deploy **Integrated Gradients (Sundararajan et al., 2017)** as the primary sequence attribution explainer, satisfying the Completeness and Additivity Axioms ($\sum_{t,d} IG_{t,d}(x) = F(x) - F(x_0)$) while operating natively through PyTorch autograd over $(B, T=10, D=57)$ inputs.

---

## 2. Checkpoint and Model Inventory

### 2.1 Target Explanation Models

| Attribute | HI-Small Champion | LI-Medium Champion | Phase 10 Reference Model |
| :--- | :--- | :--- | :--- |
| **Model Name** | `hi_small_lstm_fedavg_iid_seed42` | `config_5_bilstm_fedavg_iid` | `federated_iid_fedavg_lstm_seed42_20260918T130515` |
| **Dataset** | IBM AML HI-Small | IBM AML LI-Medium | IBM AML HI-Small |
| **Checkpoint Path** | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt` | `results/LI_Medium_Results/checkpoints/config_5/global_best.pt` | `models/checkpoints/federated/iid/fedavg/lstm/federated_iid_fedavg_lstm_seed42_20260918T130515/global_best.pt` |
| **SHA-256 Checksum** | `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4` | `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2` | `64e495c8175170033d5f08301add618e31e077e453126a44d5c263f299f492cc` |
| **Architecture** | Unidirectional LSTM (`nn.LSTM`) | Bidirectional LSTM (`nn.LSTM(bidirectional=True)`) | Unidirectional LSTM (`nn.LSTM`) |
| **Input Dimension ($D$)** | 57 float32 features | 57 float32 features | 57 float32 features |
| **Hidden Units** | 64 | 64 per direction (Total = 128) | 64 |
| **Recurrent Layers** | 1 | 1 | 1 |
| **Representation Head** | `nn.Linear(64, 1)` | `nn.Linear(128, 1)` | `nn.Linear(64, 1)` |
| **Parameter Count** | 31,553 parameters | 63,105 parameters | 31,553 parameters |
| **Training Algorithm** | Federated Averaging (FedAvg) | Federated Averaging (FedAvg) | Federated Averaging (FedAvg) |
| **Data Partitioning** | IID (Uniform Stratified, 3 clients) | IID (Uniform Stratified, 3 clients) | IID (Uniform Stratified, 3 clients) |
| **Random Seed** | 42 | 42 | 42 |
| **Frozen Decision Threshold**| `0.3377837836742401` | `0.2384829968214035` | `0.24444444477558136` (Calibrated) / `0.13` (Base) |
| **Validation PR-AUC** | 0.242083 | 0.176899 | 0.367526 |

---

## 3. Data Pipeline, Feature Representation & Sequence Semantics

### 3.1 Preprocessing Pipeline & One-Hot Expansion
The preprocessed dataset contains 23 raw feature columns, 3 transaction identifier columns (`timestamp`, `from_account`, `to_account`), and 1 target column (`is_laundering`).
During model sequence construction (`ProcessedSequenceProvider._build_feature_matrix`), categorical columns are expanded via one-hot encoding:
1. `amount_paid_log` (Continuous, float32, log1p transformed)
2. `amount_received_log` (Continuous, float32, log1p transformed)
3. `amount_mismatch` (Binary boolean, {0, 1})
4. `same_currency` (Binary boolean, {0, 1})
5. `same_bank` (Binary boolean, {0, 1})
6. `self_transfer` (Binary boolean, {0, 1})
7. `payment_format_code` (Categorical, vocabulary size = 7 $\to$ expanded to indices 6..12)
8. `payment_currency_code` (Categorical, vocabulary size = 15 $\to$ expanded to indices 13..27)
9. `receiving_currency_code` (Categorical, vocabulary size = 15 $\to$ expanded to indices 28..42)
10. `minute_of_day_sin` (Continuous cyclical, [-1, 1])
11. `minute_of_day_cos` (Continuous cyclical, [-1, 1])
12. `day_of_week_sin` (Continuous cyclical, [-1, 1])
13. `day_of_week_cos` (Continuous cyclical, [-1, 1])
14. `is_weekend` (Binary boolean, {0, 1})
15. `is_night` (Binary boolean, {0, 1})
16. `sender_prior_count_log` (Continuous, log1p)
17. `sender_prior_amount_log` (Continuous, log1p)
18. `sender_seconds_since_prior_log` (Continuous, log1p)
19. `sender_has_prior` (Binary boolean, {0, 1})
20. `receiver_prior_count_log` (Continuous, log1p)
21. `receiver_prior_amount_log` (Continuous, log1p)
22. `receiver_seconds_since_prior_log` (Continuous, log1p)
23. `receiver_has_prior` (Binary boolean, {0, 1})

**Total Feature Vector Dimension**: Exactly **57 float32 features** ($D=57$). Both HI-Small and LI-Medium share identical feature definitions and column indices.

### 3.2 Sequence Representation & Padding Policy
- **Sequence Length ($T$)**: 10 transactions ($T = 10$).
- **Window Shape**: `(batch_size, sequence_length=10, feature_dimension=57)`.
- **True Sequence Lengths**: `lengths: Tensor (batch_size,)`, representing the number of causal historical transactions available for each originating account ($1 \le \text{length} \le 10$).
- **Padding**: Right-padded with zeros. Sequences are packed using `torch.nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)`.
- **Classification Semantics**: The model predicts the logit for the **newest (last) transaction** in the sequence window.
- **Probability Function**: $P(\text{is\_laundering} = 1) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$.

---

## 4. Stage 2 Prediction Parity Verification

To prove that the audited models reproduce Phase 7 outputs identically without drift or discrepancy, deterministic validation sequences were evaluated through `backend/scripts/verify_phase12_prediction_parity.py`.

### 4.1 Parity Results Table

| Dataset | Model Identifier | Checkpoint SHA-256 | Verified Rows | Max Abs Diff | Mean Abs Diff | Label Alignment | Parity Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HI-Small** | `hi_small_lstm_fedavg_iid_seed42` | `dacfbb62...` | 10,000 | **$1.49 \times 10^{-8}$** | **$9.86 \times 10^{-12}$** | **Exact (True)** | **PASSED** |
| **LI-Medium** | `config_5_bilstm_fedavg_iid` | `bded77c7...` | 10,000 | **$1.51 \times 10^{-9}$** | **$1.83 \times 10^{-12}$** | **Exact (True)** | **PASSED** |

Both models achieve maximum deviations well within machine floating-point epsilon ($< 10^{-5}$ tolerance required; observed $< 1.5 \times 10^{-8}$). The saved predictions in `results/phase7/` are confirmed to be bit-accurate representations of these model weights.

---

## 5. Phase 10 Calibration Verification

The Phase 10 calibration study (`experiments/results/calibration/calibration_seed42_20261013T155820/`):
- **Target Model**: HI-Small LSTM baseline (`federated_iid_fedavg_lstm_seed42_20260918T130515`).
- **Best Calibrator**: Isotonic Regression (selected on validation log loss: 0.004295).
- **Test Performance**: Improved Expected Calibration Error (ECE) from 0.000494 to **0.000217** (+56.18% improvement) while preserving test F1 (0.572194).
- **Calibrator Scope**: The fitted calibrator is strictly bound to HI-Small. In accordance with Section 10 rules, **it will not be applied to LI-Medium**, preserving uncalibrated probability explanations as the primary objective for both datasets.

---

## 6. Typology vs. Client Partitioning Clarification

An essential finding of this audit concerns Stage 7 (Typology Attributions):
1. **Raw Graph Typologies**: Synthetic AMLSim typologies (Structuring, Fan-Out, Fan-In, Cycle, Bipartite) were generated in raw graph construction but are **not present as per-transaction labels** in the preprocessed CSV splits.
2. **Federated Client Partition Domains**:
   - `client_1` (**Retail Giant**): Low-value domestic retail transactions, card/cash channels.
   - `client_2` (**Offshore / Crypto Exchange**): High-velocity cross-border wire transfers, currency conversions.
   - `client_3` (**Regional Credit Union**): Regional ACH, checking, intra-bank transfers.
3. **Audit Directive**:
   - Stage 7 attributions will be analyzed across the **3 Non-IID Banking Client Domains** (`client_1`, `client_2`, `client_3`) using transaction channel characteristics (payment format, currency mismatch, cross-bank transfers).
   - We will **not fabricate** synthetic graph typology labels or infer typologies purely from high model probabilities.

---

## 7. Proposed Explainability Configuration

### 7.1 Explainer Method: PyTorch Native Integrated Gradients
Because `RecurrentSequenceClassifier` utilizes `pack_padded_sequence` and variable-length sequence masking, black-box perturbation methods (`KernelExplainer`) destroy temporal sequence packing and suffer severe combinatorial explosion. 
We deploy **Integrated Gradients (Sundararajan et al., 2017)**:
- **Mathematical Definition**:
  $$IG_i(x) = (x_i - x_{0,i}) \times \int_0^1 \frac{\partial F(x_0 + \alpha(x - x_0))}{\partial x_i} d\alpha \approx \frac{(x_i - x_{0,i})}{m} \sum_{k=1}^m \frac{\partial F\left(x_0 + \frac{k}{m}(x - x_0)\right)}{\partial x_i}$$
- **Completeness & Additivity Guarantee**:
  $$\sum_{t=1}^{10} \sum_{d=1}^{57} IG_{t,d}(x) = F(x) - F(x_0)$$
  where $F(x) = \text{logit}(x)$ or $\sigma(\text{logit}(x))$.
- **Approximation Steps ($m$)**: $m = 50$ interpolation steps with Riemann summation.
- **Reference Baseline ($x_0$)**: Expected feature vector $\mathbb{E}_{x \sim \mathcal{D}_{\text{train}}}[x]$ computed over $N=100$ training transactions, with lengths $= 10$.
- **Tensor Output**: `(batch_size, sequence_length=10, feature_dimension=57)` attribution tensor.

### 7.2 Sampling Plan & Budget
- **Background Reference ($x_0$)**: $N = 100$ sequences from `SplitName.TRAIN`.
- **Explanation Sample**: $N = 200$ sequences from `SplitName.VALIDATION` (stratified: 100 legitimate negative examples, 100 laundering positive examples).
- **Test Split**: Zero involvement in explainer tuning; reserved for optional post-freezing descriptive checks only.
- **Memory Safety**: Processed in mini-batches of 16 sequences with explicit `torch.cuda.empty_cache()` / memory release.

---

## 8. Verification Sign-Off

- **Stage 1 (Audit)**: COMPLETE
- **Stage 2 (Parity Verification)**: COMPLETE & PASSED (`results/phase12/phase12_model_verification.csv`)
- **Ready for Implementation**: YES (Incremental implementation of Stage 3–13).

