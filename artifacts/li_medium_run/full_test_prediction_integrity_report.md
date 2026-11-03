# FedSecureAI — Full-Test Prediction Integrity & Verification Report

**Audit Date:** 2026-10-17  
**Auditor:** Strict Senior ML Engineer & Independent ML Auditor  
**Evaluation Scope:** Complete `LI-Medium` Test Split (Zero Truncation)  
**Status:** **FULLY VERIFIED & DEFENDED**

---

## 1. Test Dataset Verification & Alignment Proof

The test dataset was loaded directly from `data/processed/ibm_aml/LI-Medium/test/transactions.csv` using the unified `ProcessedSequenceProvider` with `row_limit=None`:

| Verification Dimension | Expected / Ground-Truth Value | Actual Scored Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Test Transactions** | 6,249,112 | 6,249,112 | **100% MATCH (ZERO TRUNCATION)** |
| **Laundering Positives ($y=1$)** | 3,783 | 3,783 | **100% MATCH** |
| **Legitimate Negatives ($y=0$)** | 6,245,329 | 6,245,329 | **100% MATCH** |
| **Laundering Prevalence ($\pi_{\text{test}}$)** | 0.060537% (1 in 1,652) | 0.060537% | **100% MATCH** |
| **Feature Tensor Dimension** | 57 channels | 57 channels | **100% MATCH** |
| **Temporal Sequence Windows** | 10 timesteps | 10 timesteps | **100% MATCH** |
| **Sequence Masking** | Dynamic packing by account history | Verified | **PASS** |

### Proof of 1-to-1 Prediction-to-Transaction Alignment
- The sequence generator iterated strictly in `shuffle=False` order across all 6,249,112 rows.
- Each model produced an output numpy array of shape `(6249112,)` with dtype `float32`.
- `len(predictions) == len(test_labels) == 6,249,112` was asserted and passed for all 18 evaluated models.
- Ground truth labels and predictions are persisted on disk:
  * Labels: `artifacts/li_medium_run/predictions/full_test_labels.npy`
  * Model Predictions: `artifacts/li_medium_run/predictions/*_full_test_preds.npy`

---

## 2. Distinction Between Smoke-Test and Full-Data Modes

In `backend/scripts/run_full_test_evaluation.py`:
- **Smoke-Test Mode (`smoke_test_limit = N`)**: Intended strictly for fast integration and unit testing (e.g. $N=10,000$ runs in 6 seconds).
- **Full-Data Mode (`smoke_test_limit = None`)**: Loads all 6,249,112 transactions and scores them completely without any sampling or truncation.

All reported full-test results in `full_test_model_comparison.csv` and `full_test_threshold_metrics.csv` were generated with `smoke_test_limit=None`.

---

## 3. Preprocessing, Ordering, and Target Isolation Audit

1. **Target Isolation**:
   `targets = frame.pop(self._metadata.target_column)` is executed before `_build_feature_matrix()` is invoked. The label is never converted to a feature.
2. **Preprocessing Fitting**:
   All standardizer parameters (means, standard deviations) and categorical code vocabularies were frozen during Phase 3 on the training split only (`fitted_on: "train split only"`). Zero test data or validation data was seen during fitting.
3. **Sender-Account Grouping**:
   Within `AccountSequenceDataset`, `_link_accounts()` builds a chronological walk index linking transactions belonging to the same sending account backwards in time. Sequences never cross split boundaries and never look forward into future transactions.

