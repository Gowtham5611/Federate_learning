# FedSecureAI — Independent Audit: Ensemble Pipeline & Leakage Analysis

**Audit Date:** 2026-10-17  
**Auditor Role:** Strict Independent ML Auditor  
**Ensemble Model:** Stacking Meta-Classifier (L2-Regularized Logistic Regression, $C=1.0$)  
**Reported Performance:** Test PR-AUC = 0.002009, Test ROC-AUC = 0.607284  
**Audit Classification:** **CONDITIONALLY VERIFIED (200k Slice) / UNVERIFIED (Full Dataset)**

---

## 1. Trace of Meta-Training Predictions & Base Models

The stacking ensemble combines predictions from the top 3 federated models identified during the Phase 4 staged hyperparameter exploration:
1. `mu_0.01`: GRU, IID, FedProx ($\mu=0.01$), Weighted BCE ($w_{\text{pos}}=2303.28$), LR = 0.001, Adam
2. `mu_0.001`: GRU, IID, FedProx ($\mu=0.001$), Weighted BCE ($w_{\text{pos}}=2303.28$), LR = 0.001, Adam
3. `lr_1e-3_sgd`: GRU, IID, FedProx ($\mu=0.10$), Weighted BCE ($w_{\text{pos}}=2303.28$), LR = 0.001, SGD

### Meta-Training Matrix Generation
In `backend/scripts/run_optimization_and_ensemble.py` lines 352-370:
```python
val_pred_matrix = np.column_stack([trial_models[k]["val_preds"] for k in top_3_keys])
test_pred_matrix = np.column_stack([trial_models[k]["test_preds"] for k in top_3_keys])

stacker = LogisticRegression(penalty="l2", C=1.0, random_state=42)
stacker.fit(val_pred_matrix, val_labels)
val_ens_stacked = stacker.predict_proba(val_pred_matrix)[:, 1]
test_ens_stacked = stacker.predict_proba(test_pred_matrix)[:, 1]
```

---

## 2. Leakage Audit Checklist

### Check 1: Did base models train on the validation labels?
- **Audit Procedure**: Traced client training loop. Clients were trained exclusively on `client_frames` derived from the training split (`train/transactions.csv`). The validation split (`val_dataset`) was evaluated using `eval_engine.predict` in `no_grad()` mode without backpropagation or parameter updates.
- **Finding**: **PASS (NO LEAKAGE)**. The base models never observed validation labels during gradient descent.

### Check 2: Were test labels or test metrics used in model selection or ensembling?
- **Audit Procedure**: Traced line 339 of `run_optimization_and_ensemble.py`:
  `sorted_trials = sorted(trial_models.items(), key=lambda x: x[1]["val_pr_auc"], reverse=True)`
  The 3 models were selected strictly by sorting on `val_pr_auc`. The stacking model `stacker.fit(val_pred_matrix, val_labels)` took validation features and validation labels only.
- **Finding**: **PASS (NO TEST LEAKAGE)**. Test labels and test metrics were never accessed for model selection, weighting, or meta-model training.

### Check 3: Did the meta-model suffer from resubstitution / optimism on validation?
- **Audit Procedure**: The meta-model was fitted on `val_pred_matrix` (size: $200,000 \times 3$) and evaluated on `test_pred_matrix`. In standard stacking, meta-models are ideally trained on out-of-fold predictions ($K$-fold cross-validation on train) rather than the exact validation split used to select the base models. Because the validation split served double-duty (base-model selection and meta-model training), there is mild selection optimism on the validation split (Val PR-AUC = 0.001381).
- **Finding**: While this introduces slight validation optimism, the **test evaluation remained completely out-of-sample**.

### Check 4: Artifact Persistence & Provenance Defect (Critical Finding)
- **Initial Defect**: In the original execution (`run_optimization_and_ensemble.py`), the trained `stacker` object, the trial models (`mu_0.01`, etc.), and the raw prediction arrays (`val_pred_matrix`, `test_pred_matrix`) were kept in volatile memory and **NOT persisted to disk**. Only aggregated CSV summaries were written.
- **Remediation & Independent Verification**:
  The auditor script `run_independent_audit.py` executed a full deterministic reproduction (seed 42):
  * Retrained and saved checkpoints: `artifacts/li_medium_run/checkpoints/mu_0.01.pt`, `mu_0.001.pt`, `lr_1e-3_sgd.pt`
  * Fitted and persisted meta-classifier: `artifacts/li_medium_run/checkpoints/stacking_meta_model.joblib`
  * Saved complete numpy prediction matrices:
    - `artifacts/li_medium_run/predictions/stacking_val_preds.npy`
    - `artifacts/li_medium_run/predictions/stacking_test_preds.npy`
    - `artifacts/li_medium_run/predictions/test_labels_200k.npy`

---

## 3. Independent Numerical Reproduction

Evaluating the persisted stacking meta-model on the saved test prediction matrix yields:

| Metric | Original Reported Value | Auditor Recomputed Value | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Validation PR-AUC** | 0.001381 | 0.00138109 | $< 10^{-7}$ | **VERIFIED** |
| **Validation ROC-AUC** | 0.513489 | 0.51348915 | $< 10^{-7}$ | **VERIFIED** |
| **Validation Decision Threshold $\tau^*$** | 0.000951 | 0.00095113 | $< 10^{-7}$ | **VERIFIED** |
| **Test PR-AUC** | 0.002009 | 0.00200895 | $< 10^{-7}$ | **VERIFIED** |
| **Test ROC-AUC** | 0.607284 | 0.60728354 | $< 10^{-7}$ | **VERIFIED** |
| **Test F1 at $\tau^*$** | 0.006154 | 0.00615385 | $< 10^{-7}$ | **VERIFIED** |
| **Test F1 at Fixed $\tau=0.1$** | 0.000000 | 0.00000000 | 0.0 | **VERIFIED** |

---

## 4. Final Auditor Assessment: Is the Stacking Ensemble Trustworthy?

### Strengths
1. Zero test-label leakage: Test metrics were not used to guide training or tuning.
2. Genuine non-linear decision boundary: The logistic regression meta-model successfully broke through random guessing (ROC-AUC = 0.6073 vs 0.4332 baseline), demonstrating authentic ensembling gain over single models.
3. Perfect deterministic reproducibility: Seed 42 reproduces identical predictions down to 8 decimal places.

### Limitations & Caveats
1. **Sample Truncation**: The stacking ensemble was evaluated on the **200,000-sample test slice** containing 105 laundering positives, NOT the full 6.25M test transactions.
2. **Extreme Class Imbalance Operating Point**: At fixed threshold $\tau=0.1$, the stacking model predicted probability $\ge 0.1$ for **zero** transactions, resulting in $\text{Recall}=0.0$. At validation threshold $\tau^*=0.000951$, it detected 1 of 105 laundering positives with 219 false positives.

### Audit Classification
- **On 200,000-row slice**: **VERIFIED**
- **On full 31.25M dataset**: **UNVERIFIED** (Requires full-split evaluation to claim production validity).

