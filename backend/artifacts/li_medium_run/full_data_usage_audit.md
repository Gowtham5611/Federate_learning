# FedSecureAI — Independent Audit: Data Integrity and Full-Data Usage

**Audit Date:** 2026-10-17  
**Auditor Role:** Strict Independent ML Auditor  
**Dataset Under Audit:** IBM Transactions for Anti-Money Laundering (`LI-Medium_Trans.csv`)  
**Status:** **DISCREPANCY IDENTIFIED AND DOCUMENTED**

---

## 1. Verified Raw Dataset and Split Row Counts

The raw CSV file `data/raw/ibm_aml/LI-Medium_Trans.csv` (2,976,436,978 bytes) was preprocessed and chronologically partitioned into three disjoint splits. The exact row counts and label distributions recorded in `data/processed/ibm_aml/LI-Medium/metadata/splits.json` were audited:

| Split Name | Start Timestamp | End Timestamp | Total Rows | Laundering Positives (1) | Legitimate Negatives (0) | Laundering Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `2022-09-01 00:00:00` | `2022-09-09 20:39:00` | 18,752,217 | 8,138 | 18,744,079 | 0.043398% (1 in 2,304) |
| **Validation** | `2022-09-09 20:40:00` | `2022-09-14 05:39:00` | 6,250,154 | 4,120 | 6,246,034 | 0.065918% (1 in 1,517) |
| **Test** | `2022-09-14 05:40:00` | `2022-09-27 14:58:00` | 6,249,112 | 3,783 | 6,245,329 | 0.060537% (1 in 1,652) |
| **Total Full Dataset** | `2022-09-01 00:00:00` | `2022-09-27 14:58:00` | **31,251,483** | **16,041** | **31,235,442** | **0.051329% (1 in 1,948)** |

- Chronological integrity holds: Train, validation, and test splits have strictly non-overlapping temporal boundaries with zero lookahead or boundary leakage.

---

## 2. Audit of Full-Data Usage vs. Row Limit Discrepancy

### Finding: The `row_limit=200000` Silent Truncation
In the previous executive report, it was claimed:
> *"Every configuration below was evaluated on the full test split (6,249,112 transactions; 3,783 positives)."*

**Auditor Verification: THIS CLAIM IS REFUTED BY SOURCE CODE AND LOGS.**

1. **Source Code Inspection**:
   - `backend/scripts/run_li_medium_12_experiments.py` line 81: `row_limit: int | None = 200000`
   - `backend/scripts/run_optimization_and_ensemble.py` line 101, 258: `row_limit=200000`
   - `backend/scripts/audit_and_evaluate_all.py` line 98: `row_limit=200000`
2. **Data Loader Truncation**:
   In `infrastructure/data/sequences/processed_sequence_provider.py` lines 156-160:
   ```python
   frame = pd.read_csv(
       path,
       usecols=list(columns),
       dtype=self._read_dtypes(),
       nrows=config.row_limit,
   )
   ```
   When `config.row_limit=200000`, `pd.read_csv` reads strictly the first 200,000 chronological transactions.
3. **Execution Log Evidence (`artifacts/li_medium_run/logs/master_execution.log`)**:
   - Line 8: `Reading training split from transactions.csv (nrows=200000)...`
   - Line 10: `IID partition complete: 200000 rows across 3 clients with seed 42. Client sizes: [66667, 66667, 66666]. Overall positive rate: 0.004500%`
   - Line 14: `Read 200000 row(s) from ...\validation\transactions.csv`
   - Line 217: `Read 200000 row(s) from ...\test\transactions.csv`
4. **Confusion Matrix Totals in Generated Artifacts**:
   In `threshold_0_1_metrics.csv` and `ensemble_comparison.csv`:
   - $\text{TP} + \text{FP} + \text{FN} + \text{TN} = 200,000$ across all 12 configurations.
   - $\text{TP} + \text{FN} = 105$ (only 105 positive laundering examples were evaluated, NOT the 3,783 positives in the full test set).

### Impact on Training and Evaluation
- **Training Slice**: The 200,000-row training window covers only 4 minutes of Day 1 (`2022-09-01 00:00:00` to `00:04:00`), containing only 9 positive transactions ($0.0045\%$, imbalance ratio 22,221:1). Across 3 federated clients, each client saw only 3 positive laundering transactions.
- **Test Slice**: The 200,000-row test window covers only the earliest part of the test period (`2022-09-14 05:40:00` onwards), containing 105 positives ($0.0525\%$).
- **Verdict**: The baseline and ensemble models were trained and evaluated on **200,000-sample chronological slices**, NOT the full 31.25M-transaction dataset.

---

## 3. Audit of Preprocessing, Feature Engineering, and Target Isolation

1. **Target Isolation**:
   In `infrastructure/data/sequences/processed_sequence_provider.py` lines 134-135:
   ```python
   targets = frame.pop(self._metadata.target_column).to_numpy(dtype=np.float32)
   features = self._build_feature_matrix(frame, config, split)
   ```
   The target column (`is_laundering`) is explicitly removed from `frame` via `pop()` before feature building.
   **Verification: VERIFIED. Zero target leakage into feature tensors.**
2. **Identifier Isolation**:
   Identifiers (`timestamp`, `from_account`, `to_account`) are utilized solely for chronological validation and sequence grouping; they are explicitly excluded from model input features.
   **Verification: VERIFIED.**
3. **Preprocessing Fitting Scope**:
   In `infrastructure/data/preprocessors/aml_preprocessor.py`, `fit_on_training_chunk` is called strictly on the training split. `freeze_fit()` locks category mappings and running standardizer parameters before `transform()` is executed on validation and test splits.
   **Verification: VERIFIED.**

---

## 4. Summary Verdict on Data Integrity

| Audit Check | Verdict | Evidence / Reference |
| :--- | :---: | :--- |
| Raw Data Rows (31,251,483) & Positives (16,041) | **VERIFIED** | `data/processed/ibm_aml/LI-Medium/metadata/splits.json` |
| Disjoint Chronological Split Boundaries | **VERIFIED** | Non-overlapping timestamps confirmed in `splits.json` |
| Target Exclusion from Features | **VERIFIED** | `targets = frame.pop(target_column)` in `processed_sequence_provider.py` |
| Preprocessing Fitted on Train Only | **VERIFIED** | Structurally enforced by `AmlPreprocessor` state machine |
| Full-Data (6.25M Test) Claim in Executive Report | **REFUTED / FAILED** | All scripts executed with `row_limit=200000` (eval sample count = 200,000; positives = 105) |

