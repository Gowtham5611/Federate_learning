# FedSecureAI — LI-Medium 60:20:20 Chronological Data Splitting Report

**Report Date:** 2026-10-17  
**Raw Dataset:** `data/raw/ibm_aml/LI-Medium_Trans.csv` (31,251,483 rows, 2,976,436,978 bytes)  
**Processed Directory:** `data/processed/ibm_aml/LI-Medium/`  
**Execution Runtime:** 2,177.6 seconds (~36.3 minutes)  
**Verification Status:** Byte-for-byte reconciled. 31,251,483 rows read, 31,251,483 rows written, 0 rows removed.  

---

## 1. Measured Split Allocations & Class Distribution

| Split Name | Row Count | Target % | Actual Share % | Legitimate (0) | Laundering (1) | Positive Rate (%) | First Timestamp | Last Timestamp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **18,752,217** | 60.0% | **60.0042%** | 18,744,079 | **8,138** | 0.0434% | 2022-09-01 00:00:00 | 2022-09-09 20:39:00 |
| **Validation** | **6,250,154** | 20.0% | **19.9995%** | 6,246,034 | **4,120** | 0.0659% | 2022-09-09 20:40:00 | 2022-09-14 05:39:00 |
| **Test** | **6,249,112** | 20.0% | **19.9962%** | 6,245,329 | **3,783** | 0.0605% | 2022-09-14 05:40:00 | 2022-09-27 14:58:00 |
| **Total** | **31,251,483** | 100.0% | **100.0000%** | **31,235,442** | **16,041** | **0.0513%** | 2022-09-01 00:00:00 | 2022-09-27 14:58:00 |

---

## 2. Chronological Integrity & Leakage Invariants
1. **Strict Chronological Ordering**:
   - `Train Max Timestamp` (2022-09-09 20:39:00) $<$ `Validation Min Timestamp` (2022-09-09 20:40:00).
   - `Validation Max Timestamp` (2022-09-14 05:39:00) $<$ `Test Min Timestamp` (2022-09-14 05:40:00).
   - Zero timestamp overlap across splits.
2. **Zero Information Leakage**:
   - Categorical encoders (payment format, payment currency, receiving currency) and continuous feature standardizers were fitted exclusively on the `train` split.
   - Encoders and running scalers were frozen before the first row of validation was transformed.
3. **Artifacts Generated**:
   - `data/processed/ibm_aml/LI-Medium/train/transactions.csv` (4.98 GB)
   - `data/processed/ibm_aml/LI-Medium/validation/transactions.csv` (1.63 GB)
   - `data/processed/ibm_aml/LI-Medium/test/transactions.csv` (1.66 GB)
   - `data/processed/ibm_aml/LI-Medium/metadata/`: `schema.json`, `feature_metadata.json`, `transformers.json`, `splits.json`, `preprocessing_report.json`.
