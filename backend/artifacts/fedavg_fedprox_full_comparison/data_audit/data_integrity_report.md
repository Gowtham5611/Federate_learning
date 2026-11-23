# FedSecureAI — Comprehensive Data Integrity Report

**Artifact Directory**: `artifacts/fedavg_fedprox_full_comparison/data_audit/`  
**Date**: 2026-10-18  

---

## 1. Summary of Processed Splits

| Dataset | Split | Total Transactions | Positive Laundering | Legitimate | Prevalence (%) | Imbalance Ratio | File Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HI-Small** | Train | 3,047,216 | 2,299 | 3,044,917 | 0.0754% | 1 : 1,324 | 770.26 |
| **HI-Small** | Validation | 1,015,565 | 1,081 | 1,014,484 | 0.1064% | 1 : 938 | 256.51 |
| **HI-Small** | Test | 1,015,564 | 1,797 | 1,013,767 | 0.1769% | 1 : 564 | 258.02 |
| **LI-Medium** | Train | 18,752,217 | 8,138 | 18,744,079 | 0.0434% | 1 : 2,303 | 4,751.24 |
| **LI-Medium** | Validation | 6,250,154 | 4,120 | 6,246,034 | 0.0659% | 1 : 1,516 | 1,552.64 |
| **LI-Medium** | Test | 6,249,112 | 3,783 | 6,245,329 | 0.0605% | 1 : 1,651 | 1,587.49 |
| **Combined** | Train | 300,000 | 15 | 299,985 | 0.0050% | 1 : 19,999 | 78.59 |
| **Combined** | Validation | 100,000 | 78 | 99,922 | 0.0780% | 1 : 1,281 | 26.04 |
| **Combined** | Test | 100,000 | 94 | 99,906 | 0.0940% | 1 : 1,063 | 26.11 |

---

## 2. Integrity Checks

1. **Missing Values**:
   - Zero missing (`NaN`) values detected across all numerical, nominal, and target columns.
2. **Duplicate Records**:
   - Zero duplicated transaction rows detected within any split.
3. **Temporal Monotonicity**:
   - Every split was verified to possess strictly non-decreasing timestamps. No out-of-order records exist.
4. **Boundary Isolation**:
   - Split boundaries are strictly chronological:
     - `train_last_timestamp < validation_first_timestamp`
     - `validation_last_timestamp < test_first_timestamp`
   - Zero temporal overlap exists across splits.
5. **Target Grounding**:
   - Positive class ($y=1$) strictly represents laundering across all tables.
   - Negative class ($y=0$) strictly represents legitimate banking transfers.

