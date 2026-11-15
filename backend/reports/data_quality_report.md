# FedSecureAI — Dataset Quality & Provenance Audit Report

**Dataset Target:** IBM AML `HI-Small`  
**Audit Execution Time:** `2026-10-27T17:39:25.402489+00:00`  
**Audit Status:** `PASSED_RESEARCH_GRADE_STANDARDS`  

---

## 1. Dataset Integrity & Schema Conformance

- **Total Measured Transactions:** 5,078,345
- **Raw Schema Verification:** Verified 11 columns (11 columns matching verified IBM AML specification).
- **Processed Input Dimension:** 57 dimensions (26 base features expanded with nominal one-hot categorical vectors).
- **Missing / Null Values:** Zero nulls across all transactions.
- **Invalid Numerical Values:** Zero negative amounts, zero invalid currencies.
- **Duplicate Transactions:** 0 exact duplicate transaction records detected.

---

## 2. Chronological Split & Class Distribution

| Split | Row Count | Negative Rows | Positive Rows (Laundering) | Imbalance Ratio | Positive Prevalence | Temporal Span |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 3,047,216 | 3,044,917 | 2,299 | 1324.5:1 | 0.0754% | `2022-09-01 00:20:00` to `2022-09-11 12:47:00` |
| **VAL** | 1,015,739 | 1,014,301 | 1,438 | 705.4:1 | 0.1416% | `2022-09-11 12:48:00` to `2022-09-14 23:59:00` |
| **TEST** | 1,015,390 | 1,013,950 | 1,440 | 704.1:1 | 0.1418% | `2022-09-15 00:00:00` to `2022-09-18 23:59:00` |

> [!IMPORTANT]
> **Test Set Distribution Preservation:** The final evaluation test split retains the realistic natural banking imbalance of 704.1:1 (0.1418% positive). No artificial oversampling, SMOTE, or synthetic positives are injected into the authoritative test set.

---

## 3. Strict Temporal Leakage & Causality Audit

- **Train Span:** `2022-09-11 12:47:00` (Cutoff)
- **Validation Span:** `2022-09-11 12:48:00` to `2022-09-14 23:59:00`
- **Test Span:** `2022-09-15 00:00:00` (Start)
- **Train-to-Validation Causality:** `PASSED (Strictly Non-Overlapping)`
- **Validation-to-Test Causality:** `PASSED (Strictly Non-Overlapping)`
- **Overall Causality Verdict:** `STRICT_CAUSAL_ORDERING_PRESERVED`

---

## 4. Cryptographic Provenance & Artifact Hashes

| File Name | SHA-256 Digest |
| :--- | :--- |
| `transformers.json` | `d357e35b7c876572801fe8b0156f0b27532dc170a0928ede5ceb34fc4e773e2b` |

---

## 5. Auditor Recommendations

- Preserve original 0.14% positive test prevalence in all benchmark comparisons.
- Ensure all preprocessing scalers remain frozen to training statistics.
- Maintain chronological split boundaries without cross-split account sequence shuffling.