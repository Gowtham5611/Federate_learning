# FedSecureAI — LI-Medium Raw Dataset Audit & Verification Report

**Audit Date:** 2026-10-17  
**Raw Source:** `data/raw/ibm_aml/LI-Medium_Trans.csv`  
**Measurement Method:** Full sequential chunked stream (chunk size 500,000 rows, 100.5s scan time). No sampling or assumptions.

---

## 1. Physical File Metrics & Schema
- **File Size:** `2,976,436,978` bytes (Exact match to official Kaggle repository).
- **Total Rows:** `31,251,483` transactions.
- **Verified Columns (11)**:
  1. `Timestamp`: Transaction time (string `YYYY/MM/DD HH:MM`).
  2. `From Bank`: Originating bank identifier.
  3. `Account`: Originating sender account identifier (`Account`).
  4. `To Bank`: Destination bank identifier.
  5. `Account.1`: Destination receiver account identifier (`Account.1`).
  6. `Amount Received`: Numerical received amount.
  7. `Receiving Currency`: Currency code for received funds.
  8. `Amount Paid`: Numerical paid amount.
  9. `Payment Currency`: Currency code for paid funds.
  10. `Payment Format`: Transaction payment method (e.g. Credit Card, Wire, ACH, Cash, Cheque, Reinvestment).
  11. `Is Laundering`: Ground-truth binary fraud label (`0` or `1`).

---

## 2. Ground-Truth Class Distribution
- **Total Transactions:** `31,251,483`
- **Legitimate (Class 0):** `31,235,442` (`99.9487%`)
- **Money Laundering (Class 1):** `16,041` (`0.0513%`)
- **Imbalance Ratio:** ~1,947 legitimate transactions per 1 laundering transaction.
- **Timestamp Coverage:** `2022/09/01 00:00` to `2022/09/27 14:58` (~27 calendar days).

---

## 3. Disambiguation & Integrity
- Both originating account and receiving account headers (`Account` and `Account.1`) are unambiguously preserved.
- No null values detected in target or timestamp columns.
- `HI-Small_Trans.csv` (475,664,282 bytes, 5,078,345 rows) remains completely intact in `data/raw/ibm_aml/`.
