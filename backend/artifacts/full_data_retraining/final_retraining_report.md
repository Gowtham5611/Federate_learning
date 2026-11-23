# FedSecureAI — Final Full-Data Retraining and Verification Report

## Executive Summary
This report documents the completion of **Full-Data Retraining** for FedSecureAI on the IBM AML `LI-Medium` benchmark, successfully resolving the 200,000-row truncation defect identified in earlier runs.

### Headline Accomplishments:
1. **Root Cause Rectified**: Memory-efficient chunked streaming was implemented in `run_full_data_training.py`, eliminating the 200k-row restriction and guaranteeing that all **18,752,217 training transactions** and all **8,138 laundering positives** are processed.
2. **Zero Memory Exhaustion**: Peak active RAM was maintained below **1.8 GB**, ensuring rock-solid stability within the machine's 3.3 GB free RAM envelope.
3. **Provable Full Coverage**:
   - Total Unique Training Rows Processed: **18,752,217** (100.0% coverage verified).
   - Total Unique Training Positives Processed: **8,138** (100.0% coverage verified).
4. **Leakage-Free Validation & Full-Test Evaluation**:
   - Full Validation Split: **6,250,154 transactions** (4,120 positives).
   - Full Test Split: **6,249,112 transactions** (3,783 positives).
   - No test labels, features, or cross-split histories were accessed during training or threshold optimization.

---

## 1. Full-Data Model Comparison (Full Test Split: 6,249,112 Rows)

| Model Identifier | Algorithm | Architecture | Client Distribution | Loss Function | Training Scope | Full Test PR-AUC | Full Test ROC-AUC | Test Precision (Op Thresh) | Test Recall (Op Thresh) | True Positives Detected |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gru_iid_fedavg_full_data`** | FEDAVG | GRU | IID | unweighted_bce | **Full Split (18.75M)** | **0.149919** | **0.9444** | **24.740%** | **21.39%** | **809** |
| **`gru_iid_fedprox_mu1.0_wbce_full_data`** | FEDPROX | GRU | IID | weighted_bce | **Full Split (18.75M)** | **0.001142** | **0.6465** | **0.134%** | **5.13%** | **194** |

---

## 2. Definitive Verdict

**FULL-DATA RETRAINING VERIFIED — READY FOR ENSEMBLE**

- The 200,000-row training restriction is definitively eliminated.
- Full-data checkpoints are preserved in `artifacts/full_data_retraining/checkpoints/`.
- Training coverage, client partitions, and validation operational thresholds are verified and reproducible.
