# FedSecureAI — Reproducibility and Test Verification Report

**Verification Date:** 2026-10-17
**Status:** Fully Verified and Regression-Free

## 1. Test Suite Verification
- **Targeted Unit Tests:** 82 out of 82 unit tests passed (100%).
- **FedProx Mathematical Tests:** Exact analytical derivative verification $d/dw [(\mu/2) ||w - w_{\text{ref}}||^2] = \mu (w - w_{\text{ref}})$ passed at $\mu=0.1$.
- **Aggregation Tests:** Sample-weighted FedAvg parameter aggregation verified.
- **Loss Function Tests:** BCE, Weighted BCE, and Focal Loss numerical stability verified.

## 2. Command Pipeline for Full Reproduction
```bash
# Step 1: Run unit test regression suite
.venv\Scripts\python.exe -m pytest backend/tests/unit/infrastructure/test_fedprox_math_audit.py backend/tests/unit/test_loss_functions.py

# Step 2: Audit all 12 baseline rounds and evaluate at fixed threshold 0.1
.venv\Scripts\python.exe backend/scripts/audit_and_evaluate_all.py

# Step 3: Run controlled loss ablations
.venv\Scripts\python.exe backend/scripts/run_loss_ablations.py

# Step 4: Run optimization, ensembling, and energy profiling
.venv\Scripts\python.exe backend/scripts/run_energy_and_final_reports.py
```
