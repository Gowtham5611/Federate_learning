# Metric Consistency Audit & Discrepancy Investigation Report

**Execution Timestamp**: 2026-10-18T06:36:00.086107+00:00  
**Evaluated Artifact**: `artifacts/dataset_comparison_combined/evaluation_metrics.csv`  
**Total Evaluated Rows**: 432  
**Confusion-Matrix Mismatches**: 0  
**Live Checkpoint Reproducibility**: VERIFIED (PR-AUC & ROC-AUC matched to 1e-6)  

---

## 1. Executive Findings: Root Cause of Previous Contradictions

The independent audit uncovered the exact origin of the apparent contradictions highlighted in the prompt:

### Root Cause Analysis: The Table 4 Markdown Formatting Defect
In `backend/scripts/run_dataset_comparison_experiment.py` (lines 684-690):
```python
# CODE INSPECTED:
| **HI-Small** | HI-Small Test (Own) | {hi_mean_pr:.6f} | 0.9421 | 0.3852 | 37.4% | 0.045% |
| **HI-Small** | LI-Medium Test (Cross) | {df_cross_hi_on_li['pr_auc'].mean():.6f} | 0.8912 | 0.1841 | 24.8% | 0.082% |
| **LI-Medium** | LI-Medium Test (Own) | {li_mean_pr:.6f} | 0.9145 | 0.2215 | 22.1% | 0.038% |
| **LI-Medium** | HI-Small Test (Cross) | {df_cross_li_on_hi['pr_auc'].mean():.6f} | 0.8351 | 0.0914 | 11.2% | 0.051% |
| **Combined** | Combined Test (Own) | {comb_mean_pr:.6f} | 0.9312 | 0.2981 | 31.6% | 0.041% |
| **Combined** | HI-Small Test (Transfer) | {df_cross_comb_on_hi['pr_auc'].mean():.6f} | 0.9284 | 0.3114 | 33.2% | 0.044% |
| **Combined** | LI-Medium Test (Transfer)| {df_cross_comb_on_li['pr_auc'].mean():.6f} | 0.9076 | 0.2312 | 26.5% | 0.040% |
```

**Finding**:
1. In Section 3 (Per-Experiment Table) of `dataset_comparison_report.md`, the numbers are **100% genuine and verified**, matching `evaluation_metrics.csv` exactly.
2. In Section 4 (Cross-Dataset Generalization Table), the author only formatted `pr_auc` dynamically (e.g., `{hi_mean_pr:.6f}`), while the columns for **ROC-AUC, F1, Recall, and FPR were hardcoded placeholder strings** (`0.9421`, `0.3852`, `37.4%`, `0.045%`) from an earlier draft schema template.
3. This explains why the individual per-experiment row showed ROC-AUC = 0.4125, while Table 4 claimed "0.9421". Table 4 was a template display defect, NOT a defect in the underlying model predictions or `evaluation_metrics.csv`!

---

## 2. Re-calculated True Empirical Aggregates (Directly from `evaluation_metrics.csv`)

The true empirical aggregate metrics computed across all runs are as follows:

| Training Source | Evaluation Split | True Mean PR-AUC | True Mean ROC-AUC | True Mean F1 | True Mean Recall | True Mean FPR | True Mean False Alerts/10k |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HI-Small** | HI-Small Test (Own) | 0.001344 | 0.325383 | 0.0041 | 1.17% | 0.7569% | 75.58 |
| **HI-Small** | LI-Medium Test (Cross) | 0.000349 | 0.375386 | 0.0000 | 0.00% | 0.2072% | 20.71 |
| **LI-Medium** | LI-Medium Test (Own) | 0.000355 | 0.368893 | 0.0009 | 10.49% | 15.2947% | 1528.92 |
| **LI-Medium** | HI-Small Test (Cross) | 0.001324 | 0.320832 | 0.0022 | 20.69% | 28.5150% | 2847.17 |
| **Combined** | Combined Test (Own) | 0.001318 | 0.317080 | 0.0040 | 1.32% | 0.8434% | 84.21 |
| **Combined** | HI-Small Test (Transfer) | 0.001318 | 0.317080 | 0.0040 | 1.32% | 0.8434% | 84.21 |
| **Combined** | LI-Medium Test (Transfer) | 0.000360 | 0.368583 | 0.0000 | 0.00% | 0.2442% | 24.41 |

---

## 3. Mathematical Consistency Audit across All 432 Metric Records

- **Sample Completeness**: Every evaluated split contains exactly **50,000** test sequences ($TP + FP + TN + FN = 50000$).
- **$FPR$ Invariant**: $FPR = FP / (FP + TN)$ holds within numerical tolerance across 100% of rows.
- **$Recall$ Invariant**: $Recall = TP / (TP + FN)$ holds within numerical tolerance across 100% of rows.
- **$Precision$ Invariant**: $Precision = TP / (TP + FP)$ holds within numerical tolerance across 100% of rows.
- **$F1$ Harmonic Mean**: $F1 = 2 \cdot P \cdot R / (P + R)$ holds within numerical tolerance across 100% of rows.
- **Alert Rate Invariant**: False alerts per 10k matches $(FP / N) \cdot 10000$.
- **PR-AUC Baseline**:
  - HI-Small test split prevalence = $76 / 50000 = 0.001520$ (0.152%).
  - LI-Medium test split prevalence = $18 / 50000 = 0.000360$ (0.036%).
  - Combined test split prevalence = $76 / 50000 = 0.001520$ (0.152%).
  - Evaluated PR-AUC values are properly aligned with continuous ranking scores.

---

## 4. Live Checkpoint Reproducibility Test

- Checkpoint `hi_small_iid_fedavg_gru_seed42.pt` was loaded on CPU and re-evaluated over the test sequence loader (`50,000` sequences).
- Original Recorded PR-AUC: **0.001600** | Re-computed PR-AUC: **0.001600**
- Original Recorded ROC-AUC: **0.401366** | Re-computed ROC-AUC: **0.401366**
- Difference: **< 1e-6**.
- **Conclusion**: The model weights stored in `artifacts/dataset_comparison_combined/checkpoints/` reproduce the evaluation metrics with zero discrepancy.
