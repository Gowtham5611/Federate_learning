# FedSecureAI — CSV Batch Evaluation Diagnostic Report (Step 1)

**Date**: 2026-11-03  
**Branch**: `fix/eval-and-research-hardening`  
**Evaluation Engine**: `backend/interfaces/api/services/csv_evaluation_service.py` (`CsvEvaluationService`)  
**Frozen Model**: HI-Small LSTM Champion (Threshold $\theta^* = 0.337784$)  
**Baseline Checkpoints SHA-256 Audit**: 273/273 files (100%) bit-identical verified.

---

## 1. Reproduction Methodology & Dump Inventory

Each of the six test datasets was processed through the identical code path used by the frontend UI (`CsvEvaluationService.evaluate_csv_bytes` via the `/evaluate/csv` endpoint). Raw response payloads have been dumped in full to `results/diagnostics/{filename}.json`:

1. `data/demo/demo_ibm_aml_500.csv` $\to$ `results/diagnostics/demo_ibm_aml_500.csv.json`
2. `data/demo/demo_ibm_aml_5000.csv` $\to$ `results/diagnostics/demo_ibm_aml_5000.csv.json`
3. `data/demo_slices/test_slice_500.csv` $\to$ `results/diagnostics/test_slice_500.csv.json`
4. `data/demo_slices/test_slice_5000.csv` $\to$ `results/diagnostics/test_slice_5000.csv.json`
5. `data/demo_slices/test_slice_50000.csv` $\to$ `results/diagnostics/test_slice_50000.csv.json`
6. `data/demo/demo_no_labels_500.csv` (label column removed) $\to$ `results/diagnostics/demo_no_labels_500.csv.json`

---

## 2. File-by-File Diagnostic Findings

### File 1: `demo_ibm_aml_500.csv`
- **Total Rows**: 500
- **Ground Truth Labels**: Present (3 positives, prevalence $0.60\%$)
- **Score Range**: $\min = 0.000004$, $\max = 0.007693$, $\text{mean} = 0.000258$
- **Alerts at $\theta^* = 0.337784$**: **0 alerts** ($\text{TP}=0, \text{FP}=0, \text{FN}=3, \text{TN}=497$)
- **JSON Null Keys in `evaluation_metrics`**:
  - `precision`: `null` (reason: $\text{TP}+\text{FP}=0$, division by zero)
  - `f1_score`: `null` (**INCORRECT**: set to null in backend code because precision was null, whereas mathematically $F_1 = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}} = \frac{0}{3} = 0.0$)
  - `f2_score`: `null` (**INCORRECT**: mathematically $F_2 = \frac{5\text{TP}}{5\text{TP} + 4\text{FN} + \text{FP}} = \frac{0}{12} = 0.0$)
  - `false_discovery_rate`: `null` (expected: $\text{FP}/(\text{TP}+\text{FP})$ is $0/0$)
- **Frontend Display Impact**:
  - **F1 Card**: Displays `"Undefined (0 Alerts)"` instead of numeric `0.0000`.
  - **Precision Card**: Displays `"Undefined (0 Alerts)"` (mathematically correct, but confusing without budget context).
  - **Recall Card**: Displays `0.0000`.
  - **Accuracy Card**: Displays `0.9940`.
  - **Specificity / Balanced Accuracy / MCC / Cohen's Kappa**: Present in JSON (`specificity=1.0`, `balanced_acc=0.50`, `mcc=0.0`, `kappa=0.0`), but frontend lacked dedicated cards to display them.
  - **PR / ROC Curves**: Rendered, but operating point point is anchored at $(0, 0)$ with zero alerts.

---

### File 2: `demo_ibm_aml_5000.csv`
- **Total Rows**: 5,000
- **Ground Truth Labels**: Present (35 positives, prevalence $0.70\%$)
- **Score Range**: $\min = 0.000003$, $\max = 0.234477$, $\text{mean} = 0.001348$
- **Alerts at $\theta^* = 0.337784$**: **0 alerts** ($\text{TP}=0, \text{FP}=0, \text{FN}=35, \text{TN}=4965$)
- **JSON Null Keys in `evaluation_metrics`**:
  - `precision`: `null` (0 alerts generated)
  - `f1_score`: `null` (incorrect backend logic: should be $0.0$)
  - `f2_score`: `null` (incorrect backend logic: should be $0.0$)
  - `false_discovery_rate`: `null` ($0/0$)
- **Frontend Display Impact**:
  - F1 card shows `"Undefined (0 Alerts)"`.
  - Because max probability ($0.2345$) is below the canonical threshold ($0.3378$), all 35 true positives are classified as false negatives.
  - The analyst has no visibility into model ranking capability unless top-$k$ alert budget or tuned operating points are provided.

---

### File 3: `test_slice_500.csv`
- **Total Rows**: 500
- **Ground Truth Labels**: Present (1 positive, prevalence $0.20\%$)
- **Score Range**: $\min = 0.000014$, $\max = 0.236938$, $\text{mean} = 0.013033$
- **Alerts at $\theta^* = 0.337784$**: **0 alerts** ($\text{TP}=0, \text{FP}=0, \text{FN}=1, \text{TN}=499$)
- **JSON Null Keys in `evaluation_metrics`**:
  - `precision`: `null`
  - `f1_score`: `null` (should be $0.0$)
  - `f2_score`: `null` (should be $0.0$)
  - `false_discovery_rate`: `null`
  - `pr_auc_ci_lower`, `pr_auc_ci_upper`: `null` (with only 1 positive sample, bootstrap resampling generates bootstrap replicates with 0 positives where PR-AUC is degenerate).
- **Frontend Display Impact**:
  - F1 card shows `"Undefined (0 Alerts)"`.
  - Bootstrap CI text shows `[N/A, N/A]` or omitted without an informative badge indicating $positives < 2$.

---

### File 4: `test_slice_5000.csv`
- **Total Rows**: 5,000
- **Ground Truth Labels**: Present (12 positives, prevalence $0.24\%$)
- **Score Range**: $\min = 0.000003$, $\max = 0.501582$, $\text{mean} = 0.007629$
- **Alerts at $\theta^* = 0.337784$**: **3 alerts** ($\text{TP}=3, \text{FP}=0, \text{FN}=9, \text{TN}=4988$)
- **JSON Null Keys in `evaluation_metrics`**:
  - None! All core performance metrics are numeric:
    - $\text{Precision} = 1.0000$
    - $\text{Recall} = 0.2500$
    - $F_1 = 0.4000$
    - $F_2 = 0.2941$
    - $\text{Accuracy} = 0.9982$
    - $\text{Specificity} = 1.0000$
    - $\text{Balanced Accuracy} = 0.6250$
    - $\text{MCC} = 0.4996$
    - $\text{Cohen's Kappa} = 0.3994$
    - $\text{PR-AUC} = 0.4337$ (Lift $180.7\times$)
    - $\text{ROC-AUC} = 0.8793$
- **Frontend Display Impact**:
  - All cards populate cleanly.
  - Validity banner highlights `LOW_HISTORY` and `SMALL_SAMPLE` ($12 < 30$ positives).

---

### File 5: `test_slice_50000.csv`
- **Total Rows**: 50,000
- **Ground Truth Labels**: Present (76 positives, prevalence $0.152\%$)
- **Score Range**: $\min = 0.000002$, $\max = 0.757345$, $\text{mean} = 0.005523$
- **Alerts at $\theta^* = 0.337784$**: **23 alerts** ($\text{TP}=17, \text{FP}=6, \text{FN}=59, \text{TN}=49918$)
- **JSON Null Keys in `evaluation_metrics`**:
  - None.
  - $\text{Precision} = 0.7391$, $\text{Recall} = 0.2237$, $F_1 = 0.3434$, $F_2 = 0.2599$.
  - $\text{PR-AUC} = 0.3014$, $\text{ROC-AUC} = 0.9288$.
- **Frontend Display Impact**:
  - Fully populated metric cards and confusion matrix.

---

### File 6: `demo_no_labels_500.csv`
- **Total Rows**: 500
- **Ground Truth Labels**: **Absent** (`has_labels: false`)
- **Score Range**: $\min = 0.000004$, $\max = 0.007693$, $\text{mean} = 0.000258$
- **Alerts at $\theta^* = 0.337784$**: 0 alerts
- **JSON Response**:
  - `has_labels`: `false`
  - `evaluation_metrics`: `null`
- **Frontend Display Impact**:
  - Frontend displayed an amber alert box stating *"Unlabeled Transaction Batch: Ground-truth labels were not provided. Evaluation metrics cannot be calculated."*
  - The UI omitted score distribution statistics, risk decile tables, and top-risk suspect cards that are valid and meaningful for unlabeled batches in operational AML deployment.

---

## 3. Metric Nullness & UI Blank Mapping Matrix

| File | Rows | Positives | Alerts | $F_1$ in JSON | Precision in JSON | Recall in JSON | UI Blank / Issue Cause |
|---|---|---|---|---|---|---|---|
| `demo_ibm_aml_500.csv` | 500 | 3 | 0 | `null` | `null` | `0.0` | $F_1$ set to `null` due to `tp+fp==0`. Specificity, Balanced Acc, MCC, Kappa omitted from UI cards. |
| `demo_ibm_aml_5000.csv` | 5,000 | 35 | 0 | `null` | `null` | `0.0` | $F_1$ set to `null` due to `tp+fp==0`. High threshold causes 0 alerts; no alternative operating points shown. |
| `test_slice_500.csv` | 500 | 1 | 0 | `null` | `null` | `0.0` | $F_1$ set to `null`. Bootstrap CI null because positives $< 2$. |
| `test_slice_5000.csv` | 5,000 | 12 | 3 | `0.4000` | `1.0000` | `0.2500` | Full metrics present. Low sample notice ($12 < 30$) needed. |
| `test_slice_50000.csv` | 50,000 | 76 | 23 | `0.3434` | `0.7391` | `0.2237` | Full metrics present. |
| `demo_no_labels_500.csv` | 500 | 0 (Unlabeled) | 0 | `null` | `null` | `null` | No labels present. Score-only operational mode needed. |

---

## 4. Exact Root Causes Identified

1. **Flawed Nullness Logic in `evaluation/metrics/__init__.py` and `csv_evaluation_service.py`**:
   - In both files, $F_1, F_2$, MCC, and Cohen's Kappa were gated on `precision is None` or `denom_sq <= 0`.
   - By strict mathematical definition:
     $$F_1 = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}} = \frac{0}{0 + 0 + \text{FN}} = 0.0 \quad (\text{when } \text{FN} > 0)$$
     $$F_2 = \frac{5\text{TP}}{5\text{TP} + 4\text{FN} + \text{FP}} = 0.0 \quad (\text{when } \text{FN} > 0)$$
     $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = 0.0 \quad (\text{when } \text{FN} > 0)$$
     $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = 1.0 \quad (\text{when } \text{FP} = 0, \text{TN} > 0)$$
     $$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2} = 0.50$$
     $$\text{MCC} = \frac{0 - 0}{\sqrt{0 \cdot \dots}} = 0.0 \quad (\text{zero correlation when predictions are constant})$$
     $$\text{Cohen's Kappa} = 0.0 \quad (\text{agreement matches chance baseline})$$
   - Only $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ and $\text{FDR} = \frac{\text{FP}}{\text{TP} + \text{FP}}$ have a genuine $0/0$ undefined form when alerts $= 0$.

2. **Single Operating Point Constraint**:
   - The UI currently only evaluates at the canonical frozen threshold ($\theta^* = 0.3378$).
   - On short slices or files without multi-step account history, max sequence scores rarely cross $0.3378$, producing 0 alerts despite strong ranking performance (e.g. PR-AUC $= 0.1429$, ROC-AUC $= 0.7458$).
   - The system must provide three clear operating points:
     - `frozen`: canonical threshold $\theta^* = 0.337784$.
     - `alert_budget`: top-$k$ alerts per 10,000 transactions ($k \in \{10, 50, 100\}$).
     - `best_f1_on_file`: what-if threshold maximizing $F_1$ on the file.

3. **Incomplete Score-Only Experience for Unlabeled Uploads**:
   - When no ground-truth column is found, the system should gracefully transition to **Score-Only Mode**: suppress supervised metric cards, emphasize score distribution, alert count at operating thresholds, and top-$k$ high-risk suspect transactions.

4. **Missing Small-Sample Statistical Guidance**:
   - For slices with $n < 5,000$ or positives $< 30$, expected positives under training prior ($n \times 0.00177$) and bootstrap 95% confidence intervals must be prominently displayed to prevent analysts from drawing invalid conclusions from high-variance samples.

5. **UI Block Coupling**:
   - If one metric or curve is absent, the remaining cards (Score Distribution, Predictions Table, Confusion Matrix, Validity Banner) must render completely and independently.

---
*End of Step 1 Diagnosis. Proceeding to Step 2-7 Implementation.*
