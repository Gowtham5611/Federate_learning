# FedSecureAI Phase 17: Professional Research Dashboard & CSV Evaluation Report

**Document ID:** `results/phase17/phase17_implementation_report.md`  
**Phase:** 17 — Professional Research Dashboard Reconstruction & Batch CSV Evaluation  
**Date:** 2026-10-27  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, Drift-Aware Federated AML Detection  
**Status:** COMPLETE & VERIFIED (118/118 Tests Passed)

---

## 1. Executive Summary

Phase 17 successfully reconstructs the FedSecureAI frontend from a demo interface into an enterprise-grade AIML research platform and delivers an end-to-end, dataset-separated batch CSV evaluation and inference engine.

All 27 trained neural architectures, round-level federated convergence curves, empirical ablation findings, and axiomatic Integrated Gradients attributions are backed directly by authoritative Phase 7–16 on-disk artifacts without data fabrication.

Zero model retraining was conducted; all frozen decision thresholds (`0.3378` for HI-Small, `0.2385` for LI-Medium) and PyTorch checkpoint weights were strictly preserved.

---

## 2. File & Component Changes Inventory

### Backend Components Added / Modified
- [`backend/interfaces/api/services/csv_evaluation_service.py`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/backend/interfaces/api/services/csv_evaluation_service.py):
  New end-to-end service for file validation, dataset-specific preprocessing (categorical encoding and standardizer restoration from `transformers.json`), previous-row index sequence assembly `(N, 10, 57)`, forward inference through champion models, and label-aware metrics computation.
- [`backend/interfaces/api/routes/evaluation.py`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/backend/interfaces/api/routes/evaluation.py):
  New route providing `POST /evaluate/csv` accepting `CsvEvaluateRequest` (JSON payload, raw body, or multipart).
- [`backend/interfaces/api/routes/research.py`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/backend/interfaces/api/routes/research.py):
  New routes providing:
  - `GET /research/overview`: High-level verified dataset and model metadata.
  - `GET /research/models`: Full master benchmark matrix across all 27 models.
  - `GET /research/curves`: 120 round-by-round convergence datapoints (val PR-AUC, val loss).
  - `GET /research/thresholds`: Operational threshold tradeoff matrix.
  - `GET /research/ablation`: Empirical ablation findings (FedProx collapse, Non-IID drift, Ensembling, Calibration).
  - `GET /research/sample-csv/{dataset}`: One-click reference test slices.
- [`backend/interfaces/api/app.py`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/backend/interfaces/api/app.py):
  Registered `evaluation_router` and `research_router`; updated CORS allow-list for ports 3000 and 5173.
- [`tests/test_phase17_csv_evaluation.py`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/tests/test_phase17_csv_evaluation.py):
  11 comprehensive unit and integration tests covering raw format, processed format, label-aware metrics, unlabeled transactions, error handling, and research routes.

### Frontend Components Added / Modified
- [`frontend/src/types/api.ts`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/types/api.ts):
  Added TypeScript interfaces for `CsvEvaluateResponse`, `EvaluationMetrics`, `ConfusionMatrixData`, `TransactionPredictionItem`, `BenchmarkModelRecord`, `ConvergenceCurvePoint`, `ThresholdTradeoffPoint`, and `AblationSummaryResponse`.
- [`frontend/src/services/apiService.ts`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/services/apiService.ts):
  Added client methods: `evaluateCsv()`, `getResearchOverview()`, `getResearchModels()`, `getConvergenceCurves()`, `getThresholdTradeoffs()`, `getAblationSummary()`, and `getSampleCsvUrl()`.
- [`frontend/src/components/Header.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/components/Header.tsx):
  Rebuilt header with 6 clear research navigation tabs, live backend readiness indicator, and dataset pipeline switcher (`HI-Small` vs `LI-Medium`).
- [`frontend/src/pages/ResearchOverviewPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/ResearchOverviewPage.tsx):
  New Section A page detailing project purpose, federated banking topology, HI-Small vs LI-Medium pipeline statistics, champion models, and architectural pillars.
- [`frontend/src/pages/ModelRegistryPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/ModelRegistryPage.tsx):
  New Section B page providing filterable, searchable inventory of all 27 models, highlighting serving champions, parameter counts, decision thresholds, and test metrics.
- [`frontend/src/pages/ExperimentAnalysisPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/ExperimentAnalysisPage.tsx):
  New Section C & F page featuring interactive SVG convergence curves (5 communication rounds), FedProx minority suppression analysis, Non-IID skew collapse cards, hybrid ensembling $\alpha^*$ search, and calibration gains.
- [`frontend/src/pages/CsvEvaluationPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/CsvEvaluationPage.tsx):
  New Section D & E page: complete drag-and-drop CSV upload workflow, 1-click test slice loaders, live data preview, label-aware supervised metrics card with 2x2 Confusion Matrix, unlabeled batch notifications, transaction table with search and filtering, "Explain Transaction" button per row, and CSV predictions export.
- [`frontend/src/pages/SingleInferenceExplainPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/SingleInferenceExplainPage.tsx):
  Enhanced Section G page: accepts passed transactions from the CSV evaluator or demo samples, runs forward scoring, and renders PyTorch-native Integrated Gradients waterfall attribution charts.
- [`frontend/src/pages/AuditTrailPage.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/pages/AuditTrailPage.tsx):
  New Section H page displaying privacy-preserving transaction scoring audit log with export capability.
- [`frontend/src/App.tsx`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/frontend/src/App.tsx):
  Updated root application component integrating all 6 research pages, managing application state, cross-page navigation, and backend health status.

---

## 3. CSV Schema Specifications

The batch evaluation engine supports two standard input schemas:

### Format A — Raw IBM AML Transaction Format (11 Columns)
```csv
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0
```
- **Preprocessing Pipeline Applied:**
  1. Cleaning: drops rows with missing values, non-parseable timestamps, or negative amounts.
  2. Chronological Sort: stable sort by timestamp.
  3. Feature Engineering: derives cyclical temporal features (`minute_of_day_sin/cos`, `day_of_week_sin/cos`, `is_weekend`, `is_night`), transaction features (`amount_mismatch`, `same_currency`, `same_bank`, `self_transfer`), and per-account causal history (`sender/receiver_prior_count_log`, `prior_amount_log`, `seconds_since_prior_log`).
  4. Categorical Encoding: restored from `transformers.json` (Payment Format: 7 categories, Payment Currency: 15 categories, Receiving Currency: 15 categories).
  5. Continuous Scaling: standardizes log amounts and history features using fitted training means and standard deviations.
  6. Expansion: expands to 57 features and builds sequences of length $\le 10$.

### Format B — Processed IBM AML Transaction Format (26/27 Columns)
```csv
timestamp,from_account,to_account,amount_paid_log,amount_received_log,amount_mismatch,same_currency,same_bank,self_transfer,payment_format_code,payment_currency_code,receiving_currency_code,minute_of_day_sin,minute_of_day_cos,day_of_week_sin,day_of_week_cos,is_weekend,is_night,sender_prior_count_log,sender_prior_amount_log,sender_seconds_since_prior_log,sender_has_prior,receiver_prior_count_log,receiver_prior_amount_log,receiver_seconds_since_prior_log,receiver_has_prior,is_laundering
```
- Validates columns, expands nominal codes to 57 features using recorded metadata vocabulary widths, links sending accounts, and constructs temporal sequence tensors $(N, 10, 57)$.

---

## 4. Evaluation Metrics Classification

When ground-truth labels (`Is Laundering` or `is_laundering`) are present in the CSV:

### Threshold-Dependent Metrics (Evaluated at Frozen Threshold)
- **True Positives (TP)**: Actual Laundering (1) predicted as `FLAGGED_LAUNDERING`.
- **False Positives (FP)**: Legitimate (0) predicted as `FLAGGED_LAUNDERING`.
- **True Negatives (TN)**: Legitimate (0) predicted as `CLEARED_LEGITIMATE`.
- **False Negatives (FN)**: Actual Laundering (1) predicted as `CLEARED_LEGITIMATE`.
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Alerts per 10k**: $\frac{TP + FP}{\text{Total}} \times 10,000$

### Probability-Dependent Metrics (Threshold-Independent)
- **PR-AUC / Average Precision**: Trapezoidal area under Precision-Recall curve (requires both classes).
- **ROC-AUC**: Area under Receiver Operating Characteristic curve (requires both classes).
- **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$

*If labels are absent, the dashboard displays prediction probabilities and decisions, explicitly noting that supervised metrics cannot be calculated without ground-truth labels.*

---

## 5. Research Graphs Provenance

| Research Graph | Source File | Data Provenance |
| :--- | :--- | :--- |
| **Convergence Curves (5 Rounds)** | `results/phase8/round_level_analysis.csv` | 120 round-by-round validation PR-AUC, ROC-AUC, and loss measurements across all 24 FL configs. |
| **Master Model Matrix (27 models)** | `results/phase8/phase8_summary.csv` | Full test split evaluations across GRU, LSTM, and BiLSTM under IID, Non-IID, FedAvg, FedProx, and Centralized baselines. |
| **Operational Threshold Tradeoffs** | `results/phase8/threshold_tradeoff_analysis.csv` | Precision, recall, and alert tradeoffs across decision thresholds for champion models. |
| **Ablation Findings** | `results/phase8/phase8_report.json` | Quantitative comparison of FedAvg vs. FedProx and IID vs. Non-IID degradation. |
| **Integrated Gradients Waterfall** | Phase 12 PyTorch Engine | On-demand 50-step Riemann midpoint quadrature over pre-sigmoid logit. |

---

## 6. Verification Results

### Backend Test Suite
```powershell
$env:PYTHONPATH = ".;backend"; .venv\Scripts\pytest.exe tests/test_phase7_ensemble.py tests/test_phase8_analysis.py tests/test_phase12_explainability.py tests/test_phase13_api.py tests/test_phase13_schemas.py tests/test_phase13_model_service.py tests/test_phase13_security_and_contracts.py tests/test_phase14a_deployment.py tests/test_phase14b_frontend_integration.py tests/test_phase15_e2e_validation.py tests/test_phase17_csv_evaluation.py -q
```
**Outcome: 118 / 118 passed in 11.90s (0 failures, 0 regressions)**.

### Frontend Production Build
```powershell
npm --prefix frontend run build
```
**Outcome: Build completed in 5.49s**
- CSS bundle: `dist/assets/index-DISfWSEV.css` (31.23 kB, gzip: 6.01 kB)
- JS bundle: `dist/assets/index-CO3dr0w_.js` (255.68 kB, gzip: 68.86 kB)
- Zero TypeScript diagnostics.

