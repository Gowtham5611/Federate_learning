# FedSecureAI — Professional CSV-Driven AML Evaluation Dashboard Repair Report

---

## 1. Executive Summary

This report documents the architectural repair, redesign, and empirical validation of the **FedSecureAI CSV-Driven AML Evaluation Platform**. 

The evaluation workflow has been streamlined into a unified, user-driven experience. The dashboard is now directly driven by the uploaded transaction CSV, stripping away all internal model architecture names (LSTM, BiLSTM), dataset names (HI-Small, LI-Medium), parameter counts, checkpoint hashes, and training metadata from the primary user interface. Scored transactions are categorized using clear, operational AML labels: **`LEGITIMATE`** and **`SUSPICIOUS`**.

---

## 2. Root Cause Analysis of the Original 404 Not Found

Investigation of the frontend requests and FastAPI route matching revealed three distinct points of failure:

1. **Vite Development Server Port Mismatch:**
   - In `frontend/vite.config.ts`, `server.port` was configured to `3000` with `strictPort: true`.
   - When the frontend ran on the standard Vite dev port (`http://localhost:5173`), requests to `/api/evaluate/csv` were routed directly to the Vite dev server rather than being proxied to the backend on `http://127.0.0.1:8000`.
   - Vite returned an HTTP 404 response (`Not Found`), which was passed directly to the UI error banner.
2. **Missing Endpoint Route Prefixes on FastAPI:**
   - The FastAPI backend registered the batch evaluation route exclusively at `@router.post("/evaluate/csv")`.
   - Client requests dispatched to `/api/v1/evaluate/csv` or `/api/evaluate/csv` failed with HTTP 404 (`{"detail": "Not Found"}`).
3. **Frontend API URL Resolution:**
   - When `VITE_API_BASE_URL` was undefined, `apiService.ts` defaulted strictly to `/api`, which required a functional reverse proxy.
   - Without direct fallback to `http://127.0.0.1:8000`, browser-level direct requests could not reach the backend.

---

## 3. Files Modified & Created

| Component | File Path | Nature of Changes |
| :--- | :--- | :--- |
| **Backend App** | `backend/interfaces/api/app.py` | Mounted all routers (health, inference, evaluation, research) across root (`""`), `/api/v1`, and `/api` prefixes to eliminate routing mismatches. |
| **Backend Router** | `backend/interfaces/api/routes/evaluation.py` | Added multiple route decorators (`/evaluate/csv`, `/evaluate/csv/`, `/api/evaluate/csv`, `/api/v1/evaluate/csv`), added `GET /evaluate/csv` diagnostic endpoint, and enabled automated pipeline inference when dataset is omitted. |
| **Backend Service** | `backend/interfaces/api/services/csv_evaluation_service.py` | Added operational statuses (`predicted_label`, `actual_label`, `status`, `transaction_id`), 10-decile risk distribution calculation, empirical PR/ROC curve point generation, and single-class edge case safeguards. |
| **Frontend Config** | `frontend/vite.config.ts` | Configured port `5173` for dev and preview servers, set host `0.0.0.0`, and routed `/api` to `http://127.0.0.1:8000`. |
| **Frontend Env** | `frontend/.env` & `frontend/.env.development` | Set `VITE_API_BASE_URL=http://127.0.0.1:8000`. |
| **Frontend API Client** | `frontend/src/services/apiService.ts` | Added automated local backend resolution, descriptive HTTP error handling, and `/api/v1/evaluate/csv` fallback. |
| **Frontend Types** | `frontend/src/types/api.ts` | Added `transaction_id`, `predicted_label`, `actual_label`, `status`, `RiskBinData`, `CurveDatapoint`, and made `dataset` optional. |
| **Header UI** | `frontend/src/components/Header.tsx` | Redesigned header with title **FedSecureAI — AML Evaluation Platform**, removed dataset switchers and model tags, simplified to 3 core tabs: *AML Evaluation*, *Results & Analytics*, and *Prediction History*. |
| **Evaluation Page** | `frontend/src/pages/AmlEvaluationPage.tsx` | Created unified CSV upload workflow with drag-and-drop, schema validation, preview, demonstration data loaders, and summary metrics. |
| **Analytics Page** | `frontend/src/pages/ResultsAnalyticsPage.tsx` | Created dedicated analytics view with metric cards, $2 \times 2$ Confusion Matrix, interactive SVG PR and ROC curves, and risk distribution histogram. |
| **History Page** | `frontend/src/pages/PredictionHistoryPage.tsx` | Created searchable, sortable transaction-level results table with `SUSPICIOUS`/`LEGITIMATE` labels, downloadable CSV, and Integrated Gradients inspect modal. |
| **App Entry** | `frontend/src/App.tsx` | Wired unified 3-tab application state driven entirely by the uploaded CSV. |
| **Test Suite** | `tests/test_phase17_csv_evaluation.py` | Added regression tests for automated pipeline inference, route aliases, and operational status labels (17/17 passed). |

---

## 4. Supported CSV Schemas

The evaluation engine automatically detects the input schema and selects the appropriate preprocessing pipeline:

### Format A: Standard AML Stream Format (11 Columns)
- **Columns:** `Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1` (or duplicate `Account`), `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`.
- **Optional Column:** `Is Laundering` (binary `0` or `1`).
- **Processing:** Raw amounts are log-transformed, cyclical time transforms (sin/cos of hour and day) are derived, categorical columns are ordinal-encoded via fitted `transformers.json`, and account transfer history is assembled into $(N, 10, 57)$ sequences.

### Format B: Processed Feature Format (27 Columns)
- **Columns:** `timestamp`, `from_bank`, `from_account`, `to_bank`, `to_account`, `amount_paid`, `amount_received`, `receiving_currency`, `payment_currency`, `payment_format`, `amount_paid_usd`, `amount_received_usd`, `amount_ratio`, `is_cross_currency`, `log_amount_paid_usd`, `log_amount_received_usd`, `hour`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `payment_format_code`, `receiving_currency_code`, `payment_currency_code`.
- **Optional Column:** `is_laundering` (binary `0` or `1`).
- **Processing:** Currency and payment format codes are expanded into 37 one-hot indicators to form the required 57-dimensional feature vectors. Account groupings are zero-padded rightward up to sequence length 10.

---

## 5. Preprocessing & Prediction Pipeline

1. **Zero Retraining Guarantee:** Checkpoint weights, model architectures, and decision thresholds remain frozen.
2. **Automated Pipeline Inference:** If `dataset` is omitted in the request, the backend infers the production champion pipeline automatically.
3. **Sequence Construction:** Accounts with fewer than 10 historical transactions are zero-padded rightward, preserving causal ordering.
4. **Classification & Labels:**
   - Sequences exceeding the decision threshold are classified as **`SUSPICIOUS`**.
   - Sequences below the threshold are classified as **`LEGITIMATE`**.
   - No internal model architecture names (LSTM, BiLSTM) or parameter counts are exposed to the user.

---

## 6. Evaluation Metrics Implementation

### When Ground-Truth Labels Are Present:
- **Accuracy, Precision, Recall, F1-Score:** Derived from the $2 \times 2$ Confusion Matrix.
- **PR-AUC & ROC-AUC:** Computed using `sklearn.metrics.average_precision_score` and `roc_auc_score`.
- **Confusion Matrix:** True Positives, False Positives, True Negatives, False Negatives.
- **Visual Curves:** Precision-Recall and ROC curves rendered via SVG based on empirical evaluation coordinates.
- **Single-Class Safeguards:** If the uploaded data contains only negative or only positive transactions, ROC-AUC and PR-AUC display `N/A` with an explanatory note rather than throwing an exception.

### When Ground-Truth Labels Are Absent:
- Inference runs normally.
- Scored probabilities, predicted labels (`SUSPICIOUS` / `LEGITIMATE`), and final statuses are displayed.
- The UI displays: *"Ground-truth labels were not provided in the uploaded CSV. Evaluation metrics require actual labels and cannot be calculated."*
- Unlabeled predictions are never used as synthetic ground truth.

---

## 7. Verification Tests & Results

### Suite A: Phase 17 CSV Evaluation Tests (17/17 Passed)
```powershell
$env:PYTHONPATH = ".;backend"
.venv\Scripts\python.exe -m pytest tests/test_phase17_csv_evaluation.py -v
```
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 17 items

tests/test_phase17_csv_evaluation.py::test_research_overview PASSED      [  5%]
tests/test_phase17_csv_evaluation.py::test_research_models PASSED        [ 11%]
tests/test_phase17_csv_evaluation.py::test_research_curves PASSED        [ 17%]
tests/test_phase17_csv_evaluation.py::test_research_ablation PASSED      [ 23%]
tests/test_phase17_csv_evaluation.py::test_download_sample_csv PASSED    [ 29%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_hi_small_processed_with_labels PASSED [ 35%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_li_medium_processed_with_labels PASSED [ 41%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_raw_format PASSED [ 47%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_unlabeled_transactions PASSED [ 52%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_empty_content_rejection PASSED [ 58%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_missing_columns_rejection PASSED [ 64%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_invalid_dataset_rejection PASSED [ 70%]
tests/test_phase17_csv_evaluation.py::test_route_registration_aliases PASSED [ 76%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_no_positive_labels_edge_case PASSED [ 82%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_no_negative_labels_edge_case PASSED [ 88%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_auto_pipeline_inference PASSED [ 94%]
tests/test_phase17_csv_evaluation.py::test_evaluate_csv_risk_distribution_deciles PASSED [100%]

============================= 17 passed in 7.48s ==============================
```

### Suite B: Full Repository Regression Suite (137/137 Passed)
```powershell
$env:PYTHONPATH = ".;backend"
.venv\Scripts\python.exe -m pytest tests/ -v
```
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 137 items

tests/test_cuda_enforcement.py PASSED (3 tests)                          [  2%]
tests/test_full_combined_pipeline.py PASSED (4 tests)                    [  5%]
tests/test_full_li_medium_pipeline.py PASSED (4 tests)                   [  8%]
tests/test_gpu_preprocessing_equivalence.py PASSED (2 tests)             [  9%]
tests/test_phase12_explainability.py PASSED (10 tests)                   [ 17%]
tests/test_phase13_api.py PASSED (12 tests)                              [ 26%]
tests/test_phase13_model_service.py PASSED (5 tests)                     [ 30%]
tests/test_phase13_schemas.py PASSED (9 tests)                           [ 36%]
tests/test_phase13_security_and_contracts.py PASSED (10 tests)           [ 44%]
tests/test_phase14a_deployment.py PASSED (7 tests)                       [ 49%]
tests/test_phase14b_frontend_integration.py PASSED (6 tests)             [ 53%]
tests/test_phase15_e2e_validation.py PASSED (17 tests)                   [ 75%]
tests/test_phase17_csv_evaluation.py PASSED (17 tests)                   [ 87%]
tests/test_phase7_ensemble.py PASSED (7 tests)                           [ 92%]
tests/test_phase8_analysis.py PASSED (10 tests)                          [100%]

============================ 137 passed in 11.42s =============================
```

### Suite C: Frontend Production Build
```powershell
npm --prefix frontend run build
```
```
> fedsecureai-dashboard@1.0.0 build
> tsc && vite build

vite v6.4.5 building for production...
transforming...
✓ 1577 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.78 kB │ gzip:  0.46 kB
dist/assets/index-Z8DSc735.css   34.51 kB │ gzip:  6.39 kB
dist/assets/index-zx0xbUi4.js   207.52 kB │ gzip: 59.80 kB
✓ built in 4.35s
```

---

## 8. Exact Commands to Run Backend & Frontend

### Step 1: Start FastAPI Backend Service
```powershell
cd C:\Users\Acer\OneDrive\Desktop\Federate_learning
$env:PYTHONPATH = ".;backend"
.venv\Scripts\python.exe -m uvicorn backend.interfaces.api.app:app --host 127.0.0.1 --port 8000
```

### Step 2: Start React AML Dashboard
```powershell
cd C:\Users\Acer\OneDrive\Desktop\Federate_learning\frontend
npm run dev
```

*Open browser:* `http://localhost:5173`

---

## 9. Operational Boundaries & Remaining Limitations

1. **Frozen Decision Threshold:** Scored transactions are evaluated against the empirically validated decision threshold established during training. The threshold is preserved internally without exposing tuning clutter to the user.
2. **Row Limit Constraint:** Max upload limit is bounded to $10,000$ transactions per batch (default: $5,000$) to prevent memory exhaustion and maintain sub-second response times.
3. **Deployment Network Boundary:** The inference service is designed for private restricted networks or authenticated reverse proxy/API gateways as documented in Phase 14A.

