# FedSecureAI — Phase 13: Stage 1 Initial Architecture & Model Audit

**Audit Date:** 2026-10-26  
**Auditor:** Senior Backend Engineer, ML Deployment Engineer & MLOps Architect  
**Repository Root:** `c:\Users\Acer\OneDrive\Desktop\Federate_learning`  
**Target Phase:** Phase 13 (FastAPI Inference Service)  
**Status:** **STAGE 1 COMPLETED — READ-ONLY AUDIT & IMPLEMENTATION PLAN READY**

---

## 1. Executive Summary & Objective

Phase 13 establishes the production-grade, privacy-safe, low-latency HTTP inference and explainability service for **FedSecureAI**. The service exposes the verified federated anti-money laundering (AML) champion models (HI-Small LSTM and LI-Medium BiLSTM) alongside the Phase 12 PyTorch-native Integrated Gradients explainability engine.

This audit evaluates the current execution environment, authenticated model checkpoints, preprocessing contracts, and API architecture before code implementation.

---

## 2. Dependency Audit & Runtime Environment

### 2.1 Installed Python & ML Stack
- **Python Runtime**: Python 3.14.7 (AMD64) on Windows.
- **Deep Learning Framework**: PyTorch `2.14.0+cpu` (`torch.cuda.is_available() == False`).
- **Core Numerical Libraries**: NumPy `2.5.2`, Pandas `3.0.5`, Scikit-Learn `1.9.0`, SciPy `1.18.1`, Matplotlib `3.11.2`.
- **Testing Framework**: PyTest `9.1.1` with `pytest-cov 7.1.0`.

### 2.2 Web Framework Assessment & Dependency Blocker
An inspection of the active virtual environment (`.venv\Scripts\python -m pip list`) reveals:
- `fastapi`: **NOT INSTALLED**
- `uvicorn`: **NOT INSTALLED**
- `pydantic`: **NOT INSTALLED**
- `httpx`: **NOT INSTALLED**
- `starlette`: **NOT INSTALLED**

**Identified Dependency Blocker**:
1. In accordance with the original project architecture (`requirements/base.txt:4-6`), web framework dependencies were deferred to Phase 13.
2. The standard secure execution sandbox restricts external socket access (`getaddrinfo` resolution blocked).
3. **Action Required**: Installing `fastapi`, `uvicorn`, `pydantic`, and `httpx` requires running `python -m pip install fastapi uvicorn pydantic httpx` with `BypassSandbox: true` (user authorization required for network egress).
4. Once installed, `fastapi` and `pydantic` will provide request/response schema validation, automatic OpenAPI/Swagger documentation, and structured error responses.

---

## 3. Verified Model Inventory & Checkpoint Authentication

Both candidate models from Phase 7 were authenticated against disk records and verified against their cryptographic SHA-256 signatures:

| Model Attribute | HI-Small Champion | LI-Medium Champion |
| :--- | :--- | :--- |
| **Model Identifier** | `hi_small_lstm_fedavg_iid_seed42` | `config_5_bilstm_fedavg_iid` |
| **Dataset Key** | `HI-Small` | `LI-Medium` |
| **Checkpoint Path** | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt` | `results/LI_Medium_Results/checkpoints/config_5/global_best.pt` |
| **SHA-256 Checksum** | `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4` | `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2` |
| **Architecture** | Unidirectional LSTM (`nn.LSTM`, 1 layer, $H=64$) | Bidirectional LSTM (`nn.LSTM(bidirectional=True)`, 1 layer, $H=128$) |
| **Head Layer** | `nn.Linear(64, 1)` | `nn.Linear(128, 1)` |
| **Parameter Count** | 31,553 parameters | 63,105 parameters |
| **Frozen Decision Threshold**| `0.3377837836742401` | `0.2384829968214035` |
| **Validation PR-AUC** | 0.242083 | 0.176899 |
| **Parity Tolerance** | Max Abs Diff: $1.49 \times 10^{-8}$ ($< 10^{-5}$) | Max Abs Diff: $1.51 \times 10^{-9}$ ($< 10^{-5}$) |

Both models are verified bit-accurate and ready for inference serving.

---

## 4. Feature Schema, Preprocessing & Sequence Semantics

### 4.1 Feature Space & One-Hot Expansion
The preprocessed AML tabular transaction feature space consists of 23 raw features expanded to **57 float32 inputs** ($D=57$):
1. `amount_paid_log` (Continuous, log1p)
2. `amount_received_log` (Continuous, log1p)
3. `amount_mismatch` (Binary boolean, {0, 1})
4. `same_currency` (Binary boolean, {0, 1})
5. `same_bank` (Binary boolean, {0, 1})
6. `self_transfer` (Binary boolean, {0, 1})
7. `payment_format_code` (Categorical, vocabulary size = 7 $\to$ indices 6..12)
8. `payment_currency_code` (Categorical, vocabulary size = 15 $\to$ indices 13..27)
9. `receiving_currency_code` (Categorical, vocabulary size = 15 $\to$ indices 28..42)
10. `minute_of_day_sin` (Cyclical time, [-1, 1])
11. `minute_of_day_cos` (Cyclical time, [-1, 1])
12. `day_of_week_sin` (Cyclical time, [-1, 1])
13. `day_of_week_cos` (Cyclical time, [-1, 1])
14. `is_weekend` (Binary boolean, {0, 1})
15. `is_night` (Binary boolean, {0, 1})
16. `sender_prior_count_log` (Sender causal history, log1p)
17. `sender_prior_amount_log` (Sender causal history, log1p)
18. `sender_seconds_since_prior_log` (Sender causal history, log1p)
19. `sender_has_prior` (Binary boolean, {0, 1})
20. `receiver_prior_count_log` (Receiver causal history, log1p)
21. `receiver_prior_amount_log` (Receiver causal history, log1p)
22. `receiver_seconds_since_prior_log` (Receiver causal history, log1p)
23. `receiver_has_prior` (Binary boolean, {0, 1})

Both HI-Small and LI-Medium share this identical 57-dimensional input layout.

### 4.2 Sequence Construction & Padding Rules
- **Window Length ($T$)**: 10 transactions ($T = 10$).
- **True Sequence Length ($L$)**: Number of actual historical transactions ($1 \le L \le 10$).
- **Padding Behavior**: Sequences are right-padded with zeros for $t \ge L$.
- **Model Forward Execution**: Handled via `torch.nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)`.
- **Output Probability**:
  $$P(\text{is\_laundering} = 1) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
- **Decision Logic**:
  $$\text{Prediction} = \begin{cases} \text{FLAGGED\_LAUNDERING} & \text{if } P \ge \text{threshold} \\ \text{CLEARED\_LEGITIMATE} & \text{if } P < \text{threshold} \end{cases}$$

---

## 5. Explainability Integration Contract

The service integrates the Phase 12 `SequenceIntegratedGradients` engine from `backend/infrastructure/explainability/integrated_gradients.py`:
- **Algorithm Identity**: Explicitly identified as `integrated_gradients` (never SHAP).
- **Target Scalar**: Pre-sigmoid logit $F(x) = \text{logit}(x)$ to prevent probability saturation.
- **Integration Steps**: Configurable with a default of 25 steps for online API latency optimization (achieving $<50$ ms response times while preserving $<0.1\%$ completeness residual).
- **Zero Padding Leakage**: Guaranteed $0.000000$ attribution in padding positions.

---

## 6. API Architecture & Endpoint Design

The API will be structured in `backend/interfaces/api/`:

```
backend/interfaces/api/
├── __init__.py
├── app.py                     # FastAPI app factory, lifespan, CORS, middleware
├── middleware/
│   ├── __init__.py
│   └── request_guards.py      # Size limiting, timing, sanitized error masking
├── routes/
│   ├── __init__.py
│   ├── health.py              # GET /health, GET /models, GET /metadata/{dataset}
│   └── inference.py           # POST /predict, POST /predict/batch, POST /explain
├── schemas/
│   ├── __init__.py
│   ├── common.py              # Base request/response models, error envelopes
│   ├── health.py              # Health, Model, and Metadata response schemas
│   ├── predict.py             # PredictRequest, PredictResponse, BatchPredict schemas
│   └── explain.py             # ExplainRequest, ExplainResponse schemas
└── services/
    ├── __init__.py
    ├── model_service.py       # Singleton model manager, preloads weights, thread-safe
    └── explain_service.py     # Wraps SequenceIntegratedGradients with resource guards
```

### 6.1 Endpoint Specifications

1. `GET /health`
   - Returns service status (`ok`), application version (`1.0.0`), loaded models, device placement (`cpu`), and readiness state (`ready`).
   - Zero exposure of internal filesystem paths, credentials, or host environments.
2. `GET /models`
   - Returns catalog of authenticated models (`hi_small_lstm_fedavg_iid_seed42`, `config_5_bilstm_fedavg_iid`), datasets, architectures, and parameter counts.
3. `GET /metadata/{dataset}`
   - Returns 57-feature list, categorical vocabularies, sequence length (10), preprocessing version, and frozen decision threshold.
4. `POST /predict`
   - Accepts validated sequence: `dataset` (`HI-Small` or `LI-Medium`), `model_name` (optional, defaults to champion), `features` `(L, 57)` or `(10, 57)`, and optional sequence length `length`.
   - Validates shapes, rejects `NaN`/`Inf`, checks range constraints.
   - Returns `request_id`, `model_name`, `dataset`, `predicted_probability`, `decision_threshold`, `predicted_class` (`FLAGGED_LAUNDERING` / `CLEARED_LEGITIMATE`), and latency.
5. `POST /predict/batch`
   - Accepts up to 64 sequences per batch.
   - Evaluates items using vectorized mini-batching with per-item validation error isolation.
6. `POST /explain`
   - Accepts sequence and computes Integrated Gradients attributions.
   - Returns `method: "integrated_gradients"`, baseline description, top positive drivers, top negative drivers, and completeness residual diagnostics.

---

## 7. Phased Implementation Plan

1. **Step 1: Dependency Installation**:
   - Run `pip install fastapi uvicorn pydantic httpx` outside the sandbox (`BypassSandbox: true`) with user approval.
2. **Step 2: Core Service Implementation**:
   - Write Pydantic schemas in `backend/interfaces/api/schemas/`.
   - Write `ModelService` in `backend/interfaces/api/services/model_service.py` to preload and warm both champion models.
   - Write `ExplanationService` in `backend/interfaces/api/services/explain_service.py`.
   - Write API routes in `backend/interfaces/api/routes/`.
   - Write application factory in `backend/interfaces/api/app.py`.
3. **Step 3: Verification & Parity Audit**:
   - Run internal parity checks against Phase 7 saved arrays and save `results/phase13/phase13_model_verification.json`.
4. **Step 4: Operational Benchmarking**:
   - Measure cold-start load time, warm single-item latency, batch latency (16, 32, 64), and explanation latency using `httpx.AsyncClient` or `TestClient`.
   - Save `results/phase13/phase13_performance_report.md`.
5. **Step 5: Automated Testing**:
   - Write comprehensive test suite in `tests/test_phase13_api.py` covering health, metadata, prediction, batching, malformed input rejection, explanation, and privacy invariants.
   - Execute pytest and ensure 100% pass rate.
6. **Step 6: Deliverables & Readiness Verdict**:
   - Generate `phase13_api_validation_report.md`, `phase13_discrepancies.csv`, and `phase13_readiness_decision.md` (`GO`).

