# FedSecureAI Phase 13 — API Service Validation & Verification Report

**Author:** FedSecureAI Senior Backend & MLOps Team  
**Date:** October 26, 2026  
**Status:** VALIDATED & COMPLIANT  
**Target Datasets:** IBM AML HI-Small & LI-Medium  
**Framework:** FastAPI 0.162.0, Pydantic 2.14.1, PyTorch 2.14.0 CPU, Starlette 1.10.0  

---

## 1. Executive Summary

Phase 13 establishes the production-grade **FastAPI Inference Service** for the FedSecureAI platform. The service provides secure, privacy-preserving, and explainable anti-money laundering (AML) detection by serving the authoritative federated champion models selected in Phase 7 and verified across Phases 8–12.

Key achievements:
- **Zero Retraining:** Served completed checkpoints without parameter alteration or fine-tuning.
- **Strict Weight Authentication:** Cryptographic SHA-256 verification enforces checkpoint provenance before models can transition to `loaded` state.
- **Numerical Parity Verified:** Real validation sample predictions match Phase 7 references with maximum absolute error $< 6 \times 10^{-8}$ (well within the $10^{-5}$ tolerance).
- **Sub-Millisecond Inference:** Single-sequence forward passes execute in $0.73 \text{ ms}$ (HI-Small) and $1.00 \text{ ms}$ (LI-Medium).
- **Vectorized Bounded Batching:** Up to 64 sequences processed concurrently with throughput exceeding $6,900 \text{ samples/sec}$.
- **Axiomatic Attribution:** Integrated Gradients attribution with strictly compliant naming (`integrated_gradients`), bounded Riemann steps ($10 \le m \le 50$), and completeness residual checks ($| \sum \text{IG} - \Delta F | < 0.05$).
- **Privacy Protections:** Zero logging of financial payload data or client account identifiers; bounded request body size ($2 \text{ MB}$).
- **100% Test Pass Rate:** 26/26 tests passing across schema, model service, and API integration test suites.

---

## 2. API Endpoint Catalog & Contract Specifications

The service exposes the following authenticated and versioned REST endpoints:

| Endpoint | Method | Purpose | Request Body | Response Model | HTTP Codes |
|---|---|---|---|---|---|
| `/health` | `GET` | Service liveness and model readiness | None | `HealthResponse` | 200, 503 |
| `/models` | `GET` | Catalog of loaded federated models | None | `List[ModelMetadataResponse]` | 200 |
| `/metadata/{dataset}` | `GET` | Feature schema & frozen decision threshold | None | `DatasetMetadataResponse` | 200, 404 |
| `/predict` | `POST` | Single transaction sequence AML prediction | `PredictRequest` | `PredictResponse` | 200, 422, 503 |
| `/predict/batch` | `POST` | Bounded batch transaction prediction (max 64) | `BatchPredictRequest` | `BatchPredictResponse` | 200, 422, 503 |
| `/explain` | `POST` | Integrated Gradients feature attribution | `ExplainRequest` | `ExplainResponse` | 200, 422, 503 |

---

## 3. Champion Model Architecture & Frozen Thresholds

| Dataset | Model Identifier | Architecture | Parameters | Checkpoint SHA-256 | Frozen Threshold | Explainer |
|---|---|---|---|---|---|---|
| **HI-Small** | `hi_small_lstm_fedavg_iid_seed42` | LSTM (1 layer, 64 hidden) | 31,553 | `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4` | `0.3377837836742401` | `integrated_gradients` |
| **LI-Medium** | `config_5_bilstm_fedavg_iid` | BiLSTM (1 layer, 64 hidden) | 63,105 | `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2` | `0.2384829968214035` | `integrated_gradients` |

### Decision Logic:
$$\text{Decision} = \begin{cases} \text{FLAGGED\_LAUNDERING} & \text{if } P(\text{Laundering}) \ge \tau_{\text{frozen}} \\ \text{CLEARED\_LEGITIMATE} & \text{otherwise} \end{cases}$$

---

## 4. Input Validation & Strict Schema Guardrails

1. **Feature Vector Dimensions:**
   - Every transaction step must contain exactly 57 floating point features representing the canonical expanded nominal feature schema.
   - Any sequence step containing $\neq 57$ dimensions is immediately rejected with HTTP 422.
2. **Temporal Window & Zero-Padding:**
   - Sequence length $L$ is bounded: $1 \le L \le 10$.
   - Sequences shorter than 10 are right-padded with exact zero vectors of dimension 57.
   - Pydantic validators verify that padding entries ($t \ge L$) contain all zeros.
3. **Finite Numeric Verification:**
   - Values must be finite; any appearance of `NaN`, `+Inf`, or `-Inf` triggers HTTP 422.
4. **Batch Size Bounds:**
   - Batch requests strictly enforce $1 \le N \le 64$.
   - Requests with $> 64$ items are rejected with HTTP 422.
5. **Model-Dataset Alignment:**
   - If an optional `model_name` is provided in a request, it must match the designated champion for that dataset. Mismatched names trigger HTTP 422.

---

## 5. Explainability Verification

- **Method Identifier:** Labeled strictly as `integrated_gradients`. Any reference to SHAP is strictly prohibited.
- **Path Steps Bound:** Integration steps $m$ are bounded: $10 \le m \le 50$ (default 25).
- **Completeness Axiom:** Verified that $\sum_{i,j} \text{IG}_{i,j} \approx F(x) - F(x_0)$. Across test benchmarks, the absolute residual $| \sum \text{IG} - \Delta F |$ remained below $0.003$ for all configurations, fully satisfying completeness.
- **Top Driver Extraction:** Returns semantic attribution objects containing feature names, signed importance scores, semantic groups, and temporal positions.

---

## 6. Real Prediction Parity Results

Parity was verified by running 500 real validation sequences from `data/processed/` through `ModelService` and comparing against the authoritative validation predictions from Phase 7:

| Dataset | Evaluation Samples | Max Absolute Error | Mean Absolute Error | Tolerance | Parity Status |
|---|---|---|---|---|---|
| **HI-Small** | 500 | $5.9605 \times 10^{-8}$ | $1.7592 \times 10^{-9}$ | $1.0 \times 10^{-5}$ | **PASSED** |
| **LI-Medium** | 500 | $7.4506 \times 10^{-9}$ | $1.3859 \times 10^{-10}$ | $1.0 \times 10^{-5}$ | **PASSED** |

Both datasets demonstrate exact numerical parity well below the $10^{-5}$ tolerance.

---

## 7. Performance & Latency Benchmark Summary

Derived from empirical benchmarks (see `phase13_performance_report.md`):

- **Cold Start Time:**
  - HI-Small: $14.31 \text{ ms}$
  - LI-Medium: $9.42 \text{ ms}$
  - Total startup load time: **$23.73 \text{ ms}$**
- **Warm Forward Pass Latency (N=100):**
  - HI-Small: Mean $0.73 \text{ ms}$, p50 $0.62 \text{ ms}$, p95 $1.22 \text{ ms}$, p99 $1.30 \text{ ms}$
  - LI-Medium: Mean $1.00 \text{ ms}$, p50 $0.84 \text{ ms}$, p95 $1.75 \text{ ms}$, p99 $2.34 \text{ ms}$
- **Batch Processing Throughput (Batch 64):**
  - HI-Small: $9.22 \text{ ms}$ total ($0.14 \text{ ms/sample}$), $6,940 \text{ samples/sec}$
  - LI-Medium: $6.49 \text{ ms}$ total ($0.10 \text{ ms/sample}$), $9,860 \text{ samples/sec}$
- **Explanation Latency (steps=25):**
  - HI-Small: $118.22 \text{ ms}$
  - LI-Medium: $217.04 \text{ ms}$

---

## 8. Test Suite Summary

Total unit and integration tests executed: **26 passing, 0 failing, 0 skipped**.

1. `tests/test_phase13_schemas.py`: 9 passed
   - Sequence item validation, dimension checks, NaN/Inf rejection, zero-padding enforcement, batch bounds, explanation bounds, metadata responses.
2. `tests/test_phase13_model_service.py`: 5 passed
   - Checkpoint loading, SHA-256 authentication, metadata queries, single prediction, bounded batch prediction.
3. `tests/test_phase13_api.py`: 12 passed
   - Health endpoint, models catalog, metadata endpoints (valid/invalid), single predict (HI/LI), schema error handling, model name mismatch handling, batch predict success, batch size limit (65 rejected), explanation endpoint, real prediction parity verification.

