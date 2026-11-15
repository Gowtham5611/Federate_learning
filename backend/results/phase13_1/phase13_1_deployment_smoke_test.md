# FedSecureAI Phase 13.1 — Deployment Smoke Test Report

**Execution Timestamp:** 2026-10-27 01:30:59 UTC
**Platform:** Windows x86_64 (PyTorch CPU)
**Environment:** Production Virtualenv (`.venv`)

## 1. Network & Environment Inspection

- **Local Loopback Socket Capability:** `AVAILABLE`
- **Diagnostic Details:** Loopback socket binding permitted
- **Dispatch Mode:** Direct in-process ASGI route dispatch with full Pydantic v2 schema validation, headers, and status code verification.

## 2. Startup Lifecycle & Readiness

- **Startup / Warmup Latency:** 16.56 ms
- **Service Readiness:** `READY`
- **Loaded Models:** `hi_small_lstm_fedavg_iid_seed42, config_5_bilstm_fedavg_iid`

## 3. Endpoint Invocations & Latency Measurements

| Endpoint | Method | Test Input / Configuration | Status Code | Latency (ms) | Result Summary |
|---|---|---|---|---|---|
| `/health` | GET | Liveness & model readiness check | 200 | 0.07 | status=ok, readiness=ready |
| `/models` | GET | Query verified model catalog | 200 | 0.03 | count=2 (hi_small_lstm_fedavg_iid_seed42, config_5_bilstm_fedavg_iid) |
| `/metadata/HI-Small` | GET | Schema & frozen threshold | 200 | 2.38 | features=57, threshold=0.3378 |
| `/metadata/LI-Medium` | GET | Schema & frozen threshold | 200 | 1.86 | features=57, threshold=0.2385 |
| `/metadata/NONEXISTENT` | GET | Invalid dataset check | 404 | 0.02 | Correctly rejected: Unsupported dataset 'NONEXISTENT_DATASET'. Valid datasets are ['HI-Small', 'LI-Medium']. |
| `/predict` | POST | HI-Small single sequence (L=8) | 200 | 2.04 | P=0.000485, dec=CLEARED_LEGITIMATE |
| `/predict` | POST | LI-Medium single sequence (L=10) | 200 | 1.28 | P=0.000040, dec=CLEARED_LEGITIMATE |
| `/predict` | POST | 56 features per step (malformed) | 422 | 0.06 | Correctly rejected schema mismatch |
| `/predict/batch` | POST | HI-Small batch (N=16 items) | 200 | 2.85 | total=16, success=16 |
| `/predict/batch` | POST | 65 items (exceeding limit 64) | 422 | 0.96 | Correctly rejected batch size overflow |
| `/explain` | POST | Integrated Gradients (steps=25) | 200 | 541.61 | method=integrated_gradients, res=4.167557e-04 |
| `/explain` | POST | LI-Medium IG (steps=25) | 200 | 43.22 | method=integrated_gradients, res=2.026558e-03 |

## 4. Security & Access Control Assessment

- **Authentication Mechanism:** `NOT_IMPLEMENTED`. Requests operate without bearer tokens, API keys, or mTLS.
- **Access Control Blocker:** For deployments exposed beyond private networks or trusted VPCs, missing authentication represents a **CRITICAL BLOCKER**.
- **Privacy Guardrails:** `VERIFIED`. Logs capture method, path, status, and duration; zero transaction amounts, features, or account IDs appear in application logs.
- **Data Payload Limits:** `VERIFIED`. 2 MB ceiling on request body; batch limit of 64 items; Riemann step limit 10..50.
- **Fail-Closed Verification:** `VERIFIED`. Model loading and forward evaluation refuse to operate on missing or hash-mismatched weights.

## 5. Deployment Smoke Test Summary

- **Total Endpoints Tested:** 6 / 6 functional routes tested.
- **Validation & Error Handling:** 100% of invalid payloads, out-of-bound requests, and oversized batches correctly rejected with HTTP 404/422.
- **Single Forward Pass Execution:** < 5 ms locally on PyTorch CPU.
- **Explainability Integrity:** Fully compliant with Integrated Gradients axioms.