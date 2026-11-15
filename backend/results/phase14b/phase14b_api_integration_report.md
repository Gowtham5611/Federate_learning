# Phase 14B — API Integration & Contract Verification Report

**Document ID:** `results/phase14b/phase14b_api_integration_report.md`  
**Phase:** 14B — AML Analyst Dashboard & API Integration  
**Date:** 2026-10-27  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, Drift-Aware Federated AML  
**Status:** COMPLETE  

---

## 1. Executive Summary

This report documents the contract integration between the Phase 14B React/TypeScript frontend and the Phase 13 FastAPI backend (`backend/interfaces/api`). The dashboard connects seamlessly to the FastAPI service via strict Pydantic v2 data transfer models, proxy routing, and CORS policy enforcement.

---

## 2. Integrated API Endpoints

The frontend client (`frontend/src/services/apiService.ts`) implements typed methods for the five core Phase 13 endpoints:

| Endpoint | HTTP Method | Request Payload | Response Schema | Frontend Usage |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `HealthResponse` | Persistent engine status monitor and loaded model verification |
| `/models` | `GET` | None | `ModelsListResponse` | Dashboard overview inventory of available dataset models |
| `/metadata/{dataset}` | `GET` | URL param: `HI-Small` or `LI-Medium` | `DatasetMetadataResponse` | Frozen thresholds, dimensions ($T=10, D=57$), architecture details |
| `/predict` | `POST` | `PredictRequest` | `PredictResponse` | Real-time sequence scoring, risk badge determination, and latency |
| `/explain` | `POST` | `ExplainRequest` | `ExplainResponse` | Integrated Gradients feature attribution, top drivers, completeness residual |

---

## 3. Schema Parity & Type Synchronization

The frontend types defined in `frontend/src/types/api.ts` mirror the backend Pydantic schemas in `backend/interfaces/api/schemas/`:

### 3.1 Common Sequence Model
```typescript
export interface SequenceItem {
  features: number[][];   // Dimensions: Exactly [10][57]
  length: number;         // Valid range: 1 to 10
  sample_id?: string;     // Synthetic/Audit identifier
}
```
*Backend Counterpart:* `backend/interfaces/api/schemas/common.py::SequenceItem`  
- Enforces $T=10, D=57$, finite float bounds, and strict zero-padding for $t \ge \text{length}$.

### 3.2 Single Prediction Contract
```typescript
export interface PredictRequest {
  dataset: "HI-Small" | "LI-Medium";
  sequence: SequenceItem;
  model_name?: string;
}

export interface PredictResponse {
  dataset: string;
  model_name: string;
  sample_id?: string;
  probability: number;
  threshold: number;
  decision: "CLEARED_LEGITIMATE" | "FLAGGED_LAUNDERING";
  latency_ms: number;
}
```
*Backend Counterpart:* `backend/interfaces/api/schemas/predict.py`  
- Decision logic is computed server-side: $\text{decision} = \text{probability} \ge \text{threshold}$.

### 3.3 Explainability Contract
```typescript
export interface ExplainRequest {
  dataset: "HI-Small" | "LI-Medium";
  sequence: SequenceItem;
  steps?: number;        // Default 25, clamped 10..100
  top_k?: number;        // Default 5, clamped 1..20
}

export interface FeatureDriver {
  feature_index: number;
  feature_name: string;
  feature_group: string;
  attribution: number;
  rank: number;
}

export interface ExplainResponse {
  dataset: string;
  model_name: string;
  sample_id?: string;
  method: "integrated_gradients";
  steps: number;
  predicted_probability: number;
  top_positive_drivers: FeatureDriver[];
  top_negative_drivers: FeatureDriver[];
  completeness_residual: number;
  explanation_time_ms: number;
}
```
*Backend Counterpart:* `backend/interfaces/api/schemas/explain.py`  
- Explanation method is immutable: strictly `integrated_gradients` with Riemann trapezoidal integration over baseline zeroes.

---

## 4. Network Routing, CORS & Development Proxy

### 4.1 Development Proxy
In development mode (`npm run dev`), Vite proxies all `/api/*` traffic directly to the local FastAPI server:
```typescript
// frontend/vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
```

### 4.2 Cross-Origin Resource Sharing (CORS) Configuration
When deployed on distinct origins or queried directly via `VITE_API_BASE_URL=http://127.0.0.1:8000`, the FastAPI service permits authorized dashboard origins:
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5173`
- `http://127.0.0.1:5173`

The configuration was formally verified via regression test `test_backend_cors_configuration` in `tests/test_phase14b_frontend_integration.py`.

---

## 5. Error Handling & Resilience

The dashboard client handles standard and exceptional API states gracefully:
- **HTTP 422 Unprocessable Entity:** Catches validation rejections (e.g., malformed dimensions, `NaN`/`Inf` floats, non-zero padding) and formats backend error details into user-friendly diagnostic notices.
- **HTTP 503 Service Unavailable:** Notifies the analyst if a requested model has not completed initialization or failed hash verification.
- **Connection Refused / Network Timeout:** Employs an explicit 15-second request timeout, notifying the user to verify private network routing or API gateway authorization without crashing the application.

