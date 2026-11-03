# FedSecureAI Phase 14A — Network & Security Boundary Architecture

**Author:** Security Architect & Senior MLOps Engineer  
**Date:** October 27, 2026  
**Document Class:** Security & Deployment Governance  
**Service:** FedSecureAI FastAPI Inference Engine  

---

## 1. Network Boundary Architecture

The FedSecureAI inference service processes financial sequence evaluations and generates feature attributions for anti-money laundering investigations. Because the application intentionally lacks built-in authentication or access-control logic, **the network boundary serves as the primary perimeter defense.**

```
+-----------------------------------------------------------------------------+
|                           UNTRUSTED EXTERNAL CLIENTS                        |
+-----------------------------------------------------------------------------+
                                       │ (HTTPS / Public Traffic)
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                    AUTHENTICATED API GATEWAY / REVERSE PROXY                |
|                    (Envoy / Kong / Traefik / AWS API Gateway)               |
|                                                                             |
|  - TLS 1.3 Termination (mTLS / Public Certificates)                         |
|  - Token Verification (OAuth2 / JWT Bearer Validation)                      |
|  - API Key Enforcement (X-API-Key Header)                                   |
|  - Rate Limiting (Token Bucket: Predict 200 rps, Explain 5 rps)             |
|  - Request Body Filtering (Max 2 MB)                                        |
+─────────────────────────────────────────────────────────────────────────────+
                                       │ (Cleartext Internal VPC Traffic)
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|               FEDSECUREAI INFERENCE CONTAINER (RESTRICTED VPC)               |
|                                                                             |
|  - Host Loopback Binding: 127.0.0.1:8000:8000                              |
|  - Unprivileged User: fedsecure (UID 10001)                                 |
|  - Read-Only Root Filesystem + Memory tmpfs (/tmp)                          |
|  - Read-Only Checkpoint Mounts (/app/checkpoints)                           |
|  - Cryptographic Checkpoint Authentication (SHA-256 on startup)             |
|  - Privacy Logging: Zero accounts or features in logs                       |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Ingress & Access Control Rules

### 2.1 Host Binding Policy
- **Prohibited:** Binding container ports to `0.0.0.0` on any host interface with external routing.
- **Enforced:** In `docker-compose.yml`, the port is bound exclusively to `127.0.0.1:8000:8000`. This prevents external machines from communicating directly with the inference service without routing through the local host proxy.

### 2.2 API Gateway Responsibilities
Because the inference engine operates in an unauthenticated mode, any deployment that accepts client requests from external networks or distinct security zones **MUST** route through an authenticated API gateway implementing the following controls:

1. **Authentication Verification:**
   - Every request to `/predict`, `/predict/batch`, and `/explain` must present a validated authorization token (e.g. `Authorization: Bearer <JWT>`) or a registered API key (`X-API-Key`).
   - The gateway must reject unauthenticated requests with `HTTP 401 Unauthorized`.
2. **mTLS (Mutual TLS) for Enterprise Inter-Service Communication:**
   - Client certificates must be validated against the corporate PKI before proxying requests to the inference container.
3. **Compute Rate Limiting:**
   - Single prediction (`/predict`): Default quota of 200 requests/sec.
   - Batch prediction (`/predict/batch`): Default quota of 20 requests/sec (max 64 items per batch).
   - Integrated Gradients attribution (`/explain`): Default quota of 5 requests/sec. Because path integration computes multiple backward passes per sequence, rate limiting on `/explain` is mandatory to prevent compute starvation of operational scoring pipelines.

---

## 3. Data Privacy & Confidentiality Controls

1. **Zero Logging of Financial Records:**
   - The `RequestTimingAndPrivacyMiddleware` intercepts all HTTP calls and records exclusively: HTTP method, path, HTTP status code, and latency in milliseconds.
   - Raw transaction amounts, one-hot nominal vectors, client account identifiers, and risk classification rationale are never written to standard output or log streams.
2. **Payload Size Guardrail:**
   - Middleware drops requests exceeding 2 MB with `HTTP 413 Content Too Large` before deserialization.
3. **Non-Sensitive Correlation IDs:**
   - Response objects return ephemeral UUIDs (`pred_<uuid>`, `exp_<uuid>`) rather than mirroring client account IDs.

---

## 4. Cryptographic Model Provenance & Fail-Closed Behavior

1. **SHA-256 Weight Verification:**
   - The container cannot serve inferences from unknown or untrusted checkpoints.
   - The service verifies model weights against immutable SHA-256 digests:
     - HI-Small: `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4`
     - LI-Medium: `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2`
2. **Fail-Closed Guarantee:**
   - If a checkpoint is altered by even one bit, or if an adversary attempts to mount unauthorized weights, `ModelService` raises `ModelIntegrityError`, logs the cryptographic mismatch, and refuses to mark the service as ready.
   - Invocations to `/predict` while unready return `HTTP 503 Service Unavailable`.

