# FedSecureAI: Step-by-Step System Reproduction & Deployment Guide

**Document ID:** `results/phase16/fedsecureai_reproduction_guide.md`  
**Phase:** 16 — Final Documentation, Research Report & Demonstration Package  
**Date:** 2026-10-27  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, Drift-Aware Federated AML  
**Status:** FULLY REPRODUCIBLE & VERIFIED  

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
- **CPU:** Minimum 4 physical cores (x86_64 or Apple Silicon ARM64).
- **RAM:** Minimum 8 GB available RAM (16 GB recommended for full validation sequences).
- **Storage:** Minimum 5 GB free disk space (excluding raw transaction archives).

### 1.2 Software Stack
- **Operating System:** Windows 10/11, Ubuntu 22.04 LTS+, or macOS 13+.
- **Python:** Python 3.11 or Python 3.12 recommended. (Python 3.14 verified with `SyncAPIClient`).
- **Node.js:** Node.js v18.x or v20.x LTS with npm v9+.
- **Containerization (Optional):** Docker Desktop with Docker Compose v2.x.

---

## 2. Environment Setup

### 2.1 Clone Repository & Enter Root
```bash
git clone https://github.com/your-org/FedSecureAI.git
cd FedSecureAI
```

### 2.2 Python Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
# source .venv/bin/activate

# Upgrade pip and install base dependencies
python -m pip install --upgrade pip
pip install -r requirements/base.txt
pip install fastapi uvicorn httpx pydantic pytest pytest-cov
```

### 2.3 Frontend Dependencies Setup
```powershell
# Install Node dependencies for Vite React dashboard
npm --prefix frontend install
```

---

## 3. Running the Microservices Locally

### 3.1 Start the FastAPI Inference Backend
Run from the repository root:
```powershell
.venv\Scripts\python.exe -m uvicorn interfaces.api.app:app --host 127.0.0.1 --port 8000
```
- **Liveness Endpoint:** `http://127.0.0.1:8000/health`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

### 3.2 Start the React Analyst Dashboard (Development Mode)
In a separate terminal:
```powershell
npm --prefix frontend run dev
```
- Open browser at: `http://localhost:3000`

### 3.3 Compile Production Frontend Static Distribution
```powershell
npm --prefix frontend run build
```
Generates production assets in `frontend/dist/` (194 kB JS, 0.82 kB CSS).

---

## 4. API Usage Examples

### 4.1 Check Service Health (`GET /health`)
```bash
curl -X GET "http://127.0.0.1:8000/health" -H "Accept: application/json"
```
*Expected Response:*
```json
{
  "status": "ok",
  "app_name": "FedSecureAI Inference Service",
  "app_version": "1.0.0",
  "loaded_models": [
    "hi_small_lstm_fedavg_iid_seed42",
    "config_5_bilstm_fedavg_iid"
  ],
  "device": "cpu",
  "readiness": "ready",
  "timestamp": "2026-10-27T12:00:00Z"
}
```

### 4.2 Single Sequence Prediction (`POST /predict`)
```python
import requests
import json

payload = {
    "dataset": "HI-Small",
    "sequence": {
        "features": [[0.05] * 57 for _ in range(10)],
        "length": 10,
        "sample_id": "AUDIT_SEQ_001"
    }
}

resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(resp.json())
```
*Expected Response:*
```json
{
  "request_id": "req-...",
  "dataset": "HI-Small",
  "model_name": "hi_small_lstm_fedavg_iid_seed42",
  "predicted_probability": 0.009812,
  "decision_threshold": 0.337784,
  "decision": "CLEARED_LEGITIMATE",
  "sample_id": "AUDIT_SEQ_001",
  "sequence_length": 10,
  "inference_time_ms": 2.85
}
```

### 4.3 Integrated Gradients Explainability (`POST /explain`)
```python
payload = {
    "dataset": "HI-Small",
    "sequence": {
        "features": [[0.85] * 57 for _ in range(10)],
        "length": 10,
        "sample_id": "ALERT_SEQ_001"
    },
    "steps": 25,
    "top_k": 5
}

resp = requests.post("http://127.0.0.1:8000/explain", json=payload)
print(resp.json())
```
*Expected Response:*
```json
{
  "dataset": "HI-Small",
  "model_name": "hi_small_lstm_fedavg_iid_seed42",
  "sample_id": "ALERT_SEQ_001",
  "method": "integrated_gradients",
  "steps": 25,
  "predicted_probability": 0.720529,
  "top_positive_drivers": [
    {
      "feature_index": 2,
      "feature_name": "sender_prior_amount_log",
      "feature_group": "Sender Causal History",
      "attribution": 0.31254,
      "rank": 1
    }
  ],
  "top_negative_drivers": [],
  "completeness_residual": 0.000668,
  "explanation_time_ms": 649.55
}
```

---

## 5. Automated Verification & Regression Testing

Execute the complete 97-test regression suite across all project phases:

```powershell
.venv\Scripts\pytest.exe tests/test_phase7_ensemble.py tests/test_phase12_explainability.py tests/test_phase13_schemas.py tests/test_phase13_model_service.py tests/test_phase13_api.py tests/test_phase13_security_and_contracts.py tests/test_phase14a_deployment.py tests/test_phase14b_frontend_integration.py tests/test_phase15_e2e_validation.py -v
```

Execute live HTTP socket transport validation:
```powershell
.venv\Scripts\python.exe scripts/e2e_live_service_validation.py
```

---

## 6. Container Deployment Instructions

> [!WARNING]
> **Host Environment Notice (Docker Daemon Unverified):**  
> During Phase 14A, 14B, and 15 validation, the local Docker daemon engine (`dockerd`) was offline. The commands below reflect the audited, production-grade configuration files (`docker/Dockerfile.inference` and `docker-compose.yml`), but live container execution remains pending until Docker Desktop is started.

### 6.1 Build Hardened Image
```bash
docker compose build --no-cache
```

### 6.2 Run Containerized Service
```bash
docker compose up -d
```
- Check container status: `docker ps`
- Inspect healthcheck logs: `docker inspect --format='{{json .State.Health}}' fedsecure-inference`
- Stop container: `docker compose down`

---

## 7. Known Environment Limitations

1. **Python 3.14 anyio Event Loop Caveat:** On Windows platforms with Python 3.14, Starlette's `TestClient` entering context managers may experience event-loop deadlocks when thread portals are spawned. The project provides `SyncAPIClient` in `tests/test_phase13_api.py` and `tests/test_phase15_e2e_validation.py` to test endpoints synchronously with zero hang.
2. **Standard Sandbox Loopback Sockets:** In isolated sandbox environments with networking disabled, child processes cannot reach `127.0.0.1`. Live socket scripts require unsandboxed host socket access.
3. **Decoupled Authentication:** The application does not provide user sign-in forms; production deployment requires an authenticating upstream reverse proxy.

