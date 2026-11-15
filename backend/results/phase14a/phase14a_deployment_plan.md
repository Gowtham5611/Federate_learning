# FedSecureAI Phase 14A — Secure Deployment Plan & Packaging Architecture

**Author:** FedSecureAI MLOps & Systems Engineering Team  
**Date:** October 27, 2026  
**Status:** READY FOR RESTRICTED DEPLOYMENT  
**Target:** Production FastAPI Federated AML Inference Engine (`backend/interfaces/api`)  

---

## 1. Executive Summary

Phase 14A establishes the hardened, containerized packaging architecture for the FedSecureAI inference service. The design enforces strict least-privilege principles:
- **Zero Training Contamination:** Training tools (`flwr`), test harnesses (`pytest`), dataset downloaders (`kaggle`), and Jupyter notebooks are completely excluded from the runtime image.
- **Minimal Surface Area:** Built from `python:3.11-slim-bookworm` with CPU-only PyTorch, reducing image footprint by $> 70\%$.
- **Unprivileged Execution:** The service runs as unprivileged system user `fedsecure` (`UID 10001`, `GID 10001`).
- **Immutable Checkpoints:** The two authoritative model checkpoints (`hi_small_lstm_fedavg_iid_seed42` and `config_5_bilstm_fedavg_iid`) are mounted into `/app/checkpoints/` as **read-only volumes** with strict SHA-256 verification on startup.
- **Read-Only Root Filesystem:** The container runs with `read_only: true` and an in-memory `tmpfs` volume mounted at `/tmp` for temporary scratch allocation.
- **Network Boundary:** The service binds strictly to localhost (`127.0.0.1`) or an isolated internal container bridge network. Ingress authentication must be handled by an external API Gateway.

---

## 2. Packaging Architecture & Specifications

### 2.1 File Layout & Packaging Tree
```
c:\Users\Acer\OneDrive\Desktop\Federate_learning/
├── .dockerignore                     # Strict build context filter
├── docker-compose.yml                 # Hardened local deployment compose spec
├── docker/
│   ├── Dockerfile.inference          # Multi-stage production containerfile
│   └── env.inference.example         # Template for environment configuration
├── requirements/
│   ├── inference.txt                 # Runtime-only dependencies (no test/train tools)
│   └── base.txt                      # Baseline project dependencies
└── backend/interfaces/api/           # Application code
```

### 2.2 Base Image & Runtime Environment
- **Base Image:** `python:3.11-slim-bookworm` (Official Debian Bookworm LTS with active CVE remediation and glibc compatibility).
- **Python Version:** Python 3.11 LTS.
- **User Identity:** Non-root system account `fedsecure:10001:10001` with shell `/bin/false`.
- **PyTorch Engine:** CPU-only wheel from `https://download.pytorch.org/whl/cpu` (avoids 2.5 GB of unused CUDA binaries).

### 2.3 Checkpoint Distribution & Verification Strategy
Rather than baking multi-megabyte checkpoints into mutable image layers:
1. Checkpoint files remain stored in authoritative host storage (`results/HI_Small_Results/` and `results/LI_Medium_Results/`).
2. At container startup, the files are bind-mounted read-only (`:ro`) into `/app/checkpoints/`.
3. `ModelService._load_single_model` validates the SHA-256 hash of both files:
   - HI-Small: `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4`
   - LI-Medium: `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2`
4. If a checkpoint is missing, corrupted, or has an invalid hash, the container exits immediately with `ModelIntegrityError` (fail-closed behavior).

### 2.4 Data Exclusion Policy
The container **strictly excludes** raw AML CSV datasets and all partition splits (`train/`, `validation/`, `test/`). Only the lightweight JSON schema descriptors (`data/processed/ibm_aml/*/metadata/`) are packaged into `/app/data/`, ensuring total customer data isolation.

---

## 3. Step-by-Step Operator Guide

### 3.1 Building the Container Image
```bash
docker build -t fedsecureai-inference:1.0.0 -f docker/Dockerfile.inference .
```

### 3.2 Running the Container via Docker Compose
```bash
# Launch in background with read-only root and internal loopback binding
docker compose up -d

# View startup logs and model verification
docker compose logs -f fedsecure-inference
```

### 3.3 Verifying Health and Model Readiness
```bash
# Inspect container healthcheck status
docker inspect --format='{{json .State.Health}}' fedsecure-inference

# Query health endpoint via curl
curl -s http://127.0.0.1:8000/health
```

Expected output:
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
  "readiness": "ready"
}
```

### 3.4 Stopping the Service
```bash
docker compose down
```

---

## 4. Operational Ingress Restrictions

Because the FastAPI application intentionally does **NOT** embed an internal user database or token verification mechanism:
1. **Never bind to `0.0.0.0` on a public host.** The host binding must remain `127.0.0.1:8000:8000` or attach directly to an internal Docker network overlay.
2. **Mandatory Authenticated Gateway:** Ingress traffic must pass through a reverse proxy (e.g., Kong, Envoy, Traefik, NGINX) that terminates TLS and validates incoming JWT tokens, OAuth2 credentials, or API keys before proxying requests to `fedsecure-inference:8000`.

