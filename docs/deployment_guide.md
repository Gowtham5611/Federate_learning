# FedSecureAI — Enterprise Deployment & Operations Guide

**System Version:** 2.1.0 Enterprise  
**Target Environments:** On-Premise Secure Silos • Private Cloud (GCP/AWS/Azure) • Local Verification  
**Components:** FastAPI Backend Service • React/Vite Analytics Dashboard  

---

## 1. System Requirements & Prerequisites

### Hardware Requirements
- **Inference Server**: Minimum 4 CPU cores, 16 GB RAM (32 GB recommended for batch jobs $> 100\text{k}$ transactions), 20 GB SSD storage.
- **GPU Acceleration**: Optional (NVIDIA CUDA 11.8+ / 12.0+ supported; CPU is verified for inference latency $< 15$ ms/seq).
- **Client Nodes**: Minimum 2 CPU cores, 8 GB RAM per participating federated silo.

### Software Prerequisites
- **Python Runtime**: Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Node.js Runtime**: Node.js 18.x, 20.x, 22.x, or 24.x with npm 9+.
- **Git**: 2.30+ for repository provenance tracking.

---

## 2. Installation & Setup

### A. Clone and Python Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd Federate_learning

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install core Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### B. Frontend Setup & Build
```bash
# Install frontend npm dependencies
npm --prefix frontend install

# Verify clean production TypeScript build
npm --prefix frontend run build
```

---

## 3. Starting Services

### A. FastAPI Inference Backend
To start the backend in production mode with preloaded models:
```bash
# Set Python path to project root
export PYTHONPATH=.   # On Linux/macOS
$env:PYTHONPATH="."   # On Windows PowerShell

# Start ASGI production server on port 8000
python -m uvicorn backend.interfaces.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### B. React Analytics Dashboard
To run the frontend dashboard:
```bash
# Development server (port 5173):
npm --prefix frontend run dev

# Or serve production build:
npm --prefix frontend run preview -- --port 3000
```

---

## 4. Environment Variables & Production Configuration

The following configuration variables can be passed via `.env` or container environment:

| Variable Name | Default Value | Description |
|:---|:---|:---|
| `FEDSECUREAI_PRELOAD_MODELS` | `true` | Preloads PyTorch models into memory during startup. |
| `FEDSECUREAI_ENV` | `production` | Enables sanitized error responses and suppresses stack traces. |
| `FEDSECUREAI_MAX_UPLOAD_MB` | `25` | Maximum allowed CSV upload size in megabytes. |
| `FEDSECUREAI_MAX_ROWS` | `50000` | Hard cap on evaluated transactions per request. |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Frontend backend API URL. |

---

## 5. Security & Network Hardening

### Reverse Proxy (Nginx Configuration Example)
In production, place the FastAPI ASGI service behind Nginx or an enterprise load balancer with SSL/TLS termination:

```nginx
server {
    listen 443 ssl http2;
    server_name aml.internal.bank.net;

    ssl_certificate /etc/ssl/certs/aml_bank.crt;
    ssl_certificate_key /etc/ssl/private/aml_bank.key;

    client_max_body_size 30M;

    # Frontend Static Distribution
    location / {
        root /var/www/fedsecureai/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

---

## 6. Health Checks & Monitoring

- **Liveness Probe**: `GET /health` (returns HTTP 200 with status `"healthy"` when online).
- **Readiness Probe**: `GET /health` (inspects `"readiness": "ready"` confirming models are verified and resident in memory).
- **Model Inventory Probe**: `GET /api/v1/models` (validates all 27 neural checkpoints are available).

