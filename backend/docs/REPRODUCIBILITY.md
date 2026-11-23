# FedSecureAI: Reproduction Guide & Experimental Protocol

This document provides exact specifications, cryptographic checksums, and execution instructions to reproduce all empirical benchmarks, data audits, and validation runs reported in FedSecureAI.

---

## 1. System Requirements & Software Environment

### 1.1 Operating Environment
- **Operating Systems Tested:** Windows 11 Enterprise (x64) and Linux (Ubuntu 22.04 LTS).
- **Python Version:** 3.10 – 3.14 (Reference: Python 3.14.7 in local virtual environment `.venv`).
- **Node.js Environment:** Node.js v18.0.0+ (Tested with npm 10+ and Vite 6.0).
- **Hardware Profile:**
  - Minimum: 16 GB RAM, 4-core modern x86_64 CPU (CPU inference execution supported).
  - Recommended for Training: 32 GB RAM, NVIDIA RTX 3080/4090 or A100 (CUDA 12.x).

### 1.2 Core Python Dependencies
```text
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
pytest>=8.0.0
```

---

## 2. Cryptographic Checksums & Checkpoint Integrity

All model evaluations use fixed weights and immutable data splits verified via SHA-256 hashes.

### 2.1 Dataset Split Checksums
- **HI-Small Validation Split:**
  - Path: `data/processed/ibm_aml/HI-Small/validation/transactions.csv`
  - SHA-256: `bc35e07db44366c4cca133b2255adcd9340206fd4d98871a5ab114e782a8acfc`
- **LI-Medium Validation Split:**
  - Path: `data/processed/ibm_aml/LI-Medium/validation/transactions.csv`
  - SHA-256: `0e4c0369464d91ab1c3c672a105dc06a8c100f1ec341d6cb8f6820a8fc5bbcb1`

### 2.2 Model Checkpoint Checksums
- **HI-Small Lead IID Model (LSTM FedAvg IID):**
  - Path: `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt`
  - SHA-256: `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4`
  - Frozen Standalone Threshold: `0.3377837836742401`
- **HI-Small Lead Non-IID Model (LSTM FedAvg Non-IID):**
  - Path: `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_seed42/hi_small_lstm_fedavg_seed42_global_best.pt`
  - SHA-256: `1b64f79ddd4dc85d6bad953e93f2679bc102bf5fdbffeaf8b59df69ddd19ab1c`
  - Frozen Standalone Threshold: `0.3678055703639984`
- **LI-Medium Lead IID Model (LSTM FedAvg IID):**
  - Path: `results/LI_Medium_Results/checkpoints/config_5/global_best.pt`
  - SHA-256: `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2`
  - Frozen Standalone Threshold: `0.2384829968214035`
- **LI-Medium Lead Non-IID Model (BiLSTM FedAvg Non-IID):**
  - Path: `results/LI_Medium_Results/checkpoints/config_9/global_best.pt`
  - SHA-256: `91ea59ef4ad8d183006bda47c44b91d7ce8ea1f52c43f3f548b2763a2b0f3653`
  - Frozen Standalone Threshold: `0.3342503607273102`

---

## 3. Step-by-Step Reproduction Workflow

### Step 1: Environment Setup
```powershell
# Clone or navigate to the workspace
cd c:\Users\Acer\OneDrive\Desktop\Federate_learning

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python version and PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}')"
```

### Step 2: Execute Data Quality and Provenance Audit
To verify zero missing values, zero numeric anomalies, exact chronological boundaries ($T_{\text{train}} \le T_{\text{val}} \le T_{\text{test}}$), and class distribution stability:
```powershell
python backend/infrastructure/data/validators/research_data_quality_auditor.py
```
*Outputs generated:* `reports/data_quality_report.json` and `reports/data_quality_report.md`.

### Step 3: Execute Partition Heterogeneity & Atomicity Audit
To compute Total Variation Distance ($TVD$), 1-Wasserstein ($W_1$) metric shifts, and verify account atomicity:
```powershell
python backend/infrastructure/data/partitioners/partition_heterogeneity_auditor.py
```
*Outputs generated:* `reports/partition_report.json` and `reports/partition_report.md`.

### Step 4: Run Complete Automated Test Suite
To execute all unit, integration, and research verification tests:
```powershell
python -m pytest tests/ -v
```
*Expected result:* 100% test pass rate across all suites.

### Step 5: Launch FastAPI Production Inference Backend
```powershell
python -m uvicorn interfaces.api.app:app --host 127.0.0.1 --port 8000
```
Verify health endpoints:
```powershell
curl http://127.0.0.1:8000/api/v1/health
```

### Step 6: Build and Launch React Analyst Dashboard
```powershell
cd frontend
npm install
npm run build
npm run preview
```
Open `http://localhost:4173` or `http://localhost:5173` in a web browser to access the AML Research Dashboard.

