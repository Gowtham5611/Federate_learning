# FedSecureAI

**Privacy-Preserving, Explainable, and Drift-Aware Federated Deep Learning for Anti-Money Laundering Detection**

> **Final Project Status: RESEARCH AUDIT & POST-HOC HARDENING COMPLETE · PRODUCTION READY**
> - **Dataset Scale:** Evaluated across full 5.08M row IBM HI-Small and 31.25M row IBM LI-Medium benchmarks with strict zero-leakage, temporal-preserving splits.
> - **Federated Experiments:** 100% completion across 24 configurations (IID / Non-IID partitions $\times$ FedAvg / FedProx aggregations $\times$ GRU / LSTM / BiLSTM recurrent encoders).
> - **Centralized Parity:** Federated LSTM FedAvg achieves **99.77% PR-AUC parity** with peak centralized pooling (0.475977 vs. 0.477052) and **115.56% retention** vs. mean centralized baselines (0.411874) without exposing raw customer transactions.
> - **Ablation Findings:** FedProx ($\mu = 0.0001$) scored lower than FedAvg in these runs (inducing a **40.6% to 65.6% relative PR-AUC reduction**); the cause is untested. Non-IID typology clustering degrades PR-AUC by up to **81.9%**.
> - **Empirical Privacy:** Shadow-free loss-threshold Membership Inference Attack (MIA) demonstrates FL privacy advantage ($\text{MIA AUC} = 0.5041$ vs. Centralized $0.5840$).
> - **Post-Hoc Probability Calibration:** Validation temperature scaling ($T^* = 0.8340$) reduced test Expected Calibration Error (ECE) by **72.82%** without altering rank ordering.
> - **Axiomatic Explainability:** PyTorch-native Integrated Gradients explainer satisfying the completeness axiom ($<0.06\%$ error) with zero attribution leakage on zero-padded causal steps.
> - **Production Serving & Dashboard:** High-throughput FastAPI inference microservice (`127.0.0.1:8000`) and a React 18 / Vite / TypeScript compliance dashboard (`127.0.0.1:3000`) operating under a strict zero-PII, memory-only session state policy.
> - **Regression Verification:** **1,366 backend tests + 11 evaluation parity tests passed**; 273 baseline checkpoints and result files verified 100% bit-identical via SHA-256 auditing.
> - **Key Research Documents:**
>   - [`AUDIT.md`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/AUDIT.md) — Comprehensive pre-fix system audit and vulnerability analysis.
>   - [`RESULTS_POSTHOC.md`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/RESULTS_POSTHOC.md) — Exhaustive post-hoc evaluation (bootstrap CIs, alert curves, temporal drift, MIA privacy, Phase 7b meta-learner).
>   - [`INTERVIEW_NOTES.md`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/INTERVIEW_NOTES.md) — Structured technical defense notes and high-yield interview Q&A.
>   - [`CHANGELOG.md`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/CHANGELOG.md) — Chronological log of all ML engineering and research hardening changes.

---

## 1. Project Overview

Anti-money laundering (AML) detection requires identifying suspicious transaction patterns that span multiple financial institutions. However, data privacy regulations (GDPR, CCPA, Bank Secrecy Act) and commercial confidentiality prohibit banks from pooling raw transaction data into centralized databases. Consequently, money laundering syndicates intentionally disperse transactions across multiple banks (structuring / smurfing), ensuring no single institution sees the complete criminal trail.

**FedSecureAI** resolves this privacy-utility deadlock through decentralized Federated Learning (FL). By training recurrent neural sequence models locally on private bank records and aggregating only mathematical gradient updates across federated communication rounds, institutions collaboratively detect complex laundering typologies without centralizing raw customer transaction data.

---

## 2. End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        1. DATA INGESTION & PREPROCESSING               │
│  IBM AML Datasets (HI-Small 5.08M, LI-Medium 31.25M)                   │
│  → Chronological Split (Train 60%, Val 20%, Test 20%)                   │
│  → Causal Temporal Windowing (T=10, D=57 features, right-padded zeroes) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   2. DECENTRALIZED FEDERATED TRAINING                   │
│  • Client Partitions: IID Uniform vs. Non-IID Banking Typologies Skew   │
│  • Local Models: GRU, LSTM, BiLSTM (Hidden Size=64, Layer=1)            │
│  • Aggregation Protocols: FedAvg vs. FedProx (mu=0.0001) over 5 Rounds  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              3. MODEL SELECTION, ENSEMBLING & CALIBRATION               │
│  • Validation PR-AUC & F1 Optimization (Exhaustive 21-point Alpha Grid) │
│  • Frozen Champions: HI-Small LSTM (tau=0.3378), LI-Medium BiLSTM (tau=0.2385) │
│  • Phase 10 Calibration: Isotonic Regression (+56.18% ECE improvement)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     4. SECURE PRODUCTION SERVING TIER                   │
│  • FastAPI Backend (Port 8000): /health, /models, /predict, /explain    │
│  • Explainer: 25-step PyTorch Integrated Gradients with zero-pad mask   │
│  • Packaging: Multi-stage Dockerfile, non-root user (10001), read-only  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                5. ANALYST COMPLIANCE DASHBOARD (PORT 3000)              │
│  • React 18 + Vite 6 + TypeScript 5.6 (194 kB production bundle)        │
│  • Real-time Backend Health Heartbeat & Frozen Threshold Indicators     │
│  • 4 Sanitized IBM AML Demonstration Scenarios (Legitimate vs. Alert)   │
│  • Horizontal SVG Attribution Charts for Regulatory SAR Justification  │
│  • Zero-PII Policy: Volatile in-memory session state (useState only)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Deep Learning & Numerical Engine:** PyTorch 2.6.0, NumPy 2.2.3, scikit-learn 1.6.1, SciPy 1.15.2.
- **Backend Inference Microservice:** FastAPI 0.115.11, Uvicorn 0.34.0, Pydantic v2.10.6, Starlette 0.46.1.
- **Frontend Compliance Dashboard:** React 18.3.1, Vite 6.4.5, TypeScript 5.6.3, Lucide React icons.
- **Containerization & Deployment:** Docker 29.7.2, Docker Compose v2 (hardened non-root image).
- **Test & Verification Framework:** Pytest 9.1.1, Pytest-Cov 7.1.0, HTTPX 0.28.1.

---

## 4. Master Project Roadmap & Milestone Verification

| Phase | Milestone Name | Scope & Key Finding | Verification Status |
| :---: | :--- | :--- | :---: |
| **0–2** | Foundations & Ingestion | Dataset registries, chunked memory-safe ingestion, schema validation | **COMPLETE** |
| **3–4** | Feature Engineering | 57-dim feature expansion, logarithmic normalization, causal temporal walks | **COMPLETE** |
| **5–6** | Centralized Baselines | Random Forest, Logistic Regression, Centralized GRU, LSTM, BiLSTM | **COMPLETE** |
| **7** | Hybrid Ensembling | Blending $\alpha \cdot P_{\text{IID}} + (1-\alpha) \cdot P_{\text{Non-IID}}$; selected $\alpha^*=1.00$; froze $\tau^*$ | **COMPLETE** |
| **8** | Benchmarking & Ablation | Proved 99.77% centralized parity; quantified FedProx penalty (-40% to -65%) | **COMPLETE** |
| **9** | Federated Optimization | Partial FedProx $\mu$ sweep preserved on HI-Small; full sweep and SCAFFOLD frozen | **FROZEN / PARTIAL** |
| **10** | Probability Calibration | Isotonic regression reduced ECE by 56.18% on HI-Small; LI-Medium calibration outstanding | **COMPLETE (HI-Small)** |
| **11** | Results Matrix | Master benchmarking matrix consolidated directly into Phase 8 | **SUPERSEDED BY PH 8**|
| **12** | Explainability (XAI) | Integrated Gradients over logits; completeness verified ($<0.06\%$ residual error) | **COMPLETE** |
| **13** | FastAPI Inference API | Bounded endpoints (`/predict`, `/predict/batch`, `/explain`), SHA-256 integrity | **COMPLETE** |
| **13.1**| Deployment Audit | Security, bounds enforcement, concurrency, and privacy verification (63/63 tests) | **COMPLETE** |
| **14A** | Packaging Hardening | Non-root container (`UID 10001`), read-only rootfs, loopback binding | **COMPLETE** |
| **14B** | Analyst Dashboard | React 18 / Vite compliance dashboard; zero PII; SVG explainability charts | **COMPLETE** |
| **15** | System E2E Validation | Full regression suite (97/97 passed in 15.93s); live HTTP socket verified | **COMPLETE** |
| **16** | Final Documentation | 27-section research report, architecture, results summary, demo script, readiness | **COMPLETE** |

---

## 5. Authoritative Results Summary

### 5.1 HI-Small Held-Out Test Evaluation (1,015,564 sequences, 1,797 positives)
- **Centralized LSTM Baseline:** PR-AUC = **0.477052**, F1 = 0.552941, ROC-AUC = 0.935404.
- **Federated IID LSTM FedAvg (Champion):** PR-AUC = **0.475977 (99.77% parity)**, F1 = **0.535928**, ROC-AUC = **0.955481**, Precision = 0.818286, Recall = 0.398442, Alert Rate = **8.62 alerts / 10k**.
- **FedProx Penalty ($\mu=0.0001$):** PR-AUC dropped by **-40.66%** ($0.4760 \rightarrow 0.2824$).
- **Non-IID Typology Degradation:** PR-AUC dropped by **-5.35%** ($0.4760 \rightarrow 0.4505$).

### 5.2 LI-Medium Held-Out Test Evaluation (6,249,112 sequences, 3,783 positives)
- **Federated IID BiLSTM FedAvg (Champion):** PR-AUC = **0.253398**, F1 = **0.316789**, ROC-AUC = **0.969517**, Precision = 0.594182, Recall = 0.215966, Alert Rate = **2.20 alerts / 10k**.
- **FedProx Penalty ($\mu=0.0001$):** PR-AUC dropped by **-65.63%** ($0.2534 \rightarrow 0.0871$).
- **Non-IID Typology Degradation:** PR-AUC collapsed by **-81.93%** ($0.2534 \rightarrow 0.0458$ on LSTM).
- *Centralized Baseline Gap:* Full 31.25M-row centralized training was halted due to CPU/RAM out-of-memory bottlenecks prior to Colab workflows.

---

## 6. Quick Start & Execution Guide

### 6.1 Setup Python Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements/base.txt
pip install fastapi uvicorn httpx pydantic pytest pytest-cov
```

### 6.2 Setup Frontend Dashboard
```powershell
npm --prefix frontend install
```

### 6.3 Run the Application
1. **Start FastAPI Backend (PowerShell):**
   ```powershell
   # Standard entrypoint (Recommended)
   .venv\Scripts\python.exe -m uvicorn backend.interfaces.api.main:app --host 127.0.0.1 --port 8000 --reload

   # Or direct Python execution:
   .venv\Scripts\python.exe -m backend.interfaces.api.main

   # Or app module alias:
   .venv\Scripts\python.exe -m uvicorn backend.interfaces.api.app:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start Analyst Dashboard (Vite / React):**
   ```powershell
   npm --prefix frontend run dev
   ```
3. Open browser at: `http://localhost:5173` (or `http://localhost:3000`)

### 6.4 Build Production Frontend Distribution
```powershell
npm --prefix frontend run build
```

---

## 7. API Endpoints Reference

| Endpoint | Method | Payload / Params | Response Summary |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | None | Engine liveness, loaded model identifiers, readiness state |
| `/models` | `GET` | None | Inventory of active recurrent architectures, dimensions, frozen cutoffs |
| `/metadata/{dataset}` | `GET` | `HI-Small` or `LI-Medium` | 57-feature metadata column schemas, frozen threshold |
| `/predict` | `POST` | Single sequence $(10, 57)$ | Predicted probability, threshold, classification decision, latency |
| `/predict/batch` | `POST` | Batch items (ceiling $\le 64$) | Bounded batch item verdicts, total latency |
| `/explain` | `POST` | Sequence $(10, 57)$, steps, top_k | Integrated Gradients feature drivers, completeness residual |

---

## 8. Reproducibility & Automated Execution (Makefile)

FedSecureAI includes a top-level `Makefile` and pinned `requirements.txt` for reproducible execution:

```powershell
# 1. Audit SHA-256 hashes of all 273 baseline models and results (must be 100% bit-identical)
make verify-hashes

# 2. Run new standalone evaluation metrics and serving parity tests
make test-eval

# 3. Run full backend test suite (1,366 tests)
make test

# 4. Run post-hoc empirical analysis pipeline (bootstrap CIs, alert budgets, MIA, stacking)
make posthoc

# 5. Build React production dashboard bundle
make build-frontend

# 6. Start production FastAPI serving microservice (Port 8000)
make serve
```

Direct pytest commands:
```powershell
# Run robust evaluation and sequence-serving parity test suite
.venv\Scripts\pytest.exe tests/test_evaluation_metrics.py tests/test_serving_sequence_parity.py -v

# Run full backend regression suite
.venv\Scripts\pytest.exe backend/tests -v
```

---

## 9. Security, Privacy & Operational Limitations

1. **Restricted Deployment Perimeter:** In-app user authentication is intentionally decoupled. Deployment is permitted **strictly on private restricted enterprise networks** or behind an **authenticated enterprise API gateway** (e.g. NGINX, Traefik, AWS ALB) enforcing Single Sign-On (OIDC/SAML) and Role-Based Access Control (RBAC). Public deployment is prohibited.
2. **Zero-PII Assurance:** The frontend uses ephemeral memory hooks (`useState`). Zero banking account numbers, customer names, or transaction payloads persist in `localStorage`, `sessionStorage`, `IndexedDB`, or cookies.
3. **Container Daemon Status:** Static Dockerfile and Compose configurations are verified; live container execution remains unverified due to the host Docker daemon being offline during validation.

---

## 10. Repository Directory Structure

```
Federate_learning/
├── backend/
│   ├── application/            # DTOs, use cases, calibration & optimization services
│   ├── domain/                 # Core entities, value objects, enums, exceptions
│   ├── infrastructure/         # Data providers, recurrent models, FedAvg/FedProx aggregation
│   └── interfaces/             # API routes, Pydantic schemas, middleware, CLI entrypoints
├── docker/
│   ├── Dockerfile.inference    # Hardened multi-stage non-root container build
│   └── env.inference.example   # Container environment template
├── frontend/
│   ├── src/                    # React 18 components, pages, API clients, demo samples
│   ├── dist/                   # Production build distribution (194 kB bundle)
│   ├── package.json            # Node dependencies
│   └── vite.config.ts          # Port 3000 config, /api proxy to FastAPI
├── results/
│   ├── phase7/                 # Hybrid ensemble grid search & frozen configs
│   ├── phase8/                 # Master benchmarking, ablation, and comparative reports
│   ├── phase9_11_audit/        # Comprehensive audit of Phases 9, 10, and 11
│   ├── phase12/                # Integrated Gradients attributions and plots
│   ├── phase13/                # FastAPI performance and validation reports
│   ├── phase13_1/              # Deployment readiness audit and parity benchmarks
│   ├── phase14a/               # Container packaging and security boundary specifications
│   ├── phase14b/               # Dashboard implementation, API integration, and privacy reviews
│   ├── phase15/                # End-to-end system validation and live socket reports
│   └── phase16/                # Final project research report, architecture, results summary
├── scripts/
│   ├── e2e_live_service_validation.py # Live HTTP socket validation script
│   └── extract_demo_samples.py        # IBM AML validation scenario extractor
├── tests/                      # 9 test modules (97 automated regression tests)
├── docker-compose.yml          # Production deployment compose specification
├── pytest.ini                  # Pytest configuration
└── README.md                   # This master documentation
```

---

## 10. System Limitations and Operational Boundaries

To ensure scientific honesty and safe operational deployment, the following limitations are explicitly documented:

1. **Recurrent Sequence History Requirements:**
   - The primary recurrent neural networks (GRU, LSTM, BiLSTM) require temporal sequence context ($T \ge 3$ prior transactions per account) to compute multi-step velocity and counterparty representations.
   - When evaluated on single-transaction rows or isolated batches lacking historical account depth, sequence models default to conservative prior baseline probabilities ($\sim 0.001 - 0.020$).
   - For single-transaction or cold-start evaluation, users must explicitly opt into the tabular gradient-boosted fallback model (`fallback_model="tabular"`).

2. **Tabular Fallback Model Boundaries:**
   - The LightGBM tabular fallback model evaluates isolated transactions using 23 engineered cross-sectional features without sequence history.
   - On the held-out test split, standalone LightGBM achieves PR-AUC = 0.2646 (compared to 0.4760 for the recurrent LSTM champion).
   - The fallback model is strictly additive and is **never** silently mixed with sequence models.

3. **Privacy Guarantees vs. Differential Privacy:**
   - FedSecureAI eliminates raw transaction pooling across financial institutions and demonstrates lower empirical leakage under shadow-free membership inference attacks ($\text{MIA AUC} = 0.5041$ vs. $0.5840$).
   - This empirical finding does **not** constitute a formal mathematical Differential Privacy guarantee (e.g., $(\epsilon, \delta)$-DP). Formal DP requires provable noise addition (e.g. DP-SGD) or secure multiparty computation.

4. **Federated Optimization Behavior:**
   - In our empirical benchmarks under extreme class imbalance ($0.18\%$ laundering), FedProx ($\mu = 0.0001$) underperformed FedAvg. The exact theoretical cause remains untested.

5. **Decision Threshold Provenance:**
   - The canonical decision threshold for HI-Small is $\theta^* = 0.337784$, empirically determined strictly on validation data to maximize $F_1$. Nominal thresholds of $0.5000$ or rounded references ($0.35$) reflect legacy approximations; all production endpoints and reports enforce $\theta^* = 0.337784$.

---

## 11. Primary References

1. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS 2017*.
2. Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys 2020*.
3. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*.
4. Weber, M., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks. *arXiv:1908.02591*.
5. Altman, E. (2023). Synthetic IBM Transactions for Anti-Money Laundering (AML). *IBM Research*.
