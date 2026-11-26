---
title: FedSecureAI Backend
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FedSecureAI Backend Inference Service

FastAPI backend for FedSecureAI (privacy-preserving federated AML transaction evaluation).
Exposes `/health`, `/ready`, `/api/v1/csv/inspect`, `/api/v1/predict/jobs`, and `/api/v1/predict/upload` on port `7860`.
