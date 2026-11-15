# Phase 14B — AML Analyst Dashboard Implementation Report

**Document ID:** `results/phase14b/phase14b_implementation_report.md`  
**Phase:** 14B — AML Analyst Dashboard & API Integration  
**Date:** 2026-10-27  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, Drift-Aware Federated AML  
**Status:** COMPLETE  

---

## 1. Executive Summary

Phase 14B delivers an auditable, lightweight, modern web-based Anti-Money Laundering (AML) analyst dashboard designed specifically for financial crimes compliance officers and ML operations specialists. Built with React 18, Vite 6, and TypeScript 5.6, the dashboard connects directly to the Phase 13 FastAPI inference service (`http://127.0.0.1:8000`), allowing analysts to inspect serving models, evaluate real transaction sequences, examine deterministic risk scores against frozen thresholds, and visualize feature-level attributions via Integrated Gradients.

The dashboard enforces complete architectural separation: zero model preprocessing or weight calculation is duplicated in the frontend; all predictions and feature attributions are computed strictly by the backend FastAPI service.

---

## 2. Frontend Architecture & Technology Stack

### 2.1 Core Technology Selection
- **Framework:** React 18.3.1 (Functional components with React Hooks)
- **Tooling & Bundler:** Vite 6.4.5 (Lightning-fast HMR and tree-shaking ESBuild production minification)
- **Type Safety:** TypeScript 5.6.3 in `strict: true` mode
- **UI & Iconography:** Modular CSS design with Lucide React (`lucide-react`) icons
- **Visualization:** Native SVG horizontal bar chart engine for Integrated Gradients feature attributions (zero bloated charting libraries)
- **HTTP Client:** Native `fetch` with typed API adapters and automatic timeout management

### 2.2 Directory Layout

```
frontend/
├── dist/                          # Production build output (194 kB JS, 0.82 kB CSS)
├── node_modules/                  # Sandboxed NPM dependencies
├── src/
│   ├── components/                # Modular UI components
│   │   ├── AttributionBarChart.tsx # SVG bar chart for positive/negative drivers
│   │   ├── Header.tsx             # Global navigation bar with live status badge
│   │   ├── ModelCard.tsx          # Real-time model metadata and status card
│   │   └── PredictionHistoryTable.tsx # In-memory session evaluation ledger
│   ├── demo/
│   │   └── demoSamples.ts         # Sanitized IBM AML validation test sequences
│   ├── pages/
│   │   ├── DashboardOverview.tsx  # Service health, model inventory & system limits
│   │   └── PredictionPage.tsx     # Transaction scoring, thresholds & XAI analysis
│   ├── services/
│   │   └── apiService.ts          # Typed HTTP client for FastAPI endpoints
│   ├── types/
│   │   └── api.ts                 # TypeScript interfaces mirrored from Pydantic schemas
│   ├── utils/
│   │   └── formatters.ts          # Probability, risk tier, and feature name utilities
│   ├── App.tsx                    # Root application component with page routing
│   ├── index.css                  # Dark-theme compliance design stylesheet
│   ├── main.tsx                   # React DOM entrypoint
│   └── vite-env.d.ts              # Vite environment typings
├── index.html                     # HTML5 entrypoint with security meta tags
├── package.json                   # Dependency definitions and scripts
├── tsconfig.json                  # TypeScript compiler options
└── vite.config.ts                 # Vite bundler configuration (port 3000, /api proxy)
```

---

## 3. UI Components & Analyst User Flows

### 3.1 Header & System Status
The top bar persistently queries `GET /health` to confirm the operational state of the inference engine. If the backend is running and healthy, a glowing green `Engine Online` indicator is presented alongside the exact model versions (`hi_small_lstm_fedavg_iid_seed42` and `config_5_bilstm_fedavg_iid`). If the backend is unreachable, a persistent amber warning prompts the analyst to verify the API gateway.

### 3.2 Dashboard Overview Page
Provides an inventory of all active models and dataset configurations:
- **Model Cards:** Displays architecture (LSTM / BiLSTM), parameters count, hidden dimension (64), sequence length limit ($T \le 10$), input feature dimension ($D = 57$), and frozen decision thresholds.
- **Serving Architecture Specifications:** Summarizes the operational constraints: 25 Integrated Gradients interpolation steps, batch size ceiling of 64, and fails-closed SHA-256 weight integrity enforcement.

### 3.3 Transaction Evaluation & Explanation Workspace
The core compliance workspace enables investigators to inspect transaction sequences:
- **Demonstration Scenarios:** Investigators can toggle pre-loaded, sanitized IBM AML validation scenarios:
  1. `HI-Small — Routine Standard Transaction Pattern` (Expected: Cleared Legitimate, $P < 0.01$)
  2. `HI-Small — High-Risk Rapid Structuring / Dispersion Pattern` (Expected: Flagged Laundering, $P > 0.35$)
  3. `LI-Medium — Routine Standard Transaction Pattern` (Expected: Cleared Legitimate, $P < 0.01$)
  4. `LI-Medium — High-Risk Rapid Structuring / Dispersion Pattern` (Expected: Flagged Laundering, $P > 0.35$)
- **Custom JSON Ingestion:** Allows analysts to paste any valid `(10, 57)` transaction sequence JSON for real-time scoring.
- **Risk Assessment Panel:**
  - Dynamic score gauge rendering the raw calibrated probability $P$.
  - Clear compliance badge: `CLEARED_LEGITIMATE` (Green) vs. `FLAGGED_LAUNDERING` (Red).
  - Explicit threshold callout showing frozen decision cutoff ($\tau_{\text{HI}} = 0.337784$, $\tau_{\text{LI}} = 0.238483$).
  - Inference latency telemetry in milliseconds.
- **Integrated Gradients Explainability Panel:**
  - Interactive top positive drivers (factors elevating laundering risk).
  - Top negative drivers (factors mitigating laundering suspicion).
  - Categorization into semantic feature groups: *Transaction Attributes, Payment Format, Currency Rails, Temporal Dynamics, Sender History, Receiver History*.
  - Attribution completeness residual verification ($|\Delta P - \sum \text{attr}| < 0.05$).
- **In-Memory History Ledger:**
  - Session-scoped evaluation table tracking sample ID, timestamp, probability, decision, and evaluation duration.
  - Allows instant re-selection of historical scores during active triage.

---

## 4. Production Build Telemetry

The frontend was compiled and bundled into standalone static assets:

```bash
$ npm --prefix frontend run build
> fedsecureai-dashboard@1.0.0 build
> tsc && vite build

vite v6.4.5 building for production...
transforming...
✓ 1581 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.78 kB │ gzip:  0.45 kB
dist/assets/index-C15frQK3.css    0.82 kB │ gzip:  0.47 kB
dist/assets/index-D0KhqHCI.js   194.14 kB │ gzip: 57.36 kB
✓ built in 2.66s
```

- **Zero Build Warnings or TypeScript Errors:** All types strictly checked.
- **Total Bundle Size:** 194.14 kB (57.36 kB gzipped) — exceptionally lightweight and instantly loadable over restricted enterprise networks.

