# FedSecureAI — System Architecture Specification

**System Version:** 2.1.0 Enterprise  
**Framework:** PyTorch • FastAPI • React 18 • TypeScript • Scikit-Learn  
**Domain:** Cross-Silo Federated Anti-Money Laundering (AML) Detection  

---

## 1. High-Level Architectural Overview

FedSecureAI is an enterprise-grade federated learning platform engineered to detect sophisticated money laundering topologies across isolated banking institutions without sharing raw transaction records. The architecture coordinates three federated silos (client banks) with a centralized parameter aggregator, supported by a causal feature engineering engine, packed sequence recurrent encoders, and an adaptive inference web dashboard.

```
+---------------------------------------------------------------------------------------------------+
|                                      FEDSECUREAI ARCHITECTURE                                     |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Client Bank Silo 1 ]            [ Client Bank Silo 2 ]            [ Client Bank Silo 3 ]       |
|  Local Dataset (Typology A)        Local Dataset (Typology B)        Local Dataset (Typology C)   |
|  Local Adam Optimizer              Local Adam Optimizer              Local Adam Optimizer         |
|  FedProx Loss (μ = 0.0001)         FedProx Loss (μ = 0.0001)         FedProx Loss (μ = 0.0001)    |
|         │                                    │                                    │               |
|         └─── Model Weights (48KB - 162KB) ───┼─── Model Weights (48KB - 162KB) ───┘               |
|                                              ▼                                                    |
|                              +───────────────────────────────+                                    |
|                              |  CENTRAL FEDERATED AGGREGATOR |                                    |
|                              |  • Weighted FedAvg / FedProx  |                                    |
|                              |  • Global Model Broadcast     |                                    |
|                              |  • Parameter Freeze & Audit   |                                    |
|                              +───────────────┬───────────────+                                    |
|                                              │                                                    |
|                                              ▼                                                    |
|                              +───────────────────────────────+                                    |
|                              |    FROZEN MODEL REPOSITORY    |                                    |
|                              |  • HI-Small Champion (LSTM)   |                                    |
|                              |  • LI-Medium Champion (BiLSTM)|                                    |
|                              |  • Checkpoint SHA256 Registry |                                    |
|                              +───────────────┬───────────────+                                    |
|                                              │                                                    |
|        ┌─────────────────────────────────────┴─────────────────────────────────────┐              |
|        ▼                                                                           ▼              |
|  +──────────────────────────────────────────+     +─────────────────────────────────────────────+ |
|  |       CAUSAL PREPROCESSING PIPELINE      |     |           FASTAPI INFERENCE ENGINE          | |
|  |  • Universal CSV Reader (Delim/Encoding) |     |  • Model Catalog & Schema Compatibility     | |
|  |  • Causal Account History Store          |     |  • Sequence Context Diagnostics             | |
|  |  • 57-Dimensional Feature Expansion     |     |  • Mode A: Recurrent Neural Inference       | |
|  |  • Packed Sequence Dataset (Window=10)   |     |  • Mode B: Tabular Model Fallback           | |
|  +──────────────────────────────────────────+     +─────────────────────────────────────────────+ |
|                                                                    │                              |
|                                                                    ▼                              |
|                                                   +─────────────────────────────────────────────+ |
|                                                   |        REACT / TYPESCRIPT WEB DASHBOARD     | |
|                                                   |  • AML Evaluation & Batch CSV Uploader      | |
|                                                   |  • Results Analytics & Decile Distributions | |
|                                                   |  • 24-Model Comparison & Cost Analyzer      | |
|                                                   |  • Validation Ensemble Optimizer (α Sweep)  | |
|                                                   |  • Searchable Prediction History Table      | |
|                                                   +─────────────────────────────────────────────+ |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Federated Learning Subsystem

### A. Client Partitioning Strategy
- **IID Partitioning**: Uniform random sample distribution across all 3 client banks. Each client receives exactly $\frac{1}{3}$ of the training transactions, presenting identical label prevalence and feature distributions.
- **Non-IID Partitioning**: Typology-skewed and volume-skewed partitioning simulating realistic cross-silo banking networks:
  - **Client 1**: Specializes in high-frequency domestic transactions (12.0% volume).
  - **Client 2**: Specializes in cross-border wire transfers and currency exchange (4.1% volume).
  - **Client 3**: Processes high-volume retail transactions (83.9% volume).

### B. Federated Aggregation Algorithms
1. **FedAvg (Federated Averaging)**:
   In each round $t$, the central server broadcasts global model weights $\theta^t$ to all clients. Each client trains locally for $E=1$ epoch using Adam with binary cross-entropy loss, producing local weights $\theta_k^{t+1}$. The server aggregates:
   $$\theta^{t+1} = \sum_{k=1}^K \frac{n_k}{N} \theta_k^{t+1}$$
2. **FedProx (Federated Proximal)**:
   To combat client drift under severe statistical heterogeneity, FedProx introduces a proximal regularization term to each client's local loss function:
   $$\min_{\theta} \mathcal{L}_k(\theta) + \frac{\mu}{2} \|\theta - \theta^t\|^2$$
   - **Compliance Parameter**: $\mu = 0.0001$.
   - **Zero Communication Overhead**: The proximal term is evaluated entirely locally during backpropagation. Communication payload is identical to FedAvg ($1.0\times$).

---

## 3. Neural Architecture & Sequence Encoding

Recurrent neural models process financial transactions as variable-length temporal sequences to detect money dispersion, structuring (smurfing), and rapid fund pass-through.

### A. Shared Base Implementation (`RecurrentSequenceClassifier`)
All three recurrent architectures (`GruClassifier`, `LstmClassifier`, `BiLstmClassifier`) inherit from a unified base class to eliminate confounding implementation differences:
- **Feature Dimension**: 57 expanded dimensions (amounts, currency one-hot encodings, payment format one-hot encodings, temporal features, account velocity).
- **Hidden Dimension**: 64 units per direction.
- **Stacked Layers**: 2 layers with inter-layer dropout ($0.20$).
- **Head**: Linear projection from final recurrent representation to a single logit.

### B. Padding-Aware Sequence Packing
To prevent right-padded zero vectors from corrupting backward temporal representations:
- Padded inputs are wrapped using `torch.nn.utils.rnn.pack_padded_sequence` with true sequence lengths ($1 \le L \le 10$).
- The recurrent module processes only valid transactions.
- For bidirectional models (`BiLstmClassifier`), packing ensures the backward pass begins at the actual latest transaction rather than inside padding zeros.

---

## 4. Causal Feature Engineering & Universal Preprocessing

### A. Causal Account History Feature Store (`AccountHistoryFeatureStore`)
To prevent future-information data leakage during feature generation:
- The multi-tenant feature store tracks running transaction velocity, cumulative credit/debit volumes, and fan-out counts for each account.
- **Strict Temporal Causality**: Feature vectors for transaction $i$ are computed strictly using historical transactions observed *prior* to transaction $i$'s timestamp.
- **Multi-Tenant Isolation**: Tenant IDs (e.g. `tenant_id = "bank_alpha"`) maintain separated state dictionaries, preventing cross-tenant leakage.

### B. Universal CSV Reader & Schema Normalizer
- **Format Agnostic**: Detects delimiters (`,` `;` `\t` `|`), character encodings (UTF-8, UTF-8-BOM, Latin-1), and column naming variations.
- **Auto-Mapping**: Translates diverse column names (e.g. `from_acc`, `Sender`, `Account`) into standard canonical banking concepts.
- **Preserved Order**: Output predictions maintain 100% strict row order alignment with input CSV rows.

---

## 5. Dual-Mode Inference Engine

The backend provides two distinct inference pathways based on sequence context availability:

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                  INFERENCE ROUTING DECISION                                 |
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                             |
|   Uploaded CSV Transaction Batch                                                            |
|          │                                                                                  |
|          ▼                                                                                  |
|   Sequence Context Diagnostics                                                              |
|   Inspect: L_avg, Single-Tx Ratio, Mean Padding Ratio                                       |
|          │                                                                                  |
|          ├──────────────────────────────────────────────────┐                               |
|          │                                                  │                               |
|          ▼                                                  ▼                               |
|   [ L_avg >= 3.0 / Pad < 70% ]                       [ L_avg < 3.0 / Pad >= 70% ]           |
|   Sufficient Temporal History                        Truncated / Isolated Transactions      |
|          │                                                  │                               |
|          ▼                                                  ▼                               |
|   MODE A: NEURAL SEQUENCE INFERENCE                  MODE A + EXPLICIT DIAGNOSTIC WARNING   |
|   • Champion LSTM / BiLSTM Model                     • Neural Model Executes Faithfully     |
|   • Packed Sequence Dataset                          • Sequence Warning Emitted in UI/API   |
|   • Decision Threshold: τ = 0.3378                   • Recommends Tabular Model (Mode B)    |
|                                                             │                               |
|                                                             ▼                               |
|                                                      MODE B: OPTIONAL TABULAR FALLBACK      |
|                                                      • HistGradientBoosting / RandomForest  |
|                                                      • Single-Row Transaction Prior Scoring |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

### Safety Guarantee
The platform **never silently swaps models**. When a recurrent model is selected to score isolated single-transaction rows, it executes as requested, while returning an explicit diagnostic warning detailing why the predicted probabilities reflect conservative baseline priors ($\approx 0.001 - 0.015$).

---

## 6. Web Application & Dashboard Architecture

The frontend is constructed using React 18, TypeScript (Strict Mode), Tailwind CSS, and Vite:

1. **AML Evaluation (`AmlEvaluationPage`)**: File drag-and-drop, automated schema inspection, dynamic model selection, and real-time execution.
2. **Results & Analytics (`ResultsAnalyticsPage`)**: Metric summary cards (Accuracy, Precision, Recall, F1, F2, PR-AUC, ROC-AUC), confusion matrix, 10-bin risk decile histograms, and zero-alerts notification.
3. **Model Comparison (`ModelRegistryPage`)**: Complete 24-model inventory, filterable by architecture, algorithm, and paradigm; client-level partition analysis; Float32 communication bandwidth audit.
4. **Ensemble Analytics (`EnsembleAnalyticsPage`)**: Validation $\alpha$ grid search sweep ($0.0 \to 1.0$), objective tuning, standalone vs hybrid comparison, and honest non-improvement reporting.
5. **Prediction History (`PredictionHistoryPage`)**: Full searchable, paginated, and exportable table of transaction predictions.

