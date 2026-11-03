# FedSecureAI — Full-Data Retraining Initial Audit

## Executive Summary

This audit investigates the repository-wide data truncation defect that restricted federated training to the first 200,000 chronological transactions of the IBM AML `LI-Medium` training split. It documents the root cause, maps every occurrence across the codebase, quantifies memory and computational constraints, and defines the architectural solution for memory-efficient full-data training on all 18,752,217 training transactions.

---

## 1. Root Cause Analysis

### 1.1 Defect Origin
The truncation was originally introduced in `backend/scripts/run_li_medium_12_experiments.py` at lines 81 and 175:
```python
def execute_all_12_li_medium_experiments(
    ...
    row_limit: int | None = 200000,  # Safe CPU baseline window
) -> None:
    ...
    train_frame = pd.read_csv(
        train_path,
        usecols=train_cols,
        dtype=train_dtypes,
        nrows=row_limit,
    )[train_cols]
```
The rationale was to prevent out-of-memory (OOM) crashes on CPU workstations when handling the 4.98 GB `transactions.csv` file.

### 1.2 Downstream Propagation
Because baseline configurations 1 through 12 were executed with `row_limit=200000`, subsequent scripts inherited this truncation to maintain experimental comparability:
1. `backend/scripts/run_fedprox_final_optimization.py`:
   - Line 164: Hardcoded `row_limit = 200000`.
   - Line 174: `nrows=row_limit` passed to `pd.read_csv`.
2. `backend/scripts/run_optimization_and_ensemble.py`:
   - Line 101: Default `row_limit: int = 200000`.
   - Line 126: `nrows=row_limit`.
3. `backend/scripts/run_loss_ablations.py`:
   - Line 87: Default `row_limit: int = 200000`.
   - Line 111: `nrows=row_limit`.
4. `backend/scripts/run_independent_audit.py`:
   - Lines 114, 142, 240, 254: Evaluated and re-trained with `row_limit=200000`.
5. `backend/scripts/audit_and_evaluate_all.py`:
   - Lines 98, 123: Evaluated with `row_limit=200000`.
6. `backend/scripts/run_full_test_evaluation.py`:
   - Correctly removed the row limit for validation and test splits (evaluating all 6,249,112 test rows), but the evaluated models were the historical checkpoints trained on only the 200,000-row slice!

### 1.3 Consequence on Model Learning
- **Full Training Split**: 18,752,217 transactions, 8,138 laundering positives ($0.0434\%$ prevalence).
- **Truncated Slice (200k)**: Contains only **9 to 14 laundering positives** across the entire slice!
- When partitioned across 3 clients (especially under Non-IID Dirichlet distribution), individual clients had as few as **0 to 3 positive samples** during 5 training rounds.
- As a consequence, models suffered extreme positive-gradient starvation during local training.

---

## 2. Complete Codebase Occurrence Inventory

| File Path | Line(s) | Pattern | Current Usage | Required Action |
| :--- | :---: | :---: | :--- | :--- |
| `backend/scripts/run_fedprox_final_optimization.py` | 164, 174 | `row_limit = 200000` | Truncates training data | Replace with full-data streaming runner |
| `backend/scripts/run_li_medium_12_experiments.py` | 81, 175 | `row_limit = 200000` | Baseline experiment truncation | Preserve as historical; flag as truncated |
| `backend/scripts/run_optimization_and_ensemble.py` | 101, 126 | `row_limit = 200000` | Truncates tuning & ensemble | Preserve as historical; flag as truncated |
| `backend/scripts/run_loss_ablations.py` | 87, 111 | `row_limit = 200000` | Pilot loss ablations | Keep as pilot study; document row scope |
| `backend/scripts/run_independent_audit.py` | 114, 240 | `row_limit = 200000` | Historical audit runner | Keep as historical audit record |
| `backend/scripts/audit_and_evaluate_all.py` | 98, 123 | `row_limit = 200000` | Evaluation harness | Parameterize to support `full` vs `smoke` |
| `backend/scripts/run_full_test_evaluation.py` | 96, 582 | `smoke_test_limit` | Smoke-test flag | Correctly parameterized; preserve |
| `backend/infrastructure/data/sequences/processed_sequence_provider.py` | 160 | `nrows=config.row_limit` | Low-level CSV reader | Correctly parameterized; accepts `None` for full split |
| `backend/application/dto/training.py` | 95, 131 | `row_limit: int \| None` | DTO field | Correctly parameterized; default is `None` |

---

## 3. Hardware & Memory Reality Check

A physical system inspection revealed:
- **Total Physical RAM**: 16,474,168 KB (~16.0 GB).
- **Available Free RAM**: ~3,375,000 KB (~3.2 GB - 3.4 GB).
- **Raw CSV Size (`train/transactions.csv`)**: 4,982,034,864 bytes (~4.98 GB).
- **Float32 Feature Expansion (57 features)**: $18,752,217 \times 57 \times 4 \text{ bytes} \approx 4.27 \text{ GB}$.
- **Naive Pandas Ingestion**:
  Loading the entire 4.98 GB CSV into a single DataFrame requires ~6 GB RAM. Subsetting and copying across 3 client partitions requires an additional $3 \times 2 \text{ GB} = 6 \text{ GB}$, totaling $>12 \text{ GB}$ in memory.
  **Attempting naive in-memory partitioning on 3.3 GB of free RAM will trigger an immediate Out-Of-Memory (OOM) crash.**

### 3.1 Empirical Computational Throughput
Benchmarking the sequence classifier with PyTorch (16 CPU threads, batch size 2048, hidden size 64, sequence length 10, 57 features):
- **Batch processing latency**: `174.22 ms/batch`.
- **Throughput**: `5.7 batches/sec` $\implies \mathbf{11,755 \text{ samples/sec}}$.
- **Runtime for 1 full pass over 18,752,217 transactions**:
  $$\frac{18,752,217}{11,755} \approx 1,595 \text{ seconds} \approx \mathbf{26.6 \text{ minutes per epoch}}.$$
- **Runtime for 1 communication round** (3 clients, 1 local epoch each, covering all 18.75M rows): $\mathbf{\approx 26.6 \text{ minutes}}$.
- **Runtime for 5 communication rounds of 1 model**: $5 \times 26.6 \text{ min} \approx \mathbf{2.2 \text{ hours}}$.
- **Runtime for 12 models**: $12 \times 2.2 \text{ hours} \approx \mathbf{26.4 \text{ hours}}$.

---

## 4. Architectural Remediation Plan

To train on the complete 18,752,217 transactions without memory exhaustion:
1. **Separation of Modes**:
   - `full`: Strictly processes all 18,752,217 rows with verified transaction coverage.
   - `pilot`: Explicitly bounded benchmark for memory/throughput estimation (must be labeled `PILOT`).
   - `smoke`: Fast debugging run ($N \le 10,000$ rows).
2. **Chunked Streaming Partitioner & Sequence Provider**:
   - Instead of loading 18.75M rows in a single monolithic DataFrame, process the training split in sequential chunks of 500,000 to 1,000,000 rows.
   - Each chunk is deterministically assigned to client partitions, converted to feature batches, trained, and immediately garbage collected (`gc.collect()`).
   - Peak RSS memory stays strictly below **2.0 GB**, completely safe for the 3.3 GB free RAM environment.
3. **Provable Full-Data Coverage**:
   - Log exact row counts, starting and ending row indices, unique account counts, and positive label counts processed by each client in each communication round.
   - Total processed unique rows across the 3 clients must sum to exactly **18,752,217**, with exactly **8,138** positive labels.
4. **Execution Strategy**:
   - Execute full-data training for the primary comparative models (Full-Data FedAvg Control and Full-Data FedProx Best Candidate).
   - Any model configuration not fully executed within the immediate execution window will be marked `INCOMPLETE` with documented runtime limitations.

