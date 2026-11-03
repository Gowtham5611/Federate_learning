# FedSecureAI — Google Colab Full Combined & Full LI-Medium Training Guide

## 1. Executive Summary

This guide accompanies the updated, CUDA-enforced Colab deployment package `FedSecureAI_FULL_COMBINED_AND_LI_MEDIUM_COLAB_READY_v2.zip` and the runner notebook `colab_full_combined_and_li_medium_runner.ipynb`.

It provides a production-grade, RAM-bounded, GPU-accelerated pipeline for executing federated learning on:
1. **FULL_COMBINED AML Dataset**: 36,329,828 transactions (~21.80M train, ~7.27M val, ~7.26M test) constructed from the complete original IBM AML HI-Small and LI-Medium datasets, replacing the historical 500,000-row sample.
2. **Full LI-Medium AML Dataset**: 31,251,483 transactions (~18.75M train, ~6.25M val, ~6.25M test).

---

## 2. Dataset Specifications & Provenance

### Source Datasets
- **IBM AML HI-Small**:
  - Raw File: `data/raw/ibm_aml/HI-Small_Trans.csv` (SHA-256: `a09e36...acbf9`)
  - Processed Directory: `data/processed/ibm_aml/HI-Small/`
  - Total Transactions: **5,078,345** (Train: 3,047,216 | Val: 1,015,565 | Test: 1,015,564)
  - Laundering Positives: **5,177** (0.1019%)
- **IBM AML LI-Medium**:
  - Raw File: `data/raw/ibm_aml/LI-Medium_Trans.csv` (SHA-256: `0fc895...a3910`)
  - Processed Directory: `data/processed/ibm_aml/LI-Medium/`
  - Total Transactions: **31,251,483** (Train: 18,752,217 | Val: 6,250,154 | Test: 6,249,112)
  - Laundering Positives: **16,041** (0.0513%)

### Full Combined Dataset (`FULL_COMBINED`)
- **Target Path**: `data/processed/ibm_aml/FULL_COMBINED/`
- **Total Transactions**: **36,329,828** (Zero row exclusions; raw-to-processed discrepancy = 0 rows)
  - **Train**: 21,799,433 rows (60.00%)
  - **Validation**: 7,265,719 rows (20.00%)
  - **Test**: 7,264,676 rows (20.00%)
- **Total Laundering Positives**: **21,218** (Overall prevalence: 0.0584%)
- **Provenance & Schema**:
  - 28 columns (Timestamp, From Account, To Account, 23 engineered features, Target `is_laundering`, Provenance `dataset_source` ['hi_small' vs 'li_medium']).
  - Strict chronological sorting by timestamp within every split.
  - Zero cross-split temporal leakage.
  - `dataset_source` is an identifier column only; it is never fed into model input tensors.

---

## 3. Mandatory Experiment Matrix (12 Primary Configurations per Dataset)

| Config # | Distribution | Architecture | Algorithm | Proximal $\mu$ | Rounds | Epochs | Batch Size | Device |
|---|---|---|---|---|---|---|---|:---:|
| **1** | IID | GRU | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **2** | IID | GRU | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |
| **3** | IID | LSTM | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **4** | IID | LSTM | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |
| **5** | IID | BiLSTM | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **6** | IID | BiLSTM | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |
| **7** | Non-IID | GRU | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **8** | Non-IID | GRU | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |
| **9** | Non-IID | LSTM | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **10** | Non-IID | LSTM | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |
| **11** | Non-IID | BiLSTM | FedAvg | 0.0 | 5 | 1 | 4096 | CUDA |
| **12** | Non-IID | BiLSTM | FedProx | 0.0001 | 5 | 1 | 4096 | CUDA |

---

## 4. Architectural & Safety Protections

1. **Strict CUDA Enforcement**:
   - `resolve_device("cuda")` checks `torch.cuda.is_available()`. If CUDA is unavailable, it immediately raises `CUDANotAvailableError` rather than silently falling back to CPU.
   - Runners verify model parameters and training batch tensors reside on CUDA.
   - Out-of-memory errors catch `torch.cuda.OutOfMemoryError` and output an actionable advisory to adjust `--batch-size` without falling back to CPU or truncating data.
2. **GPU-Accelerated Preprocessing (`GpuAmlPreprocessor`)**:
   - RAPIDS cuDF and CuPy integration for accelerated tabular filtering, cyclical math transforms, log-scaling, and encoding.
   - Verified 100% equivalent to CPU reference implementation on test fixtures.
3. **Strict Full-Data Enforcement**:
   - Rejects `row_limit` configurations and enforces minimum row count assertions:
     - `FULL_COMBINED`: Train $\ge 21,000,000$; Val $\ge 7,000,000$; Test $\ge 7,000,000$.
     - `LI-Medium`: Train $\ge 18,000,000$; Val $\ge 6,000,000$; Test $\ge 6,000,000$.
   - Intercepts 200k-row subsets or legacy 500k-row caches with `SubsetModeDetectedError`.
4. **RAM-Safe Memory-Mapped Sequences (`np.memmap`)**:
   - Feature matrices stored as binary disk memory maps. Resident host memory remains **$< 500\text{ MB}$**, preventing Colab `SIGKILL` OOM crashes.
5. **Zero-Copy Client Partitioning & Isolated Checkpoints**:
   - Dedicated output directories: `artifacts/full_combined_experiment/` and `artifacts/full_li_medium_experiment/`.
   - Resilient `resume_manifest.json` tracks round-by-round checkpoints and metrics.

---

## 5. Google Colab Execution Step-by-Step

### Step 1: Upload Files
1. Open [Google Colab](https://colab.research.google.com/) and select a GPU runtime (**Runtime > Change runtime type > T4 GPU or A100**).
2. Upload `colab_full_combined_and_li_medium_runner.ipynb`.
3. Upload `FedSecureAI_FULL_COMBINED_AND_LI_MEDIUM_COLAB_READY_v2.zip` directly to `/content/` or to your Google Drive folder (`/content/drive/MyDrive/FedSecureAI/`).

### Step 2: Run Setup Cells
- Execute **Cells 1 to 5** to verify GPU, mount Google Drive, unzip the codebase, install dependencies (`requirements_colab.txt`), and run the automated test suite (`pytest tests/ -v`).

### Step 3: Run Full Combined Pipeline (Section A)
- **Cell A.1**: Automatically constructs or verifies `data/processed/ibm_aml/FULL_COMBINED` using streaming 2-way merge in $O(1)$ memory.
- **Cell A.2**: Performs preflight hardware, VRAM, RAM, and disk assertions.
- **Cell A.3**: Set `CONFIRM_FULL_COMBINED_TRAINING = True` to launch the 12 primary experiments on the 36.33M-row dataset.
- **Cell A.4**: Displays the resulting performance metrics table.

### Step 4: Run Full LI-Medium Pipeline (Section B, Preserved)
- **Cell B.1**: Performs preflight verification on the 31.25M-row LI-Medium dataset.
- **Cell B.2**: Set `CONFIRM_FULL_LI_MEDIUM_TRAINING = True` to launch the 12 experiments on LI-Medium.
- **Cell B.3**: Displays the LI-Medium performance table.

### Step 5: Backup & Save Results
- **Cell 6**: Automatically archives all checkpoints, predictions, and manifests into a timestamped ZIP directly in Google Drive.
