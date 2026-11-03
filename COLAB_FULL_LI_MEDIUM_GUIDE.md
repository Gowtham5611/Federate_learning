# FedSecureAI — Complete Guide: Full LI-Medium Federated Training on Google Colab GPU

This guide provides end-to-end instructions for running the **12 primary federated experiments** on the **complete 31.25-million-row IBM AML LI-Medium dataset** using Google Colab with GPU acceleration (T4/V100/A100).

---

## 1. What Was Repaired: The Memory Bottleneck Solution

In prior attempts, loading 31.25M rows simultaneously into pandas DataFrames consumed > 16 GB of host RAM, triggering the Linux kernel Out-Of-Memory Killer (`SIGKILL`). A temporary 200,000-row subset was previously used as a workaround.

**The production fix implemented in this release**:
1. **Memory-Mapped Sequence Provider (`MmapSequenceProvider`)**:
   - Streams CSV files in bounded chunks (200,000 rows) directly into pre-allocated binary memory maps on disk (`features.mmap`, `targets.npy`, `previous_row.npy`).
   - Slices windows on demand during batch generation without keeping DataFrames in RAM.
   - Resident host memory is strictly bounded to **< 1.0 GB** during the entire run.
2. **Zero-Copy Federated Client Partitioning (`MmapPartitionManager`)**:
   - Federated clients reference the shared memory map on disk using lightweight integer row index arrays.
   - Zero duplicated feature matrices or DataFrames for the 3 clients.
   - Preserves Non-IID account typology atomicity (Retail Giant, Offshore/Crypto Exchange, Regional Credit Union) and IID stratified class balance.
3. **Strict Full-Dataset Guard**:
   - Disallows silent truncation or sampling. Fails fast with `SubsetModeDetectedError` if train rows < 18,000,000 or an unauthorized `row_limit` is detected.
4. **Isolated Checkpointing & Resumption**:
   - Outputs to `artifacts/full_li_medium_experiment/` with a dedicated `resume_manifest.json`.
   - Never overwrites or touches existing HI-Small, Combined, or LI-Medium subset results.

---

## 2. File Checklist

Ensure you have the following in your Google Drive or Colab environment:
- `FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip` (Code, models, configs, tests — does not contain large CSVs)
- `colab_full_li_medium_runner.ipynb` (Dedicated Colab notebook)
- `data/processed/ibm_aml/LI-Medium/` (Your original 31.25M-row processed dataset on Google Drive):
  - `train/transactions.csv` (18,752,217 rows, ~4.98 GB)
  - `validation/transactions.csv` (6,250,154 rows, ~1.63 GB)
  - `test/transactions.csv` (6,249,112 rows, ~1.66 GB)
  - `metadata/` (`feature_metadata.json`, `splits.json`, `transformers.json`, `schema.json`)

---

## 3. Step-by-Step Execution in Google Colab

### Step 1: Open Google Colab & Select GPU Runtime
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `colab_full_li_medium_runner.ipynb`.
3. In the Colab menu, select **Runtime** $\rightarrow$ **Change runtime type**.
4. Set **Hardware accelerator** to **GPU** (T4, V100, or A100).
5. Click **Save**.

### Step 2: Upload the Code Package to Google Drive
Place `FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip` in your Google Drive under:
```
My Drive/FedSecureAI/FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip
```
*(Alternatively, drag and drop the ZIP directly into the `/content/` pane in Colab).*

### Step 3: Run Cells Sequentially
The notebook is organized into 11 self-contained cells:

- **Cell 1 & 2 — Hardware & Environment Check**:
  Verifies that CUDA is active, cuDNN is available, and displays GPU VRAM and available disk space.
- **Cell 3 — Mount Google Drive**:
  Authorizes Colab to access Google Drive for persistent checkpoints and data.
- **Cell 4 — Extract Codebase**:
  Extracts `FedSecureAI_FULL_LI_MEDIUM_COLAB_READY_v1.zip` into `/content/FedSecureAI`.
- **Cell 5 — Dependency Installation**:
  Installs required packages from `requirements_colab.txt`.
- **Cell 6 — Dataset Path Verification**:
  Detects the full LI-Medium dataset on Google Drive or local SSD.
- **Cell 7 — Run Unit & Integration Tests**:
  Runs `pytest tests/test_full_li_medium_pipeline.py -v`. Ensures all 4 pipeline tests pass.
- **Cell 8 — Preflight Verification Check**:
  Runs `run_full_li_medium_training.py --verify-only`. Validates row counts (>18M train), 57 features, and label distributions.
- **Cell 9 — Benchmark Single Configuration**:
  Runs Configuration 1 (IID GRU FedAvg) for 5 rounds to verify GPU throughput and memory stability (~3-4 minutes).
- **Cell 10 — Full 12-Experiment Suite Execution**:
  Trains all 12 configurations sequentially. Checkpoints are automatically synced to Google Drive.
- **Cell 11 — Inspect Final Results**:
  Displays `experiment_status.csv` with PR-AUC, ROC-AUC, F1, decision threshold, and false alerts per 10k legitimate transactions.
- **Cell 12 — Package Final Results**:
  Creates a timestamped ZIP archive `FedSecureAI_FULL_LI_MEDIUM_RESULTS_<TIMESTAMP>.zip` on Google Drive.

---

## 4. The 12-Experiment Primary Matrix

| Config # | Distribution | Architecture | Algorithm | Proximal $\mu$ | Rounds | Clients |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | IID | GRU | FedAvg | — | 5 | 3 |
| 2 | IID | GRU | FedProx | 0.0001 | 5 | 3 |
| 3 | IID | LSTM | FedAvg | — | 5 | 3 |
| 4 | IID | LSTM | FedProx | 0.0001 | 5 | 3 |
| 5 | IID | BiLSTM | FedAvg | — | 5 | 3 |
| 6 | IID | BiLSTM | FedProx | 0.0001 | 5 | 3 |
| 7 | Non-IID | GRU | FedAvg | — | 5 | 3 |
| 8 | Non-IID | GRU | FedProx | 0.0001 | 5 | 3 |
| 9 | Non-IID | LSTM | FedAvg | — | 5 | 3 |
| 10 | Non-IID | LSTM | FedProx | 0.0001 | 5 | 3 |
| 11 | Non-IID | BiLSTM | FedAvg | — | 5 | 3 |
| 12 | Non-IID | BiLSTM | FedProx | 0.0001 | 5 | 3 |

---

## 5. Resumption and Fault Tolerance

If your Google Colab session disconnects or reaches a timeout:
1. Re-open the notebook and re-mount Google Drive.
2. Run **Cell 10** (`run_full_li_medium_training.py`).
3. The runner reads `resume_manifest.json`:
   - Any configuration marked `COMPLETED` with a verified `global_best.pt` is **instantly skipped**.
   - An interrupted configuration automatically resumes from its latest saved round checkpoint (`round_{r}.pt`).
   - Zero completed work is repeated or lost.

---

## 6. Output Artifacts Inventory

All experiment results are saved to:
```
artifacts/full_li_medium_experiment/
├── resume_manifest.json          # Master atomic status tracker
├── checkpoints/
│   ├── config_1/
│   │   ├── round_1.pt ... round_5.pt
│   │   ├── global_best.pt
│   │   └── metrics_by_round.csv
│   └── ... (config_2 through config_12)
├── predictions/
│   ├── config_1/test_predictions.npz
│   └── ... (config_2 through config_12)
├── results/
│   └── experiment_status.csv     # Master summary table with all test metrics
└── logs/
    └── full_li_medium_execution_<timestamp>.log
```

