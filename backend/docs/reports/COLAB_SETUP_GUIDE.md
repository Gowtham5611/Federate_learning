# FedSecureAI — Google Colab GPU Execution & Setup Guide

This guide details how to execute the FedSecureAI federated comparative experiment on Google Colab with NVIDIA GPU acceleration (T4/V100/A100), including Google Drive mounting, package extraction, dataset linking, verification, and persistent checkpoint resumption.

---

## 1. Prerequisites and Expected Storage

1. **Google Drive Storage Requirements:**
   - **Code & Artifact Package (`FedSecureAI_colab.zip`):** ~45 MB.
   - **IBM AML Processed Datasets (`data/processed/ibm_aml/`):**
     - `Combined`: ~130 MB
     - `HI-Small`: ~1.28 GB
     - `LI-Medium`: ~7.89 GB
     - *Total required if all 3 datasets uploaded to Drive:* ~9.3 GB.
     - *Minimum required to run HI-Small + Combined:* ~1.45 GB.
2. **GPU Requirements:**
   - Free Google Colab tier provides an **NVIDIA T4 GPU (15.36 GB VRAM)**.
   - Batch size: 4096 fits comfortably in ~3.2 GB VRAM during GRU, LSTM, and BiLSTM training.
3. **Execution Time Comparison:**
   - **HI-Small (3.05M rows × 5 rounds):**
     - CPU (Laptop): ~25–30 minutes per model.
     - GPU (Colab T4): **~2–3 minutes per model** (10× faster!).
   - **LI-Medium (18.75M rows × 5 rounds):**
     - CPU (Laptop): ~2.5 hours per model (~15 hours total, causes overheating).
     - GPU (Colab T4): **~12–15 minutes per model** (~1.5 hours total, completely safe).

---

## 2. Directory Structure on Google Drive

In your Google Drive, create a root directory named `FedSecureAI`:

```
Google Drive/
└── FedSecureAI/
    ├── FedSecureAI_colab.zip           # Uploaded ZIP package (~45 MB)
    ├── data/
    │   └── processed/
    │       └── ibm_aml/
    │           ├── Combined/          # Uploaded folder (~130 MB)
    │           ├── HI-Small/          # Uploaded folder (~1.28 GB)
    │           └── LI-Medium/         # Uploaded folder (~7.89 GB)
    └── artifacts/
        └── fedavg_fedprox_full_comparison/  # Persistent checkpoints & CSVs saved here automatically
```

---

## 3. Step-by-Step Google Colab Execution

### Step 1: Open Google Colab and Enable GPU
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Select **Upload** and upload `colab_runner.ipynb`.
3. In the Colab menu, go to **Runtime** > **Change runtime type**.
4. Set **Hardware accelerator** to **T4 GPU** (or any available GPU). Click **Save**.

### Step 2: Run Cell 1 & 2 (GPU Verification & Drive Mount)
```python
# Verify GPU availability
!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Extract Package and Install Requirements
```bash
# Unzip code and existing baseline checkpoints to Colab local disk
mkdir -p /content/FedSecureAI
unzip -q -o "/content/drive/MyDrive/FedSecureAI/FedSecureAI_colab.zip" -d /content/FedSecureAI
cd /content/FedSecureAI

# Install requirements
pip install -q -r requirements_colab.txt
```

### Step 4: Verification Dry-Run (Zero Training)
Runs hardware check, dataset path check, checkpoint validation, and 1 forward/backward pass:
```bash
python backend/scripts/run_fedavg_fedprox_comparison_suite.py \
    --data-root "/content/drive/MyDrive/FedSecureAI/data/processed/ibm_aml" \
    --output-dir "/content/drive/MyDrive/FedSecureAI/artifacts/fedavg_fedprox_full_comparison" \
    --device auto \
    --verify-only
```
*Expected output: `=== VERIFICATION COMPLETE: ALL SYSTEMS READY FOR EXECUTION ===`*

### Step 5: Safe First Test Configuration
Benchmark 1 configuration on GPU (`HI-Small GRU FedAvg IID`) for 5 rounds:
```bash
python backend/scripts/run_fedavg_fedprox_comparison_suite.py \
    --data-root "/content/drive/MyDrive/FedSecureAI/data/processed/ibm_aml" \
    --output-dir "/content/drive/MyDrive/FedSecureAI/artifacts/fedavg_fedprox_full_comparison" \
    --device auto \
    --datasets HI-Small \
    --strategies iid \
    --architectures gru \
    --algorithms fedavg \
    --max-configs 1
```
*Expected duration on T4 GPU: ~90–120 seconds.*

### Step 6: Full Suite Resumption
Executes the remaining configurations (HI-Small IID, Combined IID, and LI-Medium IID/Non-IID), while automatically skipping the 12 already completed models:
```bash
python backend/scripts/run_fedavg_fedprox_comparison_suite.py \
    --data-root "/content/drive/MyDrive/FedSecureAI/data/processed/ibm_aml" \
    --output-dir "/content/drive/MyDrive/FedSecureAI/artifacts/fedavg_fedprox_full_comparison" \
    --device auto \
    --strategies both
```

---

## 4. Key Resumption & Safety Invariants

1. **Automatic Skip:** Any configuration whose `{exp_id}_global_best.pt` and `{exp_id}_test_preds.npy` exist on Google Drive will be skipped with 0 redundant computation.
2. **Atomic Writes:** Checkpoints and prediction arrays are written to temporary files and atomically renamed, ensuring that a Colab session disconnect never leaves a corrupted checkpoint.
3. **Continuous Metric Sync:** `all_model_metrics.csv`, `round_metrics.csv`, and `matched_fedavg_vs_fedprox.csv` are flushed to Google Drive immediately after each completed model.

