# FedSecureAI — Initial Project and Environment Audit (LI-Medium Campaign)

**Audit Date:** 2026-10-17  
**Project Root:** `c:\Users\Acer\OneDrive\Desktop\Federate_learning`  
**Target Dataset:** `LI-Medium_Trans.csv` (Expected size: ~2.98 GB)  

---

## 1. Project Structure and Runtime Analysis
- **Operating Environment:** Running locally on Windows 11 with 16 logical CPU cores and 16.0 GB RAM. (Not running inside Google Colab /content/).
- **GPU Status:** `torch.cuda.is_available() == False` (CPU-only execution).
- **Available Disk Space:** 210.3 GB available on Drive C:, 92.1 GB on Drive D:. Ample storage exists for raw extraction (~3 GB), processed splits (~3.5 GB), and checkpoints (~1 GB).
- **Existing Assets Preserved:**
  - `data/raw/ibm_aml/HI-Small_Trans.csv` (475.6 MB, 5,078,345 raw transactions) remains completely intact.
  - `data/processed/ibm_aml/HI-Small/` train, validation, and test splits remain intact.
  - All Phase 1–10 baseline models, reports, and checkpoints remain intact in `artifacts/final_baseline/` and `experiments/archive/pre_baseline_20261015_backup/`.

## 2. Component Inspection
1. **Dataset Ingestion & Extraction**:
   - `backend/infrastructure/data/downloaders/kaggle_downloader.py`: Authenticates using `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.
   - `datasets/dataset_registry.yaml`: Single source of truth for approved dataset variants. Verified that `LI-Medium_Trans.csv` exists in Kaggle dataset `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` with exact size `2,976,436,978` bytes (~2.98 GB).
2. **Preprocessing Pipeline**:
   - `AmlPreprocessor` reads in chunks of 200,000 rows, normalizes bank identifiers, currency codes, amount transformations, and maps target column to `is_laundering`.
   - Resolves repeated header names (`Account` appearing twice as sender account and receiver account).
3. **Data Splitting Architecture**:
   - `ChronologicalSplitter`: Sorts by timestamp, splits into 60% Train, 20% Validation, 20% Test without row leakage.
4. **Federated Clients**:
   - Exactly 3 clients.
   - `IIDPartitioner`: Stratified random permutation preserving class ratios across 3 clients.
   - `NonIIDPartitioner`: Typology rule routing transactions into Client 1 (Retail Giant), Client 2 (Offshore / Crypto Exchange), and Client 3 (Regional Credit Union).
5. **Models**:
   - `GRU`: 23,681 parameters (feature_dim=57, hidden_size=64).
   - `LSTM`: 31,553 parameters (feature_dim=57, hidden_size=64).
   - `BiLSTM`: 63,105 parameters (feature_dim=57, hidden_size=64).
   - All 3 models verified via 10-criteria unit test suite (100% pass rate).
6. **Federated Aggregation**:
   - `FedAvg`: Sample-weighted parameter aggregation $w_{global} = \sum p_k w_k$ with $p_k = n_k / N$.
   - `FedProx`: Implements local proximal penalty $rac{\mu}{2} ||w - w^t_{global}||_2^2$ with $\mu = 0.1$.
