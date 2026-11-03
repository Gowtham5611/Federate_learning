# FedSecureAI — LI-Medium Dataset Extraction and Verification Report

**Report Date:** 2026-10-17  
**Target Source:** `LI-Medium_Trans.csv`  
**Expected Uncompressed Size:** 2,976,436,978 bytes (~2.976 GB)  
**Destination Path:** `data/raw/ibm_aml/LI-Medium_Trans.csv`  

---

## 1. Storage & Pre-Extraction Audit
- **System Drive (C:) Free Space:** 210.3 GB
- **Secondary Drive (D:) Free Space:** 92.1 GB
- **Required Space for Download & Extraction:** ~4.0 GB (0.7 GB compressed archive + 2.98 GB extracted CSV)
- **Safety Margin:** Available disk space exceeds extraction requirement by **> 50x**. No risk of disk exhaustion.

## 2. Source File Discovery Audit
1. **Local Search Results**:
   - Searched `c:\Users\Acer\OneDrive\Desktop\Federate_learning\data\raw\ibm_aml\`: File present is `HI-Small_Trans.csv` (475,664,282 bytes).
   - Searched local drives `C:\` and `D:\` (including `Downloads`, `Desktop`, `OneDrive`, other project workspaces).
   - **Finding**: `LI-Medium_Trans.csv` is not yet stored locally on this machine or mounted Google Drive.
2. **Remote Kaggle Repository Audit**:
   - Authenticated against Kaggle API using credentials in `.env` (`KAGGLE_USERNAME=iml071`).
   - Querying dataset `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`:
     - File `LI-Medium_Trans.csv` confirmed available on Kaggle.
     - Exact remote file size: `2,976,436,978` bytes (~2.976 GB).
     - Associated files available: `LI-Medium_Patterns.txt` (390 KB), `LI-Medium_accounts.csv` (141.8 MB).

## 3. Extraction Status & Blocker Resolution
- **Status:** **READY FOR INGESTION / PENDING USER PATH SELECTION**
- **Action Required**:
  - **Option A (Automated Kaggle Download)**: If approved, the agent will execute `kaggle.api.dataset_download_file('ealtman2019/ibm-transactions-for-anti-money-laundering-aml', 'LI-Medium_Trans.csv', path='data/raw/ibm_aml')` and safely stream-extract to `data/raw/ibm_aml/LI-Medium_Trans.csv`.
  - **Option B (Local / Google Drive Path)**: If you already downloaded or mounted `LI-Medium_Trans.csv` elsewhere, specify its exact path (e.g. `D:\path\to\LI-Medium_Trans.csv` or `G:\My Drive\...`).
