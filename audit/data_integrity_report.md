# Data Integrity, Preprocessing, and Leakage Audit Report

**Execution Timestamp**: 2026-10-18T06:36:05.210163+00:00  
**Target Class Definition**: `Is Laundering` (1 = Illicit/Laundering, 0 = Legitimate)  
**Leakage Verification**: PASSED (0 temporal leakage, 0 train/val/test overlap)  
**Feature Dimension**: 57 one-hot expanded numeric features  

---

## 1. Feature Schemas & Cross-Dataset Compatibility

- **Raw Schemas**: Both `HI-Small_Trans.csv` and `LI-Medium_Trans.csv` possess identical 11-column transactional schemas (`Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`).
- **Engineered Schemas**: Both datasets produce 23 engineered features expanding to exactly 57 one-hot encoded sequence features.
- **Combined Dataset Provenance**: The combined dataset adds an explicit metadata column `dataset_source` ('hi_small' or 'li_medium') preserving source lineage without contaminating feature inputs.
- **Duplicate Analysis**: An exact hash audit confirmed **0 cross-dataset duplicate transactions** between HI-Small and LI-Medium.

---

## 2. Preprocessing & Leakage Controls

1. **Target Encoding**: The binary target is strictly non-negative: laundering transactions are encoded as positive class ($y = 1$) and legitimate as negative ($y = 0$).
2. **Scaler Isolation**: Transformers and scalers (StandardScaler on amount, MinMaxScaler on time steps) were fitted **strictly on training partitions**. No statistics from validation or test sets leaked into feature pipelines.
3. **Chronological Splitting**: Sequence splits are partitioned strictly along chronological timestamps (train $\to$ validation $\to$ test), preventing lookahead bias.
4. **Validation Isolation**: Optimal classification thresholds ($\tau^*$) were identified strictly on validation sequences by sweeping $\tau \in [0.01, 0.99]$. Test data was evaluated only with frozen thresholds.

---

## 3. Client Partitioning Analysis (3-Client Setup)

### HI-Small Partitions (100,000 Training Sequences, 137 Positives)
- **IID**:
  - Client 0: 33,334 samples, 48 positives (prevalence: 0.144%)
  - Client 1: 33,333 samples, 43 positives (prevalence: 0.129%)
  - Client 2: 33,333 samples, 46 positives (prevalence: 0.138%)
- **Non-IID (Label-Skewed)**:
  - Client 0: 33,429 samples, 95 positives (prevalence: 0.284%) — receives 70% of illicit cases
  - Client 1: 33,360 samples, 27 positives (prevalence: 0.081%) — receives 20% of illicit cases
  - Client 2: 33,211 samples, 15 positives (prevalence: 0.045%) — receives 10% of illicit cases

### LI-Medium Partitions (100,000 Training Sequences, 43 Positives)
- **IID**:
  - Client 0: 33,334 samples, 14 positives (prevalence: 0.042%)
  - Client 1: 33,333 samples, 16 positives (prevalence: 0.048%)
  - Client 2: 33,333 samples, 13 positives (prevalence: 0.039%)
- **Non-IID (Label-Skewed)**:
  - Client 0: 33,363 samples, 30 positives (prevalence: 0.090%) — receives 70% of illicit cases
  - Client 1: 33,341 samples, 8 positives (prevalence: 0.024%) — receives 20% of illicit cases
  - Client 2: 33,296 samples, 5 positives (prevalence: 0.015%) — receives 10% of illicit cases

### Combined Partitions (100,000 Training Sequences, 137 Positives)
- **IID**: 33,333 samples per client with homogeneous ~0.137% prevalence.
- **Non-IID**: Replicated 70%/20%/10% illicit concentration across clients.
