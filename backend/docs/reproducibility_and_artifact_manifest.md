# FedSecureAI — Reproducibility & Cryptographic Artifact Manifest

**Repository Version:** FedSecureAI Enterprise 2.1.0  
**Verification Date:** October 30, 2026  
**Seed Specification:** Fixed Seed 42 for all data splitting, client partitioning, weight initialization, and evaluation  

---

## 1. Experimental Dataset Provenance

The benchmarks utilize real, full-scale IBM Synthetic Financial Datasets for Anti-Money Laundering:

### A. IBM AML HI-Small Pipeline
- **Raw File**: `data/raw/ibm_aml/HI-Small_Trans.csv` (721 MB)
- **Total Transactions**: $5,078,345$
- **Distinct Accounts**: $515,080$
- **Class Imbalance**: $1,797$ positive laundering transactions in test split ($0.1769\%$)
- **Splits**:
  - Training: $3,047,216$ rows ($60\%$)
  - Validation: $1,015,565$ rows ($20\%$)
  - Held-Out Test: $1,015,564$ rows ($20\%$)

### B. IBM AML LI-Medium Pipeline
- **Raw File**: `data/raw/ibm_aml/LI-Medium_Trans.csv` (4.45 GB)
- **Total Transactions**: $31,247,404$
- **Distinct Accounts**: $3,100,000$
- **Class Imbalance**: $3,783$ positive laundering transactions in test split ($0.0605\%$)
- **Splits**:
  - Training: $18,748,442$ rows ($60\%$)
  - Validation: $6,249,850$ rows ($20\%$)
  - Held-Out Test: $6,249,112$ rows ($20\%$)

---

## 2. Feature Engineering & Preprocessing Specification

- **Input Dimension**: 57 expanded continuous and binary features.
  - Raw amount paid & received (standardized with running mean and variance).
  - Exchange rate conversion ratios.
  - One-hot currency indicators (16 currencies: US Dollar, Euro, Yen, etc.).
  - One-hot payment format indicators (Cheque, Credit Card, Wire, ACH, Cash).
  - Temporal features: Hour-of-day (sine/cosine cyclical encoding), Day-of-week, Day-of-month.
  - Causal account velocity: Rolling transaction counts (1h, 24h, 7d), cumulative flow volumes, and fan-out ratios.
- **Sequence Window Length**: 10 transactions ($L = 10$).
- **Packing Strategy**: Right-padded with zero-vectors; packed via `torch.nn.utils.rnn.pack_padded_sequence` with true lengths ($1 \le L \le 10$).

---

## 3. Federated Training Hyperparameters

- **Number of Client Silos**: $K = 3$
- **Communication Rounds**: $R = 5$
- **Local Epochs per Round**: $E = 1$
- **Local Optimizer**: Adam ($\text{lr} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Batch Size**: 256
- **Loss Function**: `BCEWithLogitsLoss` (numerically stable fused sigmoid logit loss)
- **FedProx Regularization Parameter**: $\mu = 0.0001$ strictly
- **Client Partitioning**:
  - **IID**: 33.33% uniform random allocation ($66,667$ rows/client on HI-Small).
  - **Non-IID**: Typology-based volume skew: Client 1 ($12.0\%$, $24,009$ rows), Client 2 ($4.1\%$, $8,266$ rows), Client 3 ($83.9\%$, $167,725$ rows).

---

## 4. Cryptographic Checkpoint & Prediction Manifest

The following tables list the exact relative file paths and cryptographic SHA256 hashes verified on disk:

### A. HI-Small Checkpoint Manifest

| Config ID | Model Identifier | Checkpoint Path | SHA256 Checkpoint Hash | Predictions Path | Test Samples |
|:---|:---|:---|:---|:---|:---|
| **config_1** | `hi_small_gru_fedavg_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_gru_fedavg_iid_seed42/hi_small_gru_fedavg_iid_seed42_global_best.pt` | `1da9626d3ca24dadd93b6c758d76ede7dfae4b0f245fd4468552e2476b113c04` | `..._test_preds.npy` | 1,015,564 |
| **config_2** | `hi_small_gru_fedprox_mu0.0001_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_gru_fedprox_mu0.0001_iid_seed42/hi_small_gru_fedprox_mu0.0001_iid_seed42_global_best.pt` | `fcd14fa426ed4ddbc6a00f1f5109286be0d7057648654c3c336fd6102f5cd52f` | `..._test_preds.npy` | 1,015,564 |
| **config_3** | `hi_small_lstm_fedavg_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt` | `dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4` | `..._test_preds.npy` | 1,015,564 |
| **config_4** | `hi_small_lstm_fedprox_mu0.0001_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedprox_mu0.0001_iid_seed42/hi_small_lstm_fedprox_mu0.0001_iid_seed42_global_best.pt` | `d9ddeb275aa6b83ef67cf939ecc52f897e491c9ac706c535dfefd02f672a3f74` | `..._test_preds.npy` | 1,015,564 |
| **config_5** | `hi_small_bilstm_fedavg_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_bilstm_fedavg_iid_seed42/hi_small_bilstm_fedavg_iid_seed42_global_best.pt` | `e9a8c43b09524e641d8d2f3a6d919076f589b7f77d2ab89695b66de30ba6c5aa` | `..._test_preds.npy` | 1,015,564 |
| **config_6** | `hi_small_bilstm_fedprox_mu0.0001_iid_seed42` | `results/HI_Small_Results/checkpoints/hi_small_bilstm_fedprox_mu0.0001_iid_seed42/hi_small_bilstm_fedprox_mu0.0001_iid_seed42_global_best.pt` | `ace199af13d8a86afbd16706582961ff9c5bc573e0fe8feaed76f716b75c7b8b` | `..._test_preds.npy` | 1,015,564 |
| **config_7** | `hi_small_gru_fedavg_seed42` | `results/HI_Small_Results/checkpoints/hi_small_gru_fedavg_seed42/hi_small_gru_fedavg_seed42_global_best.pt` | `d225abce33d0dd350e5d9623aed77ec71a0591daf6d2d75cc77ff8d0c2394199` | `..._test_preds.npy` | 1,015,564 |
| **config_8** | `hi_small_gru_fedprox_mu0.0001_seed42` | `results/HI_Small_Results/checkpoints/hi_small_gru_fedprox_mu0.0001_seed42/hi_small_gru_fedprox_mu0.0001_seed42_global_best.pt` | `70cee297f8caec9719b0698447901b844e01b5506f3ec3439818603cc0f4eb58` | `..._test_preds.npy` | 1,015,564 |
| **config_9** | `hi_small_lstm_fedavg_seed42` | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_seed42/hi_small_lstm_fedavg_seed42_global_best.pt` | `1b64f79ddd4dc85d6bad953e93f2679bc102bf5fdbffeaf8b59df69ddd19ab1c` | `..._test_preds.npy` | 1,015,564 |
| **config_10**| `hi_small_lstm_fedprox_mu0.0001_seed42` | `results/HI_Small_Results/checkpoints/hi_small_lstm_fedprox_mu0.0001_seed42/hi_small_lstm_fedprox_mu0.0001_seed42_global_best.pt` | `35bf9029a3f1778f359c67ad7b54b27b7e92bab476d95a505a62914e383f26fd` | `..._test_preds.npy` | 1,015,564 |
| **config_11**| `hi_small_bilstm_fedavg_seed42` | `results/HI_Small_Results/checkpoints/hi_small_bilstm_fedavg_seed42/hi_small_bilstm_fedavg_seed42_global_best.pt` | `508427bde3560f2f1641040c8b2cfb0027fc37647b8724baba4eb24548d5bcf0` | `..._test_preds.npy` | 1,015,564 |
| **config_12**| `hi_small_bilstm_fedprox_mu0.0001_seed42` | `results/HI_Small_Results/checkpoints/hi_small_bilstm_fedprox_mu0.0001_seed42/hi_small_bilstm_fedprox_mu0.0001_seed42_global_best.pt` | `17f1a3f01b7818c87ddc002024d15335af2b5c605a50cce7de5009d8159feb90` | `..._test_preds.npy` | 1,015,564 |

---

### B. LI-Medium Checkpoint Manifest

| Config ID | Model Identifier | Checkpoint Path | SHA256 Checkpoint Hash | Predictions Path | Test Samples |
|:---|:---|:---|:---|:---|:---|
| **config_1** | `config_1_gru_fedavg_iid` | `results/LI_Medium_Results/checkpoints/config_1/global_best.pt` | `110fbc328fead4ed78cd61b2f240d43502127bda5c60730bb5096d68edefb377` | `test_predictions.npz` | 6,249,112 |
| **config_2** | `config_2_gru_fedprox_iid` | `results/LI_Medium_Results/checkpoints/config_2/global_best.pt` | `9129e60c41abcc5f2e721132beb1685ba25ca388f7b0137f1cf5a26274ff9ca7` | `test_predictions.npz` | 6,249,112 |
| **config_3** | `config_3_lstm_fedavg_iid` | `results/LI_Medium_Results/checkpoints/config_3/global_best.pt` | `9025a6002a966b6481fe278207d93ed8e5c97aa03eeb6c85b9fb50e3da761df2` | `test_predictions.npz` | 6,249,112 |
| **config_4** | `config_4_lstm_fedprox_iid` | `results/LI_Medium_Results/checkpoints/config_4/global_best.pt` | `701ac1b36ae5bca5df2c5815761bcbae331aab1d7333632c0244dc08062537de` | `test_predictions.npz` | 6,249,112 |
| **config_5** | `config_5_bilstm_fedavg_iid` | `results/LI_Medium_Results/checkpoints/config_5/global_best.pt` | `bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2` | `test_predictions.npz` | 6,249,112 |
| **config_6** | `config_6_bilstm_fedprox_iid` | `results/LI_Medium_Results/checkpoints/config_6/global_best.pt` | `4267b5099cf57a74a6f51913aaade863871f83a794a56d0d03e83cda337eee2e` | `test_predictions.npz` | 6,249,112 |
| **config_7** | `config_7_gru_fedavg_non-iid` | `results/LI_Medium_Results/checkpoints/config_7/global_best.pt` | `583a5e7b5c52d058f137840af4836a4cd9b59aa10d9421a69d063bd2a47749ad` | `test_predictions.npz` | 6,249,112 |
| **config_8** | `config_8_gru_fedprox_non-iid` | `results/LI_Medium_Results/checkpoints/config_8/global_best.pt` | `37e1d655db5267e9fb6e031e92c1d234f4e9ab77dfe8f77929f2bc3e06f26ec8` | `test_predictions.npz` | 6,249,112 |
| **config_9** | `config_9_lstm_fedavg_non-iid` | `results/LI_Medium_Results/checkpoints/config_9/global_best.pt` | `91ea59ef4ad8d183006bda47c44b91d7ce8ea1f52c43f3f548b2763a2b0f3653` | `test_predictions.npz` | 6,249,112 |
| **config_10**| `config_10_lstm_fedprox_non-iid`| `results/LI_Medium_Results/checkpoints/config_10/global_best.pt` | `9f91771374e2fd434d4f187ae95d49c3364f2336d55855386a865ab121edcafb` | `test_predictions.npz` | 6,249,112 |
| **config_11**| `config_11_bilstm_fedavg_non-iid`| `results/LI_Medium_Results/checkpoints/config_11/global_best.pt` | `c2501873518f598360bc996b70d7cd0c67ad72621409d051729d6e260de4f46b` | `test_predictions.npz` | 6,249,112 |
| **config_12**| `config_12_bilstm_fedprox_non-iid`| `results/LI_Medium_Results/checkpoints/config_12/global_best.pt` | `76c70b0835bb12b8be768d0c4ec84b1d42e33f9212b272c10661209bfb389405` | `test_predictions.npz` | 6,249,112 |

