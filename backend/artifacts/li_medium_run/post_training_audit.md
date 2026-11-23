# FedSecureAI — Critical Post-Training Audit and Root-Cause Report (LI-Medium Campaign)

**Audit Date:** 2026-10-17  
**Dataset:** IBM Transactions for Anti-Money Laundering (`LI-Medium_Trans.csv`, 31,251,483 rows, ~2.98 GB)  
**Processed Directory:** `data/processed/ibm_aml/LI-Medium/`  
**Artifact Directory:** `artifacts/li_medium_run/`  
**Execution Environment:** Windows 11 (AMD64), Python 3.14.7, PyTorch 2.14.0+cpu, 16 logical cores, 16.0 GB RAM  

---

## 1. Verification of Experimental Results Across Artifacts

A full reconciliation was performed across all saved experiment artifacts:
- `artifacts/li_medium_run/all_12_experiments.csv`
- `artifacts/li_medium_run/experiment_status.csv`
- `artifacts/li_medium_run/final_report.md`
- `artifacts/li_medium_run/configs/` (12 JSON configuration files)
- `artifacts/li_medium_run/checkpoints/config_1` through `config_12` (`round_1.pt` .. `round_5.pt`, `global_best.pt`)
- Individual experiment directories: `artifacts/li_medium_run/{iid,non_iid}/{fedavg,fedprox}/{gru,lstm,bilstm}/`

### Reconciliation Result
The reported test PR-AUC of **0.000409 – 0.000758** and ROC-AUC of **0.406 – 0.521** are **100% genuine and reproducible** directly from the saved model checkpoints. They were not truncated, corrupted in storage, or fabricated.

---

## 2. Investigation of the 200,000-Sequence Limit

### Findings & Evidence
1. **Runner Configuration Parameter**:
   In `backend/scripts/run_li_medium_12_experiments.py`, line 81:
   ```python
   def execute_all_12_li_medium_experiments(..., row_limit: int | None = 200000):
   ```
2. **Dataset Loaders & Sequence Builders**:
   - `ProcessedSequenceProvider._read(...)` takes `config.row_limit` and passes `nrows=config.row_limit` to `pd.read_csv`.
   - When `row_limit=200000` was specified, **all 12 baseline configurations processed only the first 200,000 chronological transactions** of the `train` split, the first 200,000 transactions of the `validation` split, and the first 200,000 transactions of the `test` split.
3. **Full Split Sizes vs. Sampled Window**:
   - Full Train split: **18,752,217** rows (8,138 laundering cases, 0.0434% prevalence).
   - Audited Train window: **200,000** rows (only **9** laundering cases, 0.0045% prevalence).
   - Audited Validation window: **200,000** rows (only **69** laundering cases, 0.0345% prevalence).
   - Audited Test window: **200,000** rows (only **105** laundering cases, 0.0525% prevalence).
4. **Impact on Model Training**:
   Across the 3 federated clients, the 9 positive training cases were divided as follows:
   - **IID**: Exactly 3 positive sequences per client across 66,667 examples.
   - **Non-IID**:
     - `client_1` (Retail Giant): 24,009 rows, **8** positives.
     - `client_2` (Offshore/Crypto): 8,266 rows, **0** positives.
     - `client_3` (Credit Union): 167,725 rows, **1** positive.
   With only 0–8 positive examples per client in an entire local epoch (66–164 mini-batches of size 1024), standard unweighted binary cross-entropy loss gradients were dominated $>22,000:1$ by negative examples.

---

## 3. Audit of Labels, Features, and Sequence Construction

A deep audit was conducted on the pipeline components, verified by both data inspection and 82 automated unit/regression tests (`pytest backend/tests/unit/` passed 100%):

1. **Label Mapping**:
   - Raw `Is Laundering` (`0` and `1`) maps strictly to `is_laundering`.
   - Positives verified across raw and processed sets. For example, transaction at `2022-09-01 00:01:00` from account `10042B660` to `80426BB90` exists in raw row 33489 with `Is Laundering=1` and appears at processed row 44053 with `is_laundering=1`.
2. **Feature Matrix Invariant**:
   - The feature matrix consists of exactly 23 features (`amount_paid_log`, `same_currency`, `payment_format_code`, `minute_of_day_sin/cos`, etc.).
   - Verified that neither `is_laundering` nor `Is Laundering` exists in the feature matrix (zero target leakage).
3. **Account Column Disambiguation**:
   - Raw columns `Account` (sender) and `Account.1` (receiver) are cleanly disambiguated into `from_account` and `to_account`.
4. **Sequence Construction**:
   - Sequences are grouped strictly by originating account (`from_account`) in ascending chronological order.
   - Right-padded with zeros and masked using `pack_padded_sequence` so recurrent states do not absorb zero padding.
5. **Preprocessing Isolation**:
   - Running standardizers and ordinal encoders were fitted strictly on `train` timestamps (`<= 2022-09-09 20:39:00`) and frozen before processing validation and test splits.
6. **Prediction Output**:
   - The models output raw logits; `torch.sigmoid(logits)` computes probabilities during evaluation. Positive-class probability $P(Y=1)$ is correctly assigned to index 1.

---

## 4. Independent Metric Recomputation and Diagnostics

Direct re-evaluation was performed on all 12 saved checkpoints on the held-out test split (200,000 sequences, 105 positive cases, baseline prevalence $105 / 200000 = 0.000525$):

| Config # | Model | Algo | Best Round | Frozen Thresh | Pred Positives | True Pos (105) | TP | FP | FN | TN | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | GRU | FedAvg | 5 | 0.858801 | 6 | 105 | 0 | 6 | 105 | 199,889 | 0.000000 | 0.000000 | 0.000000 | 0.000690 | 0.520509 |
| **2** | GRU | FedProx | 5 | 0.953469 | 2,044 | 105 | 1 | 2,043 | 104 | 197,852 | 0.000489 | 0.009524 | 0.000931 | 0.000726 | 0.414849 |
| **3** | LSTM | FedAvg | 5 | 0.000011 | 154,951 | 105 | 84 | 154,867 | 21 | 45,028 | 0.000542 | 0.800000 | 0.001083 | 0.000440 | 0.451727 |
| **4** | LSTM | FedProx | 5 | 0.765747 | 3,152 | 105 | 6 | 3,146 | 99 | 196,749 | 0.001904 | 0.057143 | 0.003684 | 0.000758 | 0.449721 |
| **5** | BiLSTM | FedAvg | 5 | 0.000813 | 121,146 | 105 | 64 | 121,082 | 41 | 78,813 | 0.000528 | 0.609524 | 0.001056 | 0.000414 | 0.420128 |
| **6** | BiLSTM | FedProx | 1 | 0.675147 | 4,387 | 105 | 4 | 4,383 | 101 | 195,512 | 0.000912 | 0.038095 | 0.001781 | 0.000698 | 0.421655 |
| **7** | GRU | FedAvg | 1 | 0.707500 | 643 | 105 | 0 | 643 | 105 | 199,252 | 0.000000 | 0.000000 | 0.000000 | 0.000725 | 0.516661 |
| **8** | GRU | FedProx | 5 | 0.972330 | 1,856 | 105 | 1 | 1,855 | 104 | 198,040 | 0.000539 | 0.009524 | 0.001020 | 0.000719 | 0.406032 |
| **9** | LSTM | FedAvg | 1 | 0.455464 | 1,645 | 105 | 2 | 1,643 | 103 | 198,252 | 0.001216 | 0.019048 | 0.002286 | 0.000550 | 0.491823 |
| **10** | LSTM | FedProx | 5 | 0.806556 | 2,795 | 105 | 5 | 2,790 | 100 | 197,105 | 0.001789 | 0.047619 | 0.003448 | 0.000754 | 0.440472 |
| **11** | BiLSTM | FedAvg | 5 | 0.000000 | 177,348 | 105 | 85 | 177,263 | 20 | 22,632 | 0.000479 | 0.809524 | 0.000958 | 0.000409 | 0.412240 |
| **12** | BiLSTM | FedProx | 1 | 0.715956 | 2,641 | 105 | 2 | 2,639 | 103 | 197,256 | 0.000757 | 0.019048 | 0.001457 | 0.000698 | 0.416834 |

### Metric Analysis & Diagnostics
1. **Mathematical Lower Bound of PR-AUC**:
   - For binary classification, a random or uninformative classifier has an expected PR-AUC equal to the positive class prevalence:
     $$\text{PR-AUC}_{\text{baseline}} \approx \pi = \frac{105}{200,000} = 0.000525$$
   - The measured PR-AUC values ($0.000409 - 0.000758$) are centered directly around this prevalence baseline.
2. **ROC-AUC Hovering Around 0.41 – 0.52**:
   - Under unweighted BCE loss and near-zero positive signal (9 positive training cases total), gradients from the 199,991 negative samples push network outputs toward predicting negative constantly.
   - The models fail to learn discriminative rank-ordering between positive and negative sequences, resulting in random-guess ROC-AUC ($0.41 - 0.52$).

---

## 5. Root Causes Supported by Empirical Evidence

1. **Extreme Data Truncation (200k Rows vs. 18.75M Rows)**:
   - Setting `row_limit=200000` captured only the first 4 minutes of transactions on Day 1 (`2022-09-01 00:00:00` to `00:04:00`).
   - In that 4-minute window, there were only **9 positive transactions** across the entire training set.
   - A recurrent neural network with 23k–63k parameters cannot learn generalized temporal representations of money laundering topologies from 9 positive examples.
2. **Unweighted Binary Cross-Entropy Loss**:
   - Even when positive examples are encountered, standard `nn.BCEWithLogitsLoss()` weights positive and negative errors equally ($1:1$).
   - When negative examples outnumber positive examples by $22,221:1$ (in the 200k window) or $2,303:1$ (in the full training split), the gradient update from negative examples completely swamps any positive learning signal.
3. **FedProx Interaction at $\mu = 0.1$ with Severe Imbalance**:
   - When positive gradients are virtually non-existent, the FedProx proximal penalty $\mu(w - w^t)$ with $\mu=0.1$ acts as a stiff spring tethering weights to initialization, preventing client updates from moving away from the baseline prior.

---

## 6. Engineering Repairs & Validation Suite

1. **Parameterizing `row_limit`**:
   - Updated `backend/scripts/run_li_medium_12_experiments.py` to support explicit command-line flags `--row-limit` (allowing full-data or scalable subsets) and documented subset sizes accurately.
2. **Preserving Recomputation Script**:
   - Created `backend/scripts/recompute_test_metrics.py` to allow instant, non-destructive metric verification of any model checkpoint without re-running training.
3. **Regression Tests**:
   - Full test suite verified: **82 out of 82 targeted unit/regression tests passing (100%)**.
   - Verified that all models, partitioners, and metrics calculators adhere to mathematical and architectural invariants.

---

## 7. Next Steps & Recommendations

1. **Scale Training Window**:
   - Run training with a substantial chronological window (e.g. 2,000,000 to 5,000,000 rows, or the complete 18.75M training set using chunked PyTorch DataLoader streaming) so models encounter hundreds to thousands of positive laundering sequences.
2. **Loss Formulation for Severe Imbalance**:
   - To achieve PR-AUC > 0.80 as targeted, transition from unweighted BCE to **Pos-Weight BCE** (`pos_weight = N_neg / N_pos`) or **Binary Focal Loss**, ensuring rare laundering transactions generate sufficient gradient magnitude.

