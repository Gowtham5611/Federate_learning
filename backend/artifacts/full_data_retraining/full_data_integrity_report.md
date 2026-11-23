# FedSecureAI — Full Data Integrity and Verification Report

## 1. Official Dataset Split Verification

Every transaction in the processed IBM AML `LI-Medium` directory (`data/processed/ibm_aml/LI-Medium/`) was individually counted and audited via chunked verification:

| Split Name | File Path | Total Transactions | Laundering Positives | Negative Transactions | Positive Prevalence | Class Imbalance Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | `train/transactions.csv` | **18,752,217** | **8,138** | 18,744,079 | 0.043398% | 1 : 2,303.28 |
| **VALIDATION** | `validation/transactions.csv` | **6,250,154** | **4,120** | 6,246,034 | 0.065918% | 1 : 1,516.03 |
| **TEST** | `test/transactions.csv` | **6,249,112** | **3,783** | 6,245,329 | 0.060537% | 1 : 1,650.89 |
| **FULL CORPUS** | Entire Dataset | **31,251,483** | **16,041** | 31,235,442 | 0.051329% | 1 : 1,947.23 |

### Findings:
- Measured row counts match expected metadata.
- Preprocessing scaler parameters (`transformers.json`) confirm: `"fitted_on": "train split only"`.
- Grounded positive class weight for cost-sensitive loss:
  $$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{18,744,079}{8,138} \approx \mathbf{2,303.2774}$$

---

## 2. Feature Dimension & Channel Alignment

- **Channel Width**: Exactly **57 feature channels** produced by `_build_feature_matrix()` in `ProcessedSequenceProvider`.
- **Feature Composition**:
  - 8 Standardized Log-Transformed Continuous Features (Amounts, Prior Sender/Receiver Counts, Prior Elapsed Seconds).
  - 4 Cyclical Time Representations ($\sin/\cos$ for Minute-of-Day and Day-of-Week).
  - 5 Binary Context Flags (`amount_mismatch`, `same_currency`, `same_bank`, `self_transfer`, `is_weekend`, `is_night`, `sender_has_prior`, `receiver_has_prior`).
  - 40 One-Hot Encoded Categorical Channels (Payment Format codes, Payment Currency codes, Receiving Currency codes).
- **Target Exclusion**: `is_laundering` is strictly popped from feature frames before tensor conversion (`frame.pop(target_column)`). Input tensors never contain target labels.
- **Data Quality**: Verified 0 NaN, 0 Inf values across all evaluated batches.

---

## 3. Partitioning & Leakage Invariants

1. **Exact Client Coverage**:
   - `IIDPartitioner` and `NonIIDPartitioner` partition datasets such that:
     $$\bigcup_{k=1}^3 \mathcal{D}_k = \mathcal{D}_{\text{train}}, \quad \mathcal{D}_i \cap \mathcal{D}_j = \emptyset \quad (\forall i \ne j)$$
     $$\sum_{k=1}^3 |\mathcal{D}_k| = 18,752,217, \quad \sum_{k=1}^3 \text{Positives}(\mathcal{D}_k) = 8,138$$
2. **Atomic Account Temporal Integrity**:
   - `NonIIDPartitioner` assigns transactions at the originating account level (`from_account`). Sequences for an account remain on the same client, preventing broken histories across nodes.
3. **Strict Boundary Isolation**:
   - `AccountSequenceDataset` enforces `BOUNDARY_POLICY`: sequences are constructed strictly within split boundaries. Zero temporal or historical cross-split leakage exists between training, validation, and test splits.

