# FedSecureAI — Three-Client Federated Partitioning Audit (LI-Medium)

**Report Date:** 2026-10-17  
**Source Split:** `data/processed/ibm_aml/LI-Medium/train/transactions.csv`  
**Audit Window:** First 500,000 chronological training transactions  
**Number of Clients:** Exactly 3 (`client_1`, `client_2`, `client_3`)  
**Random Seed:** 42  

---

## 1. IID Client Partitions (Stratified Random Permutation)

| Client ID | Typology | Sample Count | Share (%) | Legitimate (0) | Laundering (1) | Positive Rate (%) | Aggregation Weight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `client_1` | Uniform Stratified | 166,667 | 33.33% | 166,661 | 6 | 0.0036% | 0.333334 |
| `client_2` | Uniform Stratified | 166,667 | 33.33% | 166,661 | 6 | 0.0036% | 0.333334 |
| `client_3` | Uniform Stratified | 166,666 | 33.33% | 166,660 | 6 | 0.0036% | 0.333332 |

---

## 2. Non-IID Client Partitions (Account-Level Typology Partitioning)

Partitions transactions by originating account typology to preserve sequence integrity without cross-client leakage:
- **`client_1` (Retail Giant)**: Domestic credit card and cash formats, low-risk personal transactions.
- **`client_2` (Offshore / Crypto Exchange)**: Cross-border wires, cryptocurrency rails, high-velocity transactions.
- **`client_3` (Regional Credit Union)**: Regional checking, ACH, same-bank transfers, personal loans.

| Client ID | Typology Domain | Sample Count | Share (%) | Legitimate (0) | Laundering (1) | Positive Rate (%) | Aggregation Weight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `client_1` | Retail Giant | 56,507 | 11.30% | 56,493 | 14 | 0.0248% | 0.113014 |
| `client_2` | Offshore / Crypto Exchange | 25,764 | 5.15% | 25,763 | 1 | 0.0039% | 0.051528 |
| `client_3` | Regional Credit Union | 417,729 | 83.55% | 417,726 | 3 | 0.0007% | 0.835458 |

---

## 3. Partition Verification Invariants
1. **Partition Isolation**: No sequence from an originating account spans multiple clients.
2. **Conservation of Rows**: Sum of client rows equals total audited training rows.
3. **Non-IID Heterogeneity Verified**: Distinct label prevalence and transaction channel patterns observed between Retail, Offshore/Crypto, and Credit Union domains.