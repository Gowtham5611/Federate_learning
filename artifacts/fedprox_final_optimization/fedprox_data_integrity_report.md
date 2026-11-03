# FedSecureAI — FedProx Data Integrity and Lineage Report

## 1. Split Metadata and Verification
- **Dataset**: IBM AML `LI-Medium` (`data/raw/ibm_aml/LI-Medium_Trans.csv`)
- **Total Transactions**: 31,251,483 rows, 16,041 laundering positives (0.0513% prevalence)
- **Train Split**: 18,752,217 transactions, 8,138 laundering positives (0.0434% prevalence)
- **Validation Split**: 6,250,154 transactions, 4,120 laundering positives (0.0659% prevalence)
- **Test Split**: 6,249,112 transactions, 3,783 laundering positives (0.0605% prevalence)
- **Feature Dimension**: Exactly 57 features (one-hot nominal encoding verified)

## 2. Training Slice & Client Partitions
- **Training Slice**: Strictly the first 200,000 chronological sequences from `train/transactions.csv`.
- **Client Count**: Exactly 3 federated clients.
- **IID Partitioning**: Uniform random assignment with seed 42.
- **Non-IID Partitioning**: Account-based Dirichlet distribution with seed 42.
- **Zero Data Leakage**: Sequences are built strictly within split boundaries. Model selection and operating threshold optimization were executed strictly on the validation set prior to any test evaluation.
