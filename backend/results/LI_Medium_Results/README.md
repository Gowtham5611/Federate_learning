# FedSecureAI — LI-Medium Benchmark Results

- **Dataset:** IBM AML LI-Medium (31,251,483 total transactions)
- **Train Split:** 18,752,217 rows | **Validation Split:** 6,250,154 rows | **Test Split:** 6,249,112 rows
- **Test Positive Prevalence:** 0.0605% (3,783 laundering transactions)
- **Status:** **PASS** (12/12 configurations completed on Google Colab Tesla T4 GPU)
- **Top IID Model:** `config_1` (GRU IID FedAvg: ROC-AUC: **0.969705**, PR-AUC: **0.253598**, F1: **0.316847**)
- **Top Non-IID Model:** `config_9` (LSTM Non-IID FedAvg: ROC-AUC: **0.923644**, PR-AUC: **0.045785**, F1: **0.075216**)
