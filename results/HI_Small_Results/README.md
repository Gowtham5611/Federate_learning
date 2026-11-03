# FedSecureAI — HI-Small Benchmark Results

- **Dataset:** IBM AML HI-Small (5,078,345 total transactions)
- **Train Split:** 3,047,216 rows | **Validation Split:** 1,015,565 rows | **Test Split:** 1,015,564 rows
- **Test Positive Prevalence:** 0.1769% (1,797 laundering transactions)
- **Status:** **PASS** (12/12 configurations completed, tested, and verified)
- **Top Model:** `hi_small_lstm_fedavg_iid_seed42` (ROC-AUC: **0.955481**, PR-AUC: **0.475977**, F1: **0.480196**, Precision: **90.18%**)
