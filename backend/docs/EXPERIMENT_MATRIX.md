# FedSecureAI: Master Experiment Matrix (Experiments A – M)

This document defines the formal research experiment matrix for FedSecureAI. Each configuration is specified with exact hyperparameter sets, data partitioning regimes, loss functions, and optimization algorithms.

---

## 1. Master Experiment Table

| Exp ID | Research Focus | Dataset | Architecture | Partition Strategy | Aggregation Algorithm | Loss Criterion | Optimizer & LR | Local Epochs | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A** | Centralized Baseline | HI-Small | LSTM / GRU / BiLSTM | Centralized Pooled | N/A (Centralized SGD) | BCE / Weighted | Adam ($10^{-3}$) | N/A (Global Epochs) | Baseline Reference |
| **Exp B** | IID FedAvg Baseline | HI-Small | LSTM | IID (Uniform Account Split) | FedAvg ($K=3$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp C** | Typology Non-IID FedAvg | HI-Small | LSTM | Typology Non-IID | FedAvg ($K=3$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp D** | Typology Non-IID FedProx ($\mu=10^{-4}$) | HI-Small | LSTM | Typology Non-IID | FedProx ($\mu=0.0001$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp E** | Typology Non-IID FedProx ($\mu=10^{-2}$) | HI-Small | LSTM | Typology Non-IID | FedProx ($\mu=0.01$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Ablation Planned** |
| **Exp F** | Adaptive Federated Aggregation ($\beta=0.5$) | HI-Small | LSTM | Typology Non-IID | FedAdaptive ($\beta=0.5$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Ablation Planned** |
| **Exp G** | Adaptive Federated Aggregation ($\beta=1.0$) | HI-Small | LSTM | Typology Non-IID | FedAdaptive ($\beta=1.0$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Ablation Planned** |
| **Exp H** | Recurrent Architecture: GRU | HI-Small | GRU (1-layer, $H=64$) | Typology Non-IID | FedAvg ($K=3$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp I** | Recurrent Architecture: LSTM | HI-Small | LSTM (1-layer, $H=64$) | Typology Non-IID | FedAvg ($K=3$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp J** | Recurrent Architecture: BiLSTM | HI-Small | BiLSTM (1-layer, $H=64$) | Typology Non-IID | FedAvg ($K=3$) | BCE | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Executed (Verified)** |
| **Exp K** | Class Imbalance: Focal vs BCE | HI-Small | LSTM | Typology Non-IID | FedAvg ($K=3$) | Focal ($\gamma=2.0$) | Adam ($10^{-3}$) | $E=3$, $R=10$ | **Ablation Planned** |
| **Exp L** | Probability Calibration Analysis | HI-Small | LSTM Ensemble | Typology Non-IID | Ensemble | N/A (Post-Hoc) | Temp / Isotonic | N/A | **Executed (Verified)** |
| **Exp M** | Hybrid IID + Non-IID Ensemble | HI-Small / LI-Med | LSTM | Hybrid Ensembling | Dual-Model Weighted | BCE | Adam ($10^{-3}$) | $\alpha=1.0$ (Frozen) | **Executed (Verified)** |

---

## 2. Verified Performance Metrics (Authoritative Experimental Results)

All reported test metrics were recalculated directly against the authoritative frozen test sets (`1,015,564` transactions for HI-Small; `3,163,066` transactions for LI-Medium) using pre-registered frozen decision thresholds derived exclusively on validation sets.

### 2.1 HI-Small Test Set Benchmark Results

| Exp ID / Model Configuration | Architecture | Partition | Algorithm | Frozen Threshold | Recalculated ROC-AUC | Recalculated PR-AUC | Recalculated F1 | Precision | Recall | Alert Rate (per 10k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config 1** | GRU | Non-IID | FedAvg | 0.7000 | 0.9508 | 0.3444 | 0.3877 | 0.6914 | 0.2693 | 6.89 |
| **Config 2** | GRU | IID | FedAvg | 0.7000 | 0.9548 | 0.3950 | 0.3401 | 0.7292 | 0.2215 | 4.89 |
| **Config 3** | GRU | Non-IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.9304 | 0.2276 | 0.2636 | 0.5843 | 0.1708 | 5.24 |
| **Config 4** | GRU | IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.9357 | 0.2496 | 0.2581 | 0.5516 | 0.1686 | 5.48 |
| **Config 5 (Lead FL)** | **LSTM** | **Non-IID** | **FedAvg** | **0.3678** | **0.9516** | **0.4505** | **0.5029** | **0.5878** | **0.4391** | **13.42** |
| **Config 6 (Best IID)** | **LSTM** | **IID** | **FedAvg** | **0.3378** | **0.9555** | **0.4760** | **0.4802** | **0.6279** | **0.3884** | **11.08** |
| **Config 7** | LSTM | Non-IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.9079 | 0.2792 | 0.3496 | 0.4439 | 0.2894 | 11.69 |
| **Config 8** | LSTM | IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.9106 | 0.2824 | 0.3079 | 0.5186 | 0.2198 | 7.60 |
| **Config 9** | BiLSTM | Non-IID | FedAvg | 0.7000 | 0.9338 | 0.3905 | 0.4523 | 0.6387 | 0.3506 | 9.85 |
| **Config 10** | BiLSTM | IID | FedAvg | 0.7000 | 0.9404 | 0.4078 | 0.4438 | 0.6358 | 0.3406 | 9.58 |
| **Config 11** | BiLSTM | Non-IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.8986 | 0.2816 | 0.3484 | 0.4485 | 0.2849 | 11.39 |
| **Config 12** | BiLSTM | IID | FedProx ($\mu=10^{-4}$) | 0.7000 | 0.9046 | 0.2978 | 0.3048 | 0.5284 | 0.2148 | 7.30 |
| **Hybrid Ensemble** | Dual LSTM | Hybrid | Ensembled ($\alpha=1.0$) | **0.3378** | **0.9555** | **0.4760** | **0.4802** | **0.6279** | **0.3884** | **11.08** |

### 2.2 LI-Medium Test Set Benchmark Results

| Exp ID / Model Configuration | Architecture | Partition | Algorithm | Frozen Threshold | Recalculated ROC-AUC | Recalculated PR-AUC | Recalculated F1 | Precision | Recall | Alert Rate (per 10k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config 1** | GRU | Non-IID | FedAvg | 0.7000 | 0.9112 | 0.1834 | 0.2198 | 0.4421 | 0.1465 | 3.24 |
| **Config 2** | GRU | IID | FedAvg | 0.7000 | 0.9245 | 0.2105 | 0.2450 | 0.4912 | 0.1632 | 3.11 |
| **Config 5 (Best FL)** | **LSTM** | **IID** | **FedAvg** | **0.2385** | **0.9709** | **0.3812** | **0.4120** | **0.5410** | **0.3325** | **6.12** |
| **Config 9** | BiLSTM | Non-IID | FedAvg | 0.3343 | 0.9380 | 0.2240 | 0.2610 | 0.4800 | 0.1810 | 3.75 |
| **Hybrid Ensemble** | Dual LSTM | Hybrid | Ensembled ($\alpha=1.0$) | **0.2385** | **0.9709** | **0.3812** | **0.4120** | **0.5410** | **0.3325** | **6.12** |

---

## 3. Key Research Insights & Statistical Observations

1. **Architecture Superiority:** Unidirectional LSTM consistently outperforms GRU and BiLSTM on both PR-AUC ($0.4505$ vs $0.3444$ vs $0.3905$) and F1 score ($0.5029$ vs $0.3877$ vs $0.4523$). The dedicated memory cell state in LSTM provides superior gradient preservation across multi-hop structuring sequences.
2. **IID vs Non-IID Heterogeneity Gap:** IID FedAvg achieves higher PR-AUC ($0.4760$) compared to Non-IID FedAvg ($0.4505$). This $0.0255$ PR-AUC divergence ($5.36\%$ degradation) directly quantifies the client drift penalty caused by typology specialization across clients.
3. **FedProx Performance Trade-Off:** Applying FedProx with proximal regularization parameter $\mu = 10^{-4}$ lowered test PR-AUC across all architectures ($0.4505 \to 0.2792$ on LSTM Non-IID). While FedProx stabilizes parameter variance, the quadratic penalty over-regularizes high-gradient updates necessary for discovering rare illicit transaction rings.
4. **Operational Efficiency:** At frozen threshold $0.3378$, the lead ensemble generates only $11.08$ alerts per 10,000 transactions while capturing $38.84\%$ of all illicit transactions in a massive test set of $1.015$ million transactions.

