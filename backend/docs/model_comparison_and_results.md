# FedSecureAI — Model Comparison, Empirical Results & Scientific Evaluation

**Scope:** IBM AML Benchmark (HI-Small 5M rows & LI-Medium 31M rows)  
**Evaluated Architectures:** GRU • LSTM • BiLSTM  
**Federated Algorithms:** FedAvg • FedProx ($\mu = 0.0001$)  
**Distributions:** Centralized • Federated IID • Federated Non-IID  

---

## 1. Authoritative Benchmark Matrix

All reported results were evaluated on 100% held-out test splits ($N = 1,015,564$ for HI-Small; $N = 6,249,112$ for LI-Medium). Decision thresholds were established exclusively on validation data during Phase 7 and remained strictly frozen.

### A. IBM AML HI-Small Benchmark ($5,078,345$ Total Transactions)
- **Train Split**: $3,047,216$ rows | **Validation Split**: $1,015,565$ rows | **Test Split**: $1,015,564$ rows
- **Test Positive Prevalence**: $0.1769\%$ ($1,797$ positive laundering transactions)

| Config | Architecture | Distribution | Algorithm | $\mu$ | Threshold | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Brier Score | Alerts/10k |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **config_1** | GRU | IID | FedAvg | $0.0$ | 0.5000 | 0.3950 | 0.9548 | 0.3401 | 0.2782 | 0.4374 | 0.002751 | 27.82 |
| **config_2** | GRU | IID | FedProx | $0.0001$ | 0.5000 | 0.2496 | 0.9357 | 0.2581 | 0.2211 | 0.3100 | 0.003001 | 24.80 |
| **config_3** ⭐ | **LSTM** | **IID** | **FedAvg** | **0.0** | **0.3378** | **0.4760** | **0.9555** | **0.5359** | **0.8183** | **0.3984** | **0.001220** | **6.42** |
| **config_4** | LSTM | IID | FedProx | $0.0001$ | 0.5000 | 0.2824 | 0.9106 | 0.3079 | 0.8597 | 0.1875 | 0.002946 | 3.86 |
| **config_5** | BiLSTM | IID | FedAvg | $0.0$ | 0.5000 | 0.4078 | 0.9404 | 0.4438 | 0.5289 | 0.3823 | 0.002162 | 12.79 |
| **config_6** | BiLSTM | IID | FedProx | $0.0001$ | 0.5000 | 0.2978 | 0.9046 | 0.3048 | 0.2915 | 0.3194 | 0.004199 | 19.39 |
| **config_7** | GRU | Non-IID | FedAvg | $0.0$ | 0.7000 | 0.3444 | 0.9508 | 0.3877 | 0.6914 | 0.2693 | 0.002773 | 6.89 |
| **config_8** | GRU | Non-IID | FedProx | $0.0001$ | 0.6900 | 0.2276 | 0.9304 | 0.2636 | 0.4822 | 0.1814 | 0.003295 | 6.66 |
| **config_9** 🥈 | **LSTM** | **Non-IID**| **FedAvg** | **0.0** | **0.3700** | **0.4505** | **0.9516** | **0.5029** | **0.8398** | **0.3589** | **0.001261** | **7.56** |
| **config_10**| LSTM | Non-IID | FedProx | $0.0001$ | 0.3600 | 0.2792 | 0.9079 | 0.3496 | 0.5461 | 0.2571 | 0.002482 | 8.33 |
| **config_11**| BiLSTM | Non-IID | FedAvg | $0.0$ | 0.6200 | 0.3905 | 0.9338 | 0.4523 | 0.7588 | 0.3222 | 0.002375 | 7.51 |
| **config_12**| BiLSTM | Non-IID | FedProx | $0.0001$ | 0.5800 | 0.2816 | 0.8986 | 0.3484 | 0.5694 | 0.2510 | 0.004289 | 7.80 |

---

### B. IBM AML LI-Medium Benchmark ($31,247,404$ Total Transactions)
- **Train Split**: $18,748,442$ rows | **Validation Split**: $6,249,850$ rows | **Test Split**: $6,249,112$ rows
- **Test Positive Prevalence**: $0.0605\%$ ($3,783$ positive laundering transactions)

| Config | Architecture | Distribution | Algorithm | $\mu$ | Threshold | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Brier Score | Alerts/10k |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **config_1** | GRU | IID | FedAvg | $0.0$ | 0.2362 | 0.2536 | 0.9697 | 0.3168 | 0.6223 | 0.2125 | 0.000501 | 2.07 |
| **config_2** | GRU | IID | FedProx | $0.0001$ | 0.0816 | 0.0887 | 0.9521 | 0.1418 | 0.2120 | 0.1065 | 0.000580 | 3.04 |
| **config_3** | LSTM | IID | FedAvg | $0.0$ | 0.1526 | 0.2451 | 0.9686 | 0.3038 | 0.4405 | 0.2318 | 0.000507 | 3.19 |
| **config_4** | LSTM | IID | FedProx | $0.0001$ | 0.0899 | 0.0895 | 0.9495 | 0.1289 | 0.3089 | 0.0814 | 0.000582 | 1.60 |
| **config_5** ⭐ | **BiLSTM** | **IID** | **FedAvg** | **0.0** | **0.2385** | **0.2534** | **0.9695** | **0.3168** | **0.5942** | **0.2160** | **0.000505** | **2.20** |
| **config_6** | BiLSTM | IID | FedProx | $0.0001$ | 0.0852 | 0.0871 | 0.9514 | 0.1363 | 0.2957 | 0.0886 | 0.000582 | 1.81 |
| **config_7** | GRU | Non-IID | FedAvg | $0.0$ | 0.7284 | 0.0076 | 0.8726 | 0.0236 | 0.0131 | 0.1195 | 0.010721 | 55.25 |
| **config_8** | GRU | Non-IID | FedProx | $0.0001$ | 0.6172 | 0.0051 | 0.8373 | 0.0193 | 0.0107 | 0.0973 | 0.006817 | 54.93 |
| **config_9** 🥈 | **LSTM** | **Non-IID**| **FedAvg** | **0.0** | **0.3343** | **0.0458** | **0.9236** | **0.0752** | **0.0902** | **0.0645** | **0.000821** | **4.33** |
| **config_10**| LSTM | Non-IID | FedProx | $0.0001$ | 0.4773 | 0.0054 | 0.8549 | 0.0190 | 0.0110 | 0.0708 | 0.003666 | 39.03 |
| **config_11**| BiLSTM | Non-IID | FedAvg | $0.0$ | 0.7033 | 0.0240 | 0.8952 | 0.0651 | 0.0792 | 0.0552 | 0.002642 | 4.22 |
| **config_12**| BiLSTM | Non-IID | FedProx | $0.0001$ | 0.5940 | 0.0080 | 0.8710 | 0.0221 | 0.0133 | 0.0656 | 0.003931 | 29.88 |

---

## 2. Client-Level Performance & Data Partitioning Dynamics

### A. Non-IID Volume and Typology Skew
In the Non-IID experimental partitioning, transaction records were partitioned by bank typology rather than uniform random split:
- **Client 1**: $24,009$ training samples ($12.00\%$)
- **Client 2**: $8,266$ training samples ($4.13\%$)
- **Client 3**: $167,725$ training samples ($83.86\%$)

### B. Scientific Implications
1. **Aggregator Weighting**: In FedAvg, client updates are weighted by sample volume ($w_k = \frac{n_k}{N}$). Client 3's local model parameters contribute over $83.8\%$ of the global model weight at every round.
2. **Loss Divergence**: Client 1 and Client 2 experience higher round losses ($\approx 0.18 - 0.24$) because the aggregated global model is biased toward Client 3's transaction distributions.
3. **Generalization Impact**: The resulting Non-IID models under-represent rare laundering typologies present in smaller banks, leading to lower PR-AUC ($0.4505$ vs $0.4760$ on HI-Small; $0.0458$ vs $0.2536$ on LI-Medium).

---

## 3. Communication Cost & Bandwidth Consumption

Parameter updates are transmitted in Float32 precision (4 bytes per parameter) across 5 communication rounds to 3 participating client silos:

| Model Architecture | Trainable Parameters | Size per Model Transfer | Round Uplink + Downlink (3 Clients) | Total 5-Round Bandwidth |
|:---|:---|:---|:---|:---|
| **AML_GRU** | 48,641 | 194.56 KB | 1.167 MB | **5.837 MB** |
| **AML_LSTM** | 64,833 | 259.33 KB | 1.556 MB | **7.780 MB** |
| **AML_BiLSTM** | 162,433 | 649.73 KB | 3.898 MB | **19.492 MB** |

### Communication Findings:
- **Identical FedAvg and FedProx Bandwidth**: FedProx imposes zero network communication overhead compared to FedAvg. The proximal penalty is evaluated exclusively on the client device during local gradient computation.
- **Optimal Efficiency**: AML_LSTM provides the highest PR-AUC on HI-Small ($0.4760$) while requiring only $39.9\%$ of the communication bandwidth of BiLSTM ($7.78$ MB vs $19.49$ MB).

---

## 4. Validation-Based Model Selection & Ensemble Optimization

### A. Continuous Alpha Search Sweep
Validation predictions ($N = 1,015,565$ for HI-Small) were linearly combined across candidate weights $\alpha \in [0.0, 1.0]$:

```
Validation PR-AUC vs Alpha:
α = 0.00 (Pure Non-IID) : 0.198980
α = 0.20                : 0.207093
α = 0.40                : 0.215525
α = 0.60                : 0.224069
α = 0.80                : 0.232890
α = 1.00 (Pure IID)     : 0.242083  <-- OPTIMAL CHAMPION (α* = 1.00)
```

### B. Scientific Explanation for $\alpha^* = 1.00$
- The IID stream strictly dominates the Non-IID stream across all risk thresholds.
- Mixing in degraded Non-IID predictions (which have lower confidence and higher noise due to Client 3 volume bias) pulls down the true-positive ranking score.
- **Honest Conclusion**: The optimizer correctly selects $\alpha^* = 1.00$, pruning the Non-IID stream completely. The system documents that **the Standalone IID Champion alone is optimal**, avoiding misleading claims of ensemble synergy.

---

## 5. Distinction Between Validation and Test Splits

To prevent test-set data leakage, the following protocols were enforced:
- **Validation Split**: Used strictly for hyperparameter selection, decision threshold optimization ($\tau = 0.3378$), and ensemble blending weight search ($\alpha^* = 1.00$).
- **Held-Out Test Split**: Evaluated exactly once using the frozen validation parameters. No thresholds or model weights were tuned on test data.

