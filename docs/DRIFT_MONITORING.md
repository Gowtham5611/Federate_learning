# FedSecureAI — Statistical Drift Detection & Continuous Monitoring Architecture

## 1. Overview and Operational Motivation

Anti-Money Laundering detection models operate in an intrinsically non-stationary adversarial environment. Laundering syndicates rapidly adapt tactics (e.g., altering transaction volumes, shifting payment channels, cycling through new intermediary accounts, or modifying velocity patterns) to evade automated detection rules and ML models.

The **FedSecureAI Drift Monitoring Subsystem** establishes dual-tier surveillance:
1. **Input Feature Drift:** Detects shifts in covariate transaction distributions ($P(X)$) immediately on live unlabeled streaming batches.
2. **Performance / Concept Drift:** Detects decay in classification accuracy and calibration ($P(Y \mid X)$) when delayed ground-truth investigation labels become available.

---

## 2. Input Distribution Drift: Population Stability Index (PSI)

### Mathematical Formulation
The Population Stability Index measures the distance between a baseline reference distribution $P_{\text{ref}}$ (established during the training or validation split) and a monitored evaluation stream $P_{\text{cur}}$ across $K$ quantile-based bins:

$$\text{PSI} = \sum_{k=1}^{K} \left( P_{\text{cur}}(k) - P_{\text{ref}}(k) \right) \cdot \ln\left( \frac{P_{\text{cur}}(k) + \epsilon}{P_{\text{ref}}(k) + \epsilon} \right)$$

where:
- $K = 10$ quantile bins constructed from the reference distribution.
- $\epsilon = 10^{-4}$ provides numerical smoothing against zero-probability bins.

### Alert Policy Thresholds
Financial risk management guidelines establish standardized operational thresholds:
* **$\text{PSI} < 0.10$ (Nominal / No Drift):** The current stream distribution is stable and matches the training population. Model inference continues without intervention.
* **$0.10 \le \text{PSI} < 0.25$ (Moderate Drift):** Noticeable shift detected in transaction properties (e.g. seasonal holiday spending or new regional expansion). Flags warning in the analyst dashboard for surveillance.
* **$\text{PSI} \ge 0.25$ (Significant Drift):** Severe statistical divergence detected. Triggers high-priority operational alert recommending model recalibration or federated round aggregation review.

---

## 3. Two-Sample Kolmogorov-Smirnov (KS) Test

For continuous numerical attributes (such as transaction amounts, log amounts, and account velocities), the two-sample Kolmogorov-Smirnov test evaluates whether the current stream and reference baseline share an identical continuous distribution:

$$D = \sup_{x} |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$

The two-sided asymptotic $p$-value tests the null hypothesis $H_0: F_{\text{ref}} = F_{\text{cur}}$. Rejection at $\alpha = 0.01$ indicates localized continuous distribution distortion.

---

## 4. Concept & Performance Drift Tracking

When post-investigation feedback or SAR case closures supply delayed ground-truth labels for a monitoring window $W$, the system tracks degradation relative to the frozen research baseline:

| Metric | Frozen Baseline (HI-Small) | Warning Threshold | Critical Alert Threshold |
| :--- | :---: | :---: | :---: |
| **PR-AUC** | $0.5902$ | Drop $> 0.05$ | Drop $\ge 0.15$ |
| **F1-Score** | $0.5891$ | Drop $> 0.05$ | Drop $\ge 0.15$ |
| **Precision** | $0.5855$ | Drop $> 0.08$ | Drop $\ge 0.20$ |
| **Recall** | $0.5928$ | Drop $> 0.08$ | Drop $\ge 0.20$ |

---

## 5. Non-Interference Policy

1. **No Automated Retraining Without Validation:**
   A drift alert does NOT trigger automatic, unmonitored model weight retraining. Automated retraining under malicious drift creates vulnerabilities to data poisoning and model degradation.
2. **No Future Label Leaks:**
   Delayed labels are strictly partitioned to historical evaluation windows and are never back-propagated into past inference runs.

