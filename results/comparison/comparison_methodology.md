# Scientific Comparison Methodology: HI-Small vs. LI-Medium

## 1. Overview
Direct numerical comparison between machine learning benchmarks on different financial datasets requires strict methodology. A model achieving a lower PR-AUC on LI-Medium than on HI-Small does not indicate algorithmic regression, but reflects distinct statistical properties of the transaction manifolds.

## 2. Dimensional & Statistical Discrepancies

| Property | HI-Small Benchmark | LI-Medium Benchmark | Ratio / Disparity |
| :--- | :--- | :--- | :--- |
| **Total Raw Transactions** | 5,078,345 | 31,251,483 | 6.15x larger |
| **Total Laundering Rows** | 5,177 | 16,041 | 3.10x more cases |
| **Global Prevalence** | 0.1019% (1 in 981) | 0.0513% (1 in 1,948) | **1.99x rarer in LI-Medium** |
| **Train Split Transactions** | 3,047,216 | 18,752,217 | 6.15x larger |
| **Train Laundering Cases** | 2,299 | 8,138 | 3.54x more cases |
| **Test Split Transactions** | 1,015,564 | 6,249,112 | 6.15x larger |
| **Test Laundering Cases** | 1,797 | 3,783 | 2.10x more cases |
| **Test Split Prevalence** | 0.1769% (1 in 565) | 0.0605% (1 in 1,652) | **2.92x rarer in LI-Medium** |
| **Number of Distinct Banks** | ~10 banks | >356,000 banks | Massive topological dispersion |

## 3. Mathematical Foundations of Metrics Under Imbalance

### Precision-Recall AUC (Average Precision)
The expected baseline of PR-AUC for a random uninformative classifier is equal to the positive class prevalence $P(Y=1) = \pi$:
$$\text{Baseline } PR\text{-}AUC_{HI\text{-}Small} = 0.001769$$
$$\text{Baseline } PR\text{-}AUC_{LI\text{-}Medium} = 0.000605$$

When a model on LI-Medium achieves $PR\text{-}AUC = 0.045785$ (Config 9), it is achieving an enrichment factor of:
$$\text{Enrichment}_{LI} = \frac{0.045785}{0.000605} \approx 75.7\times \text{ over random guessing}$$
On HI-Small, a model with $PR\text{-}AUC = 0.475977$ achieves:
$$\text{Enrichment}_{HI} = \frac{0.475977}{0.001769} \approx 269\times \text{ over random guessing}$$

### False Alert Burden
In production Anti-Money Laundering operations, the number of alerts investigated per 10,000 transactions determines investigator workload.
- Both HI-Small and LI-Medium Top FedAvg models maintained alerts between **1.5 and 7.5 per 10,000 transactions**, which is within viable operational thresholds (<10 alerts/10k).

## 4. Evaluation Protocol Equivalency
- Both evaluations used **frozen classification thresholds** calibrated exclusively on the validation set, preventing test-set data snooping or leakage.
- Both evaluations used identical chronological test windows with 0% train/val/test temporal overlap.
