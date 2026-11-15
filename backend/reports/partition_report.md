# FedSecureAI — Federated Partitioning & Heterogeneity Audit Report

**Dataset:** IBM AML `HI-Small` Training Split  
**Total Training Transactions:** 3,047,216  
**Global Illicit Prevalence:** 0.0754% (2,299 positive cases)  
**Partition Seed:** `42`  
**Account Atomicity Status:** `PASSED (Zero Sequence Cross-Contamination)`  

---

## 1. IID Partitioning (Stratified Uniform Control)

| Client ID | Samples | % Share | Legitimate | Illicit | Prevalence | Imbalance | Wasserstein Dist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `client_1` | 1,015,738 | 33.3% | 1,014,972 | 766 | 0.0754% | 1325.0:1 | 0.0012 |
| `client_2` | 1,015,738 | 33.3% | 1,014,972 | 766 | 0.0754% | 1325.0:1 | 0.0012 |
| `client_3` | 1,015,740 | 33.3% | 1,014,973 | 767 | 0.0755% | 1323.3:1 | 0.0012 |

---

## 2. Non-IID Typology Partitioning (Domain & Label Skew)

| Client ID | Financial Typology Description | Samples | Share | Illicit | Prevalence | Imbalance | Amount Shift | Top Channels |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `client_1` | **Retail Giant (High-Volume Consumer)**<br>*Dominant Credit Card, POS, and Cash transactions; low illicit prevalence* | 1,422,100 | 46.7% | 512 | **0.0360%** | 2776.5:1 | $W_1 = 2.25$ | Credit Card (62.4%), Cash (25.1%) |
| `client_2` | **Offshore / Crypto & FX Exchange**<br>*High-value cross-border wires, cryptocurrency, and FX; disproportionate illicit risk* | 812,400 | 26.7% | 1,438 | **0.1770%** | 564.0:1 | $W_1 = 3.38$ | Wire (58.2%), Bitcoin (24.5%) |
| `client_3` | **Regional Credit Union**<br>*Domestic clearing, payroll ACH, cheques, and intra-bank reinvestments* | 812,716 | 26.7% | 349 | **0.0429%** | 2327.7:1 | $W_1 = 0.52$ | ACH (54.3%), Cheque (31.2%) |

---

## 3. Quantitative Heterogeneity Summary

- **Mean Non-IID Label Total Variation Distance (TVD):** `0.000578`
- **Mean Non-IID Amount Wasserstein Distance:** `2.0500`
- **Sequence Integrity Guarantee:** Partitioning is enforced strictly on `from_account` IDs. All sequential temporal histories for an originating account remain contained on the exact same client node.
- **Audit Verdict:** `PARTITIONING_VERIFIED_ZERO_LEAKAGE`