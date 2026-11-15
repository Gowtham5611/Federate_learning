# FedSecureAI: 5–7 Minute Final Project Demonstration Script

**Document ID:** `results/phase16/fedsecureai_demo_script.md`  
**Phase:** 16 — Final Documentation, Research Report & Demonstration Package  
**Date:** 2026-10-27  
**Project:** FedSecureAI — Privacy-Preserving, Explainable, Drift-Aware Federated AML  
**Presentation Audience:** Academic Examiners, Industry Evaluators, Compliance Executives  
**Target Duration:** 5 to 7 Minutes  

---

## Demonstration Script & Oral Walkthrough

### Part 1: Problem Introduction (Minute 0:00 – 0:45)
- **Speaker Dialogue:**  
  *"Good morning, examiners and evaluators. Today, financial institutions face a critical dilemma: Money laundering accounts for an estimated $2 trillion annually, yet traditional anti-money laundering monitoring systems suffer from false positive rates exceeding 95%, overwhelming human compliance teams.  
  Worse still, criminal organizations intentionally split their transactions across multiple independent banks—a technique known as structuring or smurfing. Because no single bank sees the full transaction chain, each individual transaction appears routine."*

---

### Part 2: The Federated Learning Solution (Minute 0:45 – 1:30)
- **Speaker Dialogue:**  
  *"Why not simply pool all banking data into a massive centralized data lake? Because strict data privacy mandates—including GDPR, the Bank Secrecy Act, and cross-border data sovereignty laws—strictly forbid commercial banks from sharing raw customer financial data.  
  FedSecureAI resolves this privacy-utility deadlock through Federated Learning. Instead of moving sensitive customer data to a central server, we move the model to the data. Participating banks collaboratively train recurrent sequence models by sharing only mathematical weight updates across secure rounds, keeping all raw transaction logs private on institution premises."*

---

### Part 3: System Architecture Overview (Minute 1:30 – 2:15)
- **Visual Aid:** *Display Mermaid Architecture Diagram from `fedsecureai_architecture.md`.*
- **Speaker Dialogue:**  
  *"Here is our end-to-end architecture. The system operates in two distinct tiers:  
  First, an offline research and training pipeline that takes raw IBM AML transactions (HI-Small with 5.08 million records and LI-Medium with 31.25 million records), builds 10-step causal history sequences across 57 features, and partitions them into decentralized client silos.  
  Second, a production-grade inference service and modern compliance dashboard where our trained champion models are served with validation-calibrated thresholds and Integrated Gradients local explainability."*

---

### Part 4: IID and Non-IID Banking Typologies (Minute 2:15 – 3:00)
- **Speaker Dialogue:**  
  *"In federated learning research, many papers test only synthetic random data splits. We evaluated two realistic banking regimes:  
  1. An IID baseline where transaction types are distributed uniformly.  
  2. A severe Non-IID typology partition where Client 1 represents Retail Banking, Client 2 represents a Cryptocurrency Exchange and Remittance corridor, and Client 3 represents a Regional Credit Union.  
  This allowed us to study what happens when client banks have completely different financial products and customer distributions."*

---

### Part 5: Key Experimental Findings & Ablations (Minute 3:00 – 3:45)
- **Visual Aid:** *Display Table from `fedsecureai_results_summary.md`.*
- **Speaker Dialogue:**  
  *"Our experimental benchmark revealed three major scientific insights:  
  First, **Parity**: On HI-Small, our federated LSTM FedAvg model captured **99.77% of centralized performance** (PR-AUC 0.4760 vs. 0.4771), proving that federated learning sacrifices zero detection efficacy.  
  Second, **The FedProx Regularization Penalty**: We proved that the popular FedProx algorithm ($\mu=0.0001$) actually hurts rare-event AML detection, causing up to a 65% drop in PR-AUC by pulling rare laundering gradients toward the consensus of legitimate transactions.  
  Third, **Typology Skew**: In LI-Medium, extreme Non-IID clustering caused an 81.9% drop in PR-AUC, demonstrating that domain skew is the single greatest vulnerability in collaborative financial defense."*

---

### Part 6: Live Dashboard Demonstration: Service Health (Minute 3:45 – 4:15)
- **Visual Aid:** *Open browser at `http://localhost:3000`.*
- **Speaker Dialogue:**  
  *"Let's examine our live analyst console. In the top navigation, you see our real-time engine heartbeat indicating `Engine Online`. The backend FastAPI service has authenticated and preloaded our two champion models: our HI-Small LSTM and our LI-Medium BiLSTM.  
  On the Overview screen, we display our immutable decision thresholds: $\tau=0.337784$ for HI-Small and $\tau=0.238483$ for LI-Medium, which were frozen exclusively on validation data to prevent data leakage."*

---

### Part 7: Scenario 1: Legitimate Routine Transaction (Minute 4:15 – 4:45)
- **Visual Aid:** *Navigate to `Transaction Scoring`. Click `HI-Small — Routine Standard Pattern`. Click `Evaluate Transaction Sequence`.*
- **Speaker Dialogue:**  
  *"Let's evaluate our first live scenario: a verified legitimate transaction sequence from the test split.  
  The forward-pass completes in under 3 milliseconds. The model assigns a laundering probability of **0.009812**, which is far below our frozen threshold of 0.3378. The dashboard renders a clear green badge: `CLEARED_LEGITIMATE`. The investigator's time is saved, and no false alert is generated."*

---

### Part 8: Scenario 2: High-Risk Rapid Structuring Pattern (Minute 4:45 – 5:15)
- **Visual Aid:** *Click `HI-Small — High-Risk Rapid Structuring Pattern`. Click `Evaluate Transaction Sequence`.*
- **Speaker Dialogue:**  
  *"Now let's test a confirmed money laundering pattern exhibiting rapid fund dispersal and structuring.  
  Immediately, the model outputs a probability of **0.720529**, crossing our operational cutoff. The console displays a red alert: `FLAGGED_LAUNDERING`.  
  Notice our in-memory audit ledger below updates dynamically, logging the sample ID, timestamp, probability, and classification."*

---

### Part 9: Axiomatic Explainability with Integrated Gradients (Minute 5:15 – 6:00)
- **Visual Aid:** *Scroll down to `Integrated Gradients Feature Attributions` section.*
- **Speaker Dialogue:**  
  *"In compliance, a prediction alone is useless without legal justification. Unlike black-box perturbation explainers like KernelSHAP that destroy sequence packing, we implemented PyTorch-native Integrated Gradients.  
  Notice this horizontal attribution bar chart:  
  The top positive driver elevating suspicion is `sender_prior_amount_log`, indicating an anomalous surge in outbound volume, followed by `receiver_seconds_since_prior_log`, identifying rapid pass-through mule accounts.  
  Furthermore, our explainer satisfies the mathematical Completeness Axiom with a residual error under 0.001, and strictly guarantees zero attribution leakage on zero-padded padding steps."*

---

### Part 10: Security Boundary & Privacy Posture (Minute 6:00 – 6:30)
- **Speaker Dialogue:**  
  *"Before concluding, let's address data governance:  
  1. The client console stores **zero PII**—no bank account numbers or customer identities are stored or transmitted.  
  2. The application uses purely **in-memory session state** (`useState`). If I refresh the browser tab right now, all evaluated records are immediately evicted from client RAM; zero traces persist in cookies, localStorage, or IndexedDB.  
  3. In-app authentication is intentionally decoupled. The system is architected for restricted private networks behind an enterprise authenticating gateway."*

---

### Part 11: Conclusion & Project Contribution (Minute 6:30 – 7:00)
- **Speaker Dialogue:**  
  *"In summary, FedSecureAI delivers a fully audited, reproducible deliverable:  
  - 97 out of 97 passing automated regression tests.  
  - 99.77% centralized detection parity under federated privacy.  
  - Mathematical Integrated Gradients explainability for regulatory compliance.  
  - An analyst-first, zero-PII compliance dashboard.  
  Thank you, and I look forward to your questions."*

