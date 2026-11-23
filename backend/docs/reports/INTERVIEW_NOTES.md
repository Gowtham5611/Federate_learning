# FedSecureAI — Technical Interview & Defense Notes

This guide provides structured interview talking points, mathematical justifications, and answers to technical questions about the FedSecureAI platform.

---

## 1. 30-Second Elevator Pitch

> "FedSecureAI is a privacy-preserving Anti-Money Laundering detection platform that enables multiple financial institutions to collaboratively train deep sequence models without pooling raw customer transaction data. Evaluated on the 5-million-row IBM AML benchmark with an extreme 0.18% laundering imbalance, our federated LSTM achieves 99.77% PR-AUC parity with peak centralized training (0.4760 vs. 0.4771) while reducing membership inference vulnerability to random guess (AUC 0.5041 vs. 0.5840). The system features an audited, zero-leakage evaluation engine, an interactive React compliance dashboard, and mathematically grounded Integrated Gradients explainability."

---

## 2. 2-Minute Architectural Walkthrough

> "The platform addresses the classic privacy-utility deadlock in AML: money launderers intentionally disperse transactions across multiple banks to evade detection, but banking privacy laws (GDPR, Bank Secrecy Act) prevent institutions from pooling raw customer records into a central database.
>
> We solved this through three pillars:
> 1. **Decentralized Modeling:** We implemented federated learning across 3 simulated bank clients using PyTorch. We benchmarked three recurrent architectures (GRU, LSTM, BiLSTM) across IID uniform and Non-IID typology-skewed partitions using both FedAvg and FedProx ($\mu = 0.0001$).
> 2. **Rigorous, Audited Evaluation:** In an extreme rare-class regime ($0.18\%$ prevalence), accuracy is meaningless ($99.82\%$ by predicting all zeros). We designed our evaluation around PR-AUC, operational alert-budget curves (Recall at 10 to 100 alerts per 10k transactions), bootstrap 95% confidence intervals, and post-hoc temperature calibration that reduced ECE by $72.8\%$.
> 3. **Production Serving & Explainability:** The system exposes a hardened FastAPI service with causal sequence reconstruction from historical transactions and Integrated Gradients satisfying the completeness axiom. An analyst dashboard provides 'what-if' sensitivity analysis, prior-shift logit adjustments, and transparent model provenance."

---

## 3. High-Yield Technical Q&A

### Q1: Why did you choose PR-AUC as the primary metric instead of ROC-AUC or Accuracy?
- **Answer:** In AML detection, the class imbalance is extreme ($1,797$ positives out of $1,015,564$ transactions, or $0.18\%$).
  - **Accuracy** is deceptive: a trivial classifier predicting all negatives achieves $99.82\%$ accuracy while catching zero criminals.
  - **ROC-AUC** is misleading: the False Positive Rate denominator is the massive negative majority ($1,013,767$ legitimate transactions). Even $10,000$ false alerts produce an FPR of only $0.0098$, yielding an artificially inflated ROC-AUC of $>0.95$.
  - **PR-AUC** evaluates Precision ($\frac{TP}{TP+FP}$) against Recall ($\frac{TP}{TP+FN}$). It isolates the quality of positive alerts without being inflated by the sheer volume of true negatives. Random guessing yields $\text{PR-AUC} = \pi = 0.00177$, meaning our model's score of $0.4760$ represents a **$+269\%$ relative lift** over chance.

---

### Q2: In your ensemble experiments, the optimal weight was $\alpha^* = 1.00$. Why didn't combining the IID and Non-IID models improve performance?
- **Answer:** We conducted an exhaustive grid search over validation predictions and verified it with an out-of-sample Phase 7b Stacking Meta-Learner (Logistic Regression).
  - The IID champion learned a generalizable global representation across all banking typologies ($\text{PR-AUC} = 0.4760$).
  - The Non-IID model was trained on skewed, non-overlapping client partitions. When the meta-learner was fitted on both prediction streams, it assigned a large positive weight to the IID model ($+96.05$) and a large *negative weight* to the Non-IID model ($-35.38$).
  - In the presence of the superior IID predictions, the Non-IID model's skewed errors acted as structured noise. Convex blending ($\alpha < 1.0$) or linear stacking directly degraded test PR-AUC down to $0.4286$.
  - **Engineering honesty:** Rather than forcing an artificial 50/50 blend to claim 'we built an ensemble', we disclosed that $\alpha^* = 1.00$ is the scientifically optimal selection on this benchmark.

---

### Q3: Why did FedProx ($\mu = 0.0001$) underperform FedAvg in your experiments?
- **Answer:** In our empirical benchmarks under an extreme class-imbalanced setting ($0.18\%$ laundering prevalence), FedProx ($\mu = 0.0001$) scored lower than vanilla FedAvg (PR-AUC $0.2697$ vs $0.4107$).
  - While one theoretical hypothesis is that proximal regularization $\frac{\mu}{2} \|w - w_{\text{global}}\|^2$ may act as an implicit shrinkage prior that suppresses sparse minority-class parameter updates, this causal mechanism remains untested.
  - We report this transparently as an empirical benchmark observation rather than asserting an unverified causal claim.

---

### Q4: When a user uploads a 500-row CSV of single transactions, why does the model output low probabilities ($\sim 0.01$) when the decision threshold is $0.3378$?
- **Answer:** This is a fundamental property of recurrent sequence models:
  - The LSTM expects an input tensor of shape `(batch, 10, 57)` representing a sequence of up to 10 sequential transactions for an account.
  - When an uploaded CSV contains unrelated rows or single-transaction accounts without preceding history, $9$ out of the $10$ time steps are filled with zero-padding.
  - Because money laundering patterns (e.g., smurfing, structuring) require observing rapid sequential deposits followed by immediate transfers, a single isolated transaction has insufficient temporal context to trigger high risk, bounding probabilities near the background prior ($\sim 0.01$).
  - When provided with contiguous chronological history from the test set (`test_slice_500.csv`), the serving sequence builder matches the pre-computed test set predictions to within $5.01 \times 10^{-7}$.
  - We added automated `sequence_diagnostics` in the backend that detect this padding ratio and explicitly advise compliance officers when a tabular fallback model is recommended.

---

### Q5: How did you guarantee zero data leakage between splits?
- **Answer:** We enforced strict structural boundaries:
  1. **Chronological Splitting:** The dataset was partitioned sequentially by timestamp into Train (60%), Validation (20%), and Test (20%). Never random shuffle, which would leak future transaction patterns into past evaluations.
  2. **Preprocessing Isolation:** Feature scalers, log normalizers, and categorical mappings were fitted strictly on the Train split, then applied to Validation and Test.
  3. **Threshold & Hyperparameter Freezing:** Optimal decision thresholds ($\theta^* = 0.337784$) and calibration temperatures ($T^* = 0.8340$) were optimized exclusively on the Validation split and permanently frozen before scoring the Test split.
  4. **Read-Only Model Integrity:** Baseline weights were SHA-256 hashed and verified with automated test suites before and after all serving and evaluation tasks.

---

### Q6: Did Federated Learning actually match centralized training?
- **Answer:** Yes, with precision regarding the denominator:
  - **Vs. Peak Centralized Baseline:** The best centralized model was a Centralized LSTM ($\text{PR-AUC} = 0.477052$). Our federated champion (`hi_small_lstm_fedavg_iid_seed42`) achieved $\text{PR-AUC} = 0.475977$, retaining **$99.77\%$ of peak centralized performance** (delta of only $-0.0011$).
  - **Vs. Mean Centralized Baselines:** Across all three centralized architectures (GRU $0.3951$, LSTM $0.4771$, BiLSTM $0.3635$), the average centralized PR-AUC was $0.411874$. Compared to this average, our federated champion represents **$115.56\%$ retention (+15.56% lift)**.
  - Both comparisons are now explicitly documented in our codebase and UI to eliminate ambiguity.

---

### Q7: How does your system handle Prior Shift in uploaded transaction batches?
- **Answer:** When compliance teams upload targeted investigation samples, the proportion of laundering cases may be artificially elevated (e.g., $30\%$ positives instead of the natural $0.18\%$).
  - If unadjusted, a model trained on a $0.00177$ prior will underestimate risk because its prior expectations do not match the sample distribution.
  - We implement the classical **Logit Shift Adjustment** (Saerens et al.):
    $$\text{logit}(P_{\text{adj}}) = \text{logit}(P_{\text{raw}}) + \ln\left(\frac{\pi_{\text{target}}}{1-\pi_{\text{target}}}\right) - \ln\left(\frac{\pi_{\text{train}}}{1-\pi_{\text{train}}}\right)$$
  - The API returns both raw predictions and prior-adjusted probabilities, allowing compliance officers to inspect both unadjusted model outputs and context-adjusted risk scores.

---

### Q8: How did you demonstrate that Federated Learning provides real privacy benefits?
- **Answer:** We implemented a shadow-free loss-threshold Membership Inference Attack (Yeom et al., 2018):
  - In a centralized system where raw data is pooled, training sample over-fitting creates an identifiable loss disparity between train and test transactions ($\text{MIA AUC} = 0.5840$). An adversary can infer whether a customer's transaction was in the training set with noticeable advantage over chance.
  - Under FedSecureAI's federated aggregation, the adversary's capability dropped to $\text{MIA AUC} = 0.5041$ (virtually indistinguishable from a coin flip of $0.5000$). This proves an empirical privacy advantage of **$+0.0799$** without sacrificing predictive power.

---

### Q9: How does your explainability framework work?
- **Answer:** We implemented PyTorch-native Integrated Gradients:
  - It integrates the path integral of gradients from a baseline input $x'$ to the transaction input $x$:
    $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
  - We chose a zero-vector baseline representing an inactive account transaction.
  - We verified the **Completeness Axiom**: the sum of attributions equals the difference between the model output at $x$ and the model output at the baseline $x'$ within $<0.06\%$ residual error.
  - Furthermore, we mask out padded time steps so that padding artifacts never leak into regulatory Suspicious Activity Report (SAR) explanations.

---

### Q10: How would you deploy this across actual banking institutions?
- **Answer:**
  1. **Client Node Container:** A containerized agent deployed within each bank's Virtual Private Cloud (VPC) with read-only access to local transaction tables.
  2. **Federated Orchestrator:** An aggregator server running in a secure enclave (or over gRPC with mTLS) coordinating the rounds, broadcasting model weights, and collecting parameter tensors.
  3. **Differential Privacy / Secure Aggregation:** While FedAvg already resists loss-based MIA, we would add SecAgg (Diffie-Hellman secret sharing) and gradient clipping with $( \epsilon, \delta )$-differential privacy to defend against reconstruction attacks in high-dimension layers.
  4. **Serving Infrastructure:** The FastAPI inference service deploys on Kubernetes with horizontal autoscaling, serving live inference via batch micro-queues.

---

### Q11: What happened when a user uploaded a 5,000-row file that produced 0 alerts, and how did you resolve it?
- **Answer:** The issue occurred because of three simultaneous data distribution gaps:
  1. **Zero Account History:** The file contained 5,000 unique senders (100% single-transaction rows). Deep recurrent models require sequential context ($T \ge 3$ transactions per account). With zero history, 9 out of 10 sequence steps were zero-padded, bounding probabilities near the background prior (~0.01) against the decision threshold ($\theta^* = 0.3378$).
  2. **Prevalence Mismatch:** The file had 30.4% laundering prevalence compared to the training prior (0.18%).
  3. **Broken Reporting:** In the UI, missed positives were displayed with a green "CLEARED" badge, masking False Negatives.
  - **Our Fix:**
    - Added an automated `ValidityBanner` with plain-language diagnostic flags (`LOW_HISTORY`, `PREVALENCE_MISMATCH`, `NOT_CONTIGUOUS`, `SMALL_SAMPLE`).
    - Tagged rows lacking $\ge 3$ prior transactions as `LOW_CONFIDENCE_NO_HISTORY` and suppressed Integrated Gradients.
    - Badged actual suspicious missed rows prominently in red as **False Negative (Missed)**, NEVER green "CLEARED".
    - Provided an explicit opt-in tabular LightGBM fallback model (`fallback_model="tabular"`).
    - Persisted downloadable predictions to disk surviving backend reloads.

---

### Q12: How does the LightGBM fallback compare to the recurrent champion, and why didn't ensembling improve performance?
- **Answer:**
  - Standalone LightGBM on 23 tabular features achieves $\text{PR-AUC} = 0.2646$ on held-out test data.
  - Standalone recurrent LSTM champion achieves $\text{PR-AUC} = 0.4760$ (a $+79.9\%$ performance advantage).
  - When we built a validation-tuned convex blend ($\alpha^* = 0.59$), the ensemble achieved $\text{PR-AUC} = 0.4485$ (a $-0.0275$ drop vs. standalone LSTM).
  - Stacking via Logistic Regression on validation logits similarly yielded $\text{PR-AUC} = 0.4350$.
  - **Conclusion:** Recurrent inductive bias is strictly superior for capturing multi-step account velocity and layering in AML. Tabular models serve as a vital cold-start fallback when sequence history is absent, but combining them with the recurrent champion dilutes its temporal predictive signal.

---

### Q13: What is the authoritative decision threshold and where did 0.35 in the PPT come from?
- **Answer:**
  - The canonical decision threshold for HI-Small is **$\theta^* = 0.337784$**, which was empirically determined on validation data to maximize validation $F_1$.
  - Recomputing metrics on held-out test data confirms:
    - At $\theta^* = 0.337784$: $F_1 = 0.5359$, Precision $= 0.8183$, Recall $= 0.3984$, Alerts/10k $= 8.62$.
    - At $0.350000$: $F_1 = 0.5329$, Precision $= 0.8304$, Recall $= 0.3923$, Alerts/10k $= 8.36$.
  - The PPT slide previously presented $F_1 = 0.5359$, Precision $= 0.8183$, and Recall $= 0.3984$ while labeling the column as "0.35" as a visual rounding shorthand. We saved the exact comparison to `results/ppt_metrics.csv` to ensure complete mathematical transparency.

