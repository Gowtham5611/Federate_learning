# FedSecureAI: Research Methodology & Mathematical Foundations

## 1. Problem Formulation: Federated Anti-Money Laundering (AML)

Money laundering detection poses severe machine learning challenges characterized by extreme class imbalance ($< 0.2\%$ illicit transactions), non-stationary transaction patterns, and strict banking privacy regulations (GDPR, BSA, FATF). Traditional centralized model training requires pooling sensitive financial records across competing institutions into a central repository, introducing severe regulatory liabilities and single-point-of-failure vulnerabilities.

FedSecureAI frames multi-institutional transaction monitoring as a privacy-preserving distributed optimization problem. Let there be $K$ participating financial institutions (clients). Each client $k \in \{1, \dots, K\}$ holds a private local dataset:

$$\mathcal{D}_k = \left\{ (X_i^{(k)}, y_i^{(k)}) \right\}_{i=1}^{n_k}$$

where:
- $X_i^{(k)} \in \mathbb{R}^{T_i \times D}$ represents a variable-length transaction sequence belonging to an originating account, with sequence length $T_i \le T_{\max}$ and feature dimensionality $D = 13$.
- $y_i^{(k)} \in \{0, 1\}$ denotes the binary ground-truth laundering indicator ($y=1$ signifies an illicit or money-laundering transaction).
- $n_k = |\mathcal{D}_k|$ denotes the number of account sequence samples at client $k$.
- $N = \sum_{k=1}^K n_k$ denotes the total sample population across the federation.

The global optimization objective is to learn a parameter vector $w \in \mathbb{R}^d$ that minimizes the global population risk without centralizing raw transaction sequences:

$$\min_{w \in \mathbb{R}^d} F(w) \triangleq \sum_{k=1}^K \frac{n_k}{N} F_k(w)$$

where the local empirical risk at institution $k$ is defined by:

$$F_k(w) \triangleq \frac{1}{n_k} \sum_{i=1}^{n_k} \ell\left(f(X_i^{(k)}; w), y_i^{(k)}\right)$$

Here, $f(\cdot; w): \mathbb{R}^{T \times D} \to [0, 1]$ parameterizes the neural sequence classifier, and $\ell(\cdot, \cdot)$ is a differentiable loss function adapted to extreme label skew.

---

## 2. Statistical Heterogeneity & Typology-Driven Non-IID Partitioning

In real-world financial networks, data is non-identically distributed across banks due to distinct business domains, geographic footprints, and customer demographics.

### 2.1 Account-Level Atomicity (Zero Temporal Leakage)
To ensure sequence integrity and prevent data leakage across client boundaries, partitioning strictly preserves **account atomicity**:

$$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset \quad \forall i \ne j, \quad i, j \in \{1, \dots, K\}$$

where $\mathcal{A}_k$ is the set of distinct account identifiers assigned to institution $k$. All chronological transactions associated with account $a \in \mathcal{A}_k$ reside exclusively on client $k$.

### 2.2 Client Specialization Typologies
The IBM AML dataset (`HI-Small` and `LI-Medium`) is partitioned across three specialized institutional profiles:

1. **Client 1: Retail & Commercial Banking**
   - Dominant transaction modalities: Debit/credit cards, cash deposits, merchant POS, automated teller machines.
   - Low laundering base rate ($\approx 0.036\%$). High transaction volume with smaller individual payment amounts.
2. **Client 2: Offshore & Cross-Border Wealth Management**
   - Dominant modalities: High-value international cross-border wires, multi-currency foreign exchanges, rapid pass-through shell accounts.
   - High laundering base rate ($\approx 0.177\%$). Significant amount variance and structuring anomalies.
3. **Client 3: Regional & Cooperative Credit Union**
   - Dominant modalities: Domestic ACH transfers, personal loans, checks, domestic peer-to-peer transfers.
   - Moderate laundering base rate ($\approx 0.082\%$). Regular, clustered domestic transaction timing.

### 2.3 Heterogeneity Quantification Metrics
To formally measure statistical divergence between client data distributions $P_k$ and the global empirical distribution $P$, FedSecureAI computes:

#### Total Variation Distance ($TVD$) on Label Distribution:
$$TVD(P_k, P) \triangleq \frac{1}{2} \sum_{y \in \{0, 1\}} |P_k(Y = y) - P(Y = y)|$$

#### 1-Wasserstein Distance ($W_1$) on Continuous Transaction Attributes:
For continuous feature marginals (e.g., normalized transaction amount $A$):

$$W_1(P_k(A), P(A)) \triangleq \inf_{\gamma \in \Pi(P_k, P)} \mathbb{E}_{(u, v) \sim \gamma}[|u - v|] = \int_{-\infty}^{\infty} |F_k(a) - F_{\text{global}}(a)| \, da$$

where $F_k$ and $F_{\text{global}}$ are the empirical cumulative distribution functions.

---

## 3. Recurrent Sequence Architectures with Packed Sequences

Transaction streams exhibit dynamic temporal dependencies. FedSecureAI implements three recurrent architectures specifically handling variable sequence lengths via PyTorch `pack_padded_sequence` and `pad_packed_sequence`.

### 3.1 Long Short-Term Memory (LSTM)
For transaction sequence $(x_1, \dots, x_T)$:

$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

### 3.2 Gated Recurrent Unit (GRU)
GRU mitigates memory footprint and communication payload:

$$\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}$$

### 3.3 Bidirectional LSTM (BiLSTM)
BiLSTM processes transaction histories forward and backward, capturing retrospective structuring behavior:

$$h_t = \left[ \vec{h}_t \,\|\, \overleftarrow{h}_t \right] \in \mathbb{R}^{2 \cdot H}$$

### 3.4 Length Masking & Gradient Integrity
Padded zero-tokens at the end of sequences are excluded from computation using PyTorch packed sequences. This ensures:
1. Zero values do not perturb recurrent hidden states.
2. Gradients $\frac{\partial \ell}{\partial x_t}$ for padded positions $t > T_i$ are identically zero, preventing baseline corruption during backpropagation.

---

## 4. Federated Aggregation Algorithms

### 4.1 Federated Averaging (FedAvg)
At round $t$, server broadcasts $w^{(t)}$. Each client performs $E$ local epochs using local SGD:

$$w_k^{(t)} \leftarrow w^{(t)} - \eta \sum_{e=1}^E \nabla F_k(w_k^{(t, e)})$$

The server aggregates client weights via sample-weighted averaging:

$$w^{(t+1)} = \sum_{k=1}^K \frac{n_k}{N} w_k^{(t)}$$

Under severe Non-IID drift, FedAvg suffers from "client drift" where local optima diverge from the global stationary point.

### 4.2 Federated Proximal (FedProx)
FedProx adds a proximal regularization term to the local objective to penalize deviation from the global model:

$$\min_{w} F_k(w) + \frac{\mu}{2} \|w - w^{(t)}\|_2^2$$

This bounds local updates:

$$\|w_k^{(t)} - w^{(t)}\|_2 \le \frac{1}{\mu} \|\nabla F_k(w_k^{(t)})\|$$

mitigating divergence in heterogeneous financial networks.

### 4.3 Adaptive Federated Aggregation (FedAdaptive)
FedSecureAI introduces `FedAdaptive`, a drift-resilient aggregation scheme that dynamically reweights client updates based on their parameter drift relative to the federation mean.

Let $d_k^{(t)}$ denote the Frobenius parameter divergence of client $k$:

$$d_k^{(t)} \triangleq \|w_k^{(t)} - w^{(t)}\|_2 = \sqrt{\sum_{l} \|w_{k, l}^{(t)} - w_l^{(t)}\|_F^2}$$

Let $\bar{d}^{(t)}$ be the unweighted mean drift across participating clients:

$$\bar{d}^{(t)} \triangleq \frac{1}{K} \sum_{k=1}^K d_k^{(t)}$$

The adaptive aggregation weight $p_k^{(t)}$ is computed as:

$$s_k^{(t)} = \left(\frac{n_k}{N}\right) \cdot \exp\left( -\beta \frac{d_k^{(t)}}{\bar{d}^{(t)} + \epsilon} \right)$$

$$p_k^{(t)} = \frac{s_k^{(t)}}{\sum_{j=1}^K s_j^{(t)}}$$

where:
- $\beta \ge 0$ is the drift penalty sensitivity parameter (temperature).
- $\epsilon > 0$ prevents division by zero ($10^{-8}$).
- When $\beta = 0$, $s_k^{(t)} = \frac{n_k}{N}$, recovering standard sample-proportional FedAvg.
- When $\beta > 0$, clients whose local updates experience excessive drift ($d_k^{(t)} \gg \bar{d}^{(t)}$) are exponentially attenuated, preventing corrupted local updates from destabilizing global convergence.

---

## 5. Loss Formulations for Extreme Class Imbalance

With illicit base rates below $0.2\%$, standard cross-entropy suffers from majority-class gradient saturation.

### 5.1 Standard Binary Cross-Entropy (BCE)
$$\ell_{\text{BCE}}(p, y) = - [y \log p + (1 - y) \log(1 - p)]$$

### 5.2 Class-Weighted BCE
Let $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 714$ for HI-Small:

$$\ell_{\text{weighted}}(p, y) = - [w_{\text{pos}} \cdot y \log p + (1 - y) \log(1 - p)]$$

### 5.3 Binary Focal Loss with Logits
Addressing hard false negatives via modulating factor $(1 - p_t)^\gamma$:

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

$$\ell_{\text{focal}}(p, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$ dynamically downweights easy negative transactions ($p_t \to 1$), focusing gradient updates on hard borderline transactions.

---

## 6. Post-Hoc Probability Calibration

Because deep sequence classifiers trained on imbalanced datasets often produce miscalibrated, overconfident probabilities, FedSecureAI integrates post-hoc calibration fitted strictly on validation splits.

### 6.1 Temperature Scaling
For uncalibrated logit $z \in \mathbb{R}$:

$$\hat{p} = \sigma\left(\frac{z}{T}\right)$$

where temperature $T > 0$ is optimized via Nelder-Mead / L-BFGS to minimize negative log-likelihood (NLL) on the validation partition:

$$\min_{T > 0} -\sum_{i \in \mathcal{D}_{\text{val}}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log\left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$

*Theoretical Guarantee:* Because $\frac{z}{T}$ is a strictly monotonic transformation of $z$ for any fixed $T > 0$, Temperature Scaling preserves exact ranking orders: ROC-AUC and PR-AUC are strictly invariant.

### 6.2 Isotonic Regression
Non-parametric piece-wise constant monotonic mapping:

$$\min_{m} \sum_{i \in \mathcal{D}_{\text{val}}} (y_i - m(p_i))^2 \quad \text{s.t. } m(p_i) \le m(p_j) \text{ whenever } p_i \le p_j$$

---

## 7. Explainability via Sequence Integrated Gradients

Compliance with banking regulations requires transparent rationale for flagged transactions. FedSecureAI implements native PyTorch sequence Integrated Gradients adhering to Sundararajan et al. (2017).

### 7.1 Path-Integral Formulation
For an input sequence $x \in \mathbb{R}^{T \times D}$ and zero baseline $x' \equiv 0$:

$$\text{IG}_i(x) \triangleq (x_i - x_i') \times \int_{0}^1 \frac{\partial f(x' + \alpha (x - x'))}{\partial x_i} \, d\alpha$$

### 7.2 Numerical Approximation via Gauss-Legendre Quadrature
The integral is evaluated numerically across $M$ interpolation steps ($M=50$):

$$\text{IG}_i^M(x) = (x_i - x_i') \times \sum_{m=1}^M \frac{\partial f\left(x' + \frac{m}{M}(x - x')\right)}{\partial x_i} \times \frac{1}{M}$$

### 7.3 Formal Axiom Verification
1. **Completeness:** $\sum_{t=1}^T \sum_{d=1}^D \text{IG}_{t, d}(x) = f(x) - f(x')$.
2. **Implementation Invariance:** Attribution depends strictly on the input-output mapping $f$, not the specific structural implementation.
3. **Dummy Property:** If feature $d$ does not affect the prediction for any input, $\text{IG}_{t, d}(x) = 0$.

---

## 8. Operational AML Evaluation Framework

Evaluating AML systems via accuracy or standard ROC-AUC yields misleading conclusions due to extreme class imbalance ($99.86\%$ negative samples). FedSecureAI deploys an operational evaluation framework:

1. **Precision-Recall Area Under Curve (PR-AUC):** The primary benchmark metric quantifying precision across the full range of detection recall.
2. **Recall@FPR=0.1% and Recall@FPR=1.0%:** Measures the percentage of true laundering rings captured when regulatory false-alarm budgets are strictly constrained to 1 in 1,000 or 1 in 100 legitimate transactions.
3. **Alerts per 10,000 Transactions:** Operational workload metric:
   $$\text{Alert Rate}_{10k} \triangleq \frac{\text{TP} + \text{FP}}{N_{\text{total}}} \times 10,000$$
4. **Expected Calibration Error (ECE):**
   $$\text{ECE} \triangleq \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   where $B_m$ represents probability bins partitioned over $[0, 1]$.

