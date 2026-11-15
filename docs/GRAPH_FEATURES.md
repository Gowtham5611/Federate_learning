# FedSecureAI — Graph-Based Anti-Money Laundering Feature Architecture

## 1. Overview and Problem Statement

Money laundering fundamentally operates across interconnected networks of financial entities rather than isolated transactions. Criminal typologies—such as **structuring (smurfing)**, **layering via rapid fund movement**, and **round-tripping (circular flows)**—leave distinct topological and temporal footprints across the transaction graph $G = (V, E)$.

While recurrent sequence models (LSTM, BiLSTM, GRU) capture temporal dependencies for individual accounts, they lack relational awareness of counterparty networks. This module provides a strictly causal, zero-leakage graph feature extraction engine designed for high-throughput AML transaction streams.

---

## 2. Temporal Leakage Prevention in Graph Feature Extraction

### The Graph Leakage Trap in Financial ML
In academic literature, graph neural networks (GNNs) or graph metrics frequently suffer from catastrophic future information leakage:
1. Constructing a static graph over the entire dataset (including validation and test partitions).
2. Using future edges (transactions occurring after time $t$) to compute node degrees, PageRank, or neighborhood embeddings for a transaction at time $t$.
3. Leaking future transaction frequency and counterparty linkages into past predictions, producing inflated, non-deployable benchmark metrics.

### FedSecureAI Causal Graph Guarantee
In `FedSecureAI`, the `TimeAwareAmlGraph` enforces strict causal ordering:
$$\forall e = (u, v, w, \tau) \in E, \quad \tau \le t$$

For any transaction $T_i$ occurring at timestamp $t_i$:
- The feature extractor queries only historical edges recorded at $\tau \le t_i$.
- Edges occurring at $\tau > t_i$ are mathematically invisible to the feature extractor.
- Immediately after extracting causal features for $T_i$, the edge is appended to the historical graph state for subsequent transactions.

```
Incoming Stream (Sorted by Timestamp):
Tx 1 (09:00:00) ──> Extract Features (Graph empty)   ──> Add Tx 1 to Graph
Tx 2 (09:15:00) ──> Extract Features (Uses Tx 1)      ──> Add Tx 2 to Graph
Tx 3 (09:30:00) ──> Extract Features (Uses Tx 1, 2)   ──> Add Tx 3 to Graph
```

---

## 3. Mathematical Formulations of Graph Features

### 3.1 Account Activity Features
For account $u$ at time $t$:
* **In-Degree ($d_{in}(u, t)$):** Total incoming transaction edges directed to $u$ at $\tau \le t$.
* **Out-Degree ($d_{out}(u, t)$):** Total outgoing transaction edges initiated by $u$ at $\tau \le t$.
* **Unique Incoming Counterparties ($|N_{in}(u, t)|$):** Number of distinct senders that have transferred funds to $u$.
* **Unique Outgoing Counterparties ($|N_{out}(u, t)|$):** Number of distinct beneficiaries that have received funds from $u$.
* **Net Flow ($NF(u, t)$):**
  $$NF(u, t) = \sum_{e \in E_{in}(u, t)} \text{amount}(e) - \sum_{e \in E_{out}(u, t)} \text{amount}(e)$$
* **Outbound Average & Maximum Amount:**
  $$\mu_{out}(u, t) = \frac{1}{|E_{out}(u, t)|} \sum_{e \in E_{out}} \text{amount}(e), \quad M_{out}(u, t) = \max_{e \in E_{out}} \text{amount}(e)$$

### 3.2 Multi-Scale Temporal Velocity
Money laundering layering involves rapid burst transactions across short time horizons. Features are calculated across three rolling lookback windows $\Delta t \in \{5\text{ min}, 1\text{ hour}, 24\text{ hours}\}$:
* **Windowed Transaction Count:** $C_{\Delta t}(u, t) = |\{e \in E_{out}(u) \mid t - \Delta t \le \tau(e) \le t\}|$
* **Windowed Amount Sum:** $S_{\Delta t}(u, t) = \sum_{e \in E_{out}(u), \tau(e) \ge t - \Delta t} \text{amount}(e)$
* **Distinct Windowed Counterparties:** $CP_{\Delta t}(u, t) = |\{v \mid (u, v) \in E, \tau(e) \ge t - \Delta t\}|$

### 3.3 Network Typology & Structural Anomaly Indicators
* **Immediate Round-Trip Indicator ($\mathbb{I}_{\text{round\_trip}}(u, v, t)$):**
  Flags reciprocal flows where recipient $v$ has previously transferred funds back to sender $u$ within the past 24 hours:
  $$\mathbb{I}_{\text{round\_trip}} = \begin{cases} 1 & \text{if } \exists e = (v, u) \in E \text{ with } t - 24\text{h} \le \tau(e) \le t \\ 0 & \text{otherwise} \end{cases}$$
* **3-Hop Circular Flow Indicator ($\mathbb{I}_{\text{circular\_3hop}}(u, v, t)$):**
  Detects circular laundering loops $u \to v \to w \to u$ occurring within a 48-hour window.
* **Rapid Pass-Through Ratio:**
  Measures the ratio of current outbound amount to total funds received in the past 1 hour. A ratio close to $1.0$ indicates an intermediary pass-through or transit account.
* **Herfindahl-Hirschman Index of Counterparties ($HHI(u, t)$):**
  Measures counterparty concentration:
  $$HHI(u, t) = \sum_{v \in N_{out}(u, t)} \left( \frac{\text{count}(u \to v)}{\sum \text{count}} \right)^2$$
  A low $HHI$ indicates widespread fan-out smurfing; a high $HHI$ indicates concentrated bilateral flow.

---

## 4. Verification and Empirical Tests

The graph feature architecture is validated under automated tests in `tests/test_master_research_upgrade.py`:
- `test_time_aware_graph_zero_future_leakage`: Proves that a future reciprocal edge added at $t + 1\text{h}$ does NOT trigger a round-trip flag when evaluating a transaction at $t$.
- `test_circular_flow_3hop_detection`: Proves that a 3-hop loop ($A \to B \to C \to A$) is detected only after the closing edge $C \to A$ is executed.

