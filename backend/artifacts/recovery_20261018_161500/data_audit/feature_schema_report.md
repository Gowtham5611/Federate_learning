# FedSecureAI — Feature Schema & Data Representation Report

**Audit Target**: IBM AML Processed Feature Pipelines (`HI-Small`, `LI-Medium`, `Combined`)  
**Audit Location**: `artifacts/fedavg_fedprox_full_comparison/data_audit/`  
**Execution Timestamp**: 2026-10-18  

---

## 1. Feature Dimension and Column Order

All three dataset variants (`HI-Small`, `LI-Medium`, `Combined`) adhere to an identical **27-column processed schema** on disk:

| Index | Column Name | Data Type | Role | Processing / Transformation |
| :---: | :--- | :---: | :---: | :--- |
| 0 | `timestamp` | `string` / `datetime64` | Temporal index | Verified strictly monotonically increasing |
| 1 | `from_account` | `string` | Causal ID | Used exclusively to construct backward sequence links; never fed to model |
| 2 | `to_account` | `string` | Receiver ID | Historical accumulation only; never fed directly to model |
| 3 | `amount_paid_log` | `float32` | Numerical feature | Natural log of transaction amount $\ln(1 + \text{amount})$ |
| 4 | `amount_received_log` | `float32` | Numerical feature | Natural log of received amount $\ln(1 + \text{received})$ |
| 5 | `amount_mismatch` | `float32` | Boolean indicator | $1.0$ if paid amount differs from received amount, else $0.0$ |
| 6 | `same_currency` | `float32` | Boolean indicator | $1.0$ if sending currency equals receiving currency, else $0.0$ |
| 7 | `same_bank` | `float32` | Boolean indicator | $1.0$ if sender bank equals receiver bank, else $0.0$ |
| 8 | `self_transfer` | `float32` | Boolean indicator | $1.0$ if sender account equals receiver account, else $0.0$ |
| 9 | `payment_format_code` | `int32` | Nominal code | Integer category code (Cheque, Credit Card, Wire, ACH, etc.) |
| 10 | `payment_currency_code` | `int32` | Nominal code | Integer currency code |
| 11 | `receiving_currency_code`| `int32` | Nominal code | Integer currency code |
| 12 | `minute_of_day_sin` | `float32` | Cyclic temporal | $\sin(2\pi \cdot \text{minute} / 1440)$ |
| 13 | `minute_of_day_cos` | `float32` | Cyclic temporal | $\cos(2\pi \cdot \text{minute} / 1440)$ |
| 14 | `day_of_week_sin` | `float32` | Cyclic temporal | $\sin(2\pi \cdot \text{day} / 7)$ |
| 15 | `day_of_week_cos` | `float32` | Cyclic temporal | $\cos(2\pi \cdot \text{day} / 7)$ |
| 16 | `is_weekend` | `float32` | Boolean indicator | $1.0$ if Saturday/Sunday, else $0.0$ |
| 17 | `is_night` | `float32` | Boolean indicator | $1.0$ between 22:00 and 06:00, else $0.0$ |
| 18 | `sender_prior_count_log` | `float32` | Causal history | $\ln(1 + \text{prior transactions by sender})$ |
| 19 | `sender_prior_amount_log` | `float32` | Causal history | $\ln(1 + \text{sum of prior amounts})$ |
| 20 | `sender_seconds_since_prior_log` | `float32` | Causal temporal | $\ln(1 + \Delta t_{\text{sender}})$ |
| 21 | `sender_has_prior` | `float32` | Boolean indicator | $1.0$ if prior transaction exists, else $0.0$ |
| 22 | `receiver_prior_count_log`| `float32` | Causal history | $\ln(1 + \text{prior transactions by receiver})$ |
| 23 | `receiver_prior_amount_log`| `float32` | Causal history | $\ln(1 + \text{sum of prior amounts to receiver})$ |
| 24 | `receiver_seconds_since_prior_log`| `float32` | Causal temporal | $\ln(1 + \Delta t_{\text{receiver}})$ |
| 25 | `receiver_has_prior` | `float32` | Boolean indicator | $1.0$ if prior transaction exists, else $0.0$ |
| 26 | `is_laundering` | `int8` | Target label | Binary classification label ($1 = \text{Laundering}, 0 = \text{Legitimate}$) |

---

## 2. Expanded Feature Matrix (57 Dimensions)

During dataset loading by `ProcessedSequenceProvider`, nominal category columns are one-hot encoded:
- 20 continuous and binary features are passed through directly as `float32`.
- `payment_format_code`: 7 categories $\rightarrow$ 7 one-hot binary columns.
- `payment_currency_code`: 15 categories $\rightarrow$ 15 one-hot binary columns.
- `receiving_currency_code`: 15 categories $\rightarrow$ 15 one-hot binary columns.
- Total model input width: $20 + 7 + 15 + 15 = \mathbf{57}$ features.

---

## 3. Label Orientation & Verification
- Ground truth target column: `is_laundering`.
- $y = 1$: Money Laundering transaction (illicit).
- $y = 0$: Legitimate banking transaction.
- Score direction: Sigmoid probability $P(y=1) \in [0, 1]$.
- No score inversion is permitted anywhere in the pipeline.

