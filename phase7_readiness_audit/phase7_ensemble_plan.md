# Phase 7: Hybrid Federated Ensemble Specification & Implementation Plan

## 1. Ensemble Architecture & Formulation

The Phase 7 ensemble blends the representations of the best IID federated model (capturing global feature distributions) and the best Non-IID federated model (capturing specialized bank/account typologies):

$$P_{\text{ensemble}}(x) = \alpha P_{\text{IID}}(x) + (1 - \alpha) P_{\text{Non-IID}}(x)$$

where:
- $\alpha \in [0.0, 1.0]$ is the blending coefficient.
- $P_{\text{IID}}(x)$ is the continuous probability output by the frozen champion IID model.
- $P_{\text{Non-IID}}(x)$ is the continuous probability output by the frozen champion Non-IID model.

---

## 2. Strict Leakage Prevention & Protocol

1. **Zero Test-Set Tuning**: Neither $\alpha$ nor the final operational threshold $\tau_{\text{ensemble}}$ may be selected or adjusted using the test set.
2. **Validation Objective Function**: $\alpha$ is chosen to maximize validation PR-AUC (Average Precision) subject to the false alert constraint:
   $$\alpha^* = \arg\max_{\alpha} \text{PR-AUC}_{\text{val}}(\alpha) \quad \text{s.t. Alerts per 10,000 transactions} \le 10.0$$
3. **Threshold Calibration**: Once $\alpha^*$ is frozen, the classification threshold $\tau_{\text{ensemble}}$ is calibrated on the validation set to maximize $F_1$-score (or achieve target operational recall $\ge 35\%$).
4. **Frozen Evaluation**: The ensemble with frozen $(\alpha^*, \tau^*)$ is evaluated exactly once on the held-out test split.

---

## 3. Step-by-Step Implementation Workflow

### Step 1: Validation Inference Forward Pass
```bash
# Generate per-sample validation probability vectors for champions:
python backend/scripts/generate_validation_predictions.py \
  --dataset HI-Small \
  --iid-model results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt \
  --non-iid-model results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_seed42/hi_small_lstm_fedavg_seed42_global_best.pt \
  --output-dir experiments/phase7_ensemble/HI-Small
```

### Step 2: Grid Search Over $\alpha$ on Validation Split
Search $\alpha \in \{0.00, 0.05, 0.10, \dots, 0.95, 1.00\}$:
For each $\alpha$:
- Compute $P_{\alpha,\text{val}} = \alpha P_{\text{IID,val}} + (1 - \alpha) P_{\text{Non-IID,val}}$
- Calculate $\text{PR-AUC}(\alpha)$ and $\text{ROC-AUC}(\alpha)$ against ground-truth validation labels.
- Select $\alpha^*$ that maximizes validation PR-AUC.

### Step 3: Calibrate Frozen Threshold $\tau^*$
- Evaluate $F_1(\tau)$ across thresholds $\tau \in [0.01, 0.99]$.
- Select $\tau^*$ corresponding to maximum validation $F_1$.

### Step 4: Single Evaluation on Held-Out Test Split
- Compute $P_{\text{test}} = \alpha^* P_{\text{IID,test}} + (1 - \alpha^*) P_{\text{Non-IID,test}}$ using the verified test prediction arrays.
- Apply frozen $\tau^*$: $\hat{y}_{\text{test}} = \mathbb{I}(P_{\text{test}} \ge \tau^*)$.
- Compute final test metrics: ROC-AUC, PR-AUC, F1, Precision, Recall, Specificity, Alerts/10k, Confusion Matrix.
