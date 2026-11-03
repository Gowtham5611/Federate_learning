# FedSecureAI — FedProx Implementation and Mathematical Audit

## Executive Summary

This document presents a rigorous code and mathematical audit of the **FedProx** implementation in FedSecureAI for the IBM AML `LI-Medium` benchmark. The audit inspects code structure, mathematical fidelity, reference parameter life cycles, gradient isolation, and the decoupling between proximal regularization and task loss functions.

---

## 1. Mathematical Formulation

FedProx (Li et al., 2020) modifies the local client optimization problem in federated learning to handle system and data heterogeneity by adding a proximal penalty to the local objective function:

$$\min_{w \in \mathbb{R}^d} h_k(w; w^{(t)}) = F_k(w) + \frac{\mu}{2} \|w - w^{(t)}\|_2^2$$

where:
- $F_k(w) = \mathbb{E}_{(x, y) \sim \mathcal{D}_k} [\ell(f(x; w), y)]$ is client $k$'s local empirical risk.
- $w^{(t)}$ is the global model parameter vector broadcast by the server at the beginning of round $t$.
- $\mu \ge 0$ is the proximal regularization parameter controlling the constraint on local model drift.
- $\|\cdot\|_2^2$ is the squared Euclidean $\ell_2$-norm over all trainable model parameters.

### Analytical Gradient and Proximal Operator
The gradient with respect to local parameters $w$ is:
$$\nabla_w h_k(w; w^{(t)}) = \nabla_w F_k(w) + \mu (w - w^{(t)})$$

Properties:
1. When $w = w^{(t)}$: $\|w - w^{(t)}\|_2^2 = 0$, and the gradient penalty is zero.
2. When $\mu = 0$: the objective reduces identically to standard local empirical risk minimization (FedAvg).
3. The regularization is parameter-level and independent of the sample count or batch size.

---

## 2. Code Inspection & Verification

The primary implementation resides in:
- `backend/infrastructure/federated/aggregation/fedprox.py`
- `backend/infrastructure/models/pytorch/torch_training_engine.py`

### 2.1 Proximal Loss Computation (`compute_proximal_loss`)
```python
def compute_proximal_loss(
    model: nn.Module,
    global_parameters: Mapping[str, Any],
    mu: float,
    device: torch.device,
) -> Tensor:
    if mu <= 0.0:
        return torch.tensor(0.0, device=device)

    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_parameters:
            global_val = global_parameters[name]
            target = global_val.detach().to(device=device, dtype=param.dtype)
            penalty = penalty + torch.sum((param - target) ** 2)

    return (mu / 2.0) * penalty
```
**Verification findings**:
- **Identity at Reference**: When `param == target`, `torch.sum((param - target)**2) == 0.0`.
- **Zero $\mu$ Bypass**: For `mu <= 0.0`, the function immediately short-circuits to `torch.tensor(0.0)`.
- **Linear Scaling**: For any fixed drift $(w - w_{\text{ref}})$, the penalty scales strictly linearly with $\mu$.
- **Precision and Device**: Tensors are cast to `param.dtype` and placed on `device`.

### 2.2 Global Reference Parameter Handling & Gradient Isolation
In `FedProxTrainingEngine`:
- Before round training, `client.train_round(current_state, ...)` passes the round-start server weights `current_state`.
- In `_train_one_epoch`, reference tensors are cached and explicitly detached:
  ```python
  t = global_val.detach().to(device=device, dtype=param.dtype)
  cached_global[name] = t
  ```
- During backpropagation:
  ```python
  total_loss = bce_loss + prox_penalty
  total_loss.backward()
  ```
  Because `cached_global` elements are detached leaves with `requires_grad=False`, gradients flow strictly through `model.named_parameters()`. No gradient leaks back into the global model state.

### 2.3 Server-Side Aggregation
At the end of round $t$, the central server aggregates client checkpoints via sample-weighted averaging:
$$w^{(t+1)} = \sum_{k=1}^K p_k w_k^{(t+1)}, \quad p_k = \frac{n_k}{\sum_{j=1}^K n_j}$$
- Implemented in `backend/infrastructure/federated/aggregation/fedavg.py` (`compute_aggregation_weights` and `aggregate_fedavg`).
- Verified in unit tests: $\sum_k p_k = 1.0 \pm 10^{-6}$.

---

## 3. Decoupling Task Loss and Proximal Regularization

In AML detection on IBM `LI-Medium`, extreme class imbalance ($0.0434\%$ positive prevalence in train, 1 positive in ~2,304 transactions) creates severe gradient starvation for the positive class.

To address this, we evaluate two loss paradigms:
1. **Unweighted BCE**:
   $$\mathcal{L}_{\text{task}} = -\frac{1}{B} \sum_{i=1}^B [y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i))]$$
2. **Imbalance-Aware Weighted BCE**:
   $$\mathcal{L}_{\text{task}} = -\frac{1}{B} \sum_{i=1}^B [w_{\text{pos}} y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i))]$$
   where $w_{\text{pos}} = \frac{18,744,079}{8,138} \approx 2,303.2774$.

### Mathematical Decoupling:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}; w_{\text{pos}}) + \frac{\mu}{2} \sum_{p} \|w_p - w_{\text{ref}, p}\|_2^2$$
The proximal term is strictly parameter-level. It is **not** weighted by $w_{\text{pos}}$ and does not depend on whether a transaction in batch $B$ is positive or negative. This preserves both the theoretical convergence guarantees of FedProx and the cost-sensitive gradient scaling necessary for AML detection.

---

## 4. Controlled Baseline Configuration Matrix

All controlled optimization runs share the exact same structural environment:
- **Dataset**: IBM AML `LI-Medium`
- **Window Size**: 200,000 sequence training window per baseline configuration, preserving exact client partition splits (IID vs Non-IID).
- **Federated Rounds**: 5 rounds.
- **Local Epochs per Round**: 1.
- **Batch Size**: 1024.
- **Optimizer**: Adam ($\text{lr} = 0.001, \beta_1=0.9, \beta_2=0.999$).
- **Architectures**: GRU, LSTM, BiLSTM (hidden size = 64, layers = 1, dropout = 0.0).
- **Seed**: 42.

This audit confirms that the FedProx implementation is mathematically sound, free of gradient leaks, and ready for the controlled optimization experiments.

