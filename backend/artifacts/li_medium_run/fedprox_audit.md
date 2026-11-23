# FedSecureAI — FedProx Implementation and Mathematical Audit

**Audit Date:** 2026-10-17  
**Module:** `backend/infrastructure/federated/aggregation/fedprox.py`  
**Required Regularization Weight:** $\mu = 0.1$  

---

## 1. Mathematical Formulation Audit
The standard FedProx local objective is defined as:
$$\min_w h_k(w) = \mathcal{L}_{task}(w) + \frac{\mu}{2} ||w - w^t_{global}||_2^2$$

### Verification of Implementation Invariants:
1. **Initial Parameter Snapshot**: Verified. At the start of each round, `FedProxTrainingEngine.fit()` stores `self._global_parameters = initial_parameters`.
2. **Detached Reference Copy**: Verified. Global parameters are detached on the model device: `t = global_val.detach().to(device=device, dtype=param.dtype)`.
3. **No Self-Comparison or Reference Modification**: Verified. The local optimization loop updates `model.parameters()` while `cached_global` remains constant and untouched throughout the local epoch.
4. **Proximal Penalty Computation**:
   $$\text{penalty} = \frac{\mu}{2} \sum_{i} ||w_i - w^t_{global, i}||_2^2$$
   Implemented in `compute_proximal_loss(model, cached_global, self._mu, device)`.
5. **Backpropagation**: Verified. `total_loss = bce_loss + prox_penalty; total_loss.backward()` propagates both task gradients and proximal restorative gradients simultaneously.
6. **FedAvg Isolation**: Verified. FedAvg uses `TorchTrainingEngine` where $\mu = 0.0$ and no proximal term is ever added.

## 2. Unit Test Results
Automated unit tests in `backend/tests/unit/infrastructure/test_fedprox_math_audit.py`:
- `test_fedprox_penalty_zero_at_reference`: PASSED (Penalty is exactly 0.0 when $w = w^t_{global}$).
- `test_fedprox_gradient_matches_analytical_derivative`: PASSED (Autograd matches analytical derivative $\nabla_w = \mu (w - w^t_{global})$ with deviation $< 10^{-7}$).
- `test_fedavg_sample_weighted_aggregation_math`: PASSED (Aggregation weights $p_k = n_k / N$ sum to 1.0).

## 3. Diagnosis of Model Collapse at $\mu = 0.1$
In unweighted binary cross-entropy, the mini-batch task loss is normalized by batch size $B=1024$ (gradient scale $\sim \frac{1}{1024} \approx 0.00097$). The proximal penalty $\mu (w - w^t)$ with $\mu = 0.1$ is applied on every mini-batch update without batch normalization.
With rare positive prevalence ($0.075\% - 0.10\%$), the proximal anchor overpowers the rare-class gradient signal by $\approx 100\times$, pinning parameters near initialization and collapsing the output bias to the dataset's log-odds prior ($b \approx -7.2$).
