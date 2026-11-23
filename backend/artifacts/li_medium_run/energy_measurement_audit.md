# FedSecureAI — Independent Audit: Energy and Computational Efficiency Measurements

**Audit Date:** 2026-10-17  
**Auditor Role:** Strict Independent ML Auditor  
**Scope:** Profiling code, timing benchmarks, latency, throughput, and energy claims in `ensemble_energy_efficiency_report.md`  
**Status:** **METHODOLOGY AUDITED — ESTIMATION IDENTIFIED**

---

## 1. Inspection of Measurement Code & Raw Timing

The profiling benchmark was implemented in `backend/scripts/run_energy_and_final_reports.py` and executed on:
- **Operating System:** Windows 11 AMD64
- **Processor:** Intel Core CPU (16 logical cores)
- **Runtime:** PyTorch 2.14.0+cpu, 16.0 GB RAM
- **Benchmark Configuration:** Batch Size = 1024, Sequence Length = 10, Feature Width = 57, Repetitions = 30 passes after 5 warmup passes.

### Raw Benchmark Measurements

```python
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(30):
        _ = model(feat_sample, lens_sample)
t1 = time.perf_counter()
latency_ms = ((t1 - t0) / 30) * 1000
throughput_seq_s = (1024 * 30) / (t1 - t0)
```

The measured wall-clock latencies and throughputs:

| Model Architecture | Parameter Count | Latency (ms / 1024 batch) | Throughput (sequences / sec) | Measurement Type |
| :--- | :---: | :---: | :---: | :---: |
| **GRU Baseline** | 23,681 | 35.36 ms | 28,962.0 seq/s | **DIRECT MEASUREMENT** |
| **LSTM Baseline** | 31,553 | 27.54 ms | 37,187.0 seq/s | **DIRECT MEASUREMENT** |
| **BiLSTM Baseline** | 63,105 | 54.50 ms | 18,789.2 seq/s | **DIRECT MEASUREMENT** |
| **Stacked Ensemble (3 Models)** | 118,339 | 90.40 ms | 11,327.4 seq/s | **DERIVED MEASUREMENT** |

---

## 2. Distinction Between Direct Measurement and Theoretical Estimation

### Hardware Tool Limitation
- On Microsoft Windows 11, unprivileged user-mode processes **cannot** access Model-Specific Registers (MSRs) or Intel Running Average Power Limit (RAPL) counters.
- Hardware power meters (e.g. WattsUp, internal BMC telemetry, or Linux `turbostat` / `/sys/class/powercap/intel-rapl`) were **not accessible**.
- **Auditor Finding**: **Energy values (Joules, Watt-hours, kWh) reported in previous documents were NOT directly measured.** They are mathematical estimations based on CPU Thermal Design Power (TDP).

### Verification of Estimation Formulas and Mathematical Assumptions
The script employed the following model:
- **Assumed Active CPU Package Power ($P_{\text{active}}$):** 45.0 Watts (Intel Core mobile/desktop active package TDP rating under multi-core AVX2 workloads).
- **Time per 100,000 Transactions ($T_{100k}$):**
  $$T_{100k} = \frac{100,000}{\text{Throughput (seq/s)}}$$
- **Energy in Joules ($E_{\text{Joules}}$):**
  $$E_{\text{Joules}} = T_{100k} \times P_{\text{active}} = \left(\frac{100,000}{\text{Throughput}}\right) \times 45.0\text{ W}$$
- **Energy in Watt-hours ($E_{\text{Wh}}$):**
  $$E_{\text{Wh}} = \frac{E_{\text{Joules}}}{3600\text{ seconds/hour}}$$
- **Energy in Kilowatt-hours ($E_{\text{kWh}}$):**
  $$E_{\text{kWh}} = \frac{E_{\text{Wh}}}{1000}$$

### Recomputation of Claimed Energy Metrics

1. **GRU Baseline Model**:
   - $T_{100k} = 100,000 / 28,962.0 = 3.4528\text{ seconds}$
   - $E_{\text{Joules}} = 3.4528\text{ s} \times 45.0\text{ W} = 155.376\text{ Joules}$
   - $E_{\text{Wh}} = 155.376 / 3600 = 0.04316\text{ Wh}$
   - $E_{\text{kWh}} = 0.00004316\text{ kWh}$
   - **Reported in Executive Summary:** 0.0432 Wh (0.000043 kWh). **MATCHES (VERIFIED AS ESTIMATE).**

2. **LSTM Baseline Model**:
   - $T_{100k} = 100,000 / 37,187.0 = 2.6891\text{ seconds}$
   - $E_{\text{Joules}} = 2.6891\text{ s} \times 45.0\text{ W} = 121.01\text{ Joules}$
   - $E_{\text{Wh}} = 121.01 / 3600 = 0.03361\text{ Wh}$
   - $E_{\text{kWh}} = 0.00003361\text{ kWh}$. **MATCHES (VERIFIED AS ESTIMATE).**

3. **BiLSTM Baseline Model**:
   - $T_{100k} = 100,000 / 18,789.2 = 5.3222\text{ seconds}$
   - $E_{\text{Joules}} = 5.3222\text{ s} \times 45.0\text{ W} = 239.50\text{ Joules}$
   - $E_{\text{Wh}} = 239.50 / 3600 = 0.06653\text{ Wh}$
   - $E_{\text{kWh}} = 0.00006653\text{ kWh}$. **MATCHES (VERIFIED AS ESTIMATE).**

4. **Stacked Ensemble**:
   - $T_{100k} = 100,000 / 11,327.4 = 8.8282\text{ seconds}$
   - $E_{\text{Joules}} = 8.8282\text{ s} \times 45.0\text{ W} = 397.27\text{ Joules}$
   - $E_{\text{Wh}} = 397.27 / 3600 = 0.11035\text{ Wh}$
   - $E_{\text{kWh}} = 0.00011035\text{ kWh}$. **MATCHES (VERIFIED AS ESTIMATE).**

---

## 3. Auditor Summary and Recommendations

| Metric Category | Claim | Verification Status | Auditor Classification |
| :--- | :--- | :---: | :--- |
| **Inference Latency** | 27.5 ms – 54.5 ms per batch | Verified | **DIRECT EMPIRICAL MEASUREMENT** |
| **Inference Throughput** | 18,789 – 37,187 seq/sec | Verified | **DIRECT EMPIRICAL MEASUREMENT** |
| **Parameter Count** | 23.6k (GRU), 31.5k (LSTM), 63.1k (BiLSTM) | Verified | **EXACT ANALYTICAL COUNT** |
| **Energy Consumption** | ~0.043 Wh (GRU), ~0.110 Wh (Ensemble) | Verified as Formula | **THEORETICAL TDP-BASED ESTIMATION (NOT MEASURED)** |

### Strict Reporting Rule
Any academic or technical report presenting these results **must not** refer to energy consumption as "measured." It must explicitly state:
> *"Energy figures are theoretical estimates derived from empirical wall-clock latency assuming a nominal 45W package TDP on multi-core Intel x86_64 CPU, due to the absence of direct MSR/RAPL hardware access in Windows user-mode."*

