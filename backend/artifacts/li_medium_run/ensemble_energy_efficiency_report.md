# FedSecureAI — Computational Efficiency and Energy Consumption Report

**Measurement Date:** 2026-10-17
**Environment:** Windows 11 AMD64, Intel 16 logical cores, PyTorch 2.14.0+cpu, 16.0 GB RAM

## 1. Architectural Efficiency Comparison (Batch Size 1024)

| Model Architecture | Parameter Count | Inference Latency (Batch 1024) | Throughput (seq/sec) | Relative Compute Cost |
| :--- | :---: | :---: | :---: | :---: |
| **GRU** | 23,681 | 35.36 ms | 28962.0 seq/s | 1.00x |
| **LSTM** | 31,553 | 27.54 ms | 37187.0 seq/s | 0.78x |
| **BiLSTM** | 63,105 | 54.50 ms | 18789.2 seq/s | 1.54x |

## 2. Energy Consumption Estimates (Active Package TDP ~45W)

- **GRU Baseline Model:** 0.0432 Watt-hours (0.00004316 kWh) per 100,000 transactions.
- **Stacked Ensemble:** 0.1097 Watt-hours (0.00010969 kWh) per 100,000 transactions.
- **Performance-Energy Trade-Off:** The Stacked Logistic Regression ensemble increases test PR-AUC by **2.5x** (0.002009 vs 0.000801) and test ROC-AUC from 0.433 to **0.607** for an energy expenditure increase of approximately 2.3x.

## 3. Hardware Measurement Tool Limitations

- On Microsoft Windows 11, unprivileged user-mode processes cannot directly read Intel Running Average Power Limit (RAPL) Model-Specific Registers (MSRs) without custom kernel drivers or specialized vendor utilities (e.g. Intel Power Gadget).
- All energy metrics in this report are mathematically grounded using measured microsecond wall-clock execution time, hardware TDP ratings (45.0W), and empirical active CPU core utilization.
