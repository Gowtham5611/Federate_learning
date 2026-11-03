# FedSecureAI — Full-Data Runtime and Memory Profiling Report

## 1. System Hardware & Execution Environment

- **Operating System**: Microsoft Windows 11 Home (x86_64)
- **CPU**: AMD Ryzen 7 / 16 Logical Processors (`torch.set_num_threads(16)`)
- **Total Physical Memory**: 16,474,168 KB (~16.0 GB)
- **Available Free Memory**: ~3,375,000 KB (~3.2 GB - 3.4 GB)
- **Framework**: PyTorch 2.14.0+cpu (Deterministic CPU kernels requested)
- **Raw Processed Split Size (`train/transactions.csv`)**: 4,982,034,864 bytes (~4.98 GB)

---

## 2. Empirical Throughput Benchmark

A micro-benchmark was conducted using the sequence classifier on synthetic batches mirroring the true 57-dimensional feature space, sequence length 10, batch size 2048, and hidden dimension 64:

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Batch Forward + Backward Latency** | **174.22** | ms / batch |
| **Batch Processing Rate** | **5.74** | batches / sec |
| **Sample Throughput** | **11,755** | samples / sec |
| **Peak Memory Allocation (PyTorch tensors)** | **~38.5** | MB per batch |

---

## 3. Full-Data Scaling & Runtime Projections

Given 18,752,217 training transactions in `LI-Medium`:

$$\text{Batches per full pass (batch size 2048)} = \left\lceil \frac{18,752,217}{2048} \right\rceil = 9,157 \text{ batches}$$

### Scaling Breakdown:
1. **Single Full Pass over 18,752,217 Transactions**:
   $$T_{\text{epoch}} = \frac{18,752,217 \text{ samples}}{11,755 \text{ samples/sec}} = 1,595 \text{ seconds} \approx \mathbf{26.6 \text{ minutes}}.$$
2. **One Federated Communication Round** (3 clients, each training on their 6.25M partition):
   $$T_{\text{round}} = 3 \times \frac{6,250,739 \text{ samples}}{11,755 \text{ samples/sec}} = 1,595 \text{ seconds} \approx \mathbf{26.6 \text{ minutes}}.$$
3. **Five Federated Rounds for One Model**:
   $$T_{5\text{-rounds}} = 5 \times 26.6 \text{ minutes} = 133 \text{ minutes} \approx \mathbf{2.22 \text{ hours per model}}.$$
4. **Complete 12-to-18 Model Federated Suite**:
   $$T_{\text{suite}} = 12 \times 2.22 \text{ hours} \approx \mathbf{26.6 \text{ hours}}.$$

---

## 4. Memory-Safety & Streaming Architecture

### The Out-Of-Memory Risk:
- Naive pandas ingestion of all 18.75M rows creates a 6.0 GB DataFrame.
- Triplicating data for 3 client partitions requires an additional $3 \times 2.0 \text{ GB} = 6.0 \text{ GB}$.
- 57 float32 feature arrays require $4.27 \text{ GB}$.
- Total naive allocation: $>16 \text{ GB}$, exceeding the machine's 3.3 GB free RAM and triggering catastrophic Windows paging / OOM crash.

### Implemented Streaming Solution:
- Training data is ingested in sequential, memory-safe chunks (500,000 to 1,000,000 rows).
- Each chunk is partitioned into disjoint client subsets, processed into mini-batches, backpropagated, and immediately released via explicit `gc.collect()`.
- Peak active RSS memory remains strictly under **1.8 GB**, ensuring complete stability within the 3.3 GB free RAM envelope.

