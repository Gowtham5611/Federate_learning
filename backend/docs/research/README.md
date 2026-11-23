# Research Documentation

Status: **research write-up NOT YET STARTED.** No results exist to write about.

## Research question

Can federated learning train competitive AML transaction-detection models
without centralising bank transaction data, and how much does the choice of
aggregation strategy matter when client data is not identically distributed?

Concretely, the study compares:

1. Recurrent architectures — LSTM, GRU, BiLSTM — under federated training.
2. Aggregation strategies — FedAvg vs FedProx.
3. Client distributions — IID vs Non-IID.

against centralized baselines (Logistic Regression, Random Forest, and the same
three recurrent architectures trained centrally).

## What this directory will contain

| Document | Purpose |
| --- | --- |
| `literature_review.md` | Prior work on AML detection, federated learning under heterogeneity, and their intersection |
| `methodology.md` | Formal problem statement, notation, algorithms, experimental design |
| `related_work_comparison.md` | How this study's setup differs from published federated AML work |
| `findings.md` | Results and interpretation — written only after experiments run |
| `threats_to_validity.md` | Internal, external and construct validity concerns |
| `references.bib` | Bibliography |

## Intended contribution

**PLANNED — not yet demonstrated.** A controlled comparison of FedAvg and
FedProx across three recurrent architectures on a single realistic AML
transaction dataset, under both IID and Non-IID client partitions, reporting
imbalance-aware metrics (PR-AUC alongside ROC-AUC) as well as communication and
convergence cost.

## Integrity rules

1. No claim is written here without a reproducible run behind it.
2. Prior-work numbers are cited, never reproduced from memory as if measured.
3. A hypothesis that fails is reported as a finding, not removed.
4. The gap between this study's setup and a real multi-bank deployment is stated
   explicitly, including that clients are simulated by partitioning one dataset.
