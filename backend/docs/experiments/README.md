# Experiment Documentation

## Scope of the current phase

Phase 4 implements centralized GRU, LSTM, and BiLSTM baselines and the reusable
training/evaluation foundation. IID partitioning, typology-based Non-IID
partitioning, FedAvg, FedProx, and federated aggregation are intentionally
deferred to later phases.

The `train` command prints that same statement at the end of every run, so the
boundary is visible to whoever runs the tool and not only to whoever reads this
file.

## Status

The three centralized baselines have been trained on the full Phase 3 processed
splits and their results are recorded below. Every number cites the report file it
came from. No federated result exists, because no federated code exists.

## Measured baseline results

**Run configuration** — identical for all three architectures, from
`backend/config/base.yaml` with no command-line overrides:
sequence length 10, stride 1, hidden size 64 per direction, 1 layer, dropout 0.0,
batch size 1024, Adam at lr 0.001, gradient clip 5.0, 5 epochs, early-stopping
patience 2, checkpoint selected on validation PR-AUC, seed 42, CPU
(`torch 2.14.0+cpu`), deterministic kernels requested and granted.

**Data actually read** — measured by the runs, not copied from the Phase 3 report:

| Split | Rows | Positives | Positive rate |
| --- | --- | --- | --- |
| train | 3,047,216 | 2,299 | 0.0754% |
| validation | 1,015,565 | 1,081 | 0.1064% |
| test | 1,015,564 | 1,797 | 0.1769% |

Model input width 57, expanded from 23 processed feature columns. `timestamp`,
`from_account` and `to_account` were used for ordering, grouping and traceability
and excluded from the model input.

### Test split, threshold fitted on validation and frozen

| Model | Recall | F1 | PR-AUC | Precision | FN | ROC-AUC | Accuracy | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU | 0.2721 | 0.3555 | 0.2887 | 0.5126 | 1,308 | 0.9328 | 0.99825 | 0.97323 |
| LSTM | 0.4446 | 0.5529 | 0.4771 | 0.7310 | 998 | 0.9354 | 0.99873 | 0.70329 |
| BiLSTM | 0.4124 | 0.5536 | 0.4698 | 0.8420 | 1,056 | 0.9245 | 0.99882 | 0.82639 |

### Test split, fixed threshold 0.5

| Model | Recall | F1 | PR-AUC | Precision | FN | ROC-AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRU | 0.7006 | 0.0716 | 0.2887 | 0.0377 | 538 | 0.9328 | 0.96784 |
| LSTM | 0.5648 | 0.2220 | 0.4771 | 0.1381 | 782 | 0.9354 | 0.99299 |
| BiLSTM | 0.5726 | 0.1900 | 0.4698 | 0.1139 | 768 | 0.9245 | 0.99136 |

### Validation split, both policies

| Model | Policy | Recall | F1 | PR-AUC | Precision | FN | ROC-AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU | fixed | 0.6318 | 0.0387 | 0.1623 | 0.0200 | 398 | 0.9259 | 0.96662 |
| GRU | validation_f1 | 0.2220 | 0.2947 | 0.1623 | 0.4380 | 841 | 0.9259 | 0.99887 |
| LSTM | fixed | 0.4829 | 0.1222 | 0.3333 | 0.0700 | 559 | 0.9242 | 0.99262 |
| LSTM | validation_f1 | 0.3145 | 0.4285 | 0.3333 | 0.6719 | 741 | 0.9242 | 0.99911 |
| BiLSTM | fixed | 0.5005 | 0.0958 | 0.3273 | 0.0530 | 540 | 0.9167 | 0.98995 |
| BiLSTM | validation_f1 | 0.3080 | 0.4286 | 0.3273 | 0.7040 | 748 | 0.9167 | 0.99913 |

### Run identity, cost and curve

| Model | Experiment ID | Params | Selected epoch | Training | Total |
| --- | --- | --- | --- | --- | --- |
| GRU | `centralized_gru_seed42_20260905T131807` | 23,681 | 3 of 5 | 566 s | 620 s |
| LSTM | `centralized_lstm_seed42_20260905T132827` | 31,553 | 3 of 5 | 13,774 s | 13,824 s |
| BiLSTM | `centralized_bilstm_seed42_20260906T011749` | 63,105 | 3 of 5 | 2,775 s | 2,923 s |

Validation PR-AUC per epoch:

| Model | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| GRU | 0.0329 | 0.0770 | **0.1623** | 0.1401 | 0.1294 |
| LSTM | 0.0036 | 0.1491 | **0.3333** | 0.3142 | 0.3203 |
| BiLSTM | 0.0312 | 0.1272 | **0.3273** | 0.2605 | 0.2630 |

No run stopped early: each trained its full 5-epoch budget, and each selected
epoch 3, so `best.pt` and `final.pt` differ in all three cases.

The wall-clock figures are not comparable to each other. The LSTM's total is
dominated by a single 12,558 s first epoch and the BiLSTM's epochs range from 324 s
to 725 s, because these runs shared a 16 GB laptop with other work. Per-epoch cost
on an unloaded machine was around 110 s for GRU and 175–200 s for LSTM. **Do not
read the timing column as an architecture comparison.**

### What these numbers do and do not support

The three runs used one configuration, one dataset, one split, one sequence
methodology, one loss and one seed, varying only the architecture — so they are
comparable in the sense the project needs.

They are **not** sufficient to name a best architecture. Every figure above is a
single seed, and the differences between LSTM and BiLSTM on the fitted threshold
(F1 0.5529 vs 0.5536, PR-AUC 0.4771 vs 0.4698) are well inside the range a seed
change could move them. Naming a winner would need repeated seeds and a stated
criterion. **No architecture is declared best here.**

What the runs do establish: the pipeline produces real measurements on real data
end to end; the recurrent models learn something well above chance on a
0.1%-positive problem (PR-AUC 0.29–0.48 against a 0.0018 base rate); and the
threshold policy dominates the recall/precision trade-off far more than the choice
among these three architectures — the fixed 0.5 threshold roughly doubles GRU's
recall while cutting its precision by more than an order of magnitude.

### Reproducing them

```bash
python main.py train --model gru
python main.py train --model lstm
python main.py train --model bilstm
```

Each writes a fresh `experiment_id` with a new timestamp, so re-running does not
overwrite the reports above. `--model all` runs the three in one process, which is
how the recorded GRU and LSTM runs were produced; the BiLSTM run was launched
separately after that process was interrupted, with the same seed and configuration.

**Determinism caveat.** Deterministic kernels were requested and the request was
granted (`deterministic_kernels_requested: true` in each report). That is not a
claim of bit-identical reproducibility: the request is warn-only, and PyTorch has
no deterministic implementation for every recurrent kernel on every backend. Expect
the same numbers on this machine and this torch build; do not assume them across
hardware.

## What a run produces

One `train` invocation writes three artefacts, all git-ignored:

| Artefact | Path | Contents |
| --- | --- | --- |
| Report | `<results>/reports/<experiment_id>.json` | Configuration, data provenance, sequence strategy, the full training curve, and every result row |
| Results table | `<results>/experiment_results.csv` | One appended row per split per threshold policy, in the shared schema below |
| Checkpoints | `<checkpoints>/<experiment_id>/best.pt`, `final.pt` | The selected-epoch parameters and the last-epoch parameters, kept apart |

`experiment_id` is `centralized_<architecture>_seed<seed>_<UTC timestamp>`, so two
runs of the same configuration never overwrite each other.

## The results table schema

Every row carries the same 25 columns for the whole project, so a federated row
from a later phase lands in the same table as a centralized row from this one:

```
experiment_id, model, training_mode, data_distribution, federated_algorithm,
client_id, round, loss_type, mu, split, precision, recall, f1, pr_auc, roc_auc,
accuracy, tp, tn, fp, fn, threshold, training_time, evaluation_time, seed,
hyperparameters
```

The table also holds rows from three reduced wiring checks (`--row-limit` 20,000 /
50,000 / 500,000), kept rather than deleted because they are real measurements of
what they measured. They are distinguishable from the full runs by `sequence_count`
and by `row_limit` inside the `hyperparameters` cell, and their numbers are **not**
quoted as baseline results anywhere in this document.

A Phase 4 row answers `training_mode=centralized`, `data_distribution=centralized`,
`federated_algorithm=none`, `loss_type=bce_with_logits`. The three columns a
centralized run genuinely cannot answer — `client_id`, `round`, `mu` — are written
**empty**. They are not written as `0`: round zero and `mu=0` are both meaningful
federated settings, and a blank cell is the only honest way to say "this run had
none".

## Reading the metrics

Recall, F1, PR-AUC, precision and false negatives are the primary metrics.
**Accuracy is not a primary metric here and must not be quoted as one:** the
positive rate in this dataset is roughly 0.1%, so a model that flags nothing at all
scores about 99.9%. False negatives are reported as their own column because they
are the laundering transactions the model let through, which is the cost this
project exists to reduce.

Each scored split appears **twice**, once under the fixed threshold and once under
the fitted one, so the effect of the threshold choice is visible rather than folded
into a single number.

## Protocol

- **Data.** The Phase 3 processed splits under `data/processed/ibm_aml/HI-Small/`
  are read as they are. The split is never regenerated, reshuffled or re-derived,
  and the raw dataset is never read by training.
- **Sequences.** Transactions are grouped by originating account, ordered as
  stored, and windowed. The target is the newest transaction in the window, so no
  sequence contains information from after the transaction it classifies. No
  sequence crosses a split boundary.
- **Test discipline.** The test split is not read until training has finished and
  the threshold has been frozen on validation. Nothing about the model — epoch
  selection, early stopping, architecture choice, threshold — is decided using it.
- **Threshold.** Under the default policy the threshold is fitted on validation
  only, recorded, frozen, and then applied unchanged to test. The report states
  which split it was fitted on, or `null` when the configured threshold was used
  as-is.
- **Loss.** Unweighted `BCEWithLogitsLoss`. The natural class imbalance is
  preserved: no class weights, no resampling, no synthetic positives. Cost-sensitive
  loss is a later phase's subject, and these baselines are what it will be compared
  against.
- **Comparability.** `--model all` runs one configuration three times, varying only
  the architecture. No architecture gets separately tuned hyperparameters, extra
  features, or a different split.
- **Reproducibility.** One seed fixes Python, NumPy and PyTorch. Deterministic
  kernels are *requested*; whether the backend granted the request is recorded in
  the report rather than assumed. On a CPU-only build with recurrent kernels the
  honest answer may be that it was not granted.

## Running a baseline

```bash
# One architecture
python main.py train --model gru

# All three from one configuration, which is what makes them comparable
python main.py train --model all

# A reduced run for wiring checks. Its report and log are marked as reduced,
# so its numbers cannot be read as a full run's.
python main.py train --model gru --row-limit 50000 --epochs 1
```

## Reporting rules

1. Every reported number traces back to a report JSON written by a real run,
   naming the configuration, the resolved seed and the processed directory it read.
2. A result is reported only if it can be reproduced from that stored
   configuration. The GRU row above was checked this way: `best.pt` was reloaded,
   the test split re-scored at the frozen threshold, and recall 0.27212, F1
   0.355507, PR-AUC 0.28869 and FN 1,308 came back identical to the published row.
3. No architecture is declared best, and no claim is made about FedAvg, FedProx or
   any federated behaviour, until an experiment that could have shown otherwise has
   actually been run.
4. Negative and inconclusive results are documented alongside positive ones.
5. A reduced run is labelled as reduced everywhere it is quoted.
