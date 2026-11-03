# Configuration

Backend configuration lives in this directory as YAML. It is loaded by
[infrastructure/utils/config_loader.py](../infrastructure/utils/config_loader.py),
which resolves `${VAR}` references against the process environment (see
[.env.example](../../.env.example) at the repository root).

This directory is resolved as `BACKEND_ROOT / "config"` by
[infrastructure/utils/paths.py](../infrastructure/utils/paths.py). Path *values*
inside `base.yaml` (`data/raw`, `datasets/dataset_registry.yaml`, …) stay
relative to the **repository root**, because those directories are shared with
the frontend and are not owned by the backend.

## Design rule

**A configuration key exists only when code reads it.** Empty or speculative
sections are not added ahead of the implementation that consumes them — they
rot, and they make the config lie about what the system does.

## Current files

| File | Status | Contents |
| --- | --- | --- |
| `base.yaml` | Active | Application identity, random seed, project paths, logging, active dataset, preprocessing, centralized training |

## Active sections

| Section | Read by | Holds |
| --- | --- | --- |
| `app` | `config_loader` | Name, description, version, environment |
| `reproducibility` | *(seeded components, from the ML phase on)* | `random_seed` |
| `dataset` | [composition_root.build_dataset_manager](../composition_root.py) | `active_dataset_id`. Dataset *declarations* live in the registry, not here; credentials live in `.env` and are never read from YAML |
| `paths` | [infrastructure/utils/paths.py](../infrastructure/utils/paths.py) | Project-root-relative directories, including `data_raw`, `data_interim` and `data_processed` |
| `preprocessing` | [composition_root.build_preprocessing_service](../composition_root.py) | `chunk_size`, `output_subdirectory`, `split.train_fraction`, `split.validation_fraction`, `keep_interim` |
| `training` | [composition_root.build_training_session](../composition_root.py) | Every hyperparameter of a centralized baseline run |
| `logging` | [infrastructure/utils/logging_config.py](../infrastructure/utils/logging_config.py) | Level, format, console and file sinks |

### `preprocessing` in detail

| Key | Default | Effect |
| --- | --- | --- |
| `chunk_size` | `200000` | Rows held per read. Bounds peak memory independently of the file's size; `--chunk-size` overrides it for one run. Changing it must not change the output — an integration test asserts that. |
| `output_subdirectory` | `ibm_aml/HI-Small` | Appended to `data_interim` and `data_processed`, so a second dataset's outputs sit beside these rather than on top of them. |
| `split.train_fraction` | `0.60` | Chronological, never random. |
| `split.validation_fraction` | `0.20` | The test fraction is the remainder. |
| `keep_interim` | `false` | Working files are deleted after a successful run; they are a second copy of the dataset on disk. `--keep-interim` overrides it. |

The processed output never lands inside `data_raw`: the writer refuses such a
configuration at construction, before a single row exists.

### `training` in detail

One section covers all three architectures, which is the point: `train --model all`
reads these values once and varies only the architecture, so no baseline gets a
separately tuned advantage.

| Group | Keys | Notes |
| --- | --- | --- |
| Architecture choice | `architecture` | Only the fallback when no `--model` is supplied at all |
| Sequences | `sequence_length`, `sequence_stride`, `one_hot_nominal_codes` | `sequence_stride: 1` keeps every labelled transaction; above 1 it discards targets and shifts the class balance |
| Model | `hidden_size`, `num_layers`, `dropout` | `hidden_size` is per direction; the bidirectional head is sized at twice it automatically. `dropout` applies between recurrent layers only, so it has no effect at `num_layers: 1` — the report records the effective value, not the requested one |
| Optimisation | `epochs`, `batch_size`, `learning_rate`, `optimizer`, `gradient_clip_norm`, `early_stopping_patience`, `selection_metric` | `gradient_clip_norm: null` disables clipping. `early_stopping_patience: 0` trains the full budget. `selection_metric` defaults to `pr_auc`: threshold-free, and unlike ROC-AUC it does not flatter a model on a split that is 99.9% negative |
| Threshold | `threshold`, `threshold_policy` | `validation_f1` fits on validation alone, then freezes the value and applies it to test unchanged. `fixed` uses `threshold` as given |
| Runtime | `device`, `num_workers`, `row_limit` | `device: null` resolves per machine. `row_limit` marks the run as reduced in the log *and* the report, so a smoke test cannot be quoted as a full run |

Every key is overridable on the command line, and the command line wins. **No value
here is a measured optimum** — they are conventions, and the inline comments in
`base.yaml` say which.

There is no loss-weighting key. The baselines use unweighted `BCEWithLogitsLoss`
and preserve the natural class imbalance; cost-sensitive loss belongs to the later
phase that studies it, and adding a knob for it now would invite a baseline that
the later phase has nothing to compare against.

## Planned sections

These are **NOT YET IMPLEMENTED**. Each is added in the phase that introduces
the code reading it.

Phase 4 implements centralized GRU, LSTM, and BiLSTM baselines and the reusable
training/evaluation foundation. IID partitioning, typology-based Non-IID
partitioning, FedAvg, FedProx, and federated aggregation are intentionally
deferred to later phases. That is why the `federated` and `partitioning` sections
below do not exist yet rather than existing empty.

| Section | Added in phase | Will hold |
| --- | --- | --- |
| `models` | Classical baselines | Per-model hyperparameters for Logistic Regression and Random Forest. The recurrent architectures are configured under `training` |
| `federated` | Federated learning | Strategy (FedAvg / FedProx), FedProx `mu`, client count, rounds, fraction fit/eval, local epochs |
| `partitioning` | Partitioning | IID vs typology-based Non-IID scheme and its parameters |
| `experiment` | Experiment runner | Run naming, tracked metrics, artefact retention, comparison matrix |
| `evaluation` | Evaluation | Plot formats and report layout. The threshold policy and metric set are already read from `training` |

## Environment overlays

Per-environment overlay files (`development.yaml`, `testing.yaml`,
`production.yaml`) are **PLANNED**. They will be layered over `base.yaml`
rather than duplicating it.

## Related

- Experiment-specific configs (one file per experiment run) belong in
  [experiments/configs/](../../experiments/configs/), not here.
- Dataset declarations belong in
  [datasets/dataset_registry.yaml](../../datasets/dataset_registry.yaml), not here.
