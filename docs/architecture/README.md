# Architecture Documentation

Status: **skeleton only — the documents below are NOT YET WRITTEN.**

## What this directory will contain

| Document | Purpose |
| --- | --- |
| `layers.md` | The four layers (domain / application / infrastructure / interfaces), what belongs in each, and the dependency rule |
| `dependency_rules.md` | Concrete import rules and how they are enforced in CI |
| `ports_and_adapters.md` | Every port defined by the domain/application layer and its adapters |
| `data_flow.md` | End-to-end flow: Kaggle download → validation → preprocessing → sequences → partitions → federated training → evaluation |
| `federated_architecture.md` | Server/client topology, round lifecycle, message contents, aggregation |
| `decisions/` | Architecture Decision Records (ADRs), one file per decision |

## Backend / frontend separation

The repository is split into two source areas plus shared, non-code directories:

```
backend/    domain, application, infrastructure, config, interfaces/{api,cli},
            scripts, tests  — the whole Python application
frontend/   interfaces/dashboard — presentation only
data/ datasets/ reports/ docs/ requirements/  — shared, owned by neither
```

Only `backend/` is placed on `sys.path` (see `pythonpath` in `pytest.ini`,
`mypy_path` in `pyproject.toml`, and the root `main.py`). That is what keeps
`import domain`, `import application`, `import infrastructure` and
`import interfaces` resolving exactly as before the split, and it is also why
`frontend/interfaces` cannot shadow `backend/interfaces` — the frontend root is
never on the path, and it carries its own package marker.

Shared directories deliberately stay at the repository root: `data/` and
`reports/` are produced and consumed across the boundary, so nesting them inside
either area would give one side ownership of the other's inputs.

## Layering rule (already in force)

Dependencies point inward only:

```
interfaces ──▶ application ──▶ domain
                   ▲              ▲
infrastructure ────┘──────────────┘
```

- `domain/` imports nothing from the other three layers.
- `application/` imports `domain/` only.
- `infrastructure/` implements ports declared in `domain/interfaces/` and
  `application/ports/`.
- `interfaces/` (API, dashboard, CLI) drives `application/` use cases and never
  reaches into `infrastructure/` directly.

Wiring of concrete adapters to ports happens at the composition root
([backend/composition_root.py](../../backend/composition_root.py)), which is the
only place allowed to know both sides. The root
[main.py](../../main.py) is a thin shim that puts `backend/` on the path and
delegates to it, preserving the `python main.py <command>` interface.

## Current state

Implemented so far: dataset configuration, dataset acquisition, data ingestion
(source → chunked loader → schema validator), **preprocessing** (raw →
chronological train/validation/test splits plus reproducibility artefacts) and
**centralized recurrent baselines** (sequence construction → GRU/LSTM/BiLSTM →
training → evaluation → checkpoints → result storage).

Phase 4 implements centralized GRU, LSTM, and BiLSTM baselines and the reusable
training/evaluation foundation. IID partitioning, typology-based Non-IID
partitioning, FedAvg, FedProx, and federated aggregation are intentionally
deferred to later phases.

Nothing downstream of the baselines exists — no client partitioning, no federated
learning, no API, no dashboard.

### Ports declared (application/ports/output/)

| Port | Purpose | Adapter |
| --- | --- | --- |
| `DatasetRegistryPort` | Read dataset declarations | `YamlDatasetRegistry` |
| `DatasetDownloaderPort` | Transfer a dataset | `BaseDatasetDownloader` → `KaggleDownloader` |
| `TransactionDataSourcePort` | Identify and locate the transaction file | `IBMAMLDataSource` |
| `ChunkedDataLoaderPort` | Stream a tabular file in chunks | `CsvChunkLoader` |
| `SchemaValidatorPort` | Check a chunk against a schema | `DataFrameSchemaValidator` |
| `InterimStoragePort` | Day-partitioned working storage and train staging | `CsvInterimStorage` |
| `ProcessedDataWriterPort` | Write split files and metadata artefacts | `CsvProcessedDataWriter` |
| `SequenceDataProviderPort` | Build one split's sequences from the processed files | `ProcessedSequenceProvider` |
| `TrainingEnginePort` | Fit and score a recurrent classifier | `TorchTrainingEngine` |
| `MetricsCalculatorPort` | Compute metrics and fit a threshold | `SklearnMetricsCalculator` |
| `CheckpointStorePort` | Save and load model parameters | `TorchCheckpointStore` |
| `ExperimentResultStorePort` | Persist the report and append result rows | `CsvExperimentResultStore` |

Each port has exactly one abstraction. There is no competing `DataSource`,
`DataLoader`, `Validator`, `Schema`, `Preprocessor`, `Writer`, `Trainer`,
`Evaluator` or `CheckpointStore` interface anywhere in the tree.

### The reusable training foundation

Phase 4's ports are written for the federated phases that follow, and that shapes
their signatures more than centralized training alone would:

- **One training loop, not three.** `TorchTrainingEngine` fits GRU, LSTM and
  BiLSTM through the same code path. Three loops would drift, and the difference
  this project measures between architectures would become partly a difference
  between code paths.
- **`fit` is already a local round.** It takes optional starting parameters,
  returns the parameters it reached rather than mutating hidden state, and holds no
  reference to a "global" dataset. A client's local training in a later phase is
  this call on that client's own data. Nothing in the engine aggregates,
  communicates, or knows that clients will exist.
- **Parameters are plain state dicts.** `ParameterState` is a mapping of tensors,
  so the thing a server would send and receive is already the thing the engine
  consumes and produces.
- **The result schema is the final schema.** `CsvExperimentResultStore` writes the
  federated columns from the start and leaves the three a centralized run cannot
  answer empty, so a later phase appends rows to the same table instead of
  migrating it.

`CentralizedTrainingService` composes those five ports and owns the *order* of the
run — in particular that the test split is not loaded until the threshold has been
frozen on validation. It contains no PyTorch and no pandas.

### Application services

| Service | Owns |
| --- | --- |
| `DatasetManager` | Dataset lifecycle: declaration, local presence, reconciled status, explicit download |
| `DatasetIngestionService` | Composition of the three ingestion ports, and the strictness policy for an invalid chunk |
| `PreprocessingService` | The three-pass preprocessing pipeline: what happens in what order, and the measured report it produces |
| `CentralizedTrainingService` | One baseline run end to end: load, fit, freeze the threshold, score, checkpoint, report |

`PreprocessingService` orchestrates but does not transform. Cleaning, feature
construction and fitting live in `AmlPreprocessor`; boundary computation lives
in `ChronologicalSplitter`; both are injected. The service reuses
`IBMAMLDataSource` to locate the file, so preprocessing reads whatever the
registry declares and **cannot reach a different file**. It is wired with no
downloader at all — the preprocess command cannot download by construction, not
merely by policy.

### Domain

`domain/` holds value objects (`DatasetDescriptor`, `ColumnSpec`, `TableSchema`),
enums (`DatasetStatus`, `DatasetSourceKind`, `ColumnDataType`,
`UnexpectedColumnPolicy`, `ValidationSeverity`, `ValidationIssueKind`,
`SplitName`) and one exception hierarchy rooted at `FedSecureAIError`. It
imports nothing from the other three layers and does not import pandas.

`SplitName` encodes the *order of time* (`TRAIN` → `VALIDATION` → `TEST`) and
`is_fitting_split`, which answers "may statistics be learned here?" in one place
rather than restating the rule at each call site.

### Two deliberate exceptions to the purity rule

Both are recorded here rather than left to be discovered:

- `pathlib.Path` appears in application-layer DTOs and port signatures. A path is
  a filesystem concept, but the alternative — a project-specific path wrapper —
  buys nothing.
- `pandas.DataFrame` appears in `ChunkedDataLoaderPort`,
  `SchemaValidatorPort`, `InterimStoragePort` and `ProcessedDataWriterPort`. A
  chunk of a tabular file has to be *some* type, and inventing a wrapper around a
  DataFrame would add a layer with no behaviour of its own. The trade-off is
  accepted knowingly and documented in each port module.

### Data flow implemented to date

```
datasets/dataset_registry.yaml
   → YamlDatasetRegistry → DatasetManager → IBMAMLDataSource
   → CsvChunkLoader → DataFrameSchemaValidator → (chunk, ChunkValidationResult)

   → PreprocessingService (three bounded passes)
        pass 1  clean + bucket by calendar day + timestamp histogram
                    → CsvInterimStorage (data/interim, deleted after the run)
        pass 2  per day: sort, engineer features, assign split,
                    write validation/test, fit train statistics, stage train rows
        pass 3  apply the frozen transformations, write the train split
   → CsvProcessedDataWriter → data/processed/ibm_aml/HI-Small/{train,validation,test}
                            + metadata/{schema,feature_metadata,transformers,
                                        splits,preprocessing_report}.json

   → ProcessedSequenceProvider (per split: group by account, window, right-pad)
   → CentralizedTrainingService
        TorchTrainingEngine.fit(train, validation)      → best / final parameters
        SklearnMetricsCalculator.select_threshold(validation)  → frozen threshold
        TorchTrainingEngine.predict(test)               → the only read of test
   → TorchCheckpointStore    → <checkpoints>/<experiment_id>/{best,final}.pt
   → CsvExperimentResultStore → <results>/reports/<experiment_id>.json
                              + <results>/experiment_results.csv

   → [NOT IMPLEMENTED: partitions → clients → federated rounds → aggregation
                       → API → dashboard]
```

Three passes rather than one because the raw file is **not in timestamp order**
— measured, not assumed — while chronological splitting and causal history
features both require time order. See
[docs/dataset/README.md §18.1](../dataset/README.md) for the reasoning and
§14.4 for the verified schema.

**Leakage is prevented structurally, not by convention.** The transformers raise
`TransformerNotFittedError` if used before freezing and
`TransformerAlreadyFittedError` if updated after, so validation or test data
cannot reach a fit path by accident. That the fit used training rows alone is
checkable after the fact: `scaling_parameters[…]["count"]` in the written
artefact equals the training row count.

**The raw dataset is read-only by construction.** `CsvProcessedDataWriter`
refuses at *construction* to write anywhere inside the raw directory — a
configuration mistake is caught before a single row exists, not after the file
it would have overwritten is gone. The file's SHA-256 is additionally taken
before the run and re-checked after.

**Training reads the processed splits only.** `ProcessedSequenceProvider` is
constructed from a processed directory and resolves split files inside it, so
training has no path by which to reach `data/raw/` and cannot regenerate or
reshuffle the split it was given.

### Sequences, and how leakage is kept out of them

A recurrent model needs a 3D tensor, and the easy way to get one — reshape each
transaction row into a length-1 sequence — would produce the right shape and learn
nothing temporal. Instead:

- Transactions are grouped by **originating account**, which is the entity whose
  behaviour over time the model is meant to read.
- Rows are taken in stored order, which preprocessing left non-decreasing by
  timestamp; the provider verifies that rather than trusting it and raises
  `UnsortedSplitError` if it does not hold.
- A window's target is its **newest** transaction, so every feature in the window
  is at or before the moment being classified. Nothing from the future is visible.
- Windows are built **within one split**, so no sequence spans the
  train/validation or validation/test boundary.
- Short accounts are right-padded with zeros and their true length is carried
  alongside, so `pack_padded_sequence` never lets padding reach the recurrence.
- `timestamp`, `from_account` and `to_account` are used for grouping, ordering and
  traceability, and are excluded from the model input. So is the target column.

### The three architectures

`RecurrentSequenceClassifier` holds the whole forward pass — pack, run the
encoder, take the final hidden state, project to one logit — and its three
subclasses differ only in which `torch.nn` encoder they build and how wide the
resulting representation is. BiLSTM's head is sized for the concatenated forward
and backward states rather than assuming the one-directional width.

The models emit **logits**. The loss consumes logits. `torch.sigmoid` appears once
in the whole project, in the scoring path, because probabilities exist only for
evaluation.

### Composition root and startup

`backend/composition_root.py` is the only place adapters are chosen; the root
`main.py` merely delegates to it. Its first action is `load_env_file()`
(`backend/infrastructure/utils/env.py`), which applies a git-ignored `.env` to
the process environment **before** configuration is read, so both `${VAR}`
placeholders in the YAML and the Kaggle credential provider see it. An absent
file is not an error; an exported shell variable takes precedence over the file;
and no value read from the file is ever logged.

Project-root-relative paths are resolved by
`backend/infrastructure/utils/paths.py`, which distinguishes `BACKEND_ROOT`
(owning `backend/config/`) from `PROJECT_ROOT` (owning the shared `data/`,
`datasets/`, `reports/` and `.env`). No module counts `parents[n]` itself.

### Operational scripts

`backend/scripts/` holds manually-run scripts, never imported by the
application. `backend/scripts/verify_dataset.py` composes the same
`DatasetManager` / `IBMAMLDataSource` / `CsvChunkLoader` the application uses —
it introduces no parallel loader and no second downloader — and streams the real
dataset to produce a measured verification report. It downloads nothing. Its
inspection logic is unit-tested offline in `backend/tests/unit/scripts/`.

