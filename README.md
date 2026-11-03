# FedSecureAI

**Privacy-Preserving Federated Deep Learning for Anti-Money Laundering Detection**

> **Project status: PHASE 6 COMPLETE & AUDITED · PHASE 7 (HYBRID ENSEMBLE) READY WITH WARNINGS.**
> - **Dataset Scale**: Evaluated across full 5.08M row IBM HI-Small and full 31.25M row IBM LI-Medium datasets with strict zero-leakage, temporal-preserving splits.
> - **Federated Experiments**: 100% completion (24/24 configurations: 12 HI-Small + 12 LI-Medium across IID / Non-IID partitions, FedAvg / FedProx aggregations, and GRU / LSTM / BiLSTM architectures).
> - **Active Modeling Pipeline**: Strictly restricted to **GRU, LSTM, and BiLSTM**. Legacy Phase B exploratory models (TCN, Transformer, Residual MLP) have been safely quarantined to `archive/legacy_experiments/` with zero regressions (169/169 unit & integration tests passing).
> - **Phase 7 Hybrid Ensemble Decision**: **`READY_WITH_WARNINGS`**. Champion models for both datasets have been selected strictly on validation PR-AUC with zero test leakage. All audit logs, manifests, and plans are formally documented in `phase7_readiness_audit/`.

---

## 1. Project title

FedSecureAI — Privacy-Preserving Federated Deep Learning for Anti-Money
Laundering Detection.

## 2. Problem statement

Money laundering is detected from patterns that span multiple institutions, but
the transaction data revealing those patterns cannot legally or commercially be
pooled. Each bank therefore trains anti-money-laundering (AML) models on its own
narrow slice of activity, where laundering cases are extremely rare, and a
typology it has never seen is a typology it cannot detect. Centralising the data
would improve detection but is blocked by data-protection regulation, banking
secrecy, and competitive risk.

This creates a concrete technical problem: **train a shared AML detection model
across institutions without any institution exposing its raw transaction
records** — while coping with severe class imbalance and with clients whose data
distributions differ substantially from one another.

## 3. Motivation

- **Regulatory reality.** Data-protection and banking-secrecy rules make raw
  transaction sharing impractical. A method that never moves raw data sidesteps
  the blocker instead of arguing with it.
- **Rare-event learning.** Laundering is a tiny fraction of transactions. More
  effective data — even without centralisation — matters more here than in
  balanced problems.
- **Heterogeneity is the hard part.** Banks differ in size, customer base,
  geography and product mix. Federated averaging is known to degrade under
  non-identically-distributed clients; whether a proximal correction such as
  FedProx recovers that loss on realistic AML data is an open, testable question.
- **Sequence structure is underexploited.** Laundering shows up in *sequences* of
  transactions, not single rows, which motivates recurrent architectures over
  the tabular classifiers commonly used as AML baselines.

## 4. Research objective

Determine whether federated learning can train AML transaction-detection models
that are competitive with centralized training, and quantify how the choice of
aggregation strategy interacts with client heterogeneity.

Three axes are compared:

| Axis | Values |
| --- | --- |
| Recurrent architecture | LSTM, GRU, BiLSTM |
| Federated strategy | FedAvg, FedProx |
| Client distribution | IID, Non-IID |

Centralized Logistic Regression, Random Forest, LSTM, GRU and BiLSTM runs
provide the non-federated reference points.

**Status: PLANNED.** No experiment has been executed.

## 5. Primary dataset

**IBM Transactions for Anti-Money Laundering (AML)** — the only approved dataset
for this project.

| Field | Value |
| --- | --- |
| Dataset ID (project) | `ibm_transactions_aml` |
| Name | IBM Transactions for Anti-Money Laundering (AML) |
| Provider | IBM |
| Purpose | Anti-Money Laundering transaction detection |
| Dataset identifier (Kaggle) | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` |
| Selected variant | `HI-Small` |
| Expected file | `HI-Small_Trans.csv` |
| Planned local path | `data/raw/ibm_aml/` |

It must **not** be substituted with IBM AMLSim, a credit-card fraud dataset, or
any other AML dataset.

Only the `HI-Small` variant is declared, because it is the only variant whose
exact filename has been confirmed. Other variants are added to the registry only
after their filenames are verified against a real download.

## 6. Dataset source

- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The authoritative declaration is
[datasets/dataset_registry.yaml](datasets/dataset_registry.yaml).

## 7. Current dataset status

**`configured_not_downloaded`**

The dataset is registered but **no bytes exist locally**. `data/raw/` holds only
a `.gitkeep`. No synthetic stand-in has been generated — an empty directory
states the truth; a fabricated CSV would not.

A download **was attempted** on this machine with the explicit command and failed:

```
Error: Missing Kaggle credentials: KAGGLE_USERNAME, KAGGLE_KEY are not set.
Copy .env.example to .env and fill them in, or export them in the environment.
Credentials are never stored in the repository.
```

Exit code 1. No credential value appeared in the output. The download path itself
is implemented and tested; it needs the operator's own Kaggle token, which this
project will never contain.

Check the status yourself — this command **never downloads anything**:

```bash
python main.py dataset-info
```

```
========================================
FedSecureAI Dataset Information
========================================

Dataset:
IBM Transactions for Anti-Money Laundering (AML)

Source:
Kaggle

Dataset ID:
ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Variant:
HI-Small

Local Path:
data/raw/ibm_aml/

Status:
CONFIGURED - NOT DOWNLOADED

Download:
Not performed
```

The registry defines six states:

| Status | Meaning |
| --- | --- |
| `configured_not_downloaded` | Declared in the registry; nothing on disk. **← current state** |
| `downloading` | Transfer in progress; any files present are incomplete |
| `downloaded` | Raw files present and non-empty; not yet checked |
| `validated` | Raw files present and passing schema/integrity checks; checksums recorded |
| `processed` | Validated data transformed into training-ready artefacts |
| `failed` | Last download or validation attempt failed; files present are untrustworthy |

`status` in the registry is a **declaration, not an observation**. The
`DatasetManager` inspects the filesystem and reports a reconciled status, so a
stale entry can never make the project claim data it does not have. A registry
that exists does not mean a dataset that exists.

File sizes, row counts and checksums are recorded **only after being measured**
on a real local copy. They are `null` in the registry today.

## 7a. Dataset access module

**Status: IMPLEMENTED (this phase).**

```
DatasetManager                       application/services/
  ├── DatasetRegistryPort            application/ports/output/
  │     └── YamlDatasetRegistry      infrastructure/data/sources/
  └── DatasetDownloaderPort          application/ports/output/
        └── BaseDatasetDownloader    infrastructure/data/downloaders/   (template)
              └── KaggleDownloader   infrastructure/data/downloaders/
                    └── KaggleCredentialsProvider → environment
```

| Component | Responsibility |
| --- | --- |
| `DatasetManager` | Reads the declaration, observes the filesystem, reconciles the two, decides **whether** to download, and verifies afterwards that files really arrived. |
| `YamlDatasetRegistry` | Parses `datasets/dataset_registry.yaml` into a validated `DatasetDescriptor`. Read-only. Rejects any credential-like key. |
| `DatasetDownloaderPort` | The single downloader abstraction the application depends on. |
| `BaseDatasetDownloader` | Template-method **implementation** of that port (not a competing interface): destination validation + post-transfer verification, with one `_fetch()` hook. |
| `KaggleDownloader` | The only module aware of the Kaggle SDK. Implements `_fetch()`. |
| `KaggleCredentialsProvider` | The only module that reads Kaggle credentials. |

`DatasetManager` API: `get_dataset_info()`, `is_dataset_available()`,
`get_status()`, `get_local_path()`, `download_dataset(force=False)`. Only the
last can transfer data, and only when called explicitly. Metadata operations
never touch the network.

Two CLI commands expose it. `dataset-info` is read-only and cannot transfer
anything. `dataset-download` is **the only command in the project that transfers
data** — running it is the request to do so.

**Authentication.** `DatasetManager` does not own Kaggle authentication and never
sees a credential. `KaggleDownloader` asks a credential provider at the moment a
transfer starts; the default provider reads `KAGGLE_USERNAME` and `KAGGLE_KEY`
from the environment. Missing variables produce an error that names the
variables and never contains a value; a key that is read is registered with the
log-redaction filter immediately. **No credential is ever logged, committed, or
stored in the registry** — the registry loader hard-fails if it finds a
credential-like key.

Full detail: [docs/dataset/](docs/dataset/README.md).

## 7b. Data ingestion module

**Status: IMPLEMENTED (this phase).**

```
IBM AML Dataset → Dataset Source → Memory-Safe CSV Loader →
Schema Validator → Validated Chunks → Future Preprocessing
```

```
DatasetIngestionService                  application/services/
  ├── TransactionDataSourcePort          application/ports/output/
  │     └── IBMAMLDataSource             infrastructure/data/sources/
  │           └── DatasetManager         (§7a — identity and local path only)
  ├── ChunkedDataLoaderPort              application/ports/output/
  │     └── CsvChunkLoader               infrastructure/data/loaders/
  └── SchemaValidatorPort                application/ports/output/
        └── DataFrameSchemaValidator     infrastructure/data/validators/
```

| Component | Responsibility |
| --- | --- |
| `IBMAMLDataSource` | Confirms the configured dataset really is the approved IBM AML dataset, and resolves the expected local file. Downloads nothing, holds no credential, parses nothing. |
| `CsvChunkLoader` | Streams a CSV in chunks. Never reads a whole file, never concatenates, never renames a column. |
| `TableSchema` / `ColumnSpec` | Declare required/optional columns, expected types, allowed values, nullability and the unexpected-column policy. pandas-free. |
| `DataFrameSchemaValidator` | Checks a chunk against a schema and returns findings. Never mutates the caller's DataFrame and performs no preprocessing. |
| `DatasetIngestionService` | Composes the three ports and owns one decision: abort (strict) or report and continue on an invalid chunk. |

**Memory safety is structural, not measured.** The loader's only parser call is
chunked, so no code path exists that reads the whole file; a test records every
`read_csv` call and asserts each one carried `chunksize`. **No RAM figure or
reduction percentage is claimed anywhere in this project.**

**The IBM AML column schema is not defined.**
`build_ibm_aml_transaction_schema()` raises `SchemaNotVerifiedError` carrying the
procedure for defining it, because the real column names have not been read from
the real file and this project does not guess them. The pipeline is exercised
against small invented fixtures whose columns (`record_id`, `unit`, `value`, …)
are deliberately unlike anything in an AML dataset.

Full detail: [docs/dataset/README.md §14–16](docs/dataset/README.md).

## 8. Planned architecture

Clean Architecture with dependencies pointing inward only:

```
        interfaces  (API · dashboard · CLI)
              │
              ▼
        application  (use cases · services · ports)
              │
              ▼
          domain  (entities · value objects · exceptions)
              ▲
              │
       infrastructure  (adapters implementing the ports)
```

| Layer | Responsibility | Depends on |
| --- | --- | --- |
| `domain/` | Business concepts, independent of any framework | nothing |
| `application/` | Use cases orchestrating domain objects; declares ports | `domain` |
| `infrastructure/` | Concrete adapters: data, models, federated, persistence, explainability | `domain`, `application` |
| `interfaces/` | Entry points that drive use cases | `application` |

Adapters are bound to ports at the composition root
([backend/composition_root.py](backend/composition_root.py)) — the only place
permitted to know both sides. The root [main.py](main.py) is a thin shim that
puts `backend/` on the path and delegates, so `python main.py <command>` still
works. Design constraints in force: SOLID,
dependency inversion, separation of concerns, type hints throughout,
configuration over hard-coded values, and reproducibility via a single seed.

**Status: layer directories exist as empty packages**, except for the
configuration foundation in [backend/infrastructure/utils/](backend/infrastructure/utils/), the
dataset-access module described in §7a and the ingestion module described in §7b
— together the first vertical slices through all four layers.

## 9. Planned data pipeline

**Status: only the first five stages exist.**

```
Kaggle download → raw files → schema validation → cleaning → feature
engineering → train/val/test split → sequence windowing → client partitioning
→ federated training → evaluation
```

| Stage | Destination package | Status |
| --- | --- | --- |
| Registry lookup | `infrastructure/data/sources/` | **IMPLEMENTED** |
| Kaggle download | `infrastructure/data/downloaders/` | **IMPLEMENTED** (not yet executed against Kaggle) |
| File loading | `infrastructure/data/loaders/` | **IMPLEMENTED** (`CsvChunkLoader`, chunked) |
| Schema definition | `infrastructure/data/schemas/` | **MECHANISM IMPLEMENTED — IBM AML COLUMNS NOT DEFINED** (blocked on the real download) |
| Validation | `infrastructure/data/validators/` | **IMPLEMENTED** (`DataFrameSchemaValidator`) |
| Data quality report | `infrastructure/data/validators/` | NOT YET IMPLEMENTED |
| Cleaning / encoding / scaling | `infrastructure/data/preprocessors/` | NOT YET IMPLEMENTED |
| Feature engineering | `infrastructure/data/features/` | NOT YET IMPLEMENTED |
| Split + sequence windowing | `infrastructure/data/splitters/` | NOT YET IMPLEMENTED |
| IID / Non-IID partitioning | `infrastructure/data/partitioners/` | NOT YET IMPLEMENTED |

## 10. Planned centralized models

**Status: PLANNED — none implemented.**

| Model | Adapter location | Role |
| --- | --- | --- |
| Logistic Regression | `infrastructure/models/sklearn/` | Interpretable linear baseline |
| Random Forest | `infrastructure/models/sklearn/` | Strong tabular baseline |
| LSTM | `infrastructure/models/pytorch/` | Sequential baseline |
| GRU | `infrastructure/models/pytorch/` | Lighter recurrent alternative |
| BiLSTM | `infrastructure/models/pytorch/` | Bidirectional context |

## 11. Planned federated strategies

**Status: PLANNED — none implemented.**

| Strategy | Location | Rationale |
| --- | --- | --- |
| FedAvg | `infrastructure/federated/aggregation/` | Standard weighted-average baseline |
| FedProx | `infrastructure/federated/aggregation/` | Proximal term limiting client drift under heterogeneity |

Orchestration will use the Flower framework
(`infrastructure/federated/flower/`), with client and server roles in
`infrastructure/federated/clients/` and `infrastructure/federated/server/`.
Flower is **not yet a dependency** — it is added when that code is written.

## 12. IID vs Non-IID experiment

**Status: PLANNED — not executed.**

Clients are simulated by partitioning the single IBM AML dataset:

| Partitioning | Construction | Purpose |
| --- | --- | --- |
| IID | Random shuffle, equal shards, class ratio preserved per client | Upper-bound control |
| Non-IID | Skewed label and/or feature distribution across clients (e.g. Dirichlet-controlled) | Approximates real inter-bank divergence |

FedAvg and FedProx are run under both partitionings for each of LSTM, GRU and
BiLSTM, holding seeds, rounds and local epochs fixed so the partitioning is the
only variable.

## 13. Evaluation metrics

**Status: PLANNED — metric code not implemented.**

| Metric | Why it is included |
| --- | --- |
| Precision | Cost of false alerts on investigator workload |
| Recall | Fraction of laundering activity actually caught |
| F1-score | Balance of the two under imbalance |
| ROC-AUC | Threshold-independent ranking quality |
| PR-AUC | More informative than ROC-AUC for rare positives — the headline metric |
| Confusion matrix | Absolute error counts at the operating threshold |
| Training time | Wall-clock cost |
| Communication rounds | Rounds to reach a target metric — the federated bottleneck |
| Convergence behaviour | Stability and shape of the metric-vs-round curve |

Accuracy is deliberately **not** a headline metric: with extreme imbalance it is
uninformative.

## 14. Project roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project skeleton, configuration foundation, dataset registry | **COMPLETE** |
| 1 | Dataset registry access + Dataset Manager (configure, inspect, download, verify) | **COMPLETE — dataset not downloaded** |
| 2 | Data source, memory-safe loader, schema mechanism, validator | **COMPLETE — IBM AML columns still undefined; data quality report deferred** |
| 2.5 | Explicit download command, `.env` loading, memory-safe real-data verification script | **TOOLING COMPLETE — BLOCKED: dataset not obtained, no Kaggle credentials on this machine** |
| 3 | Preprocessing + feature engineering | NOT STARTED — next, after the schema is verified from the real file |
| 4 | Splitting + sequence construction | NOT STARTED |
| 5 | Centralized baselines (Logistic Regression, Random Forest) | NOT STARTED |
| 6 | Centralized LSTM / GRU / BiLSTM | NOT STARTED |
| 7 | IID and Non-IID partitioners | NOT STARTED |
| 8 | Flower integration: clients, server, FedAvg | NOT STARTED |
| 9 | FedProx | NOT STARTED |
| 10 | Evaluation, metrics, visualisations | NOT STARTED |
| 11 | Full experiment matrix + results | NOT STARTED |
| 12 | SHAP explainability | NOT STARTED |
| 13 | FastAPI service | NOT STARTED |
| 14 | Streamlit dashboard | NOT STARTED |
| 15 | CI pipeline, packaging, documentation completion | NOT STARTED |

Phases are implemented and reviewed **one module at a time**.

## 15. Installation

Requires Python 3.11 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements/dev.txt      # development (includes tests + lint)
# pip install -r requirements/prod.txt   # runtime only

# 3. Configure the environment
cp .env.example .env                     # Windows: copy .env.example .env

# 4. Verify the installation
python main.py                 # prints available commands
python main.py dataset-info    # prints dataset configuration + local status
```

Neither command downloads anything and neither trains anything.

### Obtaining the dataset

```bash
python main.py dataset-download       # the only command that transfers data
python backend/scripts/verify_dataset.py   # inspects the real file, memory-safely
```

`dataset-download` requires your own Kaggle credentials in `.env` or the
environment (see below); running it *is* the request to transfer data. Nothing
else in the project can cause a download. `verify_dataset.py` then streams the
real file in chunks — it never calls an unrestricted `pd.read_csv(path)` — and
writes a measured report to `reports/` (git-ignored). If the file is absent it
says so and exits 2 rather than inventing anything. Full procedure:
[docs/dataset/README.md §13](docs/dataset/README.md).

### Environment variables

`.env.example` documents every variable and contains **placeholders only**. Copy
it to `.env` (git-ignored) and fill in your own values. `.env` is loaded at
startup by `backend/infrastructure/utils/env.py`, before configuration is read; an absent
file is not an error, an exported shell variable takes precedence over the file, a
blank value counts as unset, and **no value from the file is ever logged**.

| Variable | Required for | Notes |
| --- | --- | --- |
| `KAGGLE_USERNAME` | `dataset-download` only | Read by the credential provider at download time. Not needed for `dataset-info` or for tests. |
| `KAGGLE_KEY` | `dataset-download` only | Your Kaggle API token, from <https://www.kaggle.com/settings> → API → Create New Token. Never logged, never committed, never stored in the registry. |

**Never** hardcode these in source, in `config/*.yaml`, or in
`datasets/dataset_registry.yaml` — the registry loader rejects the file outright
if it finds a credential-like key.

Requirement files are layered so no package is pinned twice:

| File | Contents |
| --- | --- |
| `requirements/base.txt` | Runtime dependencies of the current code |
| `requirements/test.txt` | `base` + pytest tooling |
| `requirements/dev.txt` | `test` + ruff and mypy |
| `requirements/prod.txt` | `base` only |

The remaining ML stack (numpy, scikit-learn, torch, flwr, shap, fastapi,
streamlit) is **intentionally absent**. Each package is added in the phase whose
code imports it, so the dependency list never overstates what the project does.
`pandas` is declared in `base.txt` because this phase's chunked loader and
validator use it. `kaggle` is declared there too because the downloader needs it,
but it is imported **lazily** — the test suite and `dataset-info` both work with
it uninstalled.

## 16. Testing

```bash
pytest                                  # full suite
pytest -m unit                          # unit tests only
pytest -m integration                   # registry-consistency tests
pytest --cov                            # with coverage
ruff check . && mypy .                  # lint and type-check
```

**The dataset is configured as the primary project dataset but is not required to
be downloaded for unit testing.** The suite needs **no network access, no Kaggle
credentials, no Kaggle SDK and no dataset**, and that is enforced structurally:
`DatasetManager` takes its registry and downloader by constructor injection,
metadata tests inject a downloader that fails the test if it is ever called, and
every ingestion test writes its own small CSV into `tmp_path`.

Stub files written by test doubles contain the literal string `stub`. Ingestion
fixtures are small invented CSVs whose columns (`record_id`, `unit`, `value`, …)
are chosen so they cannot be mistaken for IBM AML columns — a test asserts the
fixture header contains no AML-sounding name, and another asserts no file named
`HI-Small*` is ever created. **No synthetic transactions, schema or dataset
statistics are presented anywhere as real IBM AML data.**

Current coverage: the configuration loader, the dataset registry, the dataset
manager, the downloader template, credential handling and log redaction, `.env`
loading, the Kaggle adapter's authentication failure modes, the dataset CLI
(including `dataset-download`), the schema value objects, the chunked CSV loader,
the schema validator, the validation DTOs, the IBM AML data source, the ingestion
service, the end-to-end ingestion pipeline, and the verification script's
inspection logic. Tests are added alongside the functionality they exercise — the
suite is not padded with tests for code that does not exist yet.

**427 tests pass**; `ruff check .` and `mypy .` are clean across 124 files.

## 17. Project structure

```
Federate_Learning/
│
├── backend/                    # The entire Python application
│   ├── domain/                 # Business concepts (framework-independent)
│   │   ├── entities/           #   objects with identity
│   │   ├── value_objects/      #   dataset_descriptor.py, column_spec.py,
│   │   │                       #   table_schema.py
│   │   ├── enums/              #   dataset_status, dataset_source,
│   │   │                       #   column_data_type, unexpected_column_policy,
│   │   │                       #   validation_severity, validation_issue_kind
│   │   ├── events/             #   domain events
│   │   ├── exceptions/         #   base.py, dataset.py, download.py, data.py
│   │   └── interfaces/         #   ports the domain requires
│   │
│   ├── application/            # Use-case orchestration
│   │   ├── use_cases/
│   │   ├── services/           #   dataset_manager.py,
│   │   │                       #   dataset_ingestion_service.py
│   │   ├── dto/                #   dataset_info.py, download.py, validation.py
│   │   └── ports/{input,output}/   # dataset_registry, dataset_downloader,
│   │                               # transaction_data_source, data_loader,
│   │                               # schema_validator
│   │
│   ├── infrastructure/         # Concrete adapters
│   │   ├── data/               #   downloaders/, sources/, loaders/, schemas/,
│   │   │                       #   validators/ implemented; preprocessors,
│   │   │                       #   features, partitioners, splitters planned
│   │   ├── models/{base,sklearn,pytorch}/
│   │   ├── federated/{flower,clients,server,aggregation}/
│   │   ├── persistence/{sqlite,cache}/
│   │   ├── explainability/shap/
│   │   └── utils/              #   config_loader, paths, logging_config, env
│   │
│   ├── interfaces/             # Backend entry points
│   │   ├── api/{routes,middleware,schemas}/
│   │   └── cli/commands/       #   app.py, dataset_commands.py
│   │                           #   (dataset-info, dataset-download)
│   │
│   ├── config/                 # Backend YAML configuration (base.yaml)
│   ├── scripts/                # operational scripts
│   │   └── verify_dataset.py   #   memory-safe real-dataset inspection
│   ├── tests/
│   │   ├── unit/{domain,application,infrastructure,interfaces,scripts}/
│   │   ├── integration/
│   │   ├── e2e/
│   │   └── fixtures/
│   └── composition_root.py     # wires adapters to ports
│
├── frontend/                   # Presentation only
│   └── interfaces/
│       └── dashboard/{pages,components}/
│
├── datasets/                   # dataset_registry.yaml
├── data/{raw,processed,partitions,cache}/      # git-ignored; holds the real dataset
├── models/{checkpoints,artifacts}/             # git-ignored artefacts
├── experiments/{configs,runs,results}/         # configs tracked, output ignored
├── evaluation/{metrics,reports,visualizations}/# generated output
├── docs/{architecture,dataset,experiments,api,research}/
├── requirements/{base,dev,test,prod}.txt
├── logs/                       # git-ignored
├── reports/                    # git-ignored
├── .github/workflows/          # CI (not yet configured)
│
├── main.py                     # thin shim → backend/composition_root.py
├── README.md
├── .gitignore
├── .env.example
├── pytest.ini
└── pyproject.toml
```

`data/`, `datasets/` and `reports/` sit at the repository root on purpose: they
are shared across the boundary, so placing them inside either area would give one
side ownership of the other's inputs. Only `backend/` is put on `sys.path`, which
is why every `import domain` / `import application` / `import infrastructure` /
`import interfaces` statement is unchanged by the split.

## 18. Research contribution

**Status: PLANNED — nothing demonstrated yet.**

The intended contributions:

1. A controlled comparison of **FedAvg vs FedProx** across three recurrent
   architectures on a single realistic AML transaction dataset, with seeds,
   rounds and local epochs held fixed.
2. Quantification of the **IID → Non-IID degradation** for each strategy, rather
   than assuming FedProx helps.
3. **Imbalance-aware reporting** with PR-AUC as the headline metric alongside
   ROC-AUC, instead of accuracy-led evaluation.
4. **Cost accounting** for federation: communication rounds and training time
   reported next to detection quality, so the privacy benefit can be weighed
   against its price.
5. A **reproducible, layered reference implementation** in which every reported
   number traces back to a stored config and seed.

## 19. Limitations

Stated up front, since they bound what the eventual results can claim:

1. **Clients are simulated.** Partitioning one dataset is not the same as real
   banks with genuinely different systems, populations and labelling practices.
2. **Single dataset.** Findings will be specific to IBM Transactions for AML;
   generalisation to other AML data is untested.
3. **Synthetic origin.** The IBM AML dataset is generated, so its laundering
   typologies are cleaner and more consistent than real-world activity.
4. **Federated learning alone is not full privacy.** Gradient updates can leak
   information. Differential privacy and secure aggregation are **not** in the
   current scope, so "privacy-preserving" here means "no raw data leaves the
   client", not "provably private".
5. **No adversarial threat model.** Malicious clients, model poisoning and
   Byzantine robustness are out of scope.
6. **Labels are assumed correct.** Real AML labels are delayed, incomplete and
   partly wrong; that noise is not modelled.
7. **Simulation, not deployment.** Network failure, stragglers, client dropout
   and cross-institution latency are not measured.
8. **No regulatory validation.** Nothing here has been assessed against model
   risk management or supervisory requirements for production AML systems.

## 20. Future enhancements

Beyond the current roadmap, in rough priority order:

1. **Differential privacy** on client updates, with the resulting privacy/utility
   trade-off measured rather than asserted.
2. **Secure aggregation** so the server cannot inspect individual updates.
3. **Additional strategies** — FedAdam, FedYogi, SCAFFOLD, personalised
   federated learning.
4. **Graph-based modelling** of transaction networks (GNNs), since laundering is
   inherently relational.
5. **Attention and transformer architectures** for long sequences.
6. **Concept drift handling** for typologies that evolve after deployment.
7. **Robustness to malicious clients** via Byzantine-tolerant aggregation.
8. **Cost-sensitive thresholding** tied to investigator capacity instead of a
   fixed 0.5 cutoff.
9. **Cross-dataset validation** once a second approved AML dataset is available.
10. **Deployment hardening** — authentication, monitoring, model versioning,
    audit trails.

---

## Git safety

Nothing sensitive and nothing large enters version control:

| Never committed | Mechanism |
| --- | --- |
| Kaggle credentials | Read only from the environment; `.env`, `.env.*` and `kaggle.json` are git-ignored (`.env.example` holds placeholders) |
| Credentials in the registry | The registry loader **rejects** `datasets/dataset_registry.yaml` outright if it contains any credential-like key |
| Downloaded datasets | `data/raw/`, `data/processed/`, `data/partitions/`, `data/cache/` ignored except their `.gitkeep` |
| CSV / ZIP / archives | `*.csv`, `*.tsv`, `*.parquet`, `*.zip`, `*.tar*`, `*.gz`, `*.pkl`, `*.pt` … ignored **anywhere** in the tree |
| Model artefacts, run output, logs, reports | `models/`, `experiments/runs/`, `evaluation/`, `logs/`, `reports/` ignored except `.gitkeep` |

Kaggle API keys, passwords and authentication tokens are **never logged**: a
redaction filter scrubs every registered secret from every log record.

## Documentation

| Area | Document |
| --- | --- |
| Architecture | [docs/architecture/](docs/architecture/README.md) |
| Dataset | [docs/dataset/](docs/dataset/README.md) |
| Experiments | [docs/experiments/](docs/experiments/README.md) |
| API | [docs/api/](docs/api/README.md) |
| Research | [docs/research/](docs/research/README.md) |
| Configuration | [backend/config/](backend/config/README.md) |

## Reporting policy

This project does not claim unfinished work as complete. A component is
described as implemented only once it exists, runs and is tested. Dataset
statistics are recorded only once measured. Experimental results are reported
only once reproducible from a stored config.
