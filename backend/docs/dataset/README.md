# Dataset Documentation

> **REAL DATASET DOWNLOADED: YES.**
> **SCHEMA VERIFICATION STATUS: VERIFIED — MEASURED FROM THE REAL FILE.**
> **PREPROCESSING STATUS: COMPLETE — RUN ON THE REAL FILE (§18).**
>
> `data/raw/ibm_aml/HI-Small_Trans.csv` is present locally: 475,664,283 bytes,
> SHA-256 `b19d39f5…c5b040`, 5,078,345 data rows, 11 columns. Every figure in
> this document was measured from that file by
> [backend/scripts/verify_dataset.py](../../backend/scripts/verify_dataset.py)
> (§13.3) or produced by the preprocessing run (§18); none is recalled or
> estimated. The full measurement is in
> [reports/dataset_verification_HI-Small.md](../../reports/dataset_verification_HI-Small.md).
>
> Unit tests still do **not** require the download. They run against small
> synthetic fixtures (§10, §16), and the tests that touch the real file skip
> themselves when it is absent.

---

## 1-6. The primary dataset

| Field | Value |
| --- | --- |
| **Dataset name** | IBM Transactions for Anti-Money Laundering (AML) |
| Project dataset ID | `ibm_transactions_aml` |
| Provider | IBM |
| **Source** | Kaggle |
| **Dataset identifier** (Kaggle) | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` |
| Source URL | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml |
| **Selected variant** | `HI-Small` |
| Expected file for that variant | `HI-Small_Trans.csv` |
| **Local storage location** | `data/raw/ibm_aml/` (project-relative, planned) |
| **Status** | `configured_not_downloaded` |
| Purpose | Anti-Money Laundering transaction detection |

This is the **only approved dataset** for the project. It must not be
substituted with IBM AMLSim, a credit-card fraud dataset, or any other AML
dataset. The authoritative declaration lives in
[datasets/dataset_registry.yaml](../../datasets/dataset_registry.yaml).

### Variant selection

The Kaggle distribution ships several size / laundering-ratio variants. Only
`HI-Small` is declared, because it is the only variant whose exact filename has
been confirmed for this project. Additional variants will be added to
`supported_variants` **only after their filenames are verified against a real
download** — guessing a filename would produce a registry that lies.

### Status vocabulary

| Status | Meaning |
| --- | --- |
| `configured_not_downloaded` | Declared in the registry; **no bytes exist on this machine**. `local_path` is a planned destination only. |
| `downloading` | A transfer is in progress; any files present must be treated as incomplete. |
| `downloaded` | Every file in `expected_files` is present and non-empty, but nothing has been validated. |
| `validated` | Files are present **and** have passed schema / dtype / integrity checks. |
| `processed` | Validated raw files have been transformed into training artefacts. |
| `failed` | The last download or validation attempt failed; files present are untrustworthy. |

The registry's `status` is a **declaration, not an observation**. The
`DatasetManager` inspects the filesystem and reports a reconciled status, so a
stale registry entry can never make the project claim data it does not have. If
the registry says `downloaded` but the files are gone, the reported status is
downgraded and a warning is logged.

**Current state: nothing has been downloaded.** `data/raw/` contains only a
`.gitkeep`. No synthetic or placeholder stand-in has been generated — an empty
directory is a truthful representation of the current state; a fake
`HI-Small_Trans.csv` is not.

---

## 7. `DatasetManager` responsibility

[`application/services/dataset_manager.py`](../../backend/application/services/dataset_manager.py)

The `DatasetManager` is the single application-layer entry point for dataset
questions. It **owns**:

- reading the dataset's declaration from the registry (through a port);
- resolving `local_path` against the project root;
- observing the filesystem to determine which expected files are actually
  present and non-empty;
- reconciling the declared status with that observation;
- deciding **whether** a download is needed, and requesting one explicitly;
- verifying, after a download, that the files really arrived.

It deliberately does **not** own:

- how the registry is stored (YAML is an infrastructure detail);
- how bytes are transferred (HTTP, Kaggle SDK, anything else);
- **Kaggle authentication** — it never reads, holds, or passes a credential.

### Public API

| Method | Contract |
| --- | --- |
| `get_dataset_info()` | Full metadata + observed local state. **Never downloads.** |
| `is_dataset_available()` | `True` only if every expected file exists and is non-empty. |
| `get_status()` | The reconciled status (registry ∧ filesystem). **Never downloads.** |
| `get_local_path()` | Absolute path to the dataset directory. Does not create it. |
| `download_dataset(force=False)` | The **only** method that can transfer data, and only when called explicitly. Skips the transfer if the dataset is already available unless `force=True`. |

---

## 8. Downloader responsibility

The transfer mechanism sits behind one port:

```
DatasetManager
   └── DatasetDownloaderPort           (application/ports/output/dataset_downloader.py)
         └── BaseDatasetDownloader     (infrastructure/data/downloaders/base_downloader.py)
               └── KaggleDownloader    (infrastructure/data/downloaders/kaggle_downloader.py)
                     └── Kaggle API
```

- **`DatasetDownloaderPort`** is the *single* downloader abstraction the
  application depends on. It declares `source` and `download(request)`.
- **`BaseDatasetDownloader`** is a template-method **implementation** of that
  port — not a competing interface. It provides the behaviour every adapter must
  share (destination validation, and independent post-transfer verification that
  the expected files exist and are non-empty) and leaves exactly one hook,
  `_fetch()`, for subclasses.
- **`KaggleDownloader`** implements `_fetch()` only. It is the sole module that
  knows the Kaggle SDK exists.

The downloader receives a `DownloadRequest` value object (remote id,
destination, expected files, variant, overwrite flag) and returns a
`DownloadOutcome`. It never reads the registry and never decides *whether* a
download should happen — that is the manager's job.

A downloader that reports success without delivering files causes
`IncompleteDownloadError`. Success is verified, not trusted.

Archives are extracted defensively: an entry whose path would resolve outside the
destination directory (`../`, absolute paths) is refused rather than written.

---

## 9. Authentication approach

**The `DatasetManager` does not own Kaggle authentication.** Credentials are
resolved at the very edge of the system, as late as possible:

```
DatasetManager  →  DatasetDownloaderPort  →  KaggleDownloader  →  KaggleCredentialsProvider  →  environment
   (no secrets)        (no secrets)            (passes through)      (the only reader)
```

- [`infrastructure/data/downloaders/credentials.py`](../../backend/infrastructure/data/downloaders/credentials.py)
  is the **only** module in the project that reads Kaggle credentials.
- `KaggleCredentialsProvider` is a `Protocol`, so an alternative source (a
  secrets manager, a vault, a CI injector) can be substituted without touching
  any other module.
- The default `EnvironmentKaggleCredentialsProvider` reads two environment
  variables:

  | Variable | Meaning |
  | --- | --- |
  | `KAGGLE_USERNAME` | Kaggle account username |
  | `KAGGLE_KEY` | Kaggle API token |

- Credentials are read **only inside `download_dataset()`**, at the moment a
  transfer is actually attempted. `dataset-info` and every unit test run without
  them.
- If either variable is absent, `MissingCredentialsError` names the **missing
  variables** and never contains a value.
- `KaggleCredentials.__repr__` / `__str__` mask the key, so an accidental
  interpolation cannot leak it.
- The moment a key is read it is registered with `register_secret()`, and
  `SecretRedactingFilter` scrubs it from every log record
  ([`infrastructure/utils/logging_config.py`](../../backend/infrastructure/utils/logging_config.py)).
- Kaggle API keys, passwords and tokens are **never logged**.
- The Kaggle SDK aborts the process with `exit(1)` when it can find no usable
  credentials. `SystemExit` derives from `BaseException`, so `KaggleDownloader`
  catches it explicitly and re-raises `AuthenticationError` — a failed
  authentication surfaces through the port's error contract instead of killing
  the host process. Regression-tested.

Copy [`.env.example`](../../.env.example) to `.env` and fill in your own values.
`.env` is git-ignored and is loaded at startup by
[`backend/infrastructure/utils/env.py`](../../backend/infrastructure/utils/env.py), before
configuration is read, so `${VAR}` placeholders in the YAML and the credential
provider both see it. **No value read from the file is ever logged** — only the
file's path and how many variables it defined. An exported shell variable takes
precedence over the file, and a blank value counts as unset.

---

## 10. Offline testing strategy

**The dataset is configured as the primary project dataset but is not required
to be downloaded for unit testing.**

The unit test suite requires **no network access, no Kaggle credentials, no
Kaggle SDK installation, and no dataset**. That is enforced structurally rather
than by convention:

- `DatasetManager` receives its registry and downloader by constructor
  injection, so tests supply doubles from
  [`backend/tests/fixtures/`](../../backend/tests/fixtures/):
  - `InMemoryDatasetRegistry` — a registry with no file I/O;
  - `RecordingDownloader` — records requests and writes small stub files;
  - `ExplodingDownloader` — fails the test if `download()` is ever reached;
  - `FailingDownloader` / `LyingDownloader` — network, auth and
    "success without files" failure modes.
- Metadata tests inject `ExplodingDownloader`, which turns "a metadata
  operation triggered a download" into an immediate test failure.
- Stub files contain the string `stub`. They are **not** fabricated IBM AML
  data: no synthetic transactions, schema or statistics are generated anywhere
  in this repository.
- The Kaggle SDK is imported **lazily**, inside the download call. The full
  suite and `python main.py dataset-info` both pass with the `kaggle` package
  absent. Tests that need to exercise authentication failures inject a fake
  `kaggle.api.kaggle_api_extended` module into `sys.modules`, so they behave
  identically whether or not the real SDK is installed.
- The verification script's inspection logic is unit-tested against tiny invented
  CSVs written by the tests themselves
  ([`backend/tests/unit/scripts/`](../../backend/tests/unit/scripts/)). Those fixtures are
  labelled TEST DATA and describe nothing about the real dataset; the real file is
  never read by the suite.
- Integration tests read the real committed `dataset_registry.yaml` — metadata
  only. Reading metadata cannot trigger a transfer.

---

## 11. No credentials committed to Git

- `dataset_registry.yaml` contains **no credentials**, and the registry loader
  **rejects the file outright** if it contains any credential-like key
  (`kaggle_key`, `kaggle_username`, `api_key`, `api_token`, `token`, `password`,
  `secret`, `credentials`, `auth`). This is a hard failure, not a warning.
- `.env`, `.env.*` and `kaggle.json` are git-ignored; only `.env.example`,
  which holds placeholders, is committed.
- No credential appears in any source file, config file, test or log.

## 12. No dataset committed to Git

`.gitignore` excludes `data/raw/`, `data/processed/`, `data/partitions/` and
`data/cache/` (keeping only their `.gitkeep` files) and additionally excludes
`*.csv`, `*.zip`, `*.parquet`, `*.pkl` and similar formats **anywhere** in the
tree. Downloaded datasets, archives and derived artefacts never enter version
control.

---

## 13. Download and verification process

`dataset-download` is **IMPLEMENTED**. It is the only command in the project that
transfers data; running it *is* the request to do so. Nothing else — not
`dataset-info`, not an import, not the test suite, not the verification script —
can cause a download.

### 13.1 One-time credential setup

Credentials are supplied by the operator, locally, and never enter the
repository. Neither value is printed by any command or written to any log.

1. Install the Kaggle SDK: `pip install -r requirements/base.txt`.
2. Accept the dataset's terms once, on the Kaggle website.
3. Create the token: <https://www.kaggle.com/settings> → **API** → **Create New
   Token**. The downloaded `kaggle.json` contains both values.
4. Copy `.env.example` to `.env` and fill in `KAGGLE_USERNAME` and `KAGGLE_KEY`
   — or export them in the shell instead.

`.env` is loaded at startup by `backend/infrastructure/utils/env.py`, before
configuration is read. An absent file is not an error. **An exported shell
variable always wins over the file**, so a stale `.env` cannot silently override
a deliberate export. Blank values count as unset and fail exactly as a missing
file does.

`.env`, `.env.*` (except `.env.example`) and `kaggle.json` are all git-ignored.

### 13.2 Downloading

```bash
python main.py dataset-download           # skips an existing local copy
python main.py dataset-download --force   # re-downloads regardless
```

The command prints what it is about to do — dataset, remote id, variant, files,
target directory — then calls `DatasetManager.download_dataset()`, which:

- returns immediately if the expected files are already present (unless
  `--force`);
- resolves credentials from the environment **at that moment**, failing with a
  named-variable error if they are absent;
- authenticates and fetches **only** the files for `selected_variant`;
- extracts and removes any archive, refusing entries that would escape the
  destination directory;
- verifies every expected file exists and is non-empty, raising
  `IncompleteDownloadError` otherwise.

Failure modes are surfaced through the port's error contract, never as a crash:
`MissingCredentialsError` (nothing configured), `AuthenticationError` (rejected,
or the SDK aborting with `exit(1)` because it found no usable credentials),
`NetworkError`, `DownloadFailedError`, `IncompleteDownloadError`.

### 13.3 Verifying and inspecting the downloaded file

```bash
python backend/scripts/verify_dataset.py
```

An operational script, run manually after a download. It inventories the dataset
directory (so the actual archive layout is observed rather than assumed), locates
the transaction file through the ordinary `IBMAMLDataSource`, probes a 64 KiB
prefix for encoding, delimiter and header presence, then streams the entire file
through `CsvChunkLoader` accumulating only bounded summaries — counts, min/max,
and value counts for columns whose cardinality stays under a cap of 60.

**It never calls an unrestricted `pd.read_csv(path)`.** Peak memory is one chunk
plus the bounded summaries, whatever the file's size.

It writes `reports/dataset_verification_HI-Small.md` (git-ignored) containing the
measured file size, row count, column count, exact column names, observed dtypes,
null counts, numeric ranges, and the full value counts of every low-cardinality
column — which is the evidence a target column would be identified from. Column
groupings in that report are **by observed dtype, not by meaning**; no semantics
are inferred from a column's name.

If the file is absent the script says so and exits 2. It writes no report,
invents no value, and does not fall back to another dataset.

The inspection logic itself is unit-tested offline against tiny invented CSVs
(`backend/tests/unit/scripts/test_verify_dataset.py`) — chunk-boundary agreement, null
accumulation, cardinality abandonment, delimiter and encoding detection, malformed
row reporting — so it is known correct before it meets the real file.

### 13.4 Still planned

Validation populating `local_metadata` — checksums, row counts, file sizes. Those
fields stay `null` until measured.

---

## 14. Data ingestion pipeline

The path from a configured dataset to data a model could be trained on:

```
IBM AML Dataset → Dataset Source → Memory-Safe CSV Loader →
Schema Validator → Validated Chunks → Future Preprocessing
```

Expanded to the modules that implement it:

```
datasets/dataset_registry.yaml
   └── YamlDatasetRegistry            (infrastructure/data/sources/yaml_dataset_registry.py)
         └── DatasetManager           (application/services/dataset_manager.py)
               └── IBMAMLDataSource   (infrastructure/data/sources/ibm_aml_data_source.py)
                     │  resolve_file() → Path
                     ↓
                  CsvChunkLoader      (infrastructure/data/loaders/csv_chunk_loader.py)
                     │  stream_chunks(path, chunk_size) → Iterator[DataFrame]
                     ↓
        DataFrameSchemaValidator      (infrastructure/data/validators/dataframe_schema_validator.py)
                     │  validate_chunk(chunk, chunk_index, source) → ChunkValidationResult
                     ↓
        (chunk, result) pairs         (application/services/dataset_ingestion_service.py)
                     ↓
              Future preprocessing    NOT IMPLEMENTED
```

`DatasetIngestionService` composes the three stages and is the only component
that decides what to do with a chunk that fails validation. Each stage depends on
a port, never on the stage below it:

| Port | File | Implemented by |
| --- | --- | --- |
| `TransactionDataSourcePort` | `application/ports/output/transaction_data_source.py` | `IBMAMLDataSource` |
| `ChunkedDataLoaderPort` | `backend/application/ports/output/data_loader.py` | `CsvChunkLoader` |
| `SchemaValidatorPort` | `application/ports/output/schema_validator.py` | `DataFrameSchemaValidator` |

**Nothing beyond this point exists.** Preprocessing, feature engineering,
train/validation/test splitting, sequence construction, partitioning, models and
federated learning are not implemented.

### 14.1 `IBMAMLDataSource` responsibility

Answers two questions and nothing else: **which** dataset is being read, and
**where** its transaction file is.

It owns:

- reporting the dataset id, Kaggle id, selected variant and expected filename —
  every one of them read from the registry through the `DatasetManager`, so the
  registry stays the single source of truth;
- resolving the absolute local path of the transaction CSV, tolerating an archive
  that extracted into a nested directory;
- refusing to operate on any dataset other than the approved one. If the active
  registry entry's Kaggle id is not
  `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, it raises
  `DatasetMismatchError`. The single-approved-dataset rule is enforced in code,
  not by convention.

It does **not**: download anything, read or hold credentials, parse a CSV,
transform data, or train a model. Constructing it performs no I/O.

When the file is absent, `resolve_file()` raises `DataFileNotFoundError` carrying
acquisition instructions (which environment variables to set, and that no
substitute file is generated). On a fresh checkout that is the normal state, so
the error is written as a guide rather than a fault report.

### 14.2 `CsvChunkLoader` responsibility

Turns a file path into an iterator of `DataFrame` chunks. It owns file-level
concerns only:

- streaming the file in chunks of a configurable size (default 100 000 rows);
- validating the chunk size and the file's existence and readability **before**
  any parsing begins, so a bad call fails at the call site rather than on first
  iteration;
- preserving column names exactly as they appear in the header — no renaming, no
  case folding, no whitespace stripping;
- deterministic iteration: the same file read twice yields identical chunks;
- translating the parser's exceptions into project errors, so nothing above the
  infrastructure boundary imports pandas' exception types.

It does **not** perform AML-specific feature engineering, type coercion,
cleaning, or any interpretation of the data.

Chunks are **not concatenated**. A caller that wants the whole file in memory
must build it deliberately.

| Situation | Result |
| --- | --- |
| File absent | `DataFileNotFoundError` |
| Path is a directory | `UnreadableDataFileError` |
| Zero-byte file, or no parseable columns | `EmptyDataFileError` |
| Header present, no data rows | zero chunks — valid, not an error |
| Unterminated quote, ragged row, undecodable bytes | `MalformedDataError` |
| OS refuses the read | `UnreadableDataFileError` |
| `chunk_size` not a positive `int` | `InvalidChunkSizeError` |

### 14.3 Schema responsibility

A schema is **data**, not code: `TableSchema` holds an ordered tuple of
`ColumnSpec` value objects, both defined in `backend/domain/value_objects/` with no
pandas dependency.

A `ColumnSpec` declares:

| Field | Meaning |
| --- | --- |
| `name` | Column name as it appears in the header |
| `data_type` | Logical type: `string`, `integer`, `float`, `boolean`, `datetime` |
| `required` | Whether the column must be present at all |
| `nullable` | Whether empty cells are permitted |
| `allowed_values` | Closed vocabulary — permitted on `string` columns only |
| `minimum` / `maximum` | Inclusive bounds — permitted on numeric columns only |
| `description` | What the column means |

`allowed_values` is restricted to string columns deliberately: comparing a parsed
numeric column against string literals depends on the parser's formatting
(`1` versus `1.0`). Numeric restrictions use `minimum`/`maximum` — a 0/1 label is
an `integer` column with `minimum=0, maximum=1`.

A `TableSchema` must also state an explicit `UnexpectedColumnPolicy`. There is no
default, because both behaviours are consequential:

| Policy | Effect on an undeclared column |
| --- | --- |
| `reject` | Reported as an **error**; the chunk is invalid |
| `allow` | Reported as a **warning**; the chunk stays valid |

Under either policy the column is **reported**. It is never silently dropped —
the dangerous option is the silent one.

An incoherent schema (blank name, no columns, duplicate column names, bounds on a
string column, inverted bounds) raises `SchemaDefinitionError` at construction.

### 14.4 The IBM AML schema, as measured

> **SCHEMA VERIFICATION STATUS: VERIFIED**

The schema in
[`infrastructure/data/schemas/ibm_aml_schema.py`](../../backend/infrastructure/data/schemas/ibm_aml_schema.py)
was written from the header and contents of the real
`HI-Small_Trans.csv`, not from the dataset description or from memory.
`build_ibm_aml_transaction_schema()` returns it; it no longer raises.

**The 11 columns, in file order.** The header names `Account` twice. pandas
disambiguates the second occurrence as `Account.1`, and that parsed name — not a
name of our choosing — is what the schema and the pipeline use.

| # | Column (as parsed) | Role | Measured |
| --- | --- | --- | --- |
| 1 | `Timestamp` | Transaction time | Format `%Y/%m/%d %H:%M`, minute resolution. Range `2022/09/01 00:00` – `2022/09/18 16:18`. 15,018 distinct values. **Not sorted in the file.** |
| 2 | `From Bank` | Sending bank id | Integer-valued identifier, 356,303 distinct across both bank columns |
| 3 | `Account` | Sending account id | Hexadecimal text; 515,080 distinct across both account columns |
| 4 | `To Bank` | Receiving bank id | Integer-valued identifier |
| 5 | `Account.1` | Receiving account id | Hexadecimal text |
| 6 | `Amount Received` | Amount credited | Range 1e-06 – 1.0463e+12 |
| 7 | `Receiving Currency` | Currency of #6 | 15 distinct values |
| 8 | `Amount Paid` | Amount debited | Range 1e-06 – 1.0463e+12 |
| 9 | `Payment Currency` | Currency of #8 | 15 distinct values |
| 10 | `Payment Format` | Channel | 7 distinct values |
| 11 | `Is Laundering` | **Target** | Exactly two values: `0` and `1` |

**Measured properties of the file as a whole:**

| Property | Measured value |
| --- | --- |
| Data rows | 5,078,345 |
| File size | 475,664,283 bytes |
| SHA-256 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| Text encoding | UTF-8, no BOM |
| Null values | **0**, in every one of the 11 columns |
| Exact duplicate rows | 9 (**kept** — see §18) |
| Timestamps sorted ascending | **No** |
| `Amount Received == Amount Paid` | 5,006,187 rows (98.5791%) |
| Same currency on both sides | 5,006,175 rows (98.5789%) |
| Sender account == receiver account | 591,212 rows (11.6418%) |

**The target, identified from the data rather than from the column name:**

| Value | Rows | Share |
| --- | --- | --- |
| `0` | 5,073,168 | 99.8981% |
| `1` | 5,177 | **0.1019%** |

The positive class is roughly 1 row in 981. That ratio is a property of the
data and is preserved exactly through preprocessing — see §18.

`Is Laundering` was confirmed as the label by reading its observed values, not
by assuming from its name; the mapping `{"0": 0, "1": 1}` is recorded in
`metadata/preprocessing_report.json` alongside the values it was derived from.

**Not constrained by the schema:** account and bank identifiers carry no
`allowed_values`. They are open sets — a new account is normal data, not a
schema violation — so constraining them would reject a valid file.


### 14.5 `DataFrameSchemaValidator` responsibility

Checks one chunk against one schema and returns a `ChunkValidationResult`. Two
constraints define it:

**It does not mutate the caller's data.** Type checking is done by coercing a
*temporary* Series (`pd.to_numeric` / `pd.to_datetime` with `errors="coerce"`,
both of which return new objects) and inspecting the result. The caller's
`DataFrame` — values, dtypes and column order — is untouched. This is asserted
directly, on both conforming and non-conforming input.

**It is not preprocessing.** Nothing is cleaned, filled, cast, trimmed or
case-folded. A value of `" AAA "` where `"AAA"` is expected is a finding, not
something to fix.

Because a chunked read infers dtypes **per chunk**, the same column can arrive as
text in one chunk and as numbers in the next. The validator therefore judges
values by coercion rather than by the dtype the parser happened to choose.

What it checks, per column:

| Check | Finding kind |
| --- | --- |
| Required column present | `missing_required_column` |
| Undeclared column | `unexpected_column` |
| Values coerce to the declared type | `wrong_type` |
| Empty cells in a non-nullable column | `null_in_non_nullable` |
| Values within a closed vocabulary | `value_not_allowed` |
| Values within `minimum`/`maximum` | `value_out_of_range` |

An empty cell is reported as missing, not as malformed — so a null never
double-counts as a type failure.

### 14.6 One finding per problem, not per row

The aggregation key is **`(column, kind)`**. A chunk of 100 000 rows with a bad
value in 4 000 of them produces **one** finding with `occurrences=4000` and up to
three example row labels — not 4 000 identical messages. This is what fixes the
duplicate-error problem in the earlier implementation.

Details of the design:

- A column's two bounds are one finding, not "too low" plus "too high", so the
  aggregation key stays intact.
- `example_rows` carries the row's position **in the file**, because a chunked
  read keeps a continuing index — that is what makes a row locatable.
- Structural findings (`occurrences=0`) report no row count, because a missing
  column is not row-scoped.
- Findings are sorted on construction (structural kinds first, then by column),
  so validating identical input twice produces **equal** results.
- Each result carries `source` and `chunk_index`, so a finding stays
  interpretable after it has been separated from the loop that produced it.
- Ingestion logging happens in `DatasetIngestionService`, not in the validator,
  so an invalid chunk is reported **once** rather than once per layer. The
  validator itself has no logger — it is a pure function.

### 14.7 Strictness

`DatasetIngestionService(..., strict=True)` is the default: the first invalid
chunk aborts the run with `SchemaValidationError`, whose message quotes at most
five findings and says how many more are in the log. Training on data that failed
its schema produces results nobody can interpret.

With `strict=False`, invalid chunks are logged as warnings and still yielded
together with their report, leaving the decision to the caller.

---

## 15. Memory-safe streaming

The dataset is large enough that reading it whole is not an option. Memory safety
here is **structural, not measured**:

- `CsvChunkLoader` has **no code path that reads a whole file**. Every call to the
  CSV parser passes `chunksize`. A test records every `read_csv` call made during
  a load and asserts each one was chunked — the whole-file path must not exist,
  not merely be unused.
- Precondition checks are eager, the parse is lazy: `stream_chunks()` validates
  the chunk size and the file, then returns a generator. Nothing is parsed until
  the first chunk is requested.
- `DatasetIngestionService` **never accumulates**. Chunks are yielded as they are
  read and no reference to a previous chunk is kept, so a consumer that processes
  and discards each chunk holds one chunk's worth of rows at a time regardless of
  file size. A test asserts exactly one chunk is read per step, and that
  abandoning the stream leaves the rest of the file unread.
- Validation allocates per chunk, not per file: at most one temporary coerced
  Series per column, discarded immediately.

**No RAM figure or reduction percentage is claimed anywhere in this project.**
Nothing has been measured on the real dataset, so no number is stated. The claim
made is the one that can be verified from the code: the whole file is never read
at once.

The default chunk size is 100 000 rows, defined once in
`backend/application/ports/output/data_loader.py` and overridable per call.

---

## 16. Offline testing of the ingestion pipeline

**The real dataset is not required to test any of this, and was not downloaded.**

- Every ingestion test runs against small synthetic fixtures written to pytest's
  `tmp_path`, defined in
  [`backend/tests/fixtures/tabular_fixtures.py`](../../backend/tests/fixtures/tabular_fixtures.py)
  under a banner reading **SYNTHETIC TEST DATA — NOT IBM AML DATA**.
- The fixture columns (`record_id`, `event_time`, `sender_ref`, `receiver_ref`,
  `value`, `unit`, `flag`) are invented and deliberately unlike any real
  dataset's headers. The `unit` vocabulary is `AAA`/`BBB` — not currency codes —
  so no value in the fixtures can be mistaken for a real observation. A test
  asserts the fixture header contains no real-sounding column name.
- **No file named `HI-Small_Trans.csv` is ever created**, and a test asserts that
  no file matching `HI-Small*` appears anywhere the tests write.
- No row count, class ratio or file size in any fixture says anything about the
  real dataset.
- The integration tests exercise the real registry, source, loader, validator and
  ingestion service together. For the stages that need a local file, the registry
  entry keeps the real dataset's *identity* (the source refuses any other) while
  its `expected_files` names the synthetic stand-in. Only the wiring is under
  test.
- `ExplodingDownloader` is injected throughout, so any attempted transfer fails
  the test.
- A test deletes `KAGGLE_USERNAME`, `KAGGLE_KEY` and `KAGGLE_CONFIG_DIR` from the
  environment and runs the full pipeline, so the suite is verified to work
  without credentials.
- The suite passes with no network access and with the real dataset absent.

---


## 17. What this directory will contain

| Document | Purpose | Status |
| --- | --- | --- |
| `acquisition.md` | How to obtain the dataset (Kaggle credentials, licence terms, exact steps) | Covered by §13 |
| `schema.md` | Column-by-column schema: names, dtypes, units, semantics, nullability | **Measured — §14.4** |
| `class_distribution.md` | Measured label imbalance | **Measured — §14.4, §18.5** |
| `preprocessing.md` | Cleaning, encoding, scaling and imbalance-handling decisions with rationale | **§18** |
| `feature_engineering.md` | Derived features and why each is expected to carry AML signal | **§18.4** |
| `sequence_construction.md` | How per-account transaction sequences are windowed for LSTM/GRU/BiLSTM | NOT YET WRITTEN — a later phase |
| `partitioning.md` | IID and Non-IID client partitioning schemes and their parameters | NOT YET WRITTEN — a later phase |
| `data_quality_report.md` | Generated validation output — written by the pipeline, not by hand | **Generated: `data/processed/…/metadata/`** |

The sections marked for a later phase are not started. Nothing in this
directory describes model training, federated learning or evaluation, because
none of that exists yet.

## 18. Preprocessing

Implemented by
[`AmlPreprocessor`](../../backend/infrastructure/data/preprocessors/aml_preprocessor.py)
and orchestrated by
[`PreprocessingService`](../../backend/application/services/preprocessing_service.py).
Run with `python main.py preprocess`. **The command never downloads**; it reads
the local file and fails with acquisition instructions if it is absent.

This phase produces **data only**. No model is trained, and nothing here
depends on a later phase.

### 18.1 Why three passes

The raw file is **not in timestamp order** — that was measured, not assumed.
Chronological splitting and causal history features both need time order, and
the two obvious ways to get it both fail: sorting 5,078,345 rows in memory
defeats chunked reading, and processing the file as it lies produces account
histories that are simply wrong.

The pipeline therefore makes three bounded passes:

| Pass | Reads | Does | Bounded by |
| --- | --- | --- | --- |
| 1 `partition` | Raw CSV, 200,000 rows at a time | Cleans, appends each row to a file named for its calendar day, and builds a histogram of every distinct timestamp | One chunk + 15,018 histogram entries |
| 2 `engineer` | One day partition at a time | Sorts that day exactly, builds features, assigns the split, writes validation/test rows, fits the train statistics, stages train rows | **One day** (largest: 1,114,921 rows), plus per-account state |
| 3 `finalize_train` | The staged train rows | Applies the now-frozen transformations and writes the train split | One chunk |

The day partition is an external bucket sort with the calendar as the bucket
key. Eighteen days, so peak interim memory is one day rather than one dataset.
Working files live in `data/interim/` and are deleted after a successful run
(`keep_interim: false`); nothing downstream may read them.

The exact 60/20/20 boundaries come from the timestamp histogram, walked in
sorted order — 15,018 entries, not 5,078,345 rows.

### 18.2 Cleaning: measured, and it removed nothing

Every rule is counted whether or not it fires, because "0 rows failed" is
itself a finding.

| Rule | Rows removed |
| --- | --- |
| `missing_required_value` — any of the 11 columns null | **0** |
| `unparseable_timestamp` — did not match `%Y/%m/%d %H:%M` | **0** |
| `negative_amount` — an amount below zero | **0** |
| `unknown_target_value` — label neither `0` nor `1` | **0** |
| **Total** | **0** |

5,078,345 read − 0 removed = 5,078,345 written. The ledger reconciles.

**Large amounts are kept.** The maximum is 1.0463e+12 and it survives: in AML an
unusual value carries signal, so nothing is discarded for looking like an
outlier.

**The 9 exact duplicate rows are reported and kept.** Two identical transfers in
the same minute are plausible, and deleting real transactions from an AML
dataset is the worse error. The count is in the report; the removal count is 0.

### 18.3 Splitting: chronological, never random

60% train / 20% validation / 20% test **by timestamp**, never shuffled.

| Split | Rows | Share | Window | Positives | Positive rate |
| --- | --- | --- | --- | --- | --- |
| train | 3,047,216 | 60.0041% | `2022-09-01 00:00:00` – `2022-09-06 13:36:00` | 2,299 | 0.0754% |
| validation | 1,015,565 | 19.9980% | `2022-09-06 13:37:00` – `2022-09-08 16:12:00` | 1,081 | 0.1064% |
| test | 1,015,564 | 19.9979% | `2022-09-08 16:13:00` – `2022-09-18 16:18:00` | 1,797 | 0.1769% |

`max(train) < min(validation)` and `max(validation) < min(test)`: verified, and
checked from the written files rather than from the counters that wrote them.

**Tie policy: `whole_timestamp_to_earlier_split`.** Timestamps are minute
resolution, so thousands of rows share a boundary value. A whole timestamp goes
to the earlier split; no timestamp is ever split across two files, which is why
the realised shares are 60.0041% rather than exactly 60%. The realised shares
are reported rather than the requested ones.

The positive rate rises across the splits (0.0754% → 0.1064% → 0.1769%). That is
a property of the data in time, not an artefact: a random split would have
hidden it, which is precisely why the split is chronological.

### 18.4 Features

23 model inputs, 3 identifiers carried for traceability but **not** model
inputs (`timestamp`, `from_account`, `to_account`).

**Transaction (6):** `amount_paid_log`, `amount_received_log`,
`amount_mismatch`, `same_currency`, `same_bank`, `self_transfer`.

**Categorical (3):** `payment_format_code`, `payment_currency_code`,
`receiving_currency_code` — ordinal integer codes over a **vocabulary fitted on
train only**, with a reserved negative code for categories first seen in
validation or test.

**Temporal (6):** `minute_of_day_sin`, `minute_of_day_cos`, `day_of_week_sin`,
`day_of_week_cos`, `is_weekend`, `is_night`. Sine/cosine pairs so that 23:59 and
00:00 are adjacent rather than maximally distant.

**Behavioural, causal (8):** `sender_prior_count_log`,
`sender_prior_amount_log`, `sender_seconds_since_prior_log`,
`sender_has_prior`, and the four receiver equivalents. Every one is an
**exclusive prefix** — computed from strictly earlier transactions only. A test
asserts truncation-invariance: appending later rows cannot change an earlier
row's values.

**Features deliberately NOT built**, each recorded in
`metadata/preprocessing_report.json` rather than silently absent:

| Feature | Why not |
| --- | --- |
| `time_to_next_transaction` | Needs a later row to describe an earlier one. Future information by construction. |
| `account_total_transaction_count` | A whole-dataset per-account count uses transactions after the row it describes. The causal prior-count features replace it. |
| `is_fraud_ring_member` | Derived from the label; leaks the target directly. |
| `days_since_dataset_start` | Train range (days 0–5) and test range (days 7–17) are disjoint under the chronological split, so a model could only extrapolate it. |
| `calendar_month` | The data spans one month; constant, so no information. |
| `amount_difference` | The two amounts are in different currencies in 1.42% of rows, so their difference is not a quantity. No rate table exists and inventing one would fabricate data. `amount_mismatch` keeps the signal. |
| `amount_in_common_currency` | Would need external exchange rates — data from outside the dataset. |
| `bank_identifier_encoded` | 356,303 distinct bank ids; an integer id has no magnitude. `same_bank` keeps the part that carries signal. |
| `counterparty_pair_prior_count` | State keyed by (sender, receiver) grows with the row count rather than the account population. |
| `account_degree_over_window` | Needs a per-account window buffer; state would grow with window length × accounts. Deferred, not rejected. |

### 18.5 Encoding, scaling and the target

**High-cardinality identifiers are not encoded, and the refusal is recorded.**
`Account` / `Account.1` (515,080 distinct) and `From Bank` / `To Bank` (356,303
distinct) are neither one-hot encoded nor scaled. One-hot would produce half a
million columns; treating an account id as a number would assert a magnitude it
does not have. They are carried as identifiers, and the relational signal is
kept through `same_bank` and `self_transfer`.

**Scaling** is `log1p` then standardisation, and only on the 8 skewed
non-negative monetary and count columns. `log1p` because the amount range spans
18 orders of magnitude (1e-06 to 1.0463e+12); `log1p` rather than `log` because
zero is a legitimate value. Booleans and cyclical sin/cos features are **not**
scaled — they are already bounded, and each unscaled column records
`{"transform": "none"}` with a reason.

**Every learned statistic is fitted on train alone.** Means, standard deviations
and category vocabularies come from the 3,047,216 training rows and are then
merely *applied* to validation and test. This is enforced structurally: a
transformer raises `TransformerNotFittedError` if used before freezing and
`TransformerAlreadyFittedError` if updated after, so validation data cannot
reach a fit path by accident. The written `scaling_parameters[…]["count"]`
equals the training row count, which an integration test asserts.

**The target is `Is Laundering` → `is_laundering`, mapping `{"0": 0, "1": 1}`,
positive label `1`** — read from the observed values, not assumed from the name.

**The class distribution is NOT rebalanced.** No oversampling, no
undersampling, no class weights. 5,073,168 negative / 5,177 positive
(0.1019%) is preserved exactly, and the test split in particular keeps its
natural distribution so that later metrics mean something. Imbalance handling is
a *modelling* decision and belongs to a later phase.

### 18.6 Output

```
data/processed/ibm_aml/HI-Small/
├── train/transactions.csv          3,047,216 rows
├── validation/transactions.csv     1,015,565 rows
├── test/transactions.csv           1,015,564 rows
└── metadata/
    ├── schema.json                 the 11 verified raw columns
    ├── feature_metadata.json       feature list, output column order, roles
    ├── transformers.json           category mappings, scaling parameters
    ├── splits.json                 boundaries, tie policy, per-split counts
    └── preprocessing_report.json   the full run record
```

CSV, because no columnar writer (pyarrow / fastparquet) is installed in this
environment and this phase does not add a dependency it can do without. The
files stay streamable: one header, then rows, read back by the same
`CsvChunkLoader` the raw file uses.

The artefacts record the dataset fingerprint, the preprocessing version
(`1.0.0`), the column order, the category mappings, the scaling parameters, the
target mapping and the split boundaries — everything needed to reproduce the
run or to apply the same transformations later. **No credential can reach
them:** the metadata writer walks each payload recursively and refuses any
forbidden key, naming the key but never its value.

### 18.7 The raw file is read-only

`data/raw/ibm_aml/HI-Small_Trans.csv` is never written, renamed, moved or
copied. Three independent guards:

1. The writer **refuses at construction** to write anywhere inside the raw
   directory, before a single row exists.
2. The file's SHA-256 is taken before the run and re-checked after. A mismatch
   raises `RawDatasetModifiedError`.
3. `verified_unchanged_after_run: true` is recorded in the report.

Measured on the real run: SHA-256
`b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` before and
after, unchanged.

### 18.8 Measured performance

Real run, whole file, `chunk_size = 200000`:

| Stage | Seconds |
| --- | --- |
| `fingerprint` | 0.5 |
| `partition` (pass 1) | 25.2 |
| `boundaries` | 0.0 |
| `engineer` (pass 2) | 98.5 |
| `finalize_train` (pass 3) | 53.4 |
| `verify` | 0.5 |
| **Total** | **178.1** |

5,078,345 rows in, 5,078,345 out, 0 removed. **≈28,509 rows/second.**

Peak rows held by design: 200,000 (the chunk) in passes 1 and 3, and one day
partition in pass 2 — largest 1,114,921 rows. Behavioural state is 515,080
tracked accounts, which is what actually bounds memory: it grows with the
account population, not with the row count.

These are wall-clock measurements from the run. **No RAM figure is claimed** —
memory safety here is structural (nothing reads the whole file) and was not
measured with a profiler.


## Documentation rule

Any statistic about the dataset (row counts, file sizes, class ratios,
checksums) is recorded **only after being measured** on a real local copy. No
figure in this directory may be estimated, recalled from a paper, or inferred
from the dataset description.
