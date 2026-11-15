# FedSecureAI — Universal CSV Compatibility & Data Quality Architecture

## 1. Overview and Problem Statement

Production AML compliance systems must ingest transaction files originating from disparate core banking platforms, legacy mainframes, FinTech payment gateways, and data warehouse exports. These systems inevitably produce files with:
- Non-standard delimiters (comma, semicolon, tab, pipe)
- Differing character encodings (UTF-8, UTF-8 with BOM, Latin-1, CP1252)
- Varying column counts and column naming conventions (e.g., `From Account` vs `sender_id` vs `nameOrig`)
- Variable row batches (single transaction alert triage up to 100,000+ transaction batch reviews)
- Partial or missing non-critical attributes (e.g. empty bank identifiers or missing payment currencies)

Previous legacy implementations failed with cryptic errors such as *"Your training has only 11 rows, but the uploaded CSV does not match"*, conflating the raw 11-column dataset schema with the 57 one-hot neural sequence feature dimensions.

The **Universal CSV Compatibility Engine** resolves this completely.

---

## 2. Architecture of the Universal Pipeline

```
Raw File Upload (.csv, .txt)
   │
   ├── 1. Universal CSV Reader (Delimiter Sniffer + Multi-Encoding Prober + BOM Stripper)
   │
   ├── 2. CsvDataQualityAuditor (Missing values, Duplicates, Types, Numerical/Datetime integrity)
   │
   ├── 3. UniversalSchemaDetector (Regex semantic mapping, Leakage protection, Sequence feasibility)
   │
   ├── 4. ModelCompatibilityChecker (COMPATIBLE vs ADAPTABLE vs INCOMPATIBLE)
   │
   ├── 5. Execution Engine:
   │      ├── Mode A: Predict with Selected / Auto-Detected Model (Preserves 1-to-1 row alignment)
   │      └── Mode B: Train Custom Dynamic Classifier on User-Uploaded Labeled Data
   │
   └── 6. Output Generator:
          ├── Risk categorization: High (>= 0.70), Medium (0.30 - 0.69), Low (< 0.30)
          ├── Supervised verification metrics (if ground truth present)
          └── Aligned downloadable CSV export
```

---

## 3. The 3 Compatibility States

The `UniversalSchemaDetector` classifies uploaded CSVs against registered models into one of three distinct compatibility tiers:

| Status | Definition | Preprocessing & Inference Action | Compatibility Score |
| :--- | :--- | :--- | :---: |
| **`COMPATIBLE`** | All required semantic concepts (Timestamps, Sender Account, Receiver Account, Amount, Currencies, Formats) are directly present in the file. | Directly adapts columns to canonical dimensions; feeds temporal sliding windows into authoritative neural sequence models (`HI-Small` LSTM or `LI-Medium` BiLSTM). | $1.00$ |
| **`ADAPTABLE`** | Core transactional concepts (Accounts and Amounts) are present, but secondary topological attributes (e.g. Bank IDs or currencies) are missing or non-standard. | Synthesizes missing non-critical attributes using deterministic topology hashing and currency fallbacks; executes sequence model or tabular fallback without dropping rows. | $0.60 - 0.99$ |
| **`INCOMPATIBLE`** | Missing essential transaction concepts (e.g. no amount column or no counterparty identifiers) with no ground-truth labels available for dynamic training. | Execution rejected with clear, human-readable guidance identifying the exact missing dimensions and suggested remediation actions. | $< 0.60$ |

---

## 4. Strict Guarantees

1. **Row Count Preservation:**
   $$\text{len}(\text{predictions}) == \text{len}(\text{valid\_input\_records})$$
   Every valid input row receives an aligned prediction with its original row ID and position preserved.
2. **Target Leakage Safeguard:**
   Columns matching post-investigation cheat fields (such as `structuring_flag`, `sar_filed`, `fraud_confirmed`, `analyst_verdict`) are automatically identified and excluded from the feature tensor before model ingestion.
3. **Model Checkpoint Immutability:**
   Authoritative federated models (`HI-Small`, `LI-Medium`) remain strictly frozen with their peer-reviewed decision thresholds (`0.337784`, `0.238483`). Mode B dynamic training registers new custom models exclusively in `results/dynamic_models/`.

