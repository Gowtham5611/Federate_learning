# Dataset verification report — IBM AML HI-Small

Generated 2026-09-15 06:21:43 UTC by `backend/scripts/verify_dataset.py`.

Every figure below was measured from the real local file by a chunked pass. Nothing is estimated, remembered or inferred from a column name.

## Identity

| Field | Verified value |
| --- | --- |
| Dataset | IBM Transactions for Anti-Money Laundering (AML) |
| Project dataset id | `ibm_transactions_aml` |
| Remote id | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` |
| Variant | `HI-Small` |
| Expected filename | `HI-Small_Trans.csv` |
| Actual file | `HI-Small_Trans.csv` |
| Location | `data/raw/ibm_aml/` |
| File size | 475,664,282 bytes (453.6 MiB) |

## Structure

| Field | Verified value |
| --- | --- |
| Rows (parsed, excluding header) | 5,078,345 |
| Newline bytes in file | 5,078,345 |
| Columns | 11 |
| Encoding | `utf-8` (no BOM) |
| Delimiter | `,` (determined by csv.Sniffer) |
| Header row | present (determined by csv.Sniffer) |
| Chunked pass | 21 chunks of 250,000 rows |
| Malformed rows | none — no parse error during the full chunked pass |
### Files actually present

| Path (relative to the dataset directory) | Size |
| --- | --- |
| `HI-Small_Trans.csv` | 475,664,282 bytes |

CSV files present: **1** (`HI-Small_Trans.csv`).

## Columns as observed

| # | Column | Observed dtype(s) | Nulls | Distinct | Range / lengths | Samples |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `Timestamp` | `str` | 0 (0.00%) | > 60 | 16–16 chars | `2022/09/01 00:20`, `2022/09/01 00:20`, `2022/09/01 00:00` |
| 1 | `From Bank` | `int64` | 0 (0.00%) | > 60 | 1 … 356,303 | `10`, `3208`, `3209` |
| 2 | `Account` | `str` | 0 (0.00%) | > 60 | 9–9 chars | `8000EBD30`, `8000F4580`, `8000F4670` |
| 3 | `To Bank` | `int64` | 0 (0.00%) | > 60 | 1 … 356,294 | `10`, `1`, `3209` |
| 4 | `Account.1` | `str` | 0 (0.00%) | > 60 | 9–9 chars | `8000EBD30`, `8000F5340`, `8000F4670` |
| 5 | `Amount Received` | `float64` | 0 (0.00%) | > 60 | 1e-06 … 1.0463e+12 | `3697.34`, `0.01`, `14675.57` |
| 6 | `Receiving Currency` | `str` | 0 (0.00%) | 15 | 3–17 chars | `US Dollar`, `US Dollar`, `US Dollar` |
| 7 | `Amount Paid` | `float64` | 0 (0.00%) | > 60 | 1e-06 … 1.0463e+12 | `3697.34`, `0.01`, `14675.57` |
| 8 | `Payment Currency` | `str` | 0 (0.00%) | 15 | 3–17 chars | `US Dollar`, `US Dollar`, `US Dollar` |
| 9 | `Payment Format` | `str` | 0 (0.00%) | 7 | 3–12 chars | `Reinvestment`, `Cheque`, `Reinvestment` |
| 10 | `Is Laundering` | `int64` | 0 (0.00%) | 2 | 0 … 1 | `0`, `0`, `0` |

## Low-cardinality columns and their value counts

Reported as evidence for identifying a target column and for defining closed vocabularies. Semantics are **not** inferred from names here.

### `Receiving Currency`

| Value | Count | Share |
| --- | --- | --- |
| `US Dollar` | 1,879,341 | 37.0070% |
| `Euro` | 1,172,017 | 23.0787% |
| `Swiss Franc` | 237,884 | 4.6843% |
| `Yuan` | 206,551 | 4.0673% |
| `Shekel` | 194,988 | 3.8396% |
| `Rupee` | 192,065 | 3.7820% |
| `UK Pound` | 181,255 | 3.5692% |
| `Ruble` | 157,361 | 3.0987% |
| `Yen` | 156,319 | 3.0781% |
| `Bitcoin` | 148,151 | 2.9173% |
| `Canadian Dollar` | 141,357 | 2.7835% |
| `Australian Dollar` | 138,511 | 2.7275% |
| `Mexican Peso` | 111,030 | 2.1863% |
| `Saudi Riyal` | 89,971 | 1.7717% |
| `Brazil Real` | 71,544 | 1.4088% |

### `Payment Currency`

| Value | Count | Share |
| --- | --- | --- |
| `US Dollar` | 1,895,172 | 37.3187% |
| `Euro` | 1,168,297 | 23.0055% |
| `Swiss Franc` | 234,860 | 4.6247% |
| `Yuan` | 213,752 | 4.2091% |
| `Shekel` | 192,184 | 3.7844% |
| `Rupee` | 190,202 | 3.7454% |
| `UK Pound` | 180,738 | 3.5590% |
| `Yen` | 155,209 | 3.0563% |
| `Ruble` | 155,178 | 3.0557% |
| `Bitcoin` | 146,066 | 2.8763% |
| `Canadian Dollar` | 140,042 | 2.7576% |
| `Australian Dollar` | 136,769 | 2.6932% |
| `Mexican Peso` | 110,159 | 2.1692% |
| `Saudi Riyal` | 89,014 | 1.7528% |
| `Brazil Real` | 70,703 | 1.3922% |

### `Payment Format`

| Value | Count | Share |
| --- | --- | --- |
| `Cheque` | 1,864,331 | 36.7114% |
| `Credit Card` | 1,323,324 | 26.0582% |
| `ACH` | 600,797 | 11.8306% |
| `Cash` | 490,891 | 9.6664% |
| `Reinvestment` | 481,056 | 9.4727% |
| `Wire` | 171,855 | 3.3841% |
| `Bitcoin` | 146,091 | 2.8767% |

### `Is Laundering`

| Value | Count | Share |
| --- | --- | --- |
| `0` | 5,073,168 | 99.8981% |
| `1` | 5,177 | 0.1019% |

## Null summary

| Column | Nulls | % of rows |
| --- | --- | --- |
| `Timestamp` | 0 | 0.0000% |
| `From Bank` | 0 | 0.0000% |
| `Account` | 0 | 0.0000% |
| `To Bank` | 0 | 0.0000% |
| `Account.1` | 0 | 0.0000% |
| `Amount Received` | 0 | 0.0000% |
| `Receiving Currency` | 0 | 0.0000% |
| `Amount Paid` | 0 | 0.0000% |
| `Payment Currency` | 0 | 0.0000% |
| `Payment Format` | 0 | 0.0000% |
| `Is Laundering` | 0 | 0.0000% |

## Field groupings (by observed dtype, not by meaning)

- **Numeric (5):** `From Bank`, `To Bank`, `Amount Received`, `Amount Paid`, `Is Laundering`
- **Non-numeric (6):** `Timestamp`, `Account`, `Account.1`, `Receiving Currency`, `Payment Currency`, `Payment Format`

