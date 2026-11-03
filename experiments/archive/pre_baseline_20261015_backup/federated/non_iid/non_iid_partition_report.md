# Phase 6: Non-IID Client Partition Report

- **Total Training Rows**: 3,047,216
- **Total Positives**: 2,299 (0.075446%)
- **Total Negatives**: 3,044,917 (99.924554%)
- **Total Unique Accounts**: 492,800
- **Partition Method**: Atomic account-level typology assignment with deterministic tie-breaking (`seed=42`)

---

## Client Partition Breakdown

| Client ID | Typology | Row Count | % of Total | Positive Count | Negative Count | Positive Rate | Unique Accounts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`client_1`** | Retail Giant | 539,973 | 17.720208% | 359 | 539,614 | 0.066485% | 44,345 |
| **`client_2`** | Offshore / Crypto Exchange | 981,684 | 32.215767% | 965 | 980,719 | 0.098300% | 49,299 |
| **`client_3`** | Regional Credit Union | 1,525,559 | 50.064026% | 975 | 1,524,584 | 0.063911% | 399,156 |
| **Total / Global** | - | **3,047,216** | **100.000000%** | **2,299** | **3,044,917** | **0.075446%** | **492,800** |

---

## Domain Feature Distributions

- **Client 1 (`client_1` — Retail Giant)**:
  - Retail Payment Share (Credit Card + Cash): **59.59%**
  - Crypto / Wire / Cross-Currency FX Share: **0.00%**
  - Credit Union Share (ACH + Cheque + Reinvest): 40.98%
  - Intra-Bank Transfer Share (`same_bank == 1`): 2.25%
- **Client 2 (`client_2` — Offshore / Crypto Exchange)**:
  - Retail Payment Share: 31.09%
  - Crypto / Wire / Cross-Currency FX Share: **22.78%**
  - Credit Union Share: 53.99%
  - Intra-Bank Transfer Share: 13.47%
- **Client 3 (`client_3` — Regional Credit Union)**:
  - Retail Payment Share: 25.04%
  - Crypto / Wire / Cross-Currency FX Share: **0.00%**
  - Credit Union Share (ACH + Cheque + Reinvest): **75.58%**
  - Intra-Bank Transfer Share (`same_bank == 1`): **30.10%**

---

## Account Disjointness (Atomic Account Partitioning)

Because Phase 6 partitions data at the account level (`from_account`), no account is ever split across multiple clients:

- **Client 1 & Client 2 Shared Accounts**: **0 accounts**
- **Client 1 & Client 3 Shared Accounts**: **0 accounts**
- **Client 2 & Client 3 Shared Accounts**: **0 accounts**
- **Sum of Client Accounts**: $44,345 + 49,299 + 399,156 = 492,800$ accounts (100% of unique training accounts)

---

## Invariant Verification

- **Row Count Conservation**: $539,973 + 981,684 + 1,525,559 = 3,047,216$ [PASS]
- **Label Conservation**: $359 + 965 + 975 = 2,299$ positives [PASS]
- **Negative Conservation**: $539,614 + 980,719 + 1,524,584 = 3,044,917$ negatives [PASS]
- **Zero Account Overlap**: Pairwise account intersections are all empty [PASS]
- **Sequence Integrity**: Transactions within every account remain intact and chronologically ordered on a single client [PASS]
- **Chronological Monotonicity**: Timestamps within each partition are strictly monotonic [PASS]

