# Phase 5: IID Client Partition Report

- **Total Training Rows**: 3,047,216
- **Total Positives**: 2,299 (0.075446%)
- **Total Negatives**: 3,044,917 (99.924554%)
- **Total Unique Accounts**: 492,800
- **Partition Method**: Stratified random permutation across target classes with round-robin assignment (`seed=42`)

---

## Client Partition Breakdown

| Client ID | Row Count | % of Total | Positive Count | Negative Count | Positive Rate | Unique Accounts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`client_1`** | 1,015,740 | 33.333377% | 767 | 1,014,973 | 0.075511% | 280,720 |
| **`client_2`** | 1,015,738 | 33.333312% | 766 | 1,014,972 | 0.075413% | 280,102 |
| **`client_3`** | 1,015,738 | 33.333312% | 766 | 1,014,972 | 0.075413% | 280,956 |
| **Total / Global** | **3,047,216** | **100.000000%** | **2,299** | **3,044,917** | **0.075446%** | **492,800** |

---

## Account Overlap Under Row-Level Partitioning

Because Phase 5 IID partitions rows via stratified random shuffling across the entire table without grouping by account, single accounts frequently have transactions distributed across multiple clients:

- **Client 1 & Client 2 Shared Accounts**: 157,340 accounts
- **Client 1 & Client 3 Shared Accounts**: 157,482 accounts
- **Client 2 & Client 3 Shared Accounts**: 157,379 accounts

---

## Invariant Verification

- **Row Count Conservation**: $1,015,740 + 1,015,738 + 1,015,738 = 3,047,216$ [PASS]
- **Label Conservation**: $767 + 766 + 766 = 2,299$ positives [PASS]
- **Negative Conservation**: $1,014,973 + 1,014,972 + 1,014,972 = 3,044,917$ negatives [PASS]
- **Stratification Balance**: Max positive difference across clients is $|767 - 766| = 1 \le 1$ [PASS]
- **Size Balance**: Max row difference across clients is $|1,015,740 - 1,015,738| = 2 \le 2$ [PASS]
- **Chronological Monotonicity**: Timestamp order within each client partition is strictly monotonic [PASS]

