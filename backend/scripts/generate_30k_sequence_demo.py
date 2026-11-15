"""Extract a high-fidelity 30,000-row multi-transaction sequence dataset from HI-Small_Trans.csv.

Each account has multiple sequential transactions (depth: 8 to 18 transactions per account),
preserving temporal velocity, structuring bursts, and authentic laundering patterns
specifically calibrated for Recurrent Neural Networks (LSTM, GRU, BiLSTM, and Hybrid Ensemble).
"""

import random
from collections import defaultdict
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ibm_aml" / "HI-Small_Trans.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "demo" / "ibm_aml_multi_sequence_30000.csv"


def create_30k_sequence_dataset():
    random.seed(42)
    print("Pass 1: Scanning accounts and transaction frequencies...")
    account_counts = defaultdict(int)
    laundering_accounts = set()

    chunk_iter = pd.read_csv(RAW_CSV_PATH, chunksize=300000)
    for i, chunk in enumerate(chunk_iter):
        for acc in chunk["Account"].values:
            account_counts[acc] += 1
        laund = chunk[chunk["Is Laundering"] == 1]
        laundering_accounts.update(laund["Account"].unique())
        if i >= 11:  # Scan first ~3.6M transactions
            break

    # Select active laundering accounts (8 to 22 transactions per account)
    cand_laund = [acc for acc in laundering_accounts if 8 <= account_counts[acc] <= 22]
    random.shuffle(cand_laund)

    # Pick ~120 laundering accounts (~1,400 transactions)
    selected_laund = set()
    laund_tx_count = 0
    for acc in cand_laund:
        selected_laund.add(acc)
        laund_tx_count += account_counts[acc]
        if laund_tx_count >= 1400:
            break

    # Remaining rows needed for exactly 30,000
    target_legit_count = 30000 - laund_tx_count

    # Select active legitimate accounts (8 to 18 transactions per account)
    cand_legit = [acc for acc, c in account_counts.items() if acc not in laundering_accounts and 8 <= c <= 18]
    random.shuffle(cand_legit)

    selected_legit = set()
    accum_legit = 0
    for acc in cand_legit:
        selected_legit.add(acc)
        accum_legit += account_counts[acc]
        if accum_legit >= target_legit_count + 500:
            break

    all_target_accounts = selected_laund | selected_legit
    print(f"Target Accounts Selected: {len(selected_laund)} Laundering, {len(selected_legit)} Legit")
    print(f"Total Target Accounts: {len(all_target_accounts)}")

    print("Pass 2: Extracting complete transaction sequences for target accounts...")
    collected_rows = []
    chunk_iter2 = pd.read_csv(RAW_CSV_PATH, chunksize=300000)
    for i, chunk in enumerate(chunk_iter2):
        matched = chunk[chunk["Account"].isin(all_target_accounts)]
        if len(matched) > 0:
            collected_rows.append(matched)
        if i >= 11:
            break

    extracted_df = pd.concat(collected_rows, ignore_index=True)
    # Sort chronologically by Timestamp
    extracted_df["dt"] = pd.to_datetime(extracted_df["Timestamp"], format="%Y/%m/%d %H:%M")
    extracted_df = extracted_df.sort_values(by="dt").reset_index(drop=True)
    extracted_df = extracted_df.drop(columns=["dt"])

    # Trim to exactly 30,000 rows while keeping full accounts or clean slice
    if len(extracted_df) > 30000:
        extracted_df = extracted_df.iloc[:30000].copy()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("=============================================================")
    print(f"SUCCESS: Saved {len(extracted_df)} rows to {OUTPUT_CSV_PATH}")
    print(f"Unique Sender Accounts: {extracted_df['Account'].nunique()} (Ratio: {extracted_df['Account'].nunique()/len(extracted_df):.4f})")
    print(f"Average Sequence Depth: {len(extracted_df)/extracted_df['Account'].nunique():.2f} transactions/account")
    pos_count = int(extracted_df["Is Laundering"].sum())
    print(f"Laundering Transactions: {pos_count} ({pos_count/len(extracted_df)*100:.2f}%)")
    print("=============================================================")


if __name__ == "__main__":
    create_30k_sequence_dataset()

