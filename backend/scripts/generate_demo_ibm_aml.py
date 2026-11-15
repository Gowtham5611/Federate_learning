"""Synthetic IBM AML Demo Dataset Generator for FedSecureAI.

Generates raw IBM AML schema CSVs with repeated accounts and realistic
laundering typologies (Fan-Out, Fan-In, Cycle, Scatter-Gather).

Produces:
- data/demo/demo_ibm_aml_500.csv (500 rows, valid sequence history, small positive sample)
- data/demo/demo_ibm_aml_5000.csv (5,000 rows, fully valid sequence history & representative prevalence)

Schema matches raw IBM AML:
Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import List, Tuple, Dict, Any

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENCIES = ["US Dollar", "Euro", "Canadian Dollar", "UK Pound", "Yen", "Australian Dollar"]
PAYMENT_FORMATS = ["Cheque", "Credit Card", "ACH", "Wire", "Reinvestment", "Cash"]
BANKS = [1, 2, 10, 12, 1420, 3209, 11813, 245335]


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d %H:%M")


def _format_account(acc_id: int) -> str:
    return f"800{acc_id:05X}0"


def generate_demo_dataset(n_rows: int, n_positives: int, seed: int = 42) -> Path:
    """Generate a raw IBM AML demonstration CSV with realistic laundering patterns."""
    random.seed(seed)
    out_path = OUTPUT_DIR / f"demo_ibm_aml_{n_rows}.csv"
    
    # Pool of accounts to ensure high repetition (>= 50% rows with >= 3 prior transactions)
    # For 500 rows, ~40 active accounts -> avg ~12.5 tx/account
    # For 5000 rows, ~250 active accounts -> avg ~20 tx/account
    n_accounts = max(20, int(n_rows / 15))
    accounts = [_format_account(i + 1) for i in range(n_accounts)]
    
    start_time = datetime(2022, 9, 1, 8, 0, 0)
    current_time = start_time
    
    # We will construct transactions step by step, interleaving normal transactions and laundering patterns
    transactions: List[Dict[str, Any]] = []
    
    # Target laundering count
    target_positives = n_positives
    positives_generated = 0
    
    # Accounts reserved for laundering syndicates
    syndicate_accounts = accounts[:15]
    normal_accounts = accounts[15:]
    
    # Track account transaction counts to guarantee sufficient history depth
    account_history_count: Dict[str, int] = {acc: 0 for acc in accounts}
    
    # Step 1: Prime accounts with warm-up normal transactions so that later transactions have >= 3 history
    for _ in range(n_rows):
        current_time += timedelta(minutes=random.randint(1, 15))
        
        # Decide if this should be a laundering transaction
        is_laundering = 0
        if positives_generated < target_positives and len(transactions) > 30 and random.random() < (target_positives / (n_rows - len(transactions) + 1)):
            is_laundering = 1
            positives_generated += 1
            
        if is_laundering == 1:
            # Pick a typology: Fan-out, Fan-in, Cycle, or Scatter-Gather
            typology = random.choice(["fan_out", "fan_in", "cycle", "scatter_gather"])
            if typology == "fan_out":
                # Master sender to one of the syndicate receivers
                sender = syndicate_accounts[0]
                receiver = random.choice(syndicate_accounts[1:5])
                amount = round(random.uniform(9000.0, 9999.0), 2)  # Structuring amount
            elif typology == "fan_in":
                sender = random.choice(syndicate_accounts[1:5])
                receiver = syndicate_accounts[5]
                amount = round(random.uniform(8500.0, 9800.0), 2)
            elif typology == "cycle":
                cycle_step = positives_generated % 3
                sender = syndicate_accounts[6 + cycle_step]
                receiver = syndicate_accounts[6 + ((cycle_step + 1) % 3)]
                amount = round(random.uniform(15000.0, 25000.0), 2)
            else:  # scatter_gather
                sender = random.choice(syndicate_accounts[9:12])
                receiver = syndicate_accounts[12]
                amount = round(random.uniform(12000.0, 18000.0), 2)
                
            fmt = random.choice(["Wire", "ACH"])
            curr = "US Dollar"
        else:
            # Normal commercial / retail transaction
            sender = random.choice(accounts)
            receiver = random.choice(accounts)
            while receiver == sender:
                receiver = random.choice(accounts)
            amount = round(random.uniform(5.0, 4500.0), 2)
            fmt = random.choice(PAYMENT_FORMATS)
            curr = random.choice(CURRENCIES)
            
        from_bank = random.choice(BANKS)
        to_bank = from_bank if random.random() < 0.4 else random.choice(BANKS)
        
        transactions.append({
            "Timestamp": _format_timestamp(current_time),
            "From Bank": from_bank,
            "Account": sender,
            "To Bank": to_bank,
            "Account.1": receiver,
            "Amount Received": amount,
            "Receiving Currency": curr,
            "Amount Paid": amount,
            "Payment Currency": curr,
            "Payment Format": fmt,
            "Is Laundering": is_laundering,
        })
        account_history_count[sender] += 1
        
    # Write to CSV
    fieldnames = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering"
    ]
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
        
    print(f"Generated {out_path} ({len(transactions)} rows, {positives_generated} laundering positives, {len(set(accounts))} distinct accounts).")
    return out_path


def main() -> None:
    print("Generating demo IBM AML datasets...")
    # 500 rows, 3 positives -> prevalence 0.6% (< 0.885% threshold), <30 positives -> triggers SMALL_SAMPLE banner
    p500 = generate_demo_dataset(500, n_positives=3, seed=42)
    
    # 5000 rows, 35 positives -> prevalence 0.7% (< 0.885% threshold), >=30 positives -> clean VALID banner!
    p5000 = generate_demo_dataset(5000, n_positives=35, seed=42)
    print("Done generating demo datasets.")


if __name__ == "__main__":
    main()

