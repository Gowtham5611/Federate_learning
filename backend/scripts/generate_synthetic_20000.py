"""Generator for ibm_aml_schema_synthetic_20000.csv

Creates the benchmark synthetic test case with generator artifacts:
- 20,000 rows
- 30.0% laundering (6,000 laundering, 14,000 non-laundering)
- 20,000 unique senders and 20,000 unique receivers (unique_sender_ratio 1.0)
- Timestamps unsorted and spread across full year
- Laundering label separable by amount and payment format
- 83% currency mismatch in both classes
- Raw IBM header: Account and Account.1
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "demo" / "ibm_aml_schema_synthetic_20000.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

CURRENCIES = ["US Dollar", "Euro", "Canadian Dollar", "UK Pound", "Yen", "Australian Dollar"]
PAYMENT_FORMATS_LAUNDERING = ["Wire", "Cash"]
PAYMENT_FORMATS_NORMAL = ["Cheque", "ACH", "Credit Card"]
BANKS = [1, 2, 10, 12, 1420, 3209, 11813, 245335]

n_rows = 20000
n_positives = 6000
n_negatives = n_rows - n_positives

start_dt = datetime(2022, 1, 1, 0, 0, 0)
end_dt = datetime(2022, 12, 31, 23, 59, 59)
total_seconds = int((end_dt - start_dt).total_seconds())

rows = []

# Generate 6,000 laundering rows
for i in range(n_positives):
    sender_id = f"800{i:07X}0"
    receiver_id = f"800{(i + 50000):07X}0"
    
    # Amount shortcut: high amounts for laundering (50,000 to 250,000)
    amt = round(random.uniform(50000.0, 250000.0), 2)
    
    # Currency mismatch: 83% chance currencies differ
    cur1 = random.choice(CURRENCIES)
    if random.random() < 0.83:
        cur2 = random.choice([c for c in CURRENCIES if c != cur1])
    else:
        cur2 = cur1
        
    p_format = random.choice(PAYMENT_FORMATS_LAUNDERING)
    
    random_sec = random.randint(0, total_seconds)
    tx_dt = start_dt + timedelta(seconds=random_sec)
    ts_str = tx_dt.strftime("%Y/%m/%d %H:%M")
    
    rows.append({
        "Timestamp": ts_str,
        "From Bank": random.choice(BANKS),
        "Account": sender_id,
        "To Bank": random.choice(BANKS),
        "Account.1": receiver_id,
        "Amount Received": amt,
        "Receiving Currency": cur1,
        "Amount Paid": amt,
        "Payment Currency": cur2,
        "Payment Format": p_format,
        "Is Laundering": 1
    })

# Generate 14,000 normal rows
for i in range(n_negatives):
    sender_id = f"800{(i + 100000):07X}0"
    receiver_id = f"800{(i + 200000):07X}0"
    
    # Amount shortcut: low amounts for normal (5.0 to 4500.0)
    amt = round(random.uniform(5.0, 4500.0), 2)
    
    # Currency mismatch: 83% chance currencies differ
    cur1 = random.choice(CURRENCIES)
    if random.random() < 0.83:
        cur2 = random.choice([c for c in CURRENCIES if c != cur1])
    else:
        cur2 = cur1
        
    p_format = random.choice(PAYMENT_FORMATS_NORMAL)
    
    random_sec = random.randint(0, total_seconds)
    tx_dt = start_dt + timedelta(seconds=random_sec)
    ts_str = tx_dt.strftime("%Y/%m/%d %H:%M")
    
    rows.append({
        "Timestamp": ts_str,
        "From Bank": random.choice(BANKS),
        "Account": sender_id,
        "To Bank": random.choice(BANKS),
        "Account.1": receiver_id,
        "Amount Received": amt,
        "Receiving Currency": cur1,
        "Amount Paid": amt,
        "Payment Currency": cur2,
        "Payment Format": p_format,
        "Is Laundering": 0
    })

# Shuffle to make timestamps thoroughly unsorted
random.shuffle(rows)

header = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} synthetic benchmark rows at {OUTPUT_FILE}")

