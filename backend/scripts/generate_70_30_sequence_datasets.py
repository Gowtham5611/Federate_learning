"""Generate standardized 10K, 20K, and 30K IBM AML Multi-Sequence Datasets.

Strict 70:30 Rule Enforced:
- 10,000 rows: 7,000 Legitimate (70%) / 3,000 Laundering (30%)
- 20,000 rows: 14,000 Legitimate (70%) / 6,000 Laundering (30%)
- 30,000 rows: 21,000 Legitimate (70%) / 9,000 Laundering (30%)

Features authentic multi-step temporal sequence patterns:
1. Layering chains (10-hop accounts across banks)
2. Cycle rings (circular account loops)
3. High-velocity Fan-Out (scatter distribution)
4. Structured Fan-In (smurfing gather below currency reporting thresholds)
5. Account warmup history so recurrent neural networks (LSTM, GRU, BiLSTM)
   receive complete 10-step sequence tensors without zero-padding cold-start deficits.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def generate_sequence_dataset(total_rows: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    target_pos = int(round(total_rows * 0.30))
    target_neg = total_rows - target_pos
    
    assert target_pos + target_neg == total_rows
    print(f"\n--- Generating {total_rows:,} rows (Target: {target_neg:,} Legit / {target_pos:,} Laundering) ---")
    
    start_time = datetime(2022, 9, 1, 8, 0, 0)
    banks = [f"0{i:02d}" for i in range(1, 15)]
    legit_merchants = [f"800{i:05d}0" for i in range(1, 350)]
    currencies = ["US Dollar", "Euro", "Swiss Franc", "UK Pound"]
    formats = ["ACH", "Credit Card", "Cheque", "Wire"]
    
    events_laundering = []
    events_warmup = []
    events_background = []
    
    # Target allocations
    n_layering = int(round(target_pos * 0.30))
    n_cycle = int(round(target_pos * 0.30))
    n_fanout = int(round(target_pos * 0.20))
    n_fanin = target_pos - n_layering - n_cycle - n_fanout
    
    # ----------------------------------------------------
    # Pattern 1: Layering Networks (Chain length 10 -> 9 steps per burst, 10 bursts = 90 tx / syndicate)
    # ----------------------------------------------------
    num_syndicates = n_layering // 90
    for s_id in range(num_syndicates):
        chain_len = 10
        chain_accs = [f"81{s_id:03d}{step:02d}0" for step in range(chain_len)]
        # Initial warmup for each account in syndicate (4 warmup tx each)
        for acc in chain_accs:
            for w in range(4):
                t_w = start_time + timedelta(seconds=int(np.random.uniform(100, 500000)))
                cur = "Euro" if s_id % 2 == 0 else "US Dollar"
                amt = round(float(np.random.uniform(500, 3000)), 2)
                events_warmup.append({
                    "Timestamp": t_w,
                    "From Bank": np.random.choice(banks),
                    "Account": acc,
                    "To Bank": np.random.choice(banks),
                    "Account.1": np.random.choice(legit_merchants),
                    "Amount Received": amt,
                    "Receiving Currency": cur,
                    "Amount Paid": amt,
                    "Payment Currency": cur,
                    "Payment Format": "ACH",
                    "Is Laundering": 0,
                    "Pattern": "Warmup",
                })
        # 10 sequential bursts over 30 days
        for b_idx in range(10):
            b_time = start_time + timedelta(days=7 + b_idx * 2, hours=s_id * 1, minutes=int(np.random.uniform(0, 30)))
            amt = round(float(np.random.uniform(35000, 95000)), 2)
            cur = "Euro" if s_id % 2 == 0 else "US Dollar"
            for step in range(chain_len - 1):
                b_time += timedelta(minutes=int(np.random.uniform(1, 10)))
                events_laundering.append({
                    "Timestamp": b_time,
                    "From Bank": f"0{((s_id + step) % 12) + 1:02d}",
                    "Account": chain_accs[step],
                    "To Bank": f"0{((s_id + step + 1) % 12) + 1:02d}",
                    "Account.1": chain_accs[step + 1],
                    "Amount Received": amt,
                    "Receiving Currency": cur,
                    "Amount Paid": amt,
                    "Payment Currency": cur,
                    "Payment Format": "ACH",
                    "Is Laundering": 1,
                    "Pattern": "Layering",
                })
                amt = round(amt * 0.988, 2)
                
    assert len(events_laundering) == n_layering, f"Layering count: {len(events_laundering)} != {n_layering}"
    
    # ----------------------------------------------------
    # Pattern 2: Cycle Rings (6 accounts each, 6 steps per execution)
    # ----------------------------------------------------
    num_rings = 10 if total_rows == 10000 else 15
    exec_per_ring = n_cycle // (num_rings * 6)
    for r_id in range(num_rings):
        c_accs = [f"84{r_id:03d}{step:02d}0" for step in range(6)]
        for acc in c_accs:
            for w in range(4):
                t_w = start_time + timedelta(seconds=int(np.random.uniform(100, 500000)))
                cur = "Euro"
                amt = round(float(np.random.uniform(1000, 4000)), 2)
                events_warmup.append({
                    "Timestamp": t_w,
                    "From Bank": np.random.choice(banks),
                    "Account": acc,
                    "To Bank": np.random.choice(banks),
                    "Account.1": np.random.choice(legit_merchants),
                    "Amount Received": amt,
                    "Receiving Currency": cur,
                    "Amount Paid": amt,
                    "Payment Currency": cur,
                    "Payment Format": "ACH",
                    "Is Laundering": 0,
                    "Pattern": "Warmup",
                })
        for exec_idx in range(exec_per_ring):
            r_time = start_time + timedelta(days=6 + exec_idx * 1.2, hours=r_id * 1.5)
            amt = round(float(np.random.uniform(25000, 85000)), 2)
            for s in range(len(c_accs)):
                r_time += timedelta(minutes=int(np.random.uniform(2, 8)))
                events_laundering.append({
                    "Timestamp": r_time,
                    "From Bank": f"0{(s % 8) + 1:02d}",
                    "Account": c_accs[s],
                    "To Bank": f"0{((s + 1) % 8) + 1:02d}",
                    "Account.1": c_accs[(s + 1) % len(c_accs)],
                    "Amount Received": amt,
                    "Receiving Currency": "Euro",
                    "Amount Paid": amt,
                    "Payment Currency": "Euro",
                    "Payment Format": "ACH",
                    "Is Laundering": 1,
                    "Pattern": "Cycle",
                })
                amt = round(amt * 0.995, 2)
                
    assert len(events_laundering) == (n_layering + n_cycle), f"Cycle count: {len(events_laundering)}"
    
    # ----------------------------------------------------
    # Pattern 3: High-Velocity Fan-Out (Scatter) (12 receivers * 10 bursts = 120 tx / distributor)
    # ----------------------------------------------------
    num_distributors = n_fanout // 120
    for d_id in range(num_distributors):
        distributor = f"830{d_id:04d}0"
        receivers = [f"831{d_id:03d}{r:02d}0" for r in range(12)]
        for w in range(6):
            t_w = start_time + timedelta(seconds=int(np.random.uniform(100, 500000)))
            events_warmup.append({
                "Timestamp": t_w,
                "From Bank": "002",
                "Account": distributor,
                "To Bank": np.random.choice(banks),
                "Account.1": np.random.choice(legit_merchants),
                "Amount Received": round(float(np.random.uniform(2000, 8000)), 2),
                "Receiving Currency": "US Dollar",
                "Amount Paid": round(float(np.random.uniform(2000, 8000)), 2),
                "Payment Currency": "US Dollar",
                "Payment Format": "ACH",
                "Is Laundering": 0,
                "Pattern": "Warmup",
            })
        for b_idx in range(10):
            d_time = start_time + timedelta(days=8 + b_idx * 2, hours=d_id * 2)
            for r_idx, r in enumerate(receivers):
                d_time += timedelta(minutes=int(np.random.uniform(1, 6)))
                amt = round(float(np.random.uniform(9200, 9950)), 2)
                events_laundering.append({
                    "Timestamp": d_time,
                    "From Bank": "002",
                    "Account": distributor,
                    "To Bank": f"0{(r_idx % 12) + 1:02d}",
                    "Account.1": r,
                    "Amount Received": amt,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": amt,
                    "Payment Currency": "US Dollar",
                    "Payment Format": "ACH",
                    "Is Laundering": 1,
                    "Pattern": "Fan-Out",
                })
                
    assert len(events_laundering) == (n_layering + n_cycle + n_fanout), f"Fan-Out count: {len(events_laundering)}"
    
    # ----------------------------------------------------
    # Pattern 4: Fan-In (Smurfing Gather) (10 smurfs * 10 bursts = 100 tx / collector)
    # ----------------------------------------------------
    num_collectors = n_fanin // 100
    for col_id in range(num_collectors):
        collector = f"820{col_id:04d}0"
        smurfs = [f"821{col_id:03d}{s:02d}0" for s in range(10)]
        for s in smurfs:
            for w in range(3):
                t_w = start_time + timedelta(seconds=int(np.random.uniform(100, 500000)))
                events_warmup.append({
                    "Timestamp": t_w,
                    "From Bank": np.random.choice(banks),
                    "Account": s,
                    "To Bank": np.random.choice(banks),
                    "Account.1": np.random.choice(legit_merchants),
                    "Amount Received": round(float(np.random.uniform(500, 2000)), 2),
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": round(float(np.random.uniform(500, 2000)), 2),
                    "Payment Currency": "US Dollar",
                    "Payment Format": "ACH",
                    "Is Laundering": 0,
                    "Pattern": "Warmup",
                })
        for b_idx in range(10):
            g_time = start_time + timedelta(days=9 + b_idx * 2, hours=col_id * 1.5)
            for s_idx, smurf in enumerate(smurfs):
                g_time += timedelta(minutes=int(np.random.uniform(1, 8)))
                amt = round(float(np.random.uniform(9100, 9900)), 2)
                events_laundering.append({
                    "Timestamp": g_time,
                    "From Bank": f"0{(s_idx % 10) + 1:02d}",
                    "Account": smurf,
                    "To Bank": "001",
                    "Account.1": collector,
                    "Amount Received": amt,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": amt,
                    "Payment Currency": "US Dollar",
                    "Payment Format": "ACH",
                    "Is Laundering": 1,
                    "Pattern": "Fan-In",
                })
                
    assert len(events_laundering) == target_pos, f"Total laundering must be {target_pos}, got {len(events_laundering)}"
    
    # ----------------------------------------------------
    # Legitimate Background Traffic
    # ----------------------------------------------------
    warmup_count = len(events_warmup)
    bg_needed = target_neg - warmup_count
    print(f"Warmup events: {warmup_count:,} | Background traffic needed: {bg_needed:,}")
    
    t_bg = start_time
    for i in range(bg_needed):
        t_bg += timedelta(seconds=int(np.random.exponential(150) + 10))
        m = np.random.choice(legit_merchants)
        c = np.random.choice(legit_merchants)
        while c == m:
            c = np.random.choice(legit_merchants)
        cur = np.random.choice(currencies)
        fmt = np.random.choice(formats)
        amt = round(float(np.random.lognormal(mean=5.5, sigma=1.1)), 2)
        amt = max(10.0, min(amt, 75000.0))
        events_background.append({
            "Timestamp": t_bg,
            "From Bank": np.random.choice(banks),
            "Account": m,
            "To Bank": np.random.choice(banks),
            "Account.1": c,
            "Amount Received": amt,
            "Receiving Currency": cur,
            "Amount Paid": amt,
            "Payment Currency": cur,
            "Payment Format": fmt,
            "Is Laundering": 0,
            "Pattern": "Legitimate",
        })
        
    all_events = events_laundering + events_warmup + events_background
    df = pd.DataFrame(all_events).sort_values(by="Timestamp").reset_index(drop=True)
    
    assert len(df) == total_rows, f"Expected {total_rows} rows, got {len(df)}"
    pos_actual = int((df["Is Laundering"] == 1).sum())
    neg_actual = int((df["Is Laundering"] == 0).sum())
    assert pos_actual == target_pos, f"Expected {target_pos} positives, got {pos_actual}"
    assert neg_actual == target_neg, f"Expected {target_neg} negatives, got {neg_actual}"
    
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y/%m/%d %H:%M")
    schema_cols = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]
    return df[schema_cols]


def main():
    datasets = [
        (10000, "ibm_aml_multi_sequence_10000.csv"),
        (20000, "ibm_aml_multi_sequence_20000.csv"),
        (30000, "ibm_aml_multi_sequence_30000.csv"),
    ]
    
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    
    for count, fname in datasets:
        df = generate_sequence_dataset(count, seed=42)
        
        # Save to both root and data/demo/
        root_path = PROJECT_ROOT / fname
        demo_path = DEMO_DIR / fname
        
        df.to_csv(root_path, index=False)
        df.to_csv(demo_path, index=False)
        
        pos = int((df["Is Laundering"] == 1).sum())
        neg = int((df["Is Laundering"] == 0).sum())
        pos_pct = (pos / len(df)) * 100
        neg_pct = (neg / len(df)) * 100
        n_accounts = df["Account"].nunique()
        avg_depth = len(df) / n_accounts
        
        print(f"✓ Saved {fname}:")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Legit (0): {neg:,} ({neg_pct:.1f}%)")
        print(f"  - Laundering (1): {pos:,} ({pos_pct:.1f}%)")
        print(f"  - Unique Sender Accounts: {n_accounts:,} (Avg Depth: {avg_depth:.2f} tx/account)")
        print(f"  - Paths: {root_path.relative_to(PROJECT_ROOT)} and {demo_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

