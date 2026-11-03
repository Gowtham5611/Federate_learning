# FedSecureAI — Reproduction Commands: Baseline Recovery & Forensic Diagnosis

This document contains the exact PowerShell and Python commands required to reproduce all findings, metrics, and comparisons documented in [`artifacts/fedavg_vs_fedprox_diagnosis/`](file:///c:/Users/Acer/OneDrive/Desktop/Federate_learning/artifacts/fedavg_vs_fedprox_diagnosis/).

---

## 1. Environment & Prerequisites

Ensure the virtual environment is activated in PowerShell:
```powershell
cd c:\Users\Acer\OneDrive\Desktop\Federate_learning
$env:PYTHONPATH = "backend;$env:PYTHONPATH"
```

---

## 2. Verify Historical FedAvg Checkpoint & Recompute Metrics

To verify the recovered `federated_iid_fedavg_lstm_seed42_20260918T130515` checkpoint and evaluate it directly on the holdout test split:

```powershell
.venv\Scripts\python -c "
import sys
sys.path.insert(0, 'backend')
import torch
import numpy as np
from pathlib import Path
from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from sklearn.metrics import roc_auc_score, average_precision_score

ckpt_path = 'models/checkpoints/federated/iid/fedavg/lstm/federated_iid_fedavg_lstm_seed42_20260918T130515/global_best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

cfg = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=64, num_layers=1, row_limit=50000, batch_size=1024)
model = build_classifier(cfg, feature_dimension=57)
model.load_state_dict(ckpt['parameters'])
model.eval()

prov = ProcessedSequenceProvider(Path('data/processed/ibm_aml/HI-Small'))
ds_test = prov.load(SplitName.TEST, cfg)

all_probs = []
with torch.no_grad():
    for batch in ds_test.batches(batch_size=1024, shuffle=False):
        bx = torch.from_numpy(batch.features)
        bl = torch.from_numpy(batch.lengths)
        logits = model(bx, bl)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

probs = np.concatenate(all_probs)
labels = ds_test.labels.astype(int)

roc_auc = roc_auc_score(labels, probs)
pr_auc = average_precision_score(labels, probs)

print(f'Test on first 50k rows: sequences={len(labels)}, positives={labels.sum()}')
print(f'ROC-AUC: {roc_auc:.6f}, PR-AUC: {pr_auc:.6f}')
print(f'Mean prob pos: {probs[labels==1].mean():.6f}, Mean prob neg: {probs[labels==0].mean():.6f}')
"
```
**Expected Output**:
`Test on first 50k rows: sequences=50000, positives=76`  
`ROC-AUC: 0.980609, PR-AUC: 0.458571`  
`Mean prob pos: 0.214993, Mean prob neg: 0.000793`

---

## 3. Verify the Phase 9 FedProx Monotonic Collapse Across $\mu$

To scan all 16 saved JSON reports in `experiments/results/phase9_federated_optimization/fedprox/reports/` and print the progression of Test ROC-AUC as $\mu$ increases:

```powershell
.venv\Scripts\python -c "
import glob, json, os

reports = glob.glob('experiments/results/phase9_federated_optimization/fedprox/reports/*.json')
results = []
for r in reports:
    with open(r, 'r') as f:
        d = json.load(f)
        mu = d.get('mu')
        arch = d.get('architecture')
        val_roc = d.get('val_roc_auc', 0.0)
        test_roc = d.get('test_roc_auc', 0.0)
        val_pr = d.get('val_pr_auc', 0.0)
        test_pr = d.get('test_pr_auc', 0.0)
        results.append((arch, mu, val_roc, test_roc, val_pr, test_pr, os.path.basename(r)))

results.sort(key=lambda x: (x[0] or '', x[1] if x[1] is not None else -1))
print('Arch       mu         Val ROC    Test ROC   Val PR     Test PR    File')
for arch, mu, vr, tr, vp, tp, fn in results:
    print(f'{str(arch):<10} {str(mu):<10} {vr:<10.4f} {tr:<10.4f} {vp:<10.6f} {tp:<10.6f} {fn}')
"
```

---

## 4. Run the Full Matched Algorithm Comparison Script

To recompute all metrics across the 6 matched cohorts (Controlled 100k vs Full 3M; FedAvg vs FedProx $\mu=0.1$ and $\mu=1.0$) using strict validation-frozen thresholds:

```powershell
.venv\Scripts\python -c "
import sys
sys.path.insert(0, 'backend')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.models.pytorch.sequence_classifier import build_classifier
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider

prov = ProcessedSequenceProvider(Path('data/processed/ibm_aml/HI-Small'))
cfg_val = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=64, num_layers=1, row_limit=50000, batch_size=1024)
cfg_test = TrainingConfig(architecture=ModelArchitecture.LSTM, hidden_size=64, num_layers=1, row_limit=50000, batch_size=1024)

ds_val = prov.load(SplitName.VALIDATION, cfg_val)
ds_test = prov.load(SplitName.TEST, cfg_test)

y_val_true = ds_val.labels.astype(int)
y_test_true = ds_test.labels.astype(int)

ckpts = [
    ('HI-Small_Controlled_100k', 'LSTM', 'Non-IID', 'FedAvg', 0.0, 'artifacts/dataset_comparison_combined/checkpoints/hi_small_non_iid_fedavg_lstm_seed42.pt'),
    ('HI-Small_Controlled_100k', 'LSTM', 'Non-IID', 'FedProx', 0.1, 'artifacts/dataset_comparison_combined/checkpoints/hi_small_non_iid_fedprox_mu0.1_lstm_seed42.pt'),
    ('HI-Small_Controlled_100k', 'LSTM', 'Non-IID', 'FedProx', 1.0, 'artifacts/dataset_comparison_combined/checkpoints/hi_small_non_iid_fedprox_mu1.0_lstm_seed42.pt'),
    ('HI-Small_FullData_3M', 'LSTM', 'Non-IID', 'FedAvg', 0.0, 'models/checkpoints/federated/non_iid/fedavg/lstm/federated_non_iid_fedavg_lstm_seed42_20260922T155231/global_best.pt'),
    ('HI-Small_FullData_3M', 'LSTM', 'IID', 'FedAvg', 0.0, 'models/checkpoints/federated/iid/fedavg/lstm/federated_iid_fedavg_lstm_seed42_20260918T130515/global_best.pt'),
    ('HI-Small_FullData_3M', 'LSTM', 'Non-IID', 'FedProx', 0.1, 'models/checkpoints/federated/non_iid/fedprox_tuned/lstm/phase9_fedprox_lstm_mu_0p1_seed42/round_1.pt'),
]

records = []
for cohort, arch, dist, algo, mu, p in ckpts:
    ckpt = torch.load(p, map_location='cpu')
    params = ckpt.get('parameters', ckpt.get('state_dict'))
    m = build_classifier(cfg_test, feature_dimension=57)
    m.load_state_dict(params)
    m.eval()
    
    # Val
    v_probs = []
    with torch.no_grad():
        for b in ds_val.batches(batch_size=1024, shuffle=False):
            bx = torch.from_numpy(b.features)
            bl = torch.from_numpy(b.lengths)
            v_probs.append(torch.sigmoid(m(bx, bl)).view(-1).cpu().numpy())
    v_probs = np.concatenate(v_probs)
    
    # Select threshold on validation
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_val_true, (v_probs >= tau).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_tau = tau
            
    val_roc = roc_auc_score(y_val_true, v_probs)
    val_pr = average_precision_score(y_val_true, v_probs)
    
    # Test
    t_probs = []
    with torch.no_grad():
        for b in ds_test.batches(batch_size=1024, shuffle=False):
            bx = torch.from_numpy(b.features)
            bl = torch.from_numpy(b.lengths)
            t_probs.append(torch.sigmoid(m(bx, bl)).view(-1).cpu().numpy())
    t_probs = np.concatenate(t_probs)
    
    test_roc = roc_auc_score(y_test_true, t_probs)
    test_pr = average_precision_score(y_test_true, t_probs)
    
    y_pred = (t_probs >= best_tau).astype(int)
    cm = confusion_matrix(y_test_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_test_true, y_pred, zero_division=0)
    rec = recall_score(y_test_true, y_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    alerts_per_10k = (fp / (tn + fp)) * 10000.0
    
    records.append({
        'cohort': cohort,
        'architecture': arch,
        'partition': dist,
        'algorithm': algo,
        'mu': mu,
        'frozen_val_threshold': round(best_tau, 4),
        'val_roc_auc': round(val_roc, 6),
        'val_pr_auc': round(val_pr, 6),
        'val_f1': round(best_f1, 6),
        'test_roc_auc': round(test_roc, 6),
        'test_avg_precision': round(test_pr, 6),
        'test_f1': round(f1, 6),
        'test_precision': round(prec, 6),
        'test_recall': round(rec, 6),
        'test_fpr': round(fpr, 6),
        'test_specificity': round(spec, 6),
        'false_alerts_per_10k': round(alerts_per_10k, 2),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'n_pos_test': int(tp + fn),
        'n_neg_test': int(tn + fp),
    })

df_res = pd.DataFrame(records)
print(df_res.to_string())
df_res.to_csv('artifacts/fedavg_vs_fedprox_diagnosis/matched_algorithm_metrics.csv', index=False)
"
```

---

## 5. Verify Dataset Split Metadata & Positive Counts on Disk

To inspect the true split dimensions and positive counts recorded in `splits.json`:

```powershell
.venv\Scripts\python -c "
import json
for d in ['HI-Small', 'LI-Medium', 'Combined']:
    path = f'data/processed/ibm_aml/{d}/metadata/splits.json'
    with open(path) as f:
        meta = json.load(f)
    print(f'=== {d} ===')
    for s in meta.get('per_split', []):
        print(f'  {s[\"name\"]:12}: rows={s[\"rows\"]:<10} positives={s[\"positive\"]:<6} rate={s[\"positive_rate\"]:.6f}')
"
```

