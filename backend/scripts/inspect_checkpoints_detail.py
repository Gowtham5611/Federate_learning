import torch
import hashlib
from pathlib import Path

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

ckpts = {
    "HI-Small_Phase7_Champion": "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt",
    "LI-Medium_Phase7_Champion": "results/LI_Medium_Results/checkpoints/config_5/global_best.pt",
    "HI-Small_Phase10_Reference": "models/checkpoints/federated/iid/fedavg/lstm/federated_iid_fedavg_lstm_seed42_20260918T130515/global_best.pt"
}

for name, path in ckpts.items():
    p = Path(path)
    if p.exists():
        sha = get_sha256(p)
        data = torch.load(p, map_location='cpu', weights_only=False)
        print(f"=== {name} ===")
        print(f"Path: {p}")
        print(f"SHA-256: {sha}")
        print(f"Keys in checkpoint: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        if isinstance(data, dict):
            for k in ['architecture', 'model_architecture', 'hidden_size', 'num_layers', 'input_dim', 'input_size', 'config', 'training_config', 'round', 'epoch', 'best_val_score']:
                if k in data:
                    print(f"  {k}: {data[k]}")
            if 'state_dict' in data:
                print(f"  state_dict keys: {list(data['state_dict'].keys())}")
            elif 'model_state_dict' in data:
                print(f"  model_state_dict keys: {list(data['model_state_dict'].keys())}")
    else:
        print(f"=== {name} NOT FOUND at {p} ===")

