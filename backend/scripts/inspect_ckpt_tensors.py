import torch
from pathlib import Path

ckpts = {
    "HI-Small_Phase7_Champion": "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt",
    "LI-Medium_Phase7_Champion": "results/LI_Medium_Results/checkpoints/config_5/global_best.pt",
    "HI-Small_Phase10_Reference": "models/checkpoints/federated/iid/fedavg/lstm/federated_iid_fedavg_lstm_seed42_20260918T130515/global_best.pt"
}

for name, path in ckpts.items():
    p = Path(path)
    if not p.exists(): continue
    d = torch.load(p, map_location='cpu', weights_only=False)
    params = d.get('parameters', {})
    print(f"\n==================== {name} ====================")
    for k, v in params.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

