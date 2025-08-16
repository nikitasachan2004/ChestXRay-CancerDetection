import os, random, json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_config(cfg: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)


def metrics_binary(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    return {'acc': acc, 'f1': f1, 'auc': auc}


def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save({'state_dict': model.state_dict()}, path)