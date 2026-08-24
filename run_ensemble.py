"""Greedy ensemble selection over existing checkpoints.

Research-backed combiner: Caruana's forward selection with replacement
("ensemble selection from libraries of models"), which the 2026 TFM-ensembling
study recommends as the practical default over plain weighted averaging and
stacking. It adds the base whose inclusion maximizes validation accuracy at
each step, with replacement, so diverse-but-redundant models can each earn
weight without a meta-learner.

Add a model in one line by appending a dict to MODELS:
    {
        "name": "foo",
        "rich": False,              # True if it needs 18-channel rich features
        "build": lambda: build_foo(),  # returns an nn.Module, logits already loaded
    }

Run:
    uv run python run_ensemble.py --data dataset_augmented3.csv --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl.config import DLConfig
from dl.data_loader import DLDataLoader


def _build_lejepa() -> nn.Module:
    from dl.models.lejepa import LeJEPAModel, LeJEPAClassifier
    pretrained = LeJEPAModel(
        seq_len=10, in_channels=4, d_model=256, nhead=8,
        num_layers=4, dim_feedforward=512, pred_num_layers=2, sigreg_lambda=0.1,
    )
    return LeJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_t_jepa() -> nn.Module:
    from dl.models.t_jepa import TJEPAModel, TJEPAClassifier
    pretrained = TJEPAModel(
        n_timesteps=10, n_channels=18, d_model=512, nhead=8,
        num_layers=4, dim_feedforward=512, n_reg_tokens=2,
        pred_dim=128, pred_num_layers=2,
    )
    return TJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_ts_jepa() -> nn.Module:
    from dl.models.ts_jepa import TSJEPAModel, TSJEPAClassifier
    pretrained = TSJEPAModel(
        seq_len=10, patch_size=2, in_channels=18, embed_dim=256, nhead=8,
        num_layers=4, dim_feedforward=512, pred_dim=128, pred_num_layers=2,
    )
    return TSJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_sigreg() -> nn.Module:
    from dl.models.sigreg_classifier import SIGRegClassifier
    m = SIGRegClassifier(
        in_features=4, num_filters=192, num_blocks=3, latent_dim=128,
        num_classes=4, sigreg_lambda=0.01,
    )
    return _TupleWrap(m)


def _build_mamba3_cnn() -> nn.Module:
    from dl.models.mamba3_cnn import Mamba3CNN
    return Mamba3CNN(
        in_features=4, cnn_channels=192, d_model=192, d_state=16,
        n_mamba_layers=3, num_classes=4, dropout=0.2, mimo_rank=4,
    )


class _TupleWrap(nn.Module):
    """SIGReg returns (logits, latents); expose logits only."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: Tensor) -> Tensor:
        return self.m(x)[0]


# One entry per model. Add more by appending here.
MODELS: list[dict] = [
    {"name": "lejepa",   "rich": False, "build": _build_lejepa,   "ckpt": "models/exotic/lejepa_ft_seed42.pt"},
    {"name": "t_jepa",   "rich": True,  "build": _build_t_jepa,   "ckpt": "models/exotic/t_jepa_ft_seed42.pt"},
    {"name": "ts_jepa",  "rich": True,  "build": _build_ts_jepa,  "ckpt": "models/exotic/ts_jepa_ft_seed42.pt"},
    {"name": "sigreg",   "rich": False, "build": _build_sigreg,   "ckpt": "models/exotic/sigreg_seed42.pt"},
    {"name": "mamba3_cnn", "rich": False, "build": _build_mamba3_cnn, "ckpt": "models/exotic/mamba3_cnn_seed42.pt"},
]


def _strip_prefix(sd: dict, prefix: str = "c.") -> dict:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _add_prefix(sd: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _load_model(spec: dict, device: torch.device) -> nn.Module:
    m = spec["build"]()
    state = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    # JEPA classifier checkpoints are wrapped (keys prefixed "c."); the SIGReg
    # _TupleWrap expects a "m." prefix that its raw checkpoint does not have.
    if any(k.startswith("c.") for k in state):
        state = _strip_prefix(state, "c.")
    elif spec["name"] == "sigreg":
        state = _add_prefix(state, "m.")
    m.load_state_dict(state)
    return m.to(device).eval()


def _predict_probs(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    parts = []
    with torch.no_grad():
        for (xb,) in dl:
            parts.append(torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _load_features(data_path: Path, rich: bool) -> np.ndarray:
    cfg = DLConfig()
    cfg.data_path = data_path
    cfg.rich_features = rich
    if rich:
        cfg.in_features = 18
    dl = DLDataLoader(cfg)
    X, _ = dl.load_and_preprocess()
    return X.astype(np.float32)


def _stratified_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same split as run_exotic.load_supervised_data: 70/10/20 train/val/test."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr_val_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y,
    )
    y_tr_val = y[tr_val_idx]
    train_idx, val_idx = train_test_split(
        tr_val_idx, test_size=0.125, random_state=seed, stratify=y_tr_val,
    )
    return train_idx, val_idx, test_idx


def _greedy_weights(val_probs: list[np.ndarray], val_y: np.ndarray, steps: int = 50) -> np.ndarray:
    """Caruana ensemble selection with replacement. Returns (n_models,) counts."""
    n = len(val_probs)
    counts = np.zeros(n, dtype=np.int64)
    current = np.zeros_like(val_probs[0])
    for _ in range(steps):
        best_i, best_acc = -1, -1.0
        for i in range(n):
            cand = (current * counts.sum() + val_probs[i]) / (counts.sum() + 1)
            acc = float((cand.argmax(1) == val_y).mean())
            if acc > best_acc:
                best_acc, best_i = acc, i
        counts[best_i] += 1
        current = (current * (counts.sum() - 1) + val_probs[best_i]) / counts.sum()
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset_augmented3.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Labels drive the split once; both feature variants align to the same indices.
    import pandas as pd
    y = pd.read_csv(args.data)["label"].values
    train_idx, val_idx, test_idx = _stratified_indices(y, args.seed)
    y_val = y[val_idx]
    y_test = y[test_idx]
    from sklearn.preprocessing import LabelEncoder
    y_val = LabelEncoder().fit_transform(y_val).astype(np.int64)
    y_test = LabelEncoder().fit_transform(y_test).astype(np.int64)

    # Load both feature variants once.
    X4 = _load_features(args.data, rich=False)
    X18 = _load_features(args.data, rich=True)

    test_probs, val_probs, names = [], [], []
    print("\n=== Single-model test MCC ===")
    for spec in MODELS:
        model = _load_model(spec, device)
        X = X18 if spec["rich"] else X4
        vp = _predict_probs(model, X[val_idx], device)
        tp = _predict_probs(model, X[test_idx], device)
        mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
        print(f"  {spec['name']:<12} {mcc:.4f}")
        val_probs.append(vp)
        test_probs.append(tp)
        names.append(spec["name"])

    counts = _greedy_weights(val_probs, y_val, args.steps)
    weights = counts / counts.sum()
    print("\n=== Greedy ensemble weights ===")
    for name, w in zip(names, weights):
        print(f"  {name:<12} {w:.3f}")

    ens = sum(w * p for w, p in zip(weights, test_probs))
    ens_mcc = float(matthews_corrcoef(y_test, ens.argmax(1)))
    ens_acc = float((ens.argmax(1) == y_test).mean())
    print(f"\nEnsemble test MCC:  {ens_mcc:.4f}")
    print(f"Ensemble test acc:  {ens_acc:.4f}")


if __name__ == "__main__":
    main()
