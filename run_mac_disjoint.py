"""MAC-disjoint generalization test.

Trains a model from scratch on 9 held-out MAC devices and evaluates on 4
unseen devices. This measures whether the model learns the interference
pattern (AA/AB/BA/BB) or just memorizes per-device RSSI signatures.

Usage:
    uv run python run_mac_disjoint.py --data dataset_augmented3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dl.data_loader import DLDataLoader
from dl.config import DLConfig

# 4 test devices chosen to cover all four classes with a balanced mix.
TEST_MACS = [
    "04:b1:67:ac:8d:65",
    "e6:53:5c:2a:e8:e2",
    "4e:9a:ff:cd:da:f3",
    "f6:12:7b:46:29:86",
]


def _build_sigreg() -> torch.nn.Module:
    from dl.models.sigreg_classifier import SIGRegClassifier
    return SIGRegClassifier(
        in_features=4, num_filters=192, num_blocks=3,
        latent_dim=128, num_classes=4, sigreg_lambda=0.01,
    )


def _load_features(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = DLConfig()
    cfg.data_path = data_path
    cfg.rich_features = False
    cfg.in_features = 4
    dl = DLDataLoader(cfg)
    X, y = dl.load_and_preprocess()
    return X.astype(np.float32), y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset_augmented3.csv"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data)
    X, y = _load_features(args.data)

    mac = df["mac"].values
    label = df["label"].values
    synthetic = df["synthetic"].values

    test_mask = np.isin(mac, TEST_MACS)
    train_mask = ~test_mask

    # Label encoder is already applied by load_and_preprocess; keep int labels.
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    te_syn = synthetic[test_mask]

    print(f"Train: {len(X_tr)} rows, {np.unique(mac[train_mask]).size} MACs")
    print(f"Test:  {len(X_te)} rows, {np.unique(mac[test_mask]).size} MACs")
    print(f"  test label counts: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    # Internal val split (stratified by label) for early stopping.
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_tr)), test_size=0.15, random_state=args.seed, stratify=y_tr,
    )
    X_tr2, X_va = X_tr[tr_idx], X_tr[va_idx]
    y_tr2, y_va = y_tr[tr_idx], y_tr[va_idx]

    def loader(X, y, shuffle=True, batch=256):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch, shuffle=shuffle,
        )

    train_loader = loader(X_tr2, y_tr2, shuffle=True)
    val_loader = loader(X_va, y_va, shuffle=False)
    test_loader = loader(X_te, y_te, shuffle=False)

    model = _build_sigreg().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_mcc = -1.0
    best_state = None
    patience = 10
    patience_counter = 0

    print(f"\nTraining {args.epochs} epochs on {len(X_tr2)} train rows...")
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, latents = model(xb)
            loss, _ = model.compute_loss(logits, latents, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        # Val
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb.to(device))
                vp.extend(logits.argmax(1).cpu().tolist())
                vt.extend(yb.tolist())
        val_mcc = float(matthews_corrcoef(vt, vp))

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stop epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _ = model(xb.to(device))
            tp.extend(logits.argmax(1).cpu().tolist())
            tt.extend(yb.tolist())

    full_mcc = float(matthews_corrcoef(tt, tp))
    full_acc = float((np.array(tp) == np.array(tt)).mean())

    # Real (non-synthetic) test rows only — purest unseen-device signal.
    real_idx = np.where(~te_syn)[0]
    real_mcc = float(matthews_corrcoef(
        np.array(tt)[real_idx], np.array(tp)[real_idx],
    ))

    print(f"\nBest val MCC:        {best_val_mcc:.4f}")
    print(f"MAC-disjoint test MCC:   {full_mcc:.4f}  (acc {full_acc:.4f})")
    print(f"MAC-disjoint real-only:  {real_mcc:.4f}  (n={len(real_idx)})")


if __name__ == "__main__":
    main()
