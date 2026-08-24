"""Honest multi-model MAC-disjoint sweep on the organic dataset.csv.

Trains each model from scratch on 9 MACs, evaluates on 4 unseen MACs.
Uses the same 4-class interference labels and the same train/val/test split
logic as run_mac_disjoint.py. Purpose: a side-by-side honest comparison
of architectures, including DeepStack, MoE-Mixed, MetaFusion, and CNN.
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

from dl.config import DLConfig
from dl.data_loader import DLDataLoader

TEST_MACS = [
    "04:b1:67:ac:8d:65",
    "e6:53:5c:2a:e8:e2",
    "4e:9a:ff:cd:da:f3",
    "f6:12:7b:46:29:86",
]


def _load_features(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = DLConfig()
    cfg.data_path = data_path
    cfg.in_features = 4
    dl = DLDataLoader(cfg)
    X, y = dl.load_and_preprocess()
    return X.astype(np.float32), y


def _build(name: str, in_features: int = 4) -> torch.nn.Module:
    if name == "cnn":
        from dl.models.cnn import CNNClassifier
        return CNNClassifier(in_features=in_features, num_filters=32, num_blocks=2, num_classes=4)
    if name == "lstm":
        from dl.models.lstm import LSTMClassifier
        return LSTMClassifier(in_features=in_features, num_classes=4, hidden_size=64, num_layers=2)
    if name == "mamba":
        from dl.models.mamba import MambaClassifier
        return MambaClassifier(in_features=in_features, num_classes=4, d_model=64, num_layers=2)
    if name == "transformer":
        from dl.models.transformer import TransformerClassifier
        return TransformerClassifier(in_features=in_features, num_classes=4, d_model=64, n_heads=4, num_layers=2)
    if name == "deepstack":
        from dl.models.cnn import CNNClassifier
        from dl.models.lstm import LSTMClassifier
        from dl.models.transformer import TransformerClassifier
        from dl.models.gru import GRUClassifier
        from dl.models.deep_stack import DeepStackEnsemble, Level2Net
        base = [
            CNNClassifier(in_features=in_features, num_filters=16, num_blocks=1, num_classes=4),
            LSTMClassifier(in_features=in_features, num_classes=4, hidden_size=32, num_layers=1),
            TransformerClassifier(in_features=in_features, num_classes=4, d_model=32, n_heads=2, num_layers=1),
            GRUClassifier(in_features=in_features, num_classes=4, hidden_size=32, num_layers=1),
        ]
        l2 = [Level2Net(in_features=4 * 4, num_classes=4) for _ in range(2)]
        return DeepStackEnsemble(base_models=base, level2_nets=l2, num_classes=4, dropout=0.3)
    if name == "moe_mixed":
        # Soft mixture of experts (arbitrary count) instead of the binary-class
        # one-vs-rest MoE which expects exactly 4 experts and gets confused.
        from dl.models.moe import SoftMixtureOfExperts
        from dl.models.gru import GRUClassifier
        from dl.models.lstm import LSTMClassifier
        from dl.models.cnn import CNNClassifier
        from dl.models.transformer import TransformerClassifier
        experts = [
            GRUClassifier(in_features=in_features, num_classes=4, hidden_size=32, num_layers=1),
            LSTMClassifier(in_features=in_features, num_classes=4, hidden_size=32, num_layers=1),
            CNNClassifier(in_features=in_features, num_filters=16, num_blocks=1, num_classes=4),
            TransformerClassifier(in_features=in_features, num_classes=4, d_model=32, n_heads=2, num_layers=1),
        ]
        return SoftMixtureOfExperts(
            experts=experts, in_features=in_features, num_classes=4,
        )
    if name == "metafusion":
        from dl.models.meta_fusion import MetaFusionClassifier
        return MetaFusionClassifier(in_features=in_features, hidden_size=64, num_classes=4)
    raise ValueError(name)


def _train_eval(
    model: torch.nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    epochs: int, lr: float, device: torch.device,
) -> tuple[float, float, int]:
    def loader(X, y, shuffle=True, batch=128):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch, shuffle=shuffle,
        )

    train_loader = loader(X_tr, y_tr, shuffle=True)
    val_loader = loader(X_va, y_va, shuffle=False)
    test_loader = loader(X_te, y_te, shuffle=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()

    best_val_mcc, best_state, patience, patience_counter = -1.0, None, 12, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            loss = crit(logits, yb)
            if hasattr(model, "last_aux_loss"):
                loss = loss + model.last_aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                logits = out[0] if isinstance(out, tuple) else out
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
                break

    model.load_state_dict(best_state)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            tp.extend(logits.argmax(1).cpu().tolist())
            tt.extend(yb.tolist())

    test_mcc = float(matthews_corrcoef(tt, tp))
    test_acc = float((np.array(tp) == np.array(tt)).mean())
    return best_val_mcc, test_mcc, test_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--models", nargs="+", default=["cnn", "lstm", "transformer", "deepstack", "moe_mixed"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data)
    X, y = _load_features(args.data)
    mac = df["mac"].values

    test_mask = np.isin(mac, TEST_MACS)
    train_mask = ~test_mask
    X_tr_all, y_tr_all = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_tr_all)), test_size=0.15, random_state=args.seed, stratify=y_tr_all,
    )
    X_tr, X_va = X_tr_all[tr_idx], X_tr_all[va_idx]
    y_tr, y_va = y_tr_all[tr_idx], y_tr_all[va_idx]

    print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}  (test MACs: {len(TEST_MACS)})")
    print(f"\n{'Model':<14} {'Val MCC':>10} {'Test MCC':>10} {'Test Acc':>10}")
    print("-" * 48)
    for name in args.models:
        torch.manual_seed(args.seed)
        try:
            model = _build(name, in_features=X.shape[-1])
            val_mcc, test_mcc, test_acc = _train_eval(
                model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                args.epochs, args.lr, device,
            )
            print(f"{name:<14} {val_mcc:>10.4f} {test_mcc:>10.4f} {test_acc:>10.4f}")
        except Exception as e:
            print(f"{name:<14} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
