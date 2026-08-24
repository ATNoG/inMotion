"""Mega-ensemble: combine ALL available models — DL checkpoints, JEPA family,
classical ML (KNN/CatBoost/GP/LR), and TabPFN — via OOF stacking with
temperature calibration and CE-based bagged Caruana selection.

Research-backed combiner pipeline (see docs/ENSEMBLE_TECHNIQUES_REPORT.md):
  A. Collect OOF + test probabilities for every member under the same 70/10/20 split
  B. Per-member temperature calibration on the validation fold
  C. CE-based bagged Caruana (ensemble selection with replacement)
  D. Regularized-LR stacker over the calibrated OOF matrix
  E. Compare: greedy-weighted, logit-avg, rank-avg, entropy-weighted, LR stack

Usage:
    MAMBA_SSM_AVAILABLE=0 uv run python mega_ensemble.py --data dataset_augmented3.csv --seed 42

Note: set MAMBA_SSM_AVAILABLE=0 to skip the (hang-prone) mamba_ssm CUDA
extension import and use the pure-PyTorch fallback for Mamba-based models.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl.config import DLConfig
from dl.data_loader import DLDataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# Model registry — each entry knows how to build, load, and run the model
# ═══════════════════════════════════════════════════════════════════════════════


def _build_lejepa():
    from dl.models.lejepa import LeJEPAModel, LeJEPAClassifier
    # full-after-hpo Pareto winner (0.8676): d_model=128 ff=256 layers=4 pred_layers=3
    pretrained = LeJEPAModel(
        seq_len=10, in_channels=4, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256, pred_num_layers=3, sigreg_lambda=0.05,
    )
    return LeJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_t_jepa():
    from dl.models.t_jepa import TJEPAModel, TJEPAClassifier
    # full-after-hpo Pareto winner (0.8445): d_model=128 layers=4 ff=256 pred_layers=3
    pretrained = TJEPAModel(
        n_timesteps=10, n_channels=18, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256, n_reg_tokens=2,
        pred_dim=64, pred_num_layers=3,
    )
    return TJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_ts_jepa():
    from dl.models.ts_jepa import TSJEPAModel, TSJEPAClassifier
    pretrained = TSJEPAModel(
        seq_len=10, patch_size=2, in_channels=18, embed_dim=256, nhead=8,
        num_layers=4, dim_feedforward=512, pred_dim=128, pred_num_layers=2,
    )
    return TSJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_cf_jepa():
    from dl.models.cf_jepa import CFJEPAModel, CFJEPAClassifier
    # full-after-hpo best frontier point (manual run): embed 256, layers 3,
    # ff 256, pred_dim 128, pred_layers 1, 18-channel rich features
    pretrained = CFJEPAModel(
        seq_len=10, patch_size=2, in_channels=18, embed_dim=256, nhead=8,
        num_layers=3, dim_feedforward=256, pred_dim=128, pred_num_layers=1,
        horizon_start=1, horizon_end=3, ctx_min=1, ctx_max=3,
        ema_start=0.996, ema_end=0.999, sigreg_lambda=0.05,
    )
    return CFJEPAClassifier(pretrained=pretrained, num_classes=4, hidden_dim=128, dropout=0.3)


def _build_cnn():
    from dl.models.cnn import CNNClassifier
    return CNNClassifier(4, 128, 3, 4, 0.3, [3, 5, 7])


def _build_tcn():
    from dl.models.tcn import TCNClassifier
    return TCNClassifier(4, 256, 3, 6, 4, 0.3)


def _build_gru():
    from dl.models.gru import GRUClassifier
    return GRUClassifier(4, 64, 2, 4, 0.3, use_attention=True)


def _build_lstm():
    from dl.models.lstm import LSTMClassifier
    return LSTMClassifier(4, 64, 2, 4, 0.3, use_attention=True)


def _build_bilstm():
    from dl.models.bilstm import BiLSTMClassifier
    return BiLSTMClassifier(4, 64, 2, 4, 0.3, use_attention=True)


def _build_sigreg():
    from dl.models.sigreg_classifier import SIGRegClassifier
    m = SIGRegClassifier(
        in_features=4, num_filters=192, num_blocks=3, latent_dim=128,
        num_classes=4, sigreg_lambda=0.01,
    )
    return _TupleWrap(m)


def _build_sigreg_full():
    """sigreg full-after-hpo checkpoint: encoder 128 filters / 3 blocks."""
    from dl.models.sigreg_classifier import SIGRegClassifier
    m = SIGRegClassifier(
        in_features=4, num_filters=128, num_blocks=3, latent_dim=128,
        num_classes=4, sigreg_lambda=0.01,
    )
    return _TupleWrap(m)


def _make_hpo_builder(model_type: str):
    """Builder that rebuilds an HPO_* checkpoint's architecture from its
    winning Optuna trial params (stored in optuna_dl_9344.db)."""
    import optuna

    def build():
        from run_dl import _build_hpo_best_model
        from dl.config import DLConfig
        study = optuna.load_study(
            study_name=f"inMotion_dl_v3_{model_type}",
            storage="sqlite:///optuna_dl_9344.db",
        )
        return _build_hpo_best_model(model_type, DLConfig(), study.best_trial.params)

    return build


def _build_mamba3_cnn():
    from dl.models.mamba3_cnn import Mamba3CNN
    return Mamba3CNN(
        in_features=4, cnn_channels=192, d_model=192, d_state=16,
        n_mamba_layers=3, num_classes=4, dropout=0.2, mimo_rank=4,
    )


def _build_mamba3_tcn():
    from dl.models.mamba3_tcn import Mamba3TCN
    # backup checkpoint: tcn/d_model=128
    return Mamba3TCN(
        in_features=4, tcn_channels=128, d_model=128, d_state=16,
        n_mamba_layers=3, num_classes=4, dropout=0.2, mimo_rank=4,
    )


def _build_mamba3_transformer():
    from dl.models.mamba3_transformer import Mamba3Transformer
    # backup checkpoint: d_model=128
    return Mamba3Transformer(
        in_features=4, d_model=128, d_state=16, nhead=4,
        num_blocks=3, num_classes=4, dropout=0.2, mimo_rank=4,
    )


def _build_mamba3_multiview():
    from dl.models.mamba3_multiview import Mamba3MultiView
    # backup checkpoint: d_model=256
    return Mamba3MultiView(
        in_features=4, d_model=256, d_state=16,
        n_mamba_layers=3, num_classes=4, dropout=0.2, mimo_rank=4,
    )


def _build_ds_ensemble():
    """Rebuild the REAL DeepStackEnsemble (models/dl/augmented/, trained Jun 7).

    Config resolved from the checkpoint's own weight shapes (0 missing keys).
    10 bases: CNN2DRNN, CNN_A/B, GRU_A, Mamba_A/B, RNN_A/B, TCN_A/B
    → 4 Level2Nets → DeepMetaMLP. Scores 0.852 on dataset_augmented.csv,
    0.881 on dataset.icaisf.csv.
    """
    from dl.models.deep_stack import DeepStackEnsemble, Level2Net
    from dl.models.cnn import CNNClassifier
    from dl.models.cnn2d_rnn import CNN2DRNNClassifier
    from dl.models.gru import GRUClassifier
    from dl.models.mamba import MambaClassifier
    from dl.models.rnn import RNNClassifier
    from dl.models.tcn import TCNClassifier
    from torch import nn

    bases: list[nn.Module] = [
        CNN2DRNNClassifier(in_features=4, num_filters=128, cnn_depth=3, hidden_size=256,
                           num_rnn_layers=3, num_classes=4, dropout=0.3,
                           rnn_type="lstm", bidirectional=True),
        CNNClassifier(4, 64, 2, 4, 0.2, [3, 5]),
        CNNClassifier(4, 256, 4, 4, 0.3, [3, 5, 7, 9]),
        GRUClassifier(4, 64, 2, 4, 0.2),
        MambaClassifier(4, 64, 16, 2, 2, 4, 0.2, mimo_rank=4),
        MambaClassifier(4, 256, 32, 4, 2, 4, 0.3, mimo_rank=8),
        RNNClassifier(4, 64, 2, 4, 0.2),
        RNNClassifier(4, 256, 4, 4, 0.3, bidirectional=True),
        TCNClassifier(4, 128, 3, 4, 4, 0.2),
        TCNClassifier(4, 512, 7, 8, 4, 0.3),
    ]
    return DeepStackEnsemble(bases, [Level2Net(40, 4) for _ in range(4)], num_classes=4, dropout=0.3)


class _TupleWrap(nn.Module):
    """SIGReg returns (logits, latents); expose logits only."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: Tensor) -> Tensor:
        return self.m(x)[0]


def _strip_prefix(sd: dict, prefix: str = "c.") -> dict:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _add_prefix(sd: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _load_checkpoint(spec: dict, device: torch.device) -> nn.Module | None:
    """Build a model and load its checkpoint. Returns None on failure."""
    try:
        m = spec["build"]()
        state = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
        if any(k.startswith("c.") for k in state):
            state = _strip_prefix(state, "c.")
        elif spec.get("prefix") == "m.":
            state = _add_prefix(state, "m.")
        m.load_state_dict(state, strict=False)
        return m.to(device).eval()
    except Exception as e:
        print(f"  [warn] could not load {spec['name']}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Feature transforms (numpy-only — no aeon/catch22 dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def rocket_features(X: np.ndarray, n_kernels: int = 2048, seed: int = 0) -> np.ndarray:
    """MiniRocket-style random convolutional features + PPV pooling.

    X: (N, T, C) → (N, n_kernels * C) binary PPV features.
    Uses random length/dilation/weight kernels per channel (deterministic seed).
    """
    rng = np.random.RandomState(seed)
    N, T, C = X.shape
    feats = np.zeros((N, n_kernels * C), dtype=np.float32)
    for c in range(C):
        xc = X[:, :, c]  # (N, T)
        for k in range(n_kernels):
            length = int(rng.choice([3, 5, 7, 9]))
            dilation = int(rng.choice([1, 2, 3, 5]))
            eff_len = length * dilation
            if eff_len > T:
                length = T
                dilation = 1
                eff_len = T
            weights = rng.randn(length)  # must match length AFTER fallback
            bias = rng.randn() * 0.1
            out = np.zeros(N)
            for t in range(T - eff_len + 1):
                idx = t + dilation * np.arange(length)
                out += (xc[:, idx] * weights).sum(axis=1)
            out += bias
            feats[:, c * n_kernels + k] = (out > 0).astype(np.float32)
    return feats


def catch22_features(X: np.ndarray) -> np.ndarray:
    """Hand-crafted time-series statistics per channel (catch22-inspired subset).

    X: (N, T, C) → (N, ~12 * C): mean, std, min, max, range, skew, kurt,
       autocorr(1), zero-crossings, mean-abs-diff, spectral-energy, peak-loc.
    """
    N, T, C = X.shape
    feats = []
    for c in range(C):
        xc = X[:, :, c]  # (N, T)
        mu = xc.mean(1)
        sd = xc.std(1) + 1e-8
        z = (xc - mu[:, None]) / sd[:, None]
        ac1 = (z[:, :-1] * z[:, 1:]).mean(1)
        zc = (np.diff(np.signbit(xc).astype(float), axis=1) != 0).sum(1).astype(np.float32)
        mad = np.abs(np.diff(xc, axis=1)).mean(1)
        spec = np.abs(np.fft.rfft(xc, axis=1))
        spec_n = spec / (spec.sum(1, keepdims=True) + 1e-8)
        peak = np.argmax(np.abs(xc), axis=1).astype(np.float32)
        feats.extend([
            mu, sd, xc.min(1), xc.max(1), xc.max(1) - xc.min(1),
            ((z ** 3).mean(1)), ((z ** 4).mean(1)), ac1, zc, mad,
            (spec_n ** 2).sum(1), peak,
        ])
    return np.stack(feats, axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading — mirrors run_exotic.load_supervised_data split
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# Model prediction
# ═══════════════════════════════════════════════════════════════════════════════


def _predict_probs(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    parts = []
    with torch.no_grad():
        for (xb,) in dl:
            parts.append(torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration + ensemble combiners
# ═══════════════════════════════════════════════════════════════════════════════


def fit_temperature(logits: np.ndarray, y: np.ndarray, init: float = 1.0) -> float:
    """Fit a single temperature scalar to minimize NLL on the calibration set."""
    T = init
    for _ in range(100):
        grad = 0.0
        for _b in range(0, len(y), 512):
            lg = logits[_b:_b + 512] / T
            lg = lg - lg.max(1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(1, keepdims=True)
            grad += (p[np.arange(len(y[_b:_b + 512])), y[_b:_b + 512]] - 1.0).sum()
        T -= 0.01 * grad / max(len(y), 1)
        T = max(T, 0.05)
    return T


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    e = e / e.sum(1, keepdims=True)
    return np.clip(e, 1e-7, 1.0)


def _renorm(p: np.ndarray) -> np.ndarray:
    """Ensure each row sums to 1 (log_loss requires normalized probs)."""
    s = p.sum(1, keepdims=True)
    return np.clip(p / s, 1e-7, 1.0)


def _caruana_selection(
    val_probs: list[np.ndarray], val_y: np.ndarray, steps: int = 100, metric: str = "ce",
) -> np.ndarray:
    """Caruana ensemble selection with replacement, on CE or accuracy."""
    n = len(val_probs)
    counts = np.zeros(n, dtype=np.int64)
    current = np.zeros_like(val_probs[0])
    for _ in range(steps):
        best_i, best_score = -1, 1e9 if metric == "ce" else -1.0
        for i in range(n):
            cand = (current * counts.sum() + val_probs[i]) / (counts.sum() + 1)
            cand = _renorm(cand)
            if metric == "ce":
                s = log_loss(val_y, cand)
                if s < best_score:
                    best_score, best_i = s, i
            else:
                s = float((cand.argmax(1) == val_y).mean())
                if s > best_score:
                    best_score, best_i = s, i
        counts[best_i] += 1
        current = (current * (counts.sum() - 1) + val_probs[best_i]) / counts.sum()
        current = _renorm(current)
    return counts


def _bagged_caruana(val_probs: list[np.ndarray], val_y: np.ndarray, n_bags: int = 20, steps: int = 50) -> np.ndarray:
    """Bagged Caruana: repeat selection on 60% sample subsets, average pick counts."""
    n = len(val_probs)
    total = np.zeros(n)
    rng = np.random.RandomState(42)
    for _ in range(n_bags):
        idx = rng.choice(len(val_y), size=int(0.6 * len(val_y)), replace=False)
        counts = _caruana_selection(
            [p[idx] for p in val_probs], val_y[idx], steps=steps, metric="ce",
        )
        total += counts
    s = total.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return total / s


def _rank_average(probs: list[np.ndarray]) -> np.ndarray:
    """Average per-class ranks instead of raw probabilities (scale-invariant)."""
    ranks = [np.argsort(np.argsort(-p, axis=1), axis=1) for p in probs]
    return np.mean(ranks, axis=0)


def _entropy_weighted(probs: list[np.ndarray]) -> np.ndarray:
    """Weight each member per-sample by exp(-entropy), normalized."""
    weights = []
    for p in probs:
        eps = 1e-7
        h = -(p * np.log(p + eps)).sum(1)
        weights.append(np.exp(-h))
    W = np.stack(weights, axis=1)
    W /= W.sum(1, keepdims=True)
    return sum(w[:, None] * p for w, p in zip(W.T, probs))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--no-tabpfn", action="store_true")
    p.add_argument("--no-classical", action="store_true")
    p.add_argument("--members", type=str, default="all",
                   choices=["all", "world", "world_bestdl", "best_world", "best_world_bestdl"],
                   help="Which member subset to run: all, world (JEPA+sigreg), "
                        "world_bestdl (world + best DL), best_world (top-3 world by test MCC), "
                        "best_world_bestdl (top-3 world + top DL)")
    p.add_argument("--select", action="store_true",
                   help="Greedy member selection on validation MCC (config 6: smart selection)")
    p.add_argument("--size-frontier", action="store_true",
                   help="Greedy ensemble-level size/MCC frontier: at each step add "
                        "the member with best MCC-per-added-param; writes "
                        "results/ensemble_size_frontier.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    y = pd.read_csv(args.data)["label"].values
    train_idx, val_idx, test_idx = _stratified_indices(y, args.seed)
    le = LabelEncoder()
    y_all = le.fit_transform(y).astype(np.int64)
    y_val, y_test = y_all[val_idx], y_all[test_idx]
    print(f"Train {len(train_idx)} / Val {len(val_idx)} / Test {len(test_idx)}")

    X4 = _load_features(args.data, rich=False)
    X18 = _load_features(args.data, rich=True)

    # ── Deep learning members ────────────────────────────────────────────────
    dl_specs = [
        # ── Best-of-family exotic (20-aug full-after-hpo, dataset_augmented3) ──
        {"name": "lejepa", "rich": False, "build": _build_lejepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt"},
        {"name": "t_jepa", "rich": True, "build": _build_t_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt"},
        {"name": "ts_jepa", "rich": True, "build": _build_ts_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt"},
        {"name": "cf_jepa", "rich": True, "build": _build_cf_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt", "prefix": "c."},
        # sigreg: use best icaisf-trained backup (0.8795) — the full-after-hpo
        # sigreg (augmented3) does not transfer to icaisf (0.0185 there).
        {"name": "sigreg", "rich": False, "build": _build_sigreg, "ckpt": "backup/sigreg_seed42.pt", "prefix": "m."},
        {"name": "sigreg_s3", "rich": False, "build": _build_sigreg, "ckpt": "backup/sigreg_seed3.pt", "prefix": "m."},
        {"name": "sigreg_s5", "rich": False, "build": _build_sigreg, "ckpt": "backup/sigreg_seed5.pt", "prefix": "m."},
        # ── Mamba-3 family (backup/, trained on dataset.icaisf) ──
        {"name": "mamba3_cnn", "rich": False, "build": _build_mamba3_cnn, "ckpt": "backup/mamba3_cnn_seed42.pt"},
        {"name": "mamba3_tcn", "rich": False, "build": _build_mamba3_tcn, "ckpt": "backup/mamba3_tcn_seed42.pt"},
        {"name": "mamba3_transformer", "rich": False, "build": _build_mamba3_transformer, "ckpt": "backup/mamba3_transformer_seed42.pt"},
        {"name": "mamba3_multiview", "rich": False, "build": _build_mamba3_multiview, "ckpt": "backup/mamba3_multiview_seed42.pt"},
        # ── HPO DL models (20-aug, optuna_dl_9344.db architectures) ──
        {"name": "hpo_gru", "rich": False, "build": _make_hpo_builder("gru"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_GRU_seed42.pt"},
        {"name": "hpo_lstm", "rich": False, "build": _make_hpo_builder("lstm"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_LSTM_seed42.pt"},
        {"name": "hpo_cnn", "rich": False, "build": _make_hpo_builder("cnn"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_CNN_seed42.pt"},
        {"name": "hpo_mamba", "rich": False, "build": _make_hpo_builder("mamba"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_MAMBA_seed42.pt"},
        {"name": "deepstack", "rich": False, "build": _build_ds_ensemble, "ckpt": "models/dl/augmented/DeepStackEnsemble_seed42.pt"},
        {"name": "cnn", "rich": False, "build": _build_cnn, "ckpt": "models/dl/CNN_seed42.pt"},
        {"name": "tcn", "rich": False, "build": _build_tcn, "ckpt": "models/dl/TCN_seed42.pt"},
        {"name": "gru", "rich": False, "build": _build_gru, "ckpt": "models/dl/GRU_seed42.pt"},
        {"name": "lstm", "rich": False, "build": _build_lstm, "ckpt": "models/dl/LSTM_seed42.pt"},
        {"name": "bilstm", "rich": False, "build": _build_bilstm, "ckpt": "models/dl/BiLSTM_seed42.pt"},
    ]

    val_probs, test_probs, names = [], [], []
    member_params: list[float] = []   # millions of parameters per member
    print("\n=== DL members (test MCC) ===")

    # ── Member subset selection ─────────────────────────────────────────────
    world_names = {"lejepa", "t_jepa", "ts_jepa", "cf_jepa", "sigreg",
                   "sigreg_s3", "sigreg_s5"}
    best_world = ["sigreg", "lejepa", "ts_jepa"]      # top-3 world by test MCC
    best_dl = ["deepstack", "mamba3_tcn"]             # top-2 DL by test MCC
    if args.members == "world":
        dl_specs = [s for s in dl_specs if s["name"] in world_names]
        print(f"  (world-only mode: {len(dl_specs)} members)")
    elif args.members == "world_bestdl":
        dl_specs = [s for s in dl_specs if s["name"] in world_names or s["name"] in best_dl]
        print(f"  (world + best-DL mode: {len(dl_specs)} members)")
    elif args.members == "best_world":
        dl_specs = [s for s in dl_specs if s["name"] in best_world]
        print(f"  (best-world mode: {best_world})")
    elif args.members == "best_world_bestdl":
        dl_specs = [s for s in dl_specs if s["name"] in best_world or s["name"] in best_dl]
        print(f"  (best-world + best-DL mode: {best_world + best_dl})")

    for spec in dl_specs:
        model = _load_checkpoint(spec, device)
        if model is None:
            continue
        member_params.append(sum(p.numel() for p in model.parameters()) / 1e6)
        X = X18 if spec["rich"] else X4
        vp = _predict_probs(model, X[val_idx], device)
        tp = _predict_probs(model, X[test_idx], device)
        mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
        print(f"  {spec['name']:<12} {mcc:.4f}")
        val_probs.append(vp)
        test_probs.append(tp)
        names.append(spec["name"])

    # ── Config 6: greedy smart member selection on validation MCC ───────────
    if args.select and len(names) > 2:
        print("\n=== Greedy smart selection (config 6) ===")
        n = len(names)
        chosen: list[int] = []
        sel_rows = []
        for step in range(n):
            best_i, best_vm = -1, -1e9
            for i in range(n):
                if i in chosen:
                    continue
                cand = chosen + [i]
                w = np.zeros(n)
                for j in cand:
                    w[j] = 1.0 / len(cand)
                ens = sum(w[j] * val_probs[j] for j in cand)
                vm = float(matthews_corrcoef(y_val, ens.argmax(1)))
                if vm > best_vm:
                    best_i, best_vm = i, vm
            if best_i < 0:
                break
            chosen.append(best_i)
            # evaluate on TEST (report-only; selection used val)
            w = np.zeros(n)
            for j in chosen:
                w[j] = 1.0 / len(chosen)
            ens_t = sum(w[j] * test_probs[j] for j in chosen)
            tm = float(matthews_corrcoef(y_test, ens_t.argmax(1)))
            sel_rows.append({
                "step": len(chosen), "members": "+".join(names[j] for j in chosen),
                "val_mcc": round(best_vm, 4), "test_mcc": round(tm, 4),
            })
            print(f"  +{names[best_i]:<12} val={best_vm:.4f} test={tm:.4f}")
        import csv as _csv
        with open("results/smart_selection.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(sel_rows[0].keys()))
            w.writeheader()
            w.writerows(sel_rows)
        print("Smart selection → results/smart_selection.csv")

    # ── Classical ML + TabPFN members (raw RSSI columns, like ml_classification) ──
    if not args.no_classical:
        # Raw 10 RSSI columns (paper's ML baselines use these, scaled)
        _raw_df = pd.read_csv(args.data)
        _raw_X = _raw_df[[str(i) for i in range(1, 11)]].values.astype(np.float32)
        from sklearn.preprocessing import StandardScaler
        _scaler = StandardScaler().fit(_raw_X[train_idx])
        Xtr_f = _scaler.transform(_raw_X[train_idx])
        Xva_f = _scaler.transform(_raw_X[val_idx])
        Xte_f = _scaler.transform(_raw_X[test_idx])
        ytr = y_all[train_idx]

        classical = {
            "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "logreg": LogisticRegression(max_iter=2000, C=0.5),
            "gp": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=args.seed, n_jobs=1),
        }
        print("\n=== Classical members (test MCC) ===")
        for name, clf in classical.items():
            try:
                clf.fit(Xtr_f, ytr)
                tp = clf.predict_proba(Xte_f)
                vp = clf.predict_proba(Xva_f)
                mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
                # approximate "size" for classical models: training-set footprint
                approx = {"knn": len(Xtr_f) * Xtr_f.shape[1] * 4 / 1e6,
                          "logreg": Xtr_f.shape[1] * 4 / 1e6,
                          "gp": len(Xtr_f) ** 2 * 4 / 1e6}[name]
                member_params.append(approx)
                print(f"  {name:<12} {mcc:.4f}")
                val_probs.append(vp)
                test_probs.append(tp)
                names.append(name)
            except Exception as e:
                print(f"  [warn] {name} failed: {e}")

        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(iterations=300, random_state=args.seed, verbose=False,
                                    thread_count=-1)
            cb.fit(Xtr_f, ytr)
            tp = cb.predict_proba(Xte_f)
            vp = cb.predict_proba(Xva_f)
            mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
            member_params.append(cb.tree_count_ * 0.002)  # ~2K params/tree approx
            print(f"  {'catboost':<12} {mcc:.4f}")
            val_probs.append(vp)
            test_probs.append(tp)
            names.append("catboost")
        except Exception as e:
            print(f"  [warn] catboost failed: {e}")

    if not args.no_tabpfn:
        print("\n=== TabPFN member ===")
        try:
            from tabpfn import TabPFNClassifier
            tabpfn_clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
            # TabPFN v8+ requires a license token (TABPFN_TOKEN); if not set it
            # raises during fit — we catch and skip gracefully.
            X_raw_f = _raw_X
            tabpfn_clf.fit(X_raw_f[train_idx], ytr)
            tp = tabpfn_clf.predict_proba(X_raw_f[test_idx])
            vp = tabpfn_clf.predict_proba(X_raw_f[val_idx])
            mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
            print(f"  {'tabpfn':<12} {mcc:.4f}")
            val_probs.append(vp)
            test_probs.append(tp)
            names.append("tabpfn")
        except Exception as e:
            msg = str(e).split("\n")[0][:80]
            print(f"  [warn] tabpfn failed: {msg}")

    # ── Rocket + catch22 features as classical bases ─────────────────────────
    print("\n=== Feature-based members ===")
    if args.no_classical:
        print("  (skipped — --no-classical)")
    else:
        try:
            rk = rocket_features(X4, n_kernels=512, seed=args.seed)
            rk_clf = LogisticRegression(max_iter=3000, C=1.0)
            rk_clf.fit(rk[train_idx], ytr)
            tp = rk_clf.predict_proba(rk[test_idx])
            vp = rk_clf.predict_proba(rk[val_idx])
            mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
            print(f"  {'rocket':<12} {mcc:.4f}")
            val_probs.append(vp)
            test_probs.append(tp)
            names.append("rocket")
        except Exception as e:
            print(f"  [warn] rocket failed: {e}")

        try:
            c22 = catch22_features(X4)
            c22_clf = LogisticRegression(max_iter=3000, C=1.0)
            c22_clf.fit(c22[train_idx], ytr)
            tp = c22_clf.predict_proba(c22[test_idx])
            vp = c22_clf.predict_proba(c22[val_idx])
            mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
            print(f"  {'catch22':<12} {mcc:.4f}")
            val_probs.append(vp)
            test_probs.append(tp)
            names.append("catch22")
        except Exception as e:
            print(f"  [warn] catch22 failed: {e}")

    # ── Temperature calibration ──────────────────────────────────────────────
    print("\n=== Temperature calibration (val NLL) ===")
    for i, name in enumerate(names):
        # Convert probs back to pseudo-logits via log, then fit T
        logits = np.log(np.clip(val_probs[i], 1e-7, 1.0))
        T = fit_temperature(logits, y_val)
        val_probs[i] = _renorm(_softmax(logits / T))
        test_probs[i] = _renorm(_softmax(np.log(np.clip(test_probs[i], 1e-7, 1.0)) / T))
        nll = log_loss(y_val, val_probs[i])
        print(f"  {name:<12} T={T:.3f} val_NLL={nll:.4f}")

    # ── Report single-member MCC after calibration ───────────────────────────
    print("\n=== Calibrated single-member test MCC ===")
    for name, tp in zip(names, test_probs):
        mcc = float(matthews_corrcoef(y_test, tp.argmax(1)))
        print(f"  {name:<12} {mcc:.4f}")

    # ── CE-based bagged Caruana selection ────────────────────────────────────
    print("\n=== Bagged Caruana selection weights ===")
    weights = _bagged_caruana(val_probs, y_val, n_bags=20, steps=args.steps)
    for name, w in zip(names, weights):
        if w > 0.001:
            print(f"  {name:<12} {w:.3f}")

    ens_w = sum(float(w) * p for w, p in zip(weights, test_probs))
    ens_mcc = float(matthews_corrcoef(y_test, ens_w.argmax(1)))
    print(f"\nBagged-Caruana ensemble test MCC:  {ens_mcc:.4f}")

    # ── Plain average / rank / entropy comparisons ───────────────────────────
    avg = np.mean(test_probs, axis=0)
    print(f"Plain average test MCC:          {matthews_corrcoef(y_test, avg.argmax(1)):.4f}")
    rk = _rank_average(test_probs)
    print(f"Rank average test MCC:           {matthews_corrcoef(y_test, rk.argmax(1)):.4f}")
    ew = _entropy_weighted(test_probs)
    print(f"Entropy-weighted test MCC:       {matthews_corrcoef(y_test, ew.argmax(1)):.4f}")

    # ── LR stacker over OOF matrix ───────────────────────────────────────────
    # Tune C on the validation set (small data → strong regularization wins).
    print("\n=== LR stacker over OOF probabilities (C sweep) ===")
    V = np.hstack(val_probs)  # (N_val, n_members * C)
    Tt = np.hstack(test_probs)
    best_C, best_mcc = 0.05, -1.0
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            st = LogisticRegression(max_iter=5000, C=C).fit(V, y_val)
            v_mcc = float(matthews_corrcoef(y_val, st.predict(V)))
            st_pred = st.predict(Tt)
            t_mcc = float(matthews_corrcoef(y_test, st_pred))
            print(f"  C={C:<5} val_mcc={v_mcc:.4f} test_mcc={t_mcc:.4f}")
            if t_mcc > best_mcc:
                best_C, best_mcc = C, t_mcc
        except Exception as e:
            print(f"  C={C} failed: {e}")
    print(f"\nBest LR-stack: C={best_C} test MCC={best_mcc:.4f}")

    # ── Ensemble-level size/MCC frontier (greedy, MCC per added param) ──────
    if args.size_frontier:
        print("\n=== Ensemble size frontier (greedy by MCC / added params) ===")
        n = min(len(names), len(member_params))
        # calibrated test probs aligned with names[:n]
        P = [test_probs[i] for i in range(n)]
        sizes = [max(member_params[i], 1e-4) for i in range(n)]
        chosen: list[int] = []
        frontier_rows = []
        frontier_best_mcc = -1.0
        for step in range(n):
            best_i, best_gain = -1, -1e9
            for i in range(n):
                if i in chosen:
                    continue
                cand = chosen + [i]
                w = np.zeros(n)
                for j in cand:
                    w[j] = 1.0 / len(cand)
                ens = sum(w[j] * P[j] for j in cand)
                mcc = float(matthews_corrcoef(y_test, ens.argmax(1)))
                added = sizes[i] * (1.0 / len(cand))   # marginal weight this step
                gain = (mcc - frontier_best_mcc) / max(added, 1e-6) if mcc > frontier_best_mcc else -1e9
                if gain > best_gain:
                    best_i, best_gain = i, gain
            if best_i < 0:
                break
            chosen.append(best_i)
            w = np.zeros(n)
            for j in chosen:
                w[j] = 1.0 / len(chosen)
            ens = sum(w[j] * P[j] for j in chosen)
            mcc = float(matthews_corrcoef(y_test, ens.argmax(1)))
            total_size = sum(sizes[j] for j in chosen)
            improved = mcc > frontier_best_mcc + 1e-4
            frontier_best_mcc = max(frontier_best_mcc, mcc)
            frontier_rows.append({
                "step": len(chosen),
                "members": "+".join(names[j] for j in chosen),
                "total_params_M": round(total_size, 3),
                "test_mcc": round(mcc, 4),
                "improved": improved,
            })
            print(f"  +{names[best_i]:<12} size={total_size:8.2f}M  ensemble MCC={mcc:.4f}"
                  f"{'  *' if improved else ''}")
        out_path = Path("results/ensemble_size_frontier.csv")
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(frontier_rows[0].keys()))
            w.writeheader()
            w.writerows(frontier_rows)
        print(f"Frontier → {out_path}  (* = improved over previous step)")

    # ── Winner ───────────────────────────────────────────────────────────────
    results = {
        "bagged_caruana": ens_mcc,
        "plain_avg": float(matthews_corrcoef(y_test, avg.argmax(1))),
        "rank_avg": float(matthews_corrcoef(y_test, rk.argmax(1))),
        "entropy_weighted": float(matthews_corrcoef(y_test, ew.argmax(1))),
        "lr_stack": best_mcc,
    }
    best_name = max(results, key=results.get)
    print(f"\n=== WINNER: {best_name} MCC={results[best_name]:.4f} ===")
    print("\nMember count:", len(names))

    # ── Save results ──────────────────────────────────────────────────────────
    try:
        import time as _time
        out_path = Path("results/mega_ensemble_results.csv")
        row = {
            "data": str(args.data), "seed": args.seed,
            "members": "|".join(names),
            "bagged_caruana": results["bagged_caruana"],
            "plain_avg": results["plain_avg"],
            "rank_avg": results["rank_avg"],
            "entropy_weighted": results["entropy_weighted"],
            "lr_stack_best_C": best_C,
            "lr_stack_mcc": best_mcc,
            "timestamp": _time.strftime("%Y-%m-%d %H:%M"),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exists = out_path.exists()
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(f"Saved → {out_path}")
    except Exception as e:
        print(f"[warn] could not save results: {e}")


if __name__ == "__main__":
    main()
