"""HPO for exotic models via Optuna.

Searches over architecture + pretraining hyperparams for JEPA models,
and architecture hyperparams for Mamba-3 variants.

Usage:
    uv run python hpo_exotic.py --model ts_jepa_v2 --trials 50
    uv run python hpo_exotic.py --model mamba3_cnn --trials 30
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from dl.config import DLConfig
from dl.data_loader import DLDataLoader, RSSIDataset
from dl.training import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPO for exotic models")
    p.add_argument("--model", type=str, default="ts_jepa_v2",
                   choices=["ts_jepa_v2", "t_jepa_v2", "mamba3_cnn", "mamba3_tcn",
                            "mamba3_transformer", "mamba3_multiview"])


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu > 0 else "cuda:0")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
# TS-JEPA v2 HPO
# ═══════════════════════════════════════════════════════════════════

def objective_ts_jepa_v2(trial: optuna.Trial, args: argparse.Namespace, device: torch.device) -> float:
    from dl.models.ts_jepa_v2 import TSJEPAModelV2, TSJEPAClassifierV2

    # ── Sample hyperparams ──────────────────────────────────────────
    embed_dim = trial.suggest_categorical("embed_dim", [128, 192, 256, 384])
    nhead_options = [h for h in [2, 4, 8] if embed_dim % h == 0]
    nhead = trial.suggest_categorical("nhead", nhead_options)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dim_ff = trial.suggest_categorical("dim_ff", [embed_dim * 2, embed_dim * 4])
    pred_dim = trial.suggest_categorical("pred_dim", [embed_dim // 4, embed_dim // 2, embed_dim])
    pred_layers = trial.suggest_int("pred_layers", 1, 4)
    mask_start = trial.suggest_float("mask_start", 0.1, 0.4, step=0.1)
    mask_end = trial.suggest_float("mask_end", 0.4, 0.8, step=0.1)
    noise_std = trial.suggest_float("noise_std", 0.0, 3.0, step=0.5)
    pretrain_lr = trial.suggest_float("pretrain_lr", 1e-5, 1e-3, log=True)
    finetune_lr = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
    ema_start = trial.suggest_float("ema_start", 0.99, 0.999)
    patch_size = trial.suggest_categorical("patch_size", [2, 5])  # 5 patches or 2 patches

    model = TSJEPAModelV2(
        seq_len=10, patch_size=patch_size, embed_dim=embed_dim, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_ff, pred_dim=pred_dim,
        pred_num_layers=pred_layers, mask_ratio_start=mask_start,
        mask_ratio_end=mask_end, noise_std=noise_std,
        ema_start=ema_start, in_channels=4,
    ).to(device)

    # ── Load data ───────────────────────────────────────────────────
    config = DLConfig()
    config.data_path = args.data; config.seed = args.seed
    config.batch_size = args.batch_size; config.device = str(device)

    dl = DLDataLoader(config)
    X, y = dl.load_and_preprocess()
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # Pretrain data (no labels, use all X)
    X_pt = X_tr.astype(np.float32)  # (N, 10, 4)
    pt_mean = X_pt.mean(axis=(0, 1), keepdims=True)
    pt_std = X_pt.std(axis=(0, 1), keepdims=True) + 1e-8
    X_pt_norm = (X_pt - pt_mean) / pt_std

    class PTDataset(torch.utils.data.Dataset):
        def __init__(s, X): s.X = torch.tensor(X, dtype=torch.float32)
        def __len__(s): return len(s.X)
        def __getitem__(s, i): return (s.X[i],)

    pt_loader = DataLoader(PTDataset(X_pt_norm), batch_size=args.batch_size, shuffle=True)

    # ── Pretrain ────────────────────────────────────────────────────
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=pretrain_lr, weight_decay=1e-4)

    for epoch in range(args.pretrain_epochs):
        model.train()
        for (xb,) in pt_loader:
            xb = xb.to(device)
            loss, _ = model.pretrain_step(xb, epoch, args.pretrain_epochs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    # ── Fine-tune ───────────────────────────────────────────────────
    clf = TSJEPAClassifierV2(pretrained=model, num_classes=4, hidden_dim=128, dropout=0.3).to(device)

    # Stage 1: freeze encoder, train head
    for p in clf.encoder.parameters():
        p.requires_grad = False
    opt_ft = torch.optim.AdamW(clf.head.parameters(), lr=finetune_lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_ds = RSSIDataset(X_tr, y_tr)
    val_ds = RSSIDataset(X_va, y_va)
    tr_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Wrap: clf expects (B, 10, 4) → already matches RSSIDataset output
    # No need to extract raw RSSI — v2 uses all 4 channels
    class W(nn.Module):
        def __init__(s, c): super().__init__(); s.c = c
        def forward(s, x): return s.c(x)
    wrapped = W(clf)

    for epoch in range(args.finetune_epochs // 2):
        clf.train()
        for xb, yb in tr_ldr:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(wrapped(xb), yb)
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()

    # Stage 2: unfreeze encoder, lower LR
    for p in clf.encoder.parameters():
        p.requires_grad = True
    opt_full = torch.optim.AdamW(clf.parameters(), lr=finetune_lr * 0.1, weight_decay=1e-4)
    for epoch in range(args.finetune_epochs // 2):
        clf.train()
        for xb, yb in tr_ldr:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(wrapped(xb), yb)
            opt_full.zero_grad(); loss.backward(); opt_full.step()

    # ── Evaluate ────────────────────────────────────────────────────
    clf.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in va_ldr:
            lp = wrapped(xb.to(device)).argmax(dim=1).cpu().tolist()
            all_p.extend(lp); all_t.extend(yb.tolist())
    mcc = matthews_corrcoef(all_t, all_p)
    return float(mcc)


# ═══════════════════════════════════════════════════════════════════
# Mamba-3 CNN HPO
# ═══════════════════════════════════════════════════════════════════

def objective_mamba3_cnn(trial: optuna.Trial, args: argparse.Namespace, device: torch.device) -> float:
    from dl.models.mamba3_cnn import Mamba3CNN

    d_model = trial.suggest_categorical("d_model", [64, 128, 192, 256])
    n_mamba_layers = trial.suggest_int("n_mamba_layers", 1, 4)
    d_state = trial.suggest_categorical("d_state", [8, 16, 32])
    cnn_channels = trial.suggest_categorical("cnn_channels", [d_model, d_model // 2, d_model * 2])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    mimo_rank = trial.suggest_categorical("mimo_rank", [1, 2, 4])

    model = Mamba3CNN(
        in_features=4, cnn_channels=cnn_channels, d_model=d_model,
        d_state=d_state, n_mamba_layers=n_mamba_layers, num_classes=4,
        dropout=dropout, mimo_rank=mimo_rank,
    ).to(device)

    config = DLConfig()
    config.data_path = args.data; config.seed = args.seed
    config.batch_size = args.batch_size; config.device = str(device)
    config.num_epochs = args.finetune_epochs * 2
    config.learning_rate = lr; config.use_wandb = False

    dl = DLDataLoader(config)
    X, y = dl.load_and_preprocess()
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    tr_ldr = DataLoader(RSSIDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    va_ldr = DataLoader(RSSIDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)

    result = trainer.fit(model, tr_ldr, va_ldr)
    return float(result.best_val_mcc)


OBJECTIVES = {
    "ts_jepa_v2": objective_ts_jepa_v2,
    "t_jepa_v2": objective_t_jepa_v2,
    "mamba3_cnn": objective_mamba3_cnn,
}


def main() -> None:


def objective_t_jepa_v2(trial: optuna.Trial, args: argparse.Namespace, device: torch.device) -> float:
    from dl.models.t_jepa_v2 import TJEPAModelV2, TJEPAClassifierV2
    embed_dim = trial.suggest_categorical("embed_dim", [128, 192, 256])
    nhead_ok = [h for h in [2, 4, 8] if embed_dim % h == 0]
    nhead = trial.suggest_categorical("nhead", nhead_ok)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dim_ff = trial.suggest_categorical("dim_ff", [embed_dim * 2, embed_dim * 4])
    pred_dim = trial.suggest_categorical("pred_dim", [embed_dim // 4, embed_dim // 2])
    pred_layers = trial.suggest_int("pred_layers", 1, 4)
    n_reg = trial.suggest_int("n_reg_tokens", 1, 4)
    mask_start = trial.suggest_float("mask_start", 0.2, 0.5, step=0.1)
    mask_end = trial.suggest_float("mask_end", 0.5, 0.8, step=0.1)
    noise_std = trial.suggest_float("noise_std", 0.0, 3.0, step=0.5)
    pretrain_lr = trial.suggest_float("pretrain_lr", 1e-5, 1e-3, log=True)
    finetune_lr = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)

    m = TJEPAModelV2(n_timesteps=10, n_channels=4, d_model=embed_dim, nhead=nhead,
                     num_layers=num_layers, dim_feedforward=dim_ff, n_reg_tokens=n_reg,
                     pred_dim=pred_dim, pred_num_layers=pred_layers,
                     mask_ratio_start=mask_start, mask_ratio_end=mask_end, noise_std=noise_std)
    m.to(device)
    # Quick pretrain + fine-tune similar to ts_jepa_v2 objective
    config = DLConfig(); config.data_path = args.data; config.seed = args.seed
    config.batch_size = args.batch_size; config.device = str(device)
    dl = DLDataLoader(config); X, y = dl.load_and_preprocess()
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_pt = X_tr.astype(np.float32)
    pt_m, pt_s = X_pt.mean(axis=(0,1), keepdims=True), X_pt.std(axis=(0,1), keepdims=True) + 1e-8
    X_pt = (X_pt - pt_m) / pt_s
    class PTDS(torch.utils.data.Dataset):
        def __init__(s, X): s.X = torch.tensor(X, dtype=torch.float32)
        def __len__(s): return len(s.X)
        def __getitem__(s, i): return (s.X[i],)
    ptl = DataLoader(PTDS(X_pt), batch_size=args.batch_size, shuffle=True)
    params = list(m.context_encoder.parameters()) + list(m.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=pretrain_lr)
    for ep in range(args.pretrain_epochs):
        m.train()
        for (xb,) in ptl:
            loss, _ = m.pretrain_step(xb.to(device), ep, args.pretrain_epochs)
            opt.zero_grad(); loss.backward(); opt.step()
    clf = TJEPAClassifierV2(pretrained=m, num_classes=4, hidden_dim=128).to(device)
    for p in clf.encoder.parameters(): p.requires_grad = False
    opt_ft = torch.optim.AdamW(clf.head.parameters(), lr=finetune_lr)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    trl = DataLoader(RSSIDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(RSSIDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    class W(nn.Module):
        def __init__(s, c): super().__init__(); s.c = c
        def forward(s, x): return s.c(x)
    wr = W(clf)
    for ep in range(args.finetune_epochs // 2):
        clf.train()
        for xb, yb in trl: 
            loss = ce(wr(xb.to(device)), yb.to(device))
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
    for p in clf.encoder.parameters(): p.requires_grad = True
    opt_full = torch.optim.AdamW(clf.parameters(), lr=finetune_lr * 0.1)
    for ep in range(args.finetune_epochs // 2):
        clf.train()
        for xb, yb in trl:
            loss = ce(wr(xb.to(device)), yb.to(device))
            opt_full.zero_grad(); loss.backward(); opt_full.step()
    clf.eval(); ap, at = [], []
    with torch.no_grad():
        for xb, yb in val:
            ap.extend(wr(xb.to(device)).argmax(1).cpu().tolist()); at.extend(yb.tolist())
    return float(matthews_corrcoef(at, ap))


    args = parse_args()
    set_seed(args.seed); device = resolve_device(args.gpu)
    study_name = args.study_name or f"exotic_{args.model}_{int(time.time())}"
    storage = f"sqlite:///optuna_exotic_{args.model}.db"

    objective = OBJECTIVES[args.model]

    def obj(trial):
        return objective(trial, args, device)


    def _trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  [Trial {trial.number:03d}] PRUNED")
        elif trial.value is not None:
            print(f"  [Trial {trial.number:03d}] MCC={trial.value:.4f}")
        else:
            print(f"  [Trial {trial.number:03d}] FAILED")
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="maximize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(obj, n_trials=args.trials, show_progress_bar=True, callbacks=[_trial_cb])

    print(f"\nBest trial ({args.model}):")
    print(f"  MCC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
