"""Exotic model pipeline — SSL pretraining + supervised fine-tuning.

Three model families:
  1. T-JEPA   — tabular feature masking, learns temporal structure
  2. TS-JEPA  — time-series patch masking, learns interference-invariant features
  3. SIGReg   — CNN + Gaussian latent regularizer (no pretraining needed)

Plus Mamba-3 hybrids (supervised only).

Usage:
    # T-JEPA: pretrain + fine-tune
    uv run python run_exotic.py --model t_jepa --pretrain-epochs 300 --finetune-epochs 50

    # TS-JEPA: pretrain + fine-tune
    uv run python run_exotic.py --model ts_jepa --pretrain-epochs 300 --finetune-epochs 50

    # SIGReg: direct supervised (no pretraining)
    uv run python run_exotic.py --model sigreg --epochs 200

    # Mamba-3: direct supervised
    uv run python run_exotic.py --model mamba3_cnn --epochs 150
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch import Tensor
from torch.utils.data import DataLoader

from dl.config import DLConfig
from dl.data_loader import DLDataLoader, RSSIDataset
from dl.evaluation import evaluate_model_on_test
from dl.training import Trainer

if TYPE_CHECKING:
    from dl.jepa_viz import JEPAVisualizer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exotic model pipeline")
    p.add_argument("--model", type=str, default="t_jepa",
                   choices=["t_jepa", "ts_jepa", "sigreg",
                            "mamba3_cnn", "mamba3_tcn", "mamba3_transformer",
                            "mamba3_multiview"],
                   help="Model type")
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")

    # Pretraining
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=3e-4)
    p.add_argument("--pretrain-wd", type=float, default=1e-4)
    p.add_argument("--probe-cadence", type=int, default=10,
                   help="Pretrain: evaluate linear probe every N epochs")
    p.add_argument("--probe-patience", type=int, default=5,
                   help="Pretrain: early-stop after N probe evals without improvement")

    # Fine-tuning / Direct supervised
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=150, help="Direct supervised epochs")
    p.add_argument("--patience", type=int, default=25,
                   help="Early stopping patience for direct supervised models")
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for direct supervised models (SIGReg, Mamba)")
    p.add_argument("--finetune-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--unfreeze-encoder", action="store_true")

    # Model architecture
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--n-reg-tokens", type=int, default=2)
    p.add_argument("--pred-dim", type=int, default=128)
    p.add_argument("--pred-layers", type=int, default=2)
    p.add_argument("--ema-start", type=float, default=0.99)
    p.add_argument("--ema-end", type=float, default=0.996)
    p.add_argument("--n-preds", type=int, default=1,
                   help="T-JEPA: number of target masks per context (1 works best at this scale)")
    p.add_argument("--jepa-sigreg", type=float, default=0.05,
                   help="T-JEPA pretrain: SIGReg anti-collapse weight on context latents "
                        "(0=off; ~0.05-0.1 prevents latent collapse)")
    p.add_argument("--pooling", type=str, default="mean")

    # SIGReg specific
    p.add_argument("--latent-dim", type=int, default=128,
                   help="SIGReg: latent bottleneck dimension")
    p.add_argument("--sigreg-lambda", type=float, default=0.01,
                   help="SIGReg: Gaussian regularizer weight")

    # HPO
    p.add_argument("--hpo", action="store_true")
    p.add_argument("--hpo-trials", type=int, default=30)

    # Paths
    p.add_argument("--models-dir", type=Path, default=Path("models/exotic"))
    p.add_argument("--results-dir", type=Path, default=Path("results/exotic"))
    p.add_argument("--viz-dir", type=Path, default=Path("models/exotic/viz"))

    # WandB
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="inMotion-exotic-2")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu >= 0 else "cuda:0")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrain_data(
    config: DLConfig,
) -> tuple[DataLoader, DataLoader | None, DataLoader, DataLoader]:
    """Load data for SSL pretraining.

    Returns (train_loader, val_loader, probe_train_loader, probe_val_loader).
    The probe loaders carry labels — used by the linear probe that monitors
    representation quality during pretraining (probe MCC drives early
    stopping and checkpoint selection, not the raw val loss).
    """
    loader = DLDataLoader(config)
    X_all, y_all = loader.load_and_preprocess()
    X_all = X_all.astype(np.float32)

    # Z-score normalize per channel
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X_all - mean) / std

    # 80/20 train/val split
    n = len(X_norm)
    n_train = int(n * 0.8)
    indices = np.random.RandomState(config.seed).permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train = torch.from_numpy(X_norm[train_idx])
    X_val = torch.from_numpy(X_norm[val_idx])

    from torch.utils.data import TensorDataset
    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=config.batch_size, shuffle=True,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val), batch_size=config.batch_size, shuffle=False,
        pin_memory=True, drop_last=False,
    ) if len(X_val) > 0 else None

    # Labeled loaders for the probe (smaller batch, capped size)
    probe_batch = min(config.batch_size, 256)
    probe_train_loader = DataLoader(
        RSSIDataset(X_norm[train_idx], y_all[train_idx]),
        batch_size=probe_batch, shuffle=True, drop_last=False,
    )
    probe_val_loader = DataLoader(
        RSSIDataset(X_norm[val_idx], y_all[val_idx]),
        batch_size=probe_batch, shuffle=False, drop_last=False,
    )
    return train_loader, val_loader, probe_train_loader, probe_val_loader


def load_supervised_data(
    config: DLConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load data for supervised training.

    Returns (train_loader, val_loader, test_loader, classes).
    """
    dl = DLDataLoader(config)
    X, y = dl.load_and_preprocess()

    classes = dl.classes_
    label_map = {0: "AA", 1: "AB", 2: "BA", 3: "BB"}
    classes = [label_map.get(i, c) for i, c in enumerate(classes)]

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=config.seed, stratify=y_tr
    )

    train_loader = DataLoader(RSSIDataset(X_tr, y_tr), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(RSSIDataset(X_va, y_va), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(RSSIDataset(X_te, y_te), batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, classes


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_t_jepa(args: argparse.Namespace) -> nn.Module:
    from dl.models.t_jepa import TJEPAModel
    return TJEPAModel(
        n_timesteps=10, n_channels=4,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        n_reg_tokens=args.n_reg_tokens,
        pred_dim=args.pred_dim, pred_num_layers=args.pred_layers,
        mask_ratio=0.3,
        n_target_masks=getattr(args, "n_preds", 1),
        sigreg_lambda=getattr(args, "jepa_sigreg", 0.0),
        ema_start=args.ema_start, ema_end=args.ema_end,
    )


def _build_ts_jepa(args: argparse.Namespace) -> nn.Module:
    from dl.models.ts_jepa import TSJEPAModel
    return TSJEPAModel(
        seq_len=10, patch_size=2, in_channels=4,
        embed_dim=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        pred_dim=args.pred_dim, pred_num_layers=args.pred_layers,
        mask_ratio_start=0.4, mask_ratio_end=0.7,
        ema_start=args.ema_start, ema_end=args.ema_end,
    )


def _build_sigreg(args: argparse.Namespace) -> nn.Module:
    from dl.models.sigreg_classifier import SIGRegClassifier
    return SIGRegClassifier(
        in_features=4, num_filters=args.d_model,
        num_blocks=args.num_layers, latent_dim=args.latent_dim,
        num_classes=4, sigreg_lambda=args.sigreg_lambda,
    )


def _build_mamba3(variant: str, args: argparse.Namespace) -> nn.Module:
    kw = dict(in_features=4, d_model=args.d_model, d_state=16, num_classes=4,
              dropout=0.2, mimo_rank=4)
    if variant == "cnn":
        from dl.models.mamba3_cnn import Mamba3CNN
        return Mamba3CNN(cnn_channels=args.d_model, n_mamba_layers=args.num_layers, **kw)
    elif variant == "tcn":
        from dl.models.mamba3_tcn import Mamba3TCN
        return Mamba3TCN(tcn_channels=args.d_model, n_mamba_layers=args.num_layers, **kw)
    elif variant == "transformer":
        from dl.models.mamba3_transformer import Mamba3Transformer
        return Mamba3Transformer(nhead=args.nhead, num_blocks=args.num_layers, **kw)
    elif variant == "multiview":
        from dl.models.mamba3_multiview import Mamba3MultiView
        return Mamba3MultiView(n_mamba_layers=args.num_layers, **kw)
    raise ValueError(f"Unknown Mamba-3 variant: {variant}")


MODEL_BUILDERS: dict[str, callable] = {
    "t_jepa": _build_t_jepa,
    "ts_jepa": _build_ts_jepa,
    "sigreg": _build_sigreg,
    "mamba3_cnn": lambda a: _build_mamba3("cnn", a),
    "mamba3_tcn": lambda a: _build_mamba3("tcn", a),
    "mamba3_transformer": lambda a: _build_mamba3("transformer", a),
    "mamba3_multiview": lambda a: _build_mamba3("multiview", a),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pretraining
# ═══════════════════════════════════════════════════════════════════════════════

def _masked_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute pretrain val loss ONLY at supervised (target) positions.

    Handles both model interfaces:
      - T-JEPA:  forward → (tgt (B,total,D), preds LIST, ctx_mask, tgt_masks list, ctx_latents)
      - TS-JEPA: forward → (tgt_masked (B,M,D), preds TENSOR (B,M,D), mask_idx, non_mask_idx)
    """
    model.eval()
    val_sum, val_n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            out = model.forward(x)
            tgt, preds = out[0], out[1]
            if isinstance(preds, (list, tuple)):
                # T-JEPA: (tgt, preds list, ctx_mask, tgt_masks list, ctx_latents)
                tgt_masks = out[3]
                for p, tm in zip(preds, tgt_masks):
                    p = torch.nn.functional.normalize(p, dim=-1)
                    tgt_n = torch.nn.functional.normalize(tgt, dim=-1)
                    diff = (p - tgt_n).pow(2).sum(dim=-1) * tm
                    val_sum += diff.sum().item()
                    val_n += tm.sum().item()
            else:
                # TS-JEPA: (tgt_masked, preds tensor, mask_idx, non_mask_idx)
                # predictions already only at masked patches
                p = torch.nn.functional.normalize(preds, dim=-1)
                tgt_n = torch.nn.functional.normalize(tgt, dim=-1)
                val_sum += (p - tgt_n).pow(2).mean().item() * tgt.size(0)
                val_n += tgt.size(0)
    return val_sum / max(val_n, 1)


def _probe_mcc(
    model: nn.Module,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
) -> float:
    """Train a linear probe on frozen encoder latents, return val MCC.

    Pooled per-sample latents (mean over token dim) → Linear → 4 classes.
    This is the health metric for pretraining: a collapsed encoder scores
    near random (~0.25), a good one climbs toward 0.8+.
    """
    from sklearn.metrics import matthews_corrcoef

    d = getattr(model, "d_model", None) or getattr(model, "embed_dim")
    n_reg = getattr(model.context_encoder, "n_reg_tokens", 0)

    probe = nn.Linear(d, 4).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def pool(x: Tensor) -> Tensor:
        z = model.context_encoder(x)  # (B, N, D)
        if n_reg:
            z = z[:, n_reg:, :]
        return z.mean(dim=1)  # (B, D)

    model.eval()
    for _ in range(epochs):
        for xb, yb in probe_train_loader:
            with torch.no_grad():
                z = pool(xb.to(device))
            loss = ce(probe(z), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    probe.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in probe_val_loader:
            z = pool(xb.to(device))
            preds_all.extend(probe(z).argmax(1).cpu().tolist())
            y_all.extend(yb.tolist())
    return float(matthews_corrcoef(y_all, preds_all))


def pretrain_jepa(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    viz: JEPAVisualizer | None = None,
) -> Path:
    """Pretrain a JEPA model (T-JEPA or TS-JEPA).

    Both models share the same pretrain_step interface:
        loss, metrics = model.pretrain_step(x, epoch, total_epochs)

    Early stopping and checkpoint selection use a LINEAR PROBE MCC
    (representation quality), NOT the raw val loss. The val loss is
    logged for curves but is unreliable for model selection.

    Returns path to best checkpoint.
    """
    model.to(device)

    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pretrain_epochs,
                                                      eta_min=args.pretrain_lr * 0.01)

    # WandB
    pretrain_wandb = None
    if not args.no_wandb:
        try:
            import wandb
            pretrain_wandb = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_pretrain_seed{args.seed}",
                config={"model": args.model, "phase": "pretrain",
                        "pretrain_epochs": args.pretrain_epochs,
                        "batch_size": args.batch_size, "lr": args.pretrain_lr},
                reinit=True,
            )
        except Exception:
            pass

    best_path = save_dir / f"{args.model}_pretrain_best.pt"
    probe_cadence = getattr(args, "probe_cadence", 10)
    best_probe = -1.0
    best_val_loss = float("inf")
    val_patience_counter = 0

    print(f"Pretraining {args.pretrain_epochs} epochs ({args.model})...")
    t0 = time.time()

    for epoch in range(args.pretrain_epochs):
        # ── Training ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            loss, _ = model.pretrain_step(x, epoch, args.pretrain_epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            model._update_target_encoder()
            train_loss += loss.item()
        sch.step()
        train_loss /= max(len(train_loader), 1)

        # ── Validation (masked, target positions only) ──
        val_loss = _masked_val_loss(model, val_loader, device) if val_loader is not None else float("inf")

        # ── Probe every N epochs (logging only, not used for stopping) ──
        probe_mcc = None
        if (epoch + 1) % probe_cadence == 0:
            probe_mcc = _probe_mcc(model, probe_train_loader, probe_val_loader, device)
            if probe_mcc > best_probe:
                best_probe = probe_mcc
                torch.save({
                    "epoch": epoch,
                    "probe_mcc": probe_mcc,
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                }, best_path)

        # ── Visualization ──
        ema = getattr(model, "ema_momentum", 0.0)
        if viz is not None:
            latents = None
            if epoch % 20 == 0 and val_loader is not None:
                with torch.no_grad():
                    all_latents = []
                    for batch in val_loader:
                        x = batch[0].to(device)
                        all_latents.append(model.get_latents(x))
                    latents = torch.cat(all_latents, dim=0)
            viz.log_pretrain(epoch, train_loss, val_loss, latents, ema)
            if epoch % 50 == 0 and latents is not None:
                viz.save_checkpoint_snapshot(
                    epoch, latents,
                    title=f"{args.model.upper()} Pretrain Latent Space",
                )

        # ── WandB ──
        if pretrain_wandb is not None:
            import wandb
            log = {
                "pretrain/train_loss": train_loss,
                "pretrain/val_loss": val_loss,
                "pretrain/ema": ema,
                "pretrain/lr": sch.get_last_lr()[0],
                "pretrain/epoch": epoch,
            }
            if probe_mcc is not None:
                log["pretrain/probe_mcc"] = probe_mcc
            if viz and viz.uniformity_scores:
                log["pretrain/uniformity"] = viz.uniformity_scores[-1]
            wandb.log(log)

        # ── Logging ──
        if epoch % 20 == 0 or epoch == args.pretrain_epochs - 1 or probe_mcc is not None:
            probe_str = f" | probe={probe_mcc:.3f}" if probe_mcc is not None else ""
            print(f"  epoch {epoch + 1:3d}/{args.pretrain_epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | ema={ema:.4f}{probe_str}")

        # ── Early stopping (on val loss, not probe) ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_patience_counter = 0
        else:
            val_patience_counter += 1
            if val_patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(val loss not improving, best={best_val_loss:.4f})")
                break

    elapsed = time.time() - t0
    print(f"Pretraining done in {elapsed:.0f}s. Best probe MCC: {best_probe:.4f}")
    print(f"Checkpoint: {best_path}")

    if pretrain_wandb is not None:
        import wandb
        wandb.log({"pretrain/best_probe_mcc": best_probe})
        wandb.finish()

    return best_path


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_jepa(
    args: argparse.Namespace,
    pretrained_model: nn.Module,
    config: DLConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classes: list[str],
    device: torch.device,
    save_dir: Path,
    viz: JEPAVisualizer | None = None,
) -> dict:
    """Fine-tune a pretrained JEPA encoder for classification.

    Uses progressive unfreezing: freeze encoder → train head → unfreeze → full fine-tune.
    """
    is_t_jepa = args.model == "t_jepa"

    if is_t_jepa:
        from dl.models.t_jepa import TJEPAClassifier
        clf = TJEPAClassifier(
            pretrained=pretrained_model, num_classes=4, hidden_dim=128, dropout=0.3,
        )
    else:
        from dl.models.ts_jepa import TSJEPAClassifier
        clf = TSJEPAClassifier(
            pretrained=pretrained_model, num_classes=4, hidden_dim=128, dropout=0.3,
        )
    clf.to(device)

    # Wrap so Trainer can call model(x) → logits
    class W(nn.Module):
        def __init__(self, c): super().__init__(); self.c = c
        def forward(self, x): return self.c(x)
    wrapped = W(clf)

    config.use_wandb = not args.no_wandb
    run_name = f"{args.model}_ft_seed{args.seed}"
    save_path = save_dir / f"{run_name}.pt"

    # Stage 1: Freeze encoder, train head only (short warmup)
    for p in clf.encoder.parameters():
        p.requires_grad = False
    stage1_epochs = max(1, args.finetune_epochs // 4)
    config.num_epochs = stage1_epochs
    config.learning_rate = args.finetune_lr
    print("Stage 1: Training classification head (encoder frozen)...")
    t0 = time.time()
    r1 = Trainer(config, run_name=f"{run_name}_s1",
                 extra_wandb_config={"stage": 1, "model": args.model, "pretrained": True}
                 ).fit(wrapped, train_loader, val_loader)
    print(f"  Stage 1 best val MCC: {r1.best_val_mcc:.4f} ({time.time() - t0:.0f}s)")

    # Stage 2: Unfreeze encoder, lower LR, use the remaining budget
    for p in clf.encoder.parameters():
        p.requires_grad = True
    config.num_epochs = args.finetune_epochs - stage1_epochs
    config.learning_rate = args.finetune_lr * 0.1
    print("Stage 2: Full fine-tuning (encoder unfrozen)...")
    t0 = time.time()
    r2 = Trainer(config, run_name=f"{run_name}_s2",
                 extra_wandb_config={"stage": 2, "model": args.model, "pretrained": True}
                 ).fit(wrapped, train_loader, val_loader, save_path=save_path)

    # Evaluate on test set
    metrics = evaluate_model_on_test(
        wrapped, test_loader, config, classes, run_name, log_wandb=config.use_wandb
    )
    metrics["best_val_mcc"] = r2.best_val_mcc

    print(f"Fine-tuning done in {time.time() - t0:.0f}s.")
    print(f"  Best val MCC: {r2.best_val_mcc:.4f}")
    print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")

    if viz is not None:
        viz.log_finetune(metrics)
        # Compare pre vs post fine-tune embeddings (pooled per sample)
        if hasattr(pretrained_model, "get_latents"):
            with torch.no_grad():
                all_pre, all_post, all_labels = [], [], []
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    B = xb.size(0)
                    # Pretrain: pool per-feature latents to one vector per sample
                    pre_flat = pretrained_model.get_latents(xb)
                    all_pre.append(pre_flat.reshape(B, -1, pre_flat.size(-1)).mean(dim=1))
                    all_labels.append(yb)
                    post_out = clf.encoder(xb)
                    # T-JEPA has REG tokens to strip; TS-JEPA doesn't
                    n_reg = getattr(clf.encoder, "n_reg_tokens", 0)
                    post_feat = post_out[:, n_reg:, :]
                    all_post.append(post_feat.mean(dim=1))
                pre_latents = torch.cat(all_pre, dim=0)
                post_latents = torch.cat(all_post, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                viz.save_checkpoint_snapshot(
                    9999, pre_latents, all_labels,
                    title=f"{args.model.upper()} Pretrain Embeddings",
                )
                viz.save_checkpoint_snapshot(
                    10000, post_latents, all_labels,
                    title=f"{args.model.upper()} Fine-tuned Embeddings",
                )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SIGReg training
# ═══════════════════════════════════════════════════════════════════════════════

class SIGRegWrapper(nn.Module):
    """Wraps SIGRegClassifier so Trainer can use it with compute_loss."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

    def compute_loss(self, logits, latents, targets):
        return self.model.compute_loss(logits, latents, targets)


# ═══════════════════════════════════════════════════════════════════════════════
# HPO — Optuna hyperparameter optimization with WandB per-trial logging
# ═══════════════════════════════════════════════════════════════════════════════

def _count_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _params_str(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _efficiency(mcc: float, n_params: int) -> float:
    """Efficiency score: MCC per million parameters × 1000 (higher = more efficient)."""
    return (mcc * 1000.0) / max(n_params / 1e6, 0.001)


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_hpo(args: argparse.Namespace, device: torch.device) -> None:
    """Run Optuna HPO, update args with best hyperparams."""
    import optuna

    cfg = DLConfig()
    cfg.data_path = args.data; cfg.seed = args.seed
    cfg.batch_size = args.batch_size; cfg.device = str(device)

    dl = DLDataLoader(cfg)
    X, y = dl.load_and_preprocess()
    hpo_seed = args.seed + 0x5EED
    from sklearn.model_selection import train_test_split
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=hpo_seed, stratify=y)

    print(f"\n{'='*60}")
    print(f"HPO: {args.model} — {args.hpo_trials} trials")
    print(f"  Train: {len(Xt)} samples, Val: {len(Xv)} samples")
    print(f"{'='*60}")

    if args.model in ("t_jepa", "ts_jepa"):
        _hpo_jepa(args, device, Xt, yt, Xv, yv)
    elif args.model.startswith("mamba3"):
        _hpo_mamba3(args, device, Xt, yt, Xv, yv)
    elif args.model == "sigreg":
        _hpo_sigreg(args, device, Xt, yt, Xv, yv)


# ── WandB per-trial helpers ───────────────────────────────────────────────────

def _trial_wandb_start(trial, args: argparse.Namespace, extra_config: dict | None = None) -> object | None:
    """Start a WandB run for one Optuna trial. Returns run or None."""
    if args.no_wandb:
        return None
    try:
        import wandb
        cfg = dict(trial.params)
        cfg.update({"trial_number": trial.number, "model": args.model, "seed": args.seed})
        if extra_config:
            cfg.update(extra_config)
        name = f"hpo_{args.model}_trial{trial.number:03d}"
        return wandb.init(project=args.wandb_project, name=name, config=cfg, reinit=True)
    except Exception:
        return None


def _trial_wandb_finish(run: object | None, metrics: dict) -> None:
    """Log final metrics and close the WandB run."""
    if run is None:
        return
    try:
        import wandb
        wandb.log(metrics)
        wandb.finish()
    except Exception:
        pass


# ── JEPA HPO ──────────────────────────────────────────────────────────────────

def _hpo_jepa(args, device, Xt, yt, Xv, yv) -> None:
    import optuna
    from torch.utils.data import DataLoader, TensorDataset

    is_ts = args.model == "ts_jepa"
    pt_epochs = 50
    ft_epochs = 20

    def objective(trial):
        # ── Search space ──────────────────────────────────────────────
        d_model = trial.suggest_categorical("d_model", [128, 192, 256])
        nhead = trial.suggest_categorical("nhead", [4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dim_ff = trial.suggest_categorical("dim_ff", [256, 512, 768])
        pred_dim = trial.suggest_categorical("pred_dim", [64, 96, 128])
        pred_layers = trial.suggest_int("pred_layers", 1, 3)
        pt_lr = trial.suggest_float("pretrain_lr", 1e-5, 1e-3, log=True)
        ft_lr = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
        ema_start = trial.suggest_float("ema_start", 0.99, 0.999)

        while d_model % nhead != 0:
            nhead = max(1, nhead // 2)

        # ── Build model ───────────────────────────────────────────────
        if is_ts:
            from dl.models.ts_jepa import TSJEPAModel, TSJEPAClassifier
            m = TSJEPAModel(seq_len=10, patch_size=2, in_channels=4,
                            embed_dim=d_model, nhead=nhead, num_layers=num_layers,
                            dim_feedforward=dim_ff, pred_dim=pred_dim,
                            pred_num_layers=pred_layers, ema_start=ema_start).to(device)
        else:
            from dl.models.t_jepa import TJEPAModel, TJEPAClassifier
            m = TJEPAModel(n_timesteps=10, n_channels=4, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, dim_feedforward=dim_ff, pred_dim=pred_dim,
                           pred_num_layers=pred_layers, ema_start=ema_start).to(device)

        n_params = _count_params(m) + _count_params(m.context_encoder)

        # ── WandB: start trial run ────────────────────────────────────
        run = _trial_wandb_start(trial, args, {"n_params": n_params,
                                                "params_str": _params_str(n_params)})

        # ── Pretrain (50 epochs) ──────────────────────────────────────
        Xp = Xt.astype(np.float32)
        pm, ps = Xp.mean(axis=(0, 1), keepdims=True), Xp.std(axis=(0, 1), keepdims=True) + 1e-8
        Xp = (Xp - pm) / ps
        ptds = TensorDataset(torch.tensor(Xp, dtype=torch.float32))
        ptl = DataLoader(ptds, batch_size=args.batch_size, shuffle=True)

        params_list = list(m.context_encoder.parameters()) + list(m.predictor.parameters())
        opt = torch.optim.AdamW(params_list, lr=pt_lr)
        for ep in range(pt_epochs):
            m.train()
            pt_loss_sum = 0.0; pt_n = 0
            for (xb,) in ptl:
                loss, _ = m.pretrain_step(xb.to(device), ep, pt_epochs)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params_list, 1.0); opt.step()
                m._update_target_encoder()
                pt_loss_sum += loss.item(); pt_n += 1
            avg_pt_loss = pt_loss_sum / max(pt_n, 1)
            if run is not None:
                import wandb
                wandb.log({"pretrain/epoch": ep, "pretrain/loss": avg_pt_loss,
                            "pretrain/ema": m.ema_momentum})

        # ── Fine-tune (stage 1: head only, stage 2: full) ────────────
        if is_ts:
            clf = TSJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)
        else:
            clf = TJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)

        class W(nn.Module):
            def __init__(self, c): super().__init__(); self.c = c
            def forward(self, x): return self.c(x)
        wr = W(clf)

        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Stage 1: freeze encoder, train head
        for p in clf.encoder.parameters():
            p.requires_grad = False
        opt1 = torch.optim.AdamW(clf.head.parameters(), lr=ft_lr)
        for ep in range(ft_epochs // 2):
            clf.train()
            for xb, yb in trl:
                loss = ce(wr(xb.to(device)), yb.to(device))
                opt1.zero_grad(); loss.backward(); opt1.step()
            # Val MCC
            clf.eval(); vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist()); vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if run is not None:
                import wandb; wandb.log({"ft/stage1_epoch": ep, "ft/stage1_val_mcc": vmcc})

        # Stage 2: unfreeze, lower LR
        for p in clf.encoder.parameters():
            p.requires_grad = True
        opt2 = torch.optim.AdamW(clf.parameters(), lr=ft_lr * 0.1)
        best_mcc = -1.0
        for ep in range(ft_epochs - ft_epochs // 2):
            clf.train()
            for xb, yb in trl:
                loss = ce(wr(xb.to(device)), yb.to(device))
                opt2.zero_grad(); loss.backward(); opt2.step()
            clf.eval(); vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist()); vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if vmcc > best_mcc:
                best_mcc = vmcc
            if run is not None:
                import wandb; wandb.log({"ft/stage2_epoch": ep, "ft/stage2_val_mcc": vmcc})

        # ── Final metrics ─────────────────────────────────────────────
        eff = _efficiency(best_mcc, n_params)
        final_metrics = {
            "trial_mcc": best_mcc,
            "trial_n_params": n_params,
            "trial_params_M": n_params / 1e6,
            "trial_efficiency": eff,
        }
        _trial_wandb_finish(run, final_metrics)
        return best_mcc

    from dl.optimization import _get_optuna_storage, _log_trial_callback
    study = optuna.create_study(
        direction="maximize", study_name=f"exotic_{args.model}",
        storage=_get_optuna_storage("sqlite:///optuna_exotic.db"), load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True,
                   callbacks=[_log_trial_callback])

    _apply_best_jepa_params(args, study)


def _apply_best_jepa_params(args, study) -> None:
    """Copy best JEPA trial params to args."""
    bp = study.best_params
    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Params: {bp}")
    for k, v in bp.items():
        if hasattr(args, k):
            setattr(args, k, v)
    # Map study params to CLI arg names
    _map = {"d_model": "d_model", "nhead": "nhead", "num_layers": "num_layers",
            "dim_ff": "dim_ff", "pred_dim": "pred_dim", "pred_layers": "pred_layers",
            "pretrain_lr": "pretrain_lr", "finetune_lr": "finetune_lr",
            "ema_start": "ema_start"}
    for sk, ak in _map.items():
        if sk in bp:
            setattr(args, ak, bp[sk])


# ── SIGReg HPO ────────────────────────────────────────────────────────────────

def _hpo_sigreg(args, device, Xt, yt, Xv, yv) -> None:
    import optuna
    from dl.models.sigreg_classifier import SIGRegClassifier

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [128, 192, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
        sigreg_lambda = trial.suggest_float("sigreg_lambda", 0.001, 0.5, log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        m = SIGRegClassifier(in_features=4, num_filters=d_model, num_blocks=num_layers,
                             latent_dim=latent_dim, num_classes=4,
                             sigreg_lambda=sigreg_lambda).to(device)
        n_params = _count_params(m)
        run = _trial_wandb_start(trial, args, {"n_params": n_params,
                                                "params_str": _params_str(n_params)})

        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        best_mcc = -1.0

        for ep in range(40):
            m.train()
            for xb, yb in trl:
                xb, yb = xb.to(device), yb.to(device)
                logits, latents = m(xb)
                loss, loss_dict = m.compute_loss(logits, latents, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval(); vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    logits, _ = m(xb.to(device))
                    vap.extend(logits.argmax(1).cpu().tolist()); vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if vmcc > best_mcc:
                best_mcc = vmcc
            if run is not None:
                import wandb
                wandb.log({"epoch": ep, "val_mcc": vmcc, "ce_loss": loss_dict.get("ce_loss", 0),
                            "sigreg_loss": loss_dict.get("sigreg_loss", 0)})

        eff = _efficiency(best_mcc, n_params)
        _trial_wandb_finish(run, {"trial_mcc": best_mcc, "trial_n_params": n_params,
                                   "trial_params_M": n_params / 1e6, "trial_efficiency": eff})
        return best_mcc

    from dl.optimization import _get_optuna_storage
    study = optuna.create_study(direction="maximize", study_name=f"exotic_{args.model}",
                                storage=_get_optuna_storage("sqlite:///optuna_exotic.db"),
                                load_if_exists=True)
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True)
    bp = study.best_params
    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Params: {bp}")
    for k, v in bp.items():
        if hasattr(args, k):
            setattr(args, k, v)


# ── Mamba-3 HPO ───────────────────────────────────────────────────────────────

def _hpo_mamba3(args, device, Xt, yt, Xv, yv) -> None:
    import optuna

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128, 192])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        d_state = trial.suggest_categorical("d_state", [8, 16, 32])
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        mimo_rank = trial.suggest_categorical("mimo_rank", [1, 2, 4])

        m = _build_mamba3(args.model.replace("mamba3_", ""),
                          argparse.Namespace(d_model=d_model, num_layers=num_layers,
                                             nhead=4)).to(device)
        n_params = _count_params(m)
        run = _trial_wandb_start(trial, args, {"n_params": n_params,
                                                "params_str": _params_str(n_params)})

        cfg = DLConfig(); cfg.seed = args.seed; cfg.batch_size = args.batch_size
        cfg.device = str(device); cfg.num_epochs = 30; cfg.learning_rate = lr
        cfg.use_wandb = False
        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)

        from dl.training import Trainer
        result = Trainer(cfg).fit(m, trl, val)
        mcc = float(result.best_val_mcc)

        if run is not None:
            import wandb
            for ep, (tl, vl, vm) in enumerate(zip(result.train_losses, result.val_losses,
                                                   result.val_mccs), start=1):
                wandb.log({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_mcc": vm})

        eff = _efficiency(mcc, n_params)
        _trial_wandb_finish(run, {"trial_mcc": mcc, "trial_n_params": n_params,
                                   "trial_params_M": n_params / 1e6, "trial_efficiency": eff})
        return mcc

    from dl.optimization import _get_optuna_storage, _log_trial_callback
    study = optuna.create_study(direction="maximize", study_name=f"exotic_{args.model}",
                                storage=_get_optuna_storage("sqlite:///optuna_exotic.db"),
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True,
                   callbacks=[_log_trial_callback])
    bp = study.best_params
    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Params: {bp}")
    for k, v in bp.items():
        if hasattr(args, k):
            setattr(args, k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.gpu)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Data:   {args.data}")
    print(f"Seed:   {args.seed}")

    config = DLConfig()
    config.data_path = args.data
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.device = str(device)
    config.use_wandb = not args.no_wandb
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    config.models_dir = args.models_dir
    config.results_dir = args.results_dir

    # ── HPO (optional) ────────────────────────────────────────────────
    if args.hpo:
        _run_hpo(args, device)

    # ── Pretraining (JEPA models only) ────────────────────────────────
    pretrained_model: nn.Module | None = None
    viz = None

    is_jepa = args.model in ("t_jepa", "ts_jepa")

    if is_jepa and not args.finetune_only and args.checkpoint is None:
        print(f"\n{'=' * 60}")
        print(f"Phase 1: SSL Pretraining ({args.model})")
        print(f"{'=' * 60}")

        # Setup visualization
        from dl.jepa_viz import JEPAVisualizer
        viz = JEPAVisualizer(
            save_dir=args.viz_dir, model_name=args.model, seed=args.seed,
        )

        pt_config = DLConfig()
        pt_config.data_path = args.data
        pt_config.seed = args.seed
        pt_config.batch_size = args.batch_size
        pt_config.device = str(device)

        (pt_train_loader, pt_val_loader,
         probe_train_loader, probe_val_loader) = load_pretrain_data(pt_config)
        pretrained_model = MODEL_BUILDERS[args.model](args)

        ckpt_path = pretrain_jepa(
            args, pretrained_model,
            pt_train_loader, pt_val_loader,
            probe_train_loader, probe_val_loader,
            device, args.models_dir, viz,
        )
        args.checkpoint = ckpt_path

        # Generate final summary plots
        if viz is not None:
            summary_path = viz.finalize()
            if summary_path:
                print(f"Visualization summary: {summary_path}")

    elif args.checkpoint is not None and args.checkpoint.exists():
        print(f"\nLoading checkpoint: {args.checkpoint}")
        pretrained_model = MODEL_BUILDERS[args.model](args)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        pretrained_model.context_encoder.load_state_dict(ckpt["context_encoder"])
        pretrained_model.to(device)
        print(f"  Loaded epoch {ckpt['epoch']}")

    # ── Phase 2: Supervised training ──────────────────────────────────
    print(f"\n{'=' * 60}")
    phase_label = "Fine-tuning" if pretrained_model is not None else "Direct supervised"
    print(f"Phase 2: {phase_label} ({args.model})")
    print(f"{'=' * 60}")

    train_loader, val_loader, test_loader, classes = load_supervised_data(config)

    if pretrained_model is not None:
        # Fine-tune JEPA model
        metrics = finetune_jepa(
            args, pretrained_model, config,
            train_loader, val_loader, test_loader, classes,
            device, args.models_dir, viz,
        )
    elif args.model == "sigreg":
        # SIGReg direct supervised (uses compute_loss)
        model = MODEL_BUILDERS[args.model](args).to(device)
        run_name = f"{args.model}_seed{args.seed}"
        save_path = args.models_dir / f"{run_name}.pt"

        config.num_epochs = args.epochs
        config.learning_rate = args.lr
        config.use_mixup = False
        config.label_smoothing = 0.0
        config.l1_lambda = 0.0
        config.weight_decay = 0.0
        config.dropout = 0.0

        # Manual training loop with compute_loss
        opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)
        best_val_mcc = -1.0
        patience_counter = 0

        print(f"Training {config.num_epochs} epochs...")
        for epoch in range(config.num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, latents = model(xb)
                loss, _ = model.compute_loss(logits, latents, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()

            # Validation
            model.eval()
            vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits, _ = model(xb.to(device))
                    vap.extend(logits.argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            val_mcc = float(matthews_corrcoef(vat, vap))

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(f"  epoch {epoch + 1:3d}: val_mcc={val_mcc:.4f}")

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(patience={args.patience}, best={best_val_mcc:.4f})")
                break
        # Load best and evaluate — wrap to return logits only
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        class _EvalWrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)[0]
        eval_model = _EvalWrap(model)
        metrics = evaluate_model_on_test(
            eval_model, test_loader, config, classes, run_name, log_wandb=config.use_wandb
        )
        metrics["best_val_mcc"] = best_val_mcc
        print(f"  Best val MCC: {best_val_mcc:.4f}")
        print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")

    else:
        # Direct supervised (Mamba-3 variants)
        model = MODEL_BUILDERS[args.model](args).to(device)
        run_name = f"{args.model}_seed{args.seed}"
        save_path = args.models_dir / f"{run_name}.pt"
        config.num_epochs = args.epochs
        config.learning_rate = args.lr

        trainer = Trainer(config, run_name=run_name)
        result = trainer.fit(model, train_loader, val_loader, save_path=save_path)
        metrics = evaluate_model_on_test(
            model, test_loader, config, classes, run_name, log_wandb=config.use_wandb
        )
        metrics["best_val_mcc"] = result.best_val_mcc
        print(f"  Best val MCC: {result.best_val_mcc:.4f}")
        print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")
    if viz is not None:
        viz.finalize()

    # ── Save results CSV ──────────────────────────────────────────────
    results_file = args.results_dir / "exotic_results.csv"
    file_exists = results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "data", "seed", "pretrained", "test_mcc", "test_acc",
            "best_val_mcc", "timestamp",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model": args.model,
            "data": str(args.data),
            "seed": str(args.seed),
            "pretrained": str(pretrained_model is not None),
            "test_mcc": metrics.get("mcc", ""),
            "test_acc": metrics.get("accuracy", ""),
            "best_val_mcc": metrics.get("best_val_mcc", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        })
    print(f"\nResults appended to {results_file}")


if __name__ == "__main__":
    main()
