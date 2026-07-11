"""Exotic model pipeline — SSL pretraining + supervised fine-tuning.

Extends the standard run_dl.py pattern with a pretraining phase for
self-supervised models (T-JEPA, TS-JEPA, LeWM, Mamba-3 hybrids).

Usage:
    # T-JEPA: pretrain + fine-tune
    uv run python run_exotic.py --model t_jepa --pretrain-epochs 300

    # T-JEPA: fine-tune only (from checkpoint)
    uv run python run_exotic.py --model t_jepa --finetune-only \\
        --checkpoint models/exotic/t_jepa_pretrain_best.pt

    # T-JEPA on augmented data
    uv run python run_exotic.py --model t_jepa --data dataset_augmented.csv \\
        --pretrain-epochs 200 --finetune-epochs 50

    # Mamba-3 CNN (direct supervised training, no pretrain)
    uv run python run_exotic.py --model mamba3_cnn --epochs 150

Options:
    --model NAME          Model type: t_jepa, mamba3_cnn, mamba3_tcn, ...
    --data PATH           Dataset CSV (default: dataset.csv)
    --pretrain-epochs N   SSL pretraining epochs (default: 300)
    --finetune-epochs N   Supervised fine-tuning epochs (default: 50)
    --epochs N            Direct supervised epochs (default: 150)
    --finetune-only       Skip pretraining, load checkpoint
    --unfreeze-encoder    Unfreeze pretrained encoder during fine-tuning
    --checkpoint PATH     Path to pretrained checkpoint for fine-tuning
    --seed INT            Random seed (default: 42)
    --batch-size INT      Batch size (default: 128)
    --lr FLOAT            Learning rate (default: 3e-4 pretrain, 1e-3 finetune)
    --no-wandb            Disable WandB

"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl.config import DLConfig
from dl.data_loader import DLDataLoader, RSSIDataset
from dl.evaluation import evaluate_model_on_test
from dl.training import Trainer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exotic model pipeline")
    p.add_argument("--model", type=str, default="t_jepa",
                   choices=["t_jepa", "t_jepa_v2", "ts_jepa", "ts_jepa_v2",
                            "mamba3_cnn", "mamba3_tcn", "mamba3_transformer",
                            "mamba3_multiview", "sigreg", "lejepa"],
                   help="Model type")
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gpu", type=int, default=0, help="GPU device index (0=auto)")

    # Pretraining
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=3e-4)
    p.add_argument("--pretrain-wd", type=float, default=1e-4)
    p.add_argument("--finetune-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)

    # Fine-tuning / Direct training
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=150, help="Direct supervised epochs")
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-encoder", action="store_true")

    # T-JEPA specific
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--n-reg-tokens", type=int, default=2)
    p.add_argument("--pred-dim", type=int, default=64)
    p.add_argument("--pred-layers", type=int, default=2)
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=0.999)
    p.add_argument("--mask-min-ctx", type=float, default=0.4)
    p.add_argument("--mask-max-ctx", type=float, default=0.75)
    p.add_argument("--mask-min-tgt", type=float, default=0.15)
    p.add_argument("--mask-max-tgt", type=float, default=0.35)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])

    p.add_argument("--hpo", action="store_true", help="Run HPO before training")
    p.add_argument("--hpo-trials", type=int, default=30, help="Optuna trials for HPO")
    p.add_argument("--models-dir", type=Path, default=Path("models/exotic"))
    p.add_argument("--results-dir", type=Path, default=Path("results/exotic"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="inMotion-exotic")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        if gpu == 0:
            return torch.device("cuda:0")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders  (extensible — add new models here)
# ═══════════════════════════════════════════════════════════════════════════════

def build_t_jepa(args: argparse.Namespace) -> nn.Module:
    """Build T-JEPA model (pretrainable)."""
    from dl.models.t_jepa import TJEPAModel

    return TJEPAModel(
        n_features=10,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        n_reg_tokens=args.n_reg_tokens,
        pred_dim=args.pred_dim,
        pred_num_layers=args.pred_layers,
        ema_momentum=args.ema_start,
        target_ema_start=args.ema_start,
        target_ema_end=args.ema_end,
        mask_min_ctx=args.mask_min_ctx,
        mask_max_ctx=args.mask_max_ctx,
        mask_min_tgt=args.mask_min_tgt,
        mask_max_tgt=args.mask_max_tgt,
    )


def build_t_jepa_classifier(
    args: argparse.Namespace, pretrained: nn.Module
) -> nn.Module:
    """Build T-JEPA classifier wrapper from pretrained encoder."""
    from dl.models.t_jepa import TJEPAClassifier

    clf = TJEPAClassifier(
        pretrained=pretrained,
        num_classes=4,
        pooling=args.pooling,
        dropout=0.3,
    )
    # Wrap: extract raw RSSI from (B, 10, 4) standard input
    class Wrapper(nn.Module):
        def __init__(self, c: nn.Module):
            super().__init__()
            self.clf = c

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.clf(x[:, :, 0])

    return Wrapper(clf)


def build_ts_jepa(args: argparse.Namespace) -> nn.Module:
    """Build TS-JEPA model (pretrainable)."""
    from dl.models.ts_jepa import TSJEPAModel
    return TSJEPAModel(
        seq_len=10, patch_size=2, embed_dim=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        pred_dim=args.pred_dim, pred_num_layers=args.pred_layers,
        mask_ratio=0.5, in_channels=1,
    )


def build_ts_jepa_classifier(args: argparse.Namespace, pretrained: nn.Module) -> nn.Module:
    """Build TS-JEPA classifier wrapper from pretrained encoder."""
    from dl.models.ts_jepa import TSJEPAClassifier
    clf = TSJEPAClassifier(
        pretrained=pretrained, num_classes=4, pooling=args.pooling, dropout=0.3,
    )
    class Wrapper(nn.Module):
        def __init__(self, c: nn.Module): super().__init__(); self.clf = c
        def forward(self, x: torch.Tensor) -> torch.Tensor: return self.clf(x[:, :, 0])
    return Wrapper(clf)





def _build_t_jepa_v2(args: argparse.Namespace) -> nn.Module:
    """Build improved T-JEPA v2 model."""
    from dl.models.t_jepa_v2 import TJEPAModelV2
    return TJEPAModelV2(
        n_timesteps=10, n_channels=4, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        n_reg_tokens=args.n_reg_tokens, pred_dim=args.pred_dim,
        pred_num_layers=args.pred_layers, mask_ratio_start=0.3,
        mask_ratio_end=0.6, noise_std=1.0,
    )

def _build_ts_jepa_v2(args: argparse.Namespace) -> nn.Module:
    """Build improved TS-JEPA v2 model."""
    from dl.models.ts_jepa_v2 import TSJEPAModelV2
    return TSJEPAModelV2(
        seq_len=10, patch_size=2, embed_dim=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        pred_dim=args.pred_dim, pred_num_layers=args.pred_layers,
        mask_ratio_start=0.4, mask_ratio_end=0.7, noise_std=1.0,
        in_channels=4,
    )


def _build_lejepa(args: argparse.Namespace) -> nn.Module:
    """Build LeJEPA model for SSL pretraining."""
    from dl.models.lejepa_model import LeJEPAModel
    return LeJEPAModel(
        in_features=4, num_filters=args.d_model, num_blocks=args.num_layers,
        latent_dim=256, proj_dim=args.pred_dim, lamb=0.02, V=2,
    )

MODEL_BUILDERS: dict[str, callable] = {
    "t_jepa": build_t_jepa,
    "t_jepa_v2": lambda args: _build_t_jepa_v2(args),
    "ts_jepa": build_ts_jepa,
    "ts_jepa_v2": lambda args: _build_ts_jepa_v2(args),
    "mamba3_cnn": lambda args: _build_mamba3("cnn", args),
    "mamba3_tcn": lambda args: _build_mamba3("tcn", args),
    "mamba3_multiview": lambda args: _build_mamba3("multiview", args),
    "sigreg": lambda args: _build_sigreg_model(args),
    "lejepa": lambda args: _build_lejepa(args),
}


def _build_mamba3(variant: str, args: argparse.Namespace) -> nn.Module:
    """Build one of the four Mamba-3 hybrid variants."""
    kw = dict(
        in_features=4, d_model=args.d_model, d_state=16,
        num_classes=4, dropout=0.2, mimo_rank=4,
    )
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


def _build_sigreg_model(args: argparse.Namespace) -> nn.Module:
    """Build SIGReg classifier with latent bottleneck + dynamic augmentation."""
    from dl.models.sigreg_classifier import SIGRegClassifier
    return SIGRegClassifier(
        in_features=4, num_filters=args.d_model, num_blocks=args.num_layers,
        latent_dim=args.d_model, num_classes=4,
    )


def pretrain_lejepa(args, model, train_loader, val_loader, device, save_dir):
    """Pretrain LeJEPA — exactly follows the paper's training loop."""
    import torch.nn.functional as F
    from dl.models.lejepa_model import LeJEPAModel
    assert isinstance(model, LeJEPAModel)
    model.to(device)

    # Online probe
    probe = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 4)).to(device)

    params = list(model.backbone.parameters()) + list(probe.parameters())
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.pretrain_lr, "weight_decay": 5e-2},
        {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7},
    ])

    best_path = save_dir / f"{args.model}_pretrain_best.pt"
    best_loss = float("inf")
    patience = 25; patience_counter = 0

    print(f"Pretraining {args.pretrain_epochs} epochs (LeJEPA, V={model.V})...")
    for epoch in range(args.pretrain_epochs):
        model.train(); probe.train()
        epoch_loss = 0.0; epoch_probe = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            emb, proj = model(x)
            # LeJEPA losses — exactly per paper
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = model.sigreg(proj)
            lejepa_loss = model.lamb * sigreg_loss + (1.0 - model.lamb) * inv_loss
            # Online probe on detached embeddings
            y_dummy = torch.zeros(x.size(0), dtype=torch.long, device=device)
            probe_loss = F.cross_entropy(probe(emb.detach()), y_dummy.repeat_interleave(model.V))
            loss = lejepa_loss + probe_loss * 0.1  # downweight probe
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters()) + list(probe.parameters()), 1.0)
            opt.step()
            epoch_loss += lejepa_loss.item(); epoch_probe += probe_loss.item()

        epoch_loss /= max(len(train_loader), 1)
        if epoch % 20 == 0 or epoch == args.pretrain_epochs - 1:
            print(f"  epoch {epoch+1:3d}: lejepa={epoch_loss:.4f}, probe={epoch_probe/len(train_loader):.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss; patience_counter = 0
            torch.save({"backbone": model.backbone.state_dict(), "epoch": epoch}, best_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}"); break

    print(f"Pretraining done. Best lejepa loss: {best_loss:.4f}")
    return best_path


# ═══════════════════════════════════════════════════════════════════════════════
# Pretraining
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_t_jepa(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader | None,  # type: ignore[type-arg]
    device: torch.device,
    save_dir: Path,
) -> Path:
    """Pretrain T-JEPA or TS-JEPA. Returns path to best checkpoint."""
    # Both TJEPAModel and TSJEPAModel share the same interface
    model.to(device)

    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    # ── WandB init ─────────────────────────────────────────────────
    pretrain_wandb: object | None = None
    if not args.no_wandb:
        try:
            import wandb
            pretrain_wandb = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_pretrain_seed{args.seed}",
                config={
                    "model": args.model, "phase": "pretrain",
                    "pretrain_epochs": args.pretrain_epochs,
                    "batch_size": args.batch_size, "lr": args.pretrain_lr,
                },
                reinit=True,
            )
        except Exception:
            pass

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pretrain_epochs)

    best_val = float("inf")
    best_path = save_dir / f"{args.model}_pretrain_best.pt"
    patience = 25
    patience_counter = 0

    print(f"Pretraining {args.pretrain_epochs} epochs ({args.model})...")
    t0 = time.time()

    for epoch in range(args.pretrain_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            loss, _ = model.pretrain_step(x, epoch, args.pretrain_epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            train_loss += loss.item()
        sch.step()
        train_loss /= max(len(train_loader), 1)

        # Validation
        val_loss = float("inf")
        if val_loader is not None:
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    tgt, preds, _, _ = model.forward(x)
                    for p in preds:
                        val_sum += (p - tgt).pow(2).mean().item()
            val_loss = val_sum / max(len(val_loader), 1)


        # ── WandB logging ───────────────────────────────────────────
        if pretrain_wandb is not None:
            import wandb
            wandb.log({
                "pretrain/train_loss": train_loss,
                "pretrain/val_loss": val_loss,
                "pretrain/ema": getattr(model, "ema_momentum", 0.0),
                "pretrain/lr": sch.get_last_lr()[0],
                "pretrain/epoch": epoch,
            })
        # Logging
        if epoch % 20 == 0 or epoch == args.pretrain_epochs - 1:
            ema = getattr(model, "ema_momentum", 0.0)
            print(f"  epoch {epoch + 1:3d}/{args.pretrain_epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | ema={ema:.4f}")

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
            }, best_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    elapsed = time.time() - t0
    print(f"Pretraining done in {elapsed:.0f}s. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")
    return best_path

    if pretrain_wandb is not None:
        import wandb
        wandb.log({"pretrain/best_val_loss": best_val})
        wandb.finish()


PRETRAINERS: dict[str, callable] = {
    "t_jepa": pretrain_t_jepa,
    "t_jepa_v2": pretrain_t_jepa,
    "ts_jepa": pretrain_t_jepa,
    "ts_jepa_v2": pretrain_t_jepa,
    "lejepa": pretrain_lejepa,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning / Direct supervised training
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_t_jepa(
    args: argparse.Namespace,
    pretrained_model: nn.Module,
    config: DLConfig,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    test_loader: DataLoader,  # type: ignore[type-arg]
    classes: list[str],
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Fine-tune a pretrained T-JEPA encoder for classification."""
    from dl.models.t_jepa import TJEPAModel

    assert isinstance(pretrained_model, TJEPAModel)

    # Build classifier
    wrapped = build_t_jepa_classifier(args, pretrained_model)

    if not args.unfreeze_encoder:
        # Freeze encoder
        encoder = pretrained_model.context_encoder
        for p in encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — training head only")
    else:
        print("Encoder unfrozen — full fine-tuning")

    config.num_epochs = args.finetune_epochs
    config.learning_rate = args.finetune_lr
    config.use_wandb = not args.no_wandb

    run_name = f"{args.model}_ft_seed{args.seed}"
    save_path = save_dir / f"{run_name}.pt"

    trainer = Trainer(config, run_name=run_name, extra_wandb_config={
        "model": args.model,
        "pretrained": True,
        "finetune_epochs": args.finetune_epochs,
        "unfreeze": args.unfreeze_encoder,
    })

    t0 = time.time()
    result = trainer.fit(wrapped, train_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    metrics = evaluate_model_on_test(
        wrapped, test_loader, config, classes, run_name, log_wandb=config.use_wandb
    )

    print(f"Fine-tuning done in {elapsed:.0f}s.")
    print(f"  Best val MCC: {result.best_val_mcc:.4f}")
    metrics["best_val_mcc"] = result.best_val_mcc
    print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrain_data(
    config: DLConfig, device: torch.device
) -> tuple[DataLoader, DataLoader | None]:  # type: ignore[type-arg]
    """Load data for SSL pretraining (no labels needed).

    Returns (train_loader, val_loader) with (B, 10) raw RSSI.
    """
    from dl.jepa_training import prepare_jepa_data
    return prepare_jepa_data(config)


def load_supervised_data(
    config: DLConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:  # type: ignore[type-arg]
    """Load data for supervised training (standard pipeline).

    Returns (train_loader, val_loader, test_loader, classes).
    """
    dl = DLDataLoader(config)
    X, y = dl.load_and_preprocess()

    classes = dl.classes_
    # Replace generic names
    label_map = {0: "AA", 1: "AB", 2: "BA", 3: "BB"}
    classes = [label_map.get(i, c) for i, c in enumerate(classes)]

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.seed, stratify=y
    )
    # Further split train into train/val


    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=config.seed, stratify=y_tr
    )

    train_ds = RSSIDataset(X_tr, y_tr)
    val_ds = RSSIDataset(X_va, y_va)
    test_ds = RSSIDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes




def _finetune_ts_jepa(
    args: argparse.Namespace,
    pretrained_model: nn.Module,
    config: DLConfig,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    test_loader: DataLoader,  # type: ignore[type-arg]
    classes: list[str],
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Fine-tune a pretrained TS-JEPA encoder for classification."""
    from dl.models.ts_jepa import TSJEPAModel
    assert isinstance(pretrained_model, TSJEPAModel)

    wrapped = build_ts_jepa_classifier(args, pretrained_model)

    if not args.unfreeze_encoder:
        for p in pretrained_model.context_encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — training head only")
    else:
        print("Encoder unfrozen — full fine-tuning")

    config.num_epochs = args.finetune_epochs
    config.learning_rate = args.finetune_lr
    config.use_wandb = not args.no_wandb

    run_name = f"{args.model}_ft_seed{args.seed}"
    save_path = save_dir / f"{run_name}.pt"

    trainer = Trainer(config, run_name=run_name, extra_wandb_config={
        "model": args.model, "pretrained": True,
        "finetune_epochs": args.finetune_epochs, "unfreeze": args.unfreeze_encoder,
    })

    t0 = time.time()
    result = trainer.fit(wrapped, train_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    metrics = evaluate_model_on_test(
        wrapped, test_loader, config, classes, run_name, log_wandb=config.use_wandb
    )

    print(f"Fine-tuning done in {elapsed:.0f}s.")
    print(f"  Best val MCC: {result.best_val_mcc:.4f}")
    metrics["best_val_mcc"] = result.best_val_mcc
    print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")
    return metrics




def _finetune_ts_jepa_v2(
    args: argparse.Namespace,
    pretrained_model: nn.Module,
    config: DLConfig,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    test_loader: DataLoader,  # type: ignore[type-arg]
    classes: list[str],
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Fine-tune TS-JEPA v2 with progressive unfreezing."""
    from dl.models.ts_jepa_v2 import TSJEPAModelV2, TSJEPAClassifierV2
    clf = TSJEPAClassifierV2(pretrained=pretrained_model, num_classes=4, hidden_dim=128).to(device)
    class W(nn.Module):
        def __init__(self, c): super().__init__(); self.c = c
        def forward(self, x): return self.c(x)
    wrapped = W(clf)
    config.use_wandb = not args.no_wandb
    run_name = f"{args.model}_ft_seed{args.seed}"
    save_path = save_dir / f"{run_name}.pt"
    # Stage 1: freeze encoder
    for p in clf.encoder.parameters(): p.requires_grad = False
    config.num_epochs = args.finetune_epochs // 2
    config.learning_rate = args.finetune_lr
    r1 = Trainer(config, run_name=f"{run_name}_s1", extra_wandb_config={"stage":1}).fit(wrapped, train_loader, val_loader)
    # Stage 2: unfreeze
    for p in clf.encoder.parameters(): p.requires_grad = True
    config.num_epochs = args.finetune_epochs // 2
    config.learning_rate = args.finetune_lr * 0.1
    r2 = Trainer(config, run_name=f"{run_name}_s2", extra_wandb_config={"stage":2}).fit(wrapped, train_loader, val_loader, save_path=save_path)
    metrics = evaluate_model_on_test(wrapped, test_loader, config, classes, run_name, log_wandb=config.use_wandb)
    metrics["best_val_mcc"] = r2.best_val_mcc
    return metrics




def _finetune_t_jepa_v2(
    args: argparse.Namespace,
    pretrained_model: nn.Module,
    config: DLConfig,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    test_loader: DataLoader,  # type: ignore[type-arg]
    classes: list[str],
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Fine-tune T-JEPA v2 with progressive unfreezing."""
    from dl.models.t_jepa_v2 import TJEPAModelV2, TJEPAClassifierV2
    clf = TJEPAClassifierV2(pretrained=pretrained_model, num_classes=4, hidden_dim=128).to(device)
    class W(nn.Module):
        def __init__(self, c): super().__init__(); self.c = c
        def forward(self, x): return self.c(x)
    wrapped = W(clf)
    config.use_wandb = not args.no_wandb
    run_name = f"{args.model}_ft_seed{args.seed}"
    save_path = save_dir / f"{run_name}.pt"
    for p in clf.encoder.parameters(): p.requires_grad = False
    config.num_epochs = args.finetune_epochs // 2
    config.learning_rate = args.finetune_lr
    r1 = Trainer(config, run_name=f"{run_name}_s1", extra_wandb_config={"stage":1}).fit(wrapped, train_loader, val_loader)
    for p in clf.encoder.parameters(): p.requires_grad = True
    config.num_epochs = args.finetune_epochs // 2
    config.learning_rate = args.finetune_lr * 0.1
    r2 = Trainer(config, run_name=f"{run_name}_s2", extra_wandb_config={"stage":2}).fit(wrapped, train_loader, val_loader, save_path=save_path)
    metrics = evaluate_model_on_test(wrapped, test_loader, config, classes, run_name, log_wandb=config.use_wandb)
    metrics["best_val_mcc"] = r2.best_val_mcc
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════



def _hpo_trial_name(prefix: str, trial, params: dict) -> str:
    """Build compact WandB run name from trial params — matches run_dl.py style."""
    p = params
    parts = [prefix, f"t{trial.number}"]
    if "d_model" in p: parts.append(f"d{p['d_model']}")
    if "nhead" in p: parts.append(f"nh{p['nhead']}")
    if "num_layers" in p: parts.append(f"l{p['num_layers']}")
    if "dim_ff" in p: parts.append(f"ff{p['dim_ff']}")
    if "pred_dim" in p: parts.append(f"pd{p['pred_dim']}")
    if "pred_layers" in p: parts.append(f"pl{p['pred_layers']}")
    if "mask_start" in p: parts.append(f"ms{float(p['mask_start']):.1f}")
    if "mask_end" in p: parts.append(f"me{float(p['mask_end']):.1f}")
    if "noise_std" in p: parts.append(f"ns{float(p['noise_std']):.1f}")
    if "pretrain_lr" in p: parts.append(f"ptlr{float(p['pretrain_lr']):.1e}")
    if "finetune_lr" in p: parts.append(f"ftlr{float(p['finetune_lr']):.1e}")
    if "ema_start" in p: parts.append(f"ema{float(p['ema_start']):.3f}")
    if "n_reg_tokens" in p: parts.append(f"reg{p['n_reg_tokens']}")
    if "d_state" in p: parts.append(f"st{p['d_state']}")
    if "dropout" in p: parts.append(f"do{float(p['dropout']):.2f}")
    if "lr" in p: parts.append(f"lr{float(p['lr']):.1e}")
    if "mimo_rank" in p: parts.append(f"mr{p['mimo_rank']}")
    return "_".join(parts)


def _run_hpo(args: argparse.Namespace, device: torch.device) -> None:
    """Run Optuna HPO, then update args with best hyperparams."""
    import optuna
    from dl.config import DLConfig as Cfg
    from dl.data_loader import DLDataLoader as DL

    print(f"\n{'='*60}")
    print(f"HPO: {args.model} — {args.hpo_trials} trials")
    print(f"{'='*60}")

    cfg = Cfg(); cfg.data_path = args.data; cfg.seed = args.seed
    cfg.batch_size = args.batch_size; cfg.device = str(device)
    dl = DL(cfg); X, y = dl.load_and_preprocess()
    hpo_seed = args.seed + 0x5EED  # avoid collision with final train/test split
    from sklearn.model_selection import train_test_split
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=hpo_seed, stratify=y)

    if args.model in ("t_jepa_v2", "ts_jepa_v2"):
        _hpo_jepa(args, device, Xt, yt, Xv, yv)
    elif args.model.startswith("mamba3"):
        _hpo_mamba3(args, device, Xt, yt, Xv, yv)
    elif args.model == "sigreg":
        _hpo_sigreg(args, device, Xt, yt, Xv, yv)
    else:
        print(f"  No HPO search space for '{args.model}', skipping.")

def _hpo_jepa(args, device, Xt, yt, Xv, yv) -> None:
    import optuna
    from sklearn.metrics import matthews_corrcoef
    from torch.utils.data import DataLoader, TensorDataset
    from dl.data_loader import RSSIDataset

    is_ts = args.model.startswith("ts")
    trial_counter = [0]  # mutable counter for trial number

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [128, 192, 256, 384])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 6)
        dim_ff = trial.suggest_categorical("dim_ff", [256, 512, 768, 1024])
        pred_dim = trial.suggest_categorical("pred_dim", [32, 64, 128, 256])
        pred_layers = trial.suggest_int("pred_layers", 1, 4)
        mask_start = trial.suggest_float("mask_start", 0.1, 0.5, step=0.1)
        mask_end = trial.suggest_float("mask_end", 0.4, 0.8, step=0.1)
        noise_std = trial.suggest_float("noise_std", 0.0, 3.0, step=0.5)
        pt_lr = trial.suggest_float("pretrain_lr", 1e-5, 1e-3, log=True)
        ft_lr = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
        ema_start = trial.suggest_float("ema_start", 0.99, 0.999)
        # ── WandB per-trial init (after suggest so trial.params is populated) ─
        trial_wandb = None
        if not args.no_wandb:
            try:
                import wandb
                name = _hpo_trial_name(f"hpo_{args.model}", trial, trial.params)
                trial_wandb = wandb.init(
                    project=args.wandb_project,
                    name=name,
                    config=trial.params,
                )
            except Exception as e:
                print(f"  [WandB] trial {trial.number} init failed: {e}")

        # Ensure nhead divides d_model
        while d_model % nhead != 0:
            nhead = max(1, nhead // 2)
        kw = dict(seq_len=10, patch_size=2, embed_dim=d_model, nhead=nhead,
                  num_layers=num_layers, dim_feedforward=dim_ff, pred_dim=pred_dim,
                  pred_num_layers=pred_layers, mask_ratio_start=mask_start,
                  mask_ratio_end=mask_end, noise_std=noise_std,
                  ema_start=ema_start, in_channels=4)

        if is_ts:
            from dl.models.ts_jepa_v2 import TSJEPAModelV2, TSJEPAClassifierV2
            m = TSJEPAModelV2(**kw).to(device)
        else:
            from dl.models.t_jepa_v2 import TJEPAModelV2, TJEPAClassifierV2
            nhead_use = nhead
            while d_model % nhead_use != 0:
                nhead_use = max(1, nhead_use // 2)
            m = TJEPAModelV2(n_timesteps=10, n_channels=4, d_model=d_model, nhead=nhead_use,
                             num_layers=num_layers, dim_feedforward=dim_ff,
                             n_reg_tokens=2, pred_dim=pred_dim, pred_num_layers=pred_layers,
                             mask_ratio_start=mask_start, mask_ratio_end=mask_end,
                             noise_std=noise_std).to(device)

        Xp = Xt.astype(np.float32)
        pm, ps = Xp.mean(axis=(0,1), keepdims=True), Xp.std(axis=(0,1), keepdims=True) + 1e-8
        Xp = (Xp - pm) / ps
        ptds = TensorDataset(torch.tensor(Xp, dtype=torch.float32))
        ptl = DataLoader(ptds, batch_size=args.batch_size, shuffle=True)
        params = list(m.context_encoder.parameters()) + list(m.predictor.parameters())
        opt = torch.optim.AdamW(params, lr=pt_lr)
        for ep in range(50):
            m.train()
            pt_loss_sum = 0.0
            pt_n = 0
            for (xb,) in ptl:
                loss, _ = m.pretrain_step(xb.to(device), ep, 50)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                pt_loss_sum += loss.item()
                pt_n += 1
            pt_avg_loss = pt_loss_sum / max(pt_n, 1)
            if trial_wandb is not None:
                import wandb
                wandb.log({
                    "pretrain/epoch": ep,
                    "pretrain/train_loss": pt_avg_loss,
                    "pretrain/lr": pt_lr,
                })

        if is_ts:
            clf = TSJEPAClassifierV2(pretrained=m, num_classes=4, hidden_dim=128).to(device)
        else:
            clf = TJEPAClassifierV2(pretrained=m, num_classes=4, hidden_dim=128).to(device)

        class W(nn.Module):
            def __init__(s, c): super().__init__(); s.c = c
            def forward(s, x): return s.c(x)
        wr = W(clf)

        for p in clf.encoder.parameters(): p.requires_grad = False
        opt_ft = torch.optim.AdamW(clf.head.parameters(), lr=ft_lr)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
        for ep in range(15):
            clf.train()
            ft_loss_sum = 0.0
            ft_n = 0
            for xb, yb in trl:
                loss = ce(wr(xb.to(device)), yb.to(device))
                opt_ft.zero_grad(); loss.backward(); opt_ft.step()
                ft_loss_sum += loss.item()
                ft_n += 1
            # Evaluate on validation set
            clf.eval(); vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if trial_wandb is not None:
                import wandb
                wandb.log({
                    "ft/stage1_epoch": ep,
                    "ft/stage1_train_loss": ft_loss_sum / max(ft_n, 1),
                    "ft/stage1_val_mcc": vmcc,
                })
        for p in clf.encoder.parameters(): p.requires_grad = True
        opt_all = torch.optim.AdamW(clf.parameters(), lr=ft_lr * 0.1)
        for ep in range(15):
            clf.train()
            ft_loss_sum = 0.0
            ft_n = 0
            for xb, yb in trl:
                loss = ce(wr(xb.to(device)), yb.to(device))
                opt_all.zero_grad(); loss.backward(); opt_all.step()
                ft_loss_sum += loss.item()
                ft_n += 1
            # Evaluate on validation set
            clf.eval(); vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if trial_wandb is not None:
                import wandb
                wandb.log({
                    "ft/stage2_epoch": ep,
                    "ft/stage2_train_loss": ft_loss_sum / max(ft_n, 1),
                    "ft/stage2_val_mcc": vmcc,
                })
        clf.eval(); ap, at = [], []
        with torch.no_grad():
            for xb, yb in val:
                ap.extend(wr(xb.to(device)).argmax(1).cpu().tolist()); at.extend(yb.tolist())
        mcc = float(matthews_corrcoef(at, ap))

        # ── WandB final log + finish ──────────────────────────────────
        if trial_wandb is not None:
            try:
                import wandb
                wandb.log({"trial_mcc": float(mcc)})
                wandb.finish()
            except Exception as e:
                print(f"  [WandB] trial {trial.number} finish failed: {e}")

        return mcc

    from dl.optimization import _get_optuna_storage, _log_trial_callback
    storage_url = "sqlite:///optuna_exotic.db"
    study_name = f"exotic_{args.model}"
    storage = _get_optuna_storage(storage_url)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True, callbacks=[_log_trial_callback])

    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    for k, v in study.best_params.items():
        if hasattr(args, k):
            setattr(args, k, v)
    if "d_model" in study.best_params:
        args.d_model = study.best_params.get("d_model", args.d_model)
        args.nhead = study.best_params.get("nhead", args.nhead)
        args.num_layers = study.best_params.get("num_layers", args.num_layers)
        args.dim_ff = study.best_params.get("dim_ff", args.dim_ff)
        args.pred_dim = study.best_params.get("pred_dim", args.pred_dim)
        args.pred_layers = study.best_params.get("pred_layers", args.pred_layers)
        if "n_reg_tokens" in study.best_params:
            args.n_reg_tokens = study.best_params["n_reg_tokens"]



def _hpo_sigreg(args, device, Xt, yt, Xv, yv) -> None:
    import optuna
    from dl.data_loader import RSSIDataset
    from dl.models.sigreg_classifier import SIGRegClassifier

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [128, 192, 256, 384])
        num_layers = trial.suggest_int("num_layers", 1, 6)
        latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 384, 512])
        sigreg_lambda = trial.suggest_float("sigreg_lambda", 0.001, 0.5, log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        aug_drop_t = trial.suggest_float("aug_drop_t", 0.0, 0.3, step=0.05)
        aug_drop_c = trial.suggest_float("aug_drop_c", 0.0, 0.2, step=0.05)
        aug_block_p = trial.suggest_float("aug_block_p", 0.0, 0.5, step=0.1)
        aug_noise = trial.suggest_float("aug_noise", 0.0, 2.0, step=0.5)

        m = SIGRegClassifier(
            in_features=4, num_filters=d_model, num_blocks=num_layers,
            latent_dim=latent_dim, num_classes=4, sigreg_lambda=sigreg_lambda,
            aug_drop_t=aug_drop_t, aug_drop_c=aug_drop_c,
            aug_block_p=aug_block_p, aug_noise=aug_noise,
        ).to(device)

        cfg = DLConfig(); cfg.seed = args.seed; cfg.batch_size = args.batch_size
        cfg.device = str(device); cfg.num_epochs = 40; cfg.learning_rate = lr
        cfg.use_wandb = False; cfg.use_mixup = False; cfg.label_smoothing = 0.0
        cfg.l1_lambda = 0.0; cfg.weight_decay = 0.0; cfg.dropout = 0.0

        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
        from dl.training import Trainer
        result = Trainer(cfg).fit(m, trl, val)
        mcc = float(result.best_val_mcc)

        if not args.no_wandb:
            try:
                import wandb
                name = _hpo_trial_name(f"hpo_{args.model}", trial, trial.params)
                run = wandb.init(project=args.wandb_project, name=name, config=trial.params)
                wandb.log({"trial_mcc": float(mcc)})
                wandb.finish()
            except Exception as e:
                print(f"  [WandB] trial {trial.number}: {e}")
        return mcc

    from dl.optimization import _get_optuna_storage
    storage = _get_optuna_storage("sqlite:///optuna_exotic.db")
    study = optuna.create_study(direction="maximize", study_name=f"exotic_{args.model}",
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=False)

    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    for k, v in study.best_params.items():
        if hasattr(args, k):
            setattr(args, k, v)


def _hpo_mamba3(args, device, Xt, yt, Xv, yv) -> None:
    import optuna
    from dl.data_loader import RSSIDataset

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128, 192, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        d_state = trial.suggest_categorical("d_state", [8, 16, 32])
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        mimo_rank = trial.suggest_categorical("mimo_rank", [1, 2, 4])
        # ── WandB per-trial init (after suggest so trial.params is populated) ─
        trial_wandb = None
        if not args.no_wandb:
            try:
                import wandb
                name = _hpo_trial_name(f"hpo_{args.model}", trial, trial.params)
                trial_wandb = wandb.init(
                    project=args.wandb_project,
                    name=name,
                    config=trial.params,
                )
            except Exception as e:
                print(f"  [WandB] trial {trial.number} init failed: {e}")

        if args.model == "mamba3_cnn":
            from dl.models.mamba3_cnn import Mamba3CNN
            m = Mamba3CNN(in_features=4, cnn_channels=d_model, d_model=d_model,
                          d_state=d_state, n_mamba_layers=num_layers, num_classes=4,
                          dropout=dropout, mimo_rank=mimo_rank).to(device)
        elif args.model == "mamba3_tcn":
            from dl.models.mamba3_tcn import Mamba3TCN
            m = Mamba3TCN(in_features=4, tcn_channels=d_model, d_model=d_model,
                          d_state=d_state, n_mamba_layers=num_layers, num_classes=4,
                          dropout=dropout, mimo_rank=mimo_rank).to(device)
        elif args.model == "mamba3_transformer":
            from dl.models.mamba3_transformer import Mamba3Transformer
            m = Mamba3Transformer(in_features=4, d_model=d_model, d_state=d_state,
                                  nhead=4, num_blocks=num_layers, num_classes=4,
                                  dropout=dropout, mimo_rank=mimo_rank).to(device)
        else:
            from dl.models.mamba3_multiview import Mamba3MultiView
            m = Mamba3MultiView(in_features=4, d_model=d_model, d_state=d_state,
                                n_mamba_layers=num_layers, num_classes=4,
                                dropout=dropout, mimo_rank=mimo_rank).to(device)

        cfg = DLConfig(); cfg.seed = args.seed; cfg.batch_size = args.batch_size
        cfg.device = str(device); cfg.num_epochs = 30; cfg.learning_rate = lr
        cfg.use_wandb = False
        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
        from dl.training import Trainer
        result = Trainer(cfg).fit(m, trl, val)
        mcc = float(result.best_val_mcc)
        # ── WandB per-trial logging ───────────────────────────────────
        if trial_wandb is not None:
            try:
                import wandb
                # Log per-epoch metrics from Trainer result
                for ep, (tl, vl, vm) in enumerate(zip(
                    result.train_losses, result.val_losses, result.val_mccs
                ), start=1):
                    wandb.log({
                        "epoch": ep,
                        "train_loss": tl,
                        "val_loss": vl,
                        "val_mcc": vm,
                    })
                wandb.log({"trial_mcc": float(mcc)})
                wandb.finish()
            except Exception as e:
                print(f"  [WandB] trial {trial.number} finish failed: {e}")
        return mcc

    from dl.optimization import _get_optuna_storage, _log_trial_callback
    storage_url = "sqlite:///optuna_exotic.db"
    study_name = f"exotic_{args.model}"
    storage = _get_optuna_storage(storage_url)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True, callbacks=[_log_trial_callback])

    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    for k, v in study.best_params.items():
        if hasattr(args, k):
            setattr(args, k, v)

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


    # ── Load pretrained checkpoint if provided ────────────────────────
    pretrained_model: nn.Module | None = None

    if args.checkpoint and args.checkpoint.exists():
        print(f"\nLoading checkpoint: {args.checkpoint}")
        builder = MODEL_BUILDERS.get(args.model)
        if builder is None:
            raise ValueError(f"No builder for model '{args.model}'. "
                             f"Available: {list(MODEL_BUILDERS)}")
        pretrained_model = builder(args)
        from dl.jepa_training import JEPATrainer
        jt = JEPATrainer(DLConfig(), pretrained_model)
        epoch = jt.load_checkpoint(args.checkpoint)
        pretrained_model.to(device)
        print(f"  Loaded epoch {epoch}")

    # ── Phase 1: Pretraining (skip if finetune-only or checkpoint provided) ──
    if not args.finetune_only and pretrained_model is None:
        pretrainer = PRETRAINERS.get(args.model)
        if pretrainer is not None:
            print(f"\n{'=' * 60}")
            print(f"Phase 1: SSL Pretraining ({args.model})")
            print(f"{'=' * 60}")

            pt_config = DLConfig()
            pt_config.data_path = args.data
            pt_config.seed = args.seed
            pt_config.batch_size = args.batch_size
            pt_config.device = str(device)

            pt_train_loader, pt_val_loader = load_pretrain_data(pt_config, device)

            pretrained_model = MODEL_BUILDERS[args.model](args)

            ckpt_path = pretrainer(
                args, pretrained_model,
                pt_train_loader, pt_val_loader,
                device, args.models_dir,
            )
            args.checkpoint = ckpt_path
        else:
            print(f"Note: '{args.model}' has no pretrainer. "
                  f"Running direct supervised training.")

    # ── Phase 2: Fine-tuning / Direct supervised ──────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Supervised Training ({args.model})")
    print(f"{'=' * 60}")

    train_loader, val_loader, test_loader, classes = load_supervised_data(config)

    if pretrained_model is not None:
        # Fine-tune mode — dispatch based on model type
        if args.model == "t_jepa":
            metrics = finetune_t_jepa(
                args, pretrained_model, config,
                train_loader, val_loader, test_loader, classes,
                device, args.models_dir,
            )
        elif args.model == "ts_jepa":
            metrics = _finetune_ts_jepa(
                args, pretrained_model, config,
                train_loader, val_loader, test_loader, classes,
                device, args.models_dir,
            )
        elif args.model == "ts_jepa_v2":
            metrics = _finetune_ts_jepa_v2(
                args, pretrained_model, config,
                train_loader, val_loader, test_loader, classes,
                device, args.models_dir,
            )
        elif args.model == "t_jepa_v2":
            metrics = _finetune_t_jepa_v2(
                args, pretrained_model, config,
                train_loader, val_loader, test_loader, classes,
                device, args.models_dir,
            )
        else:
            raise ValueError(f"No fine-tuner for model '{args.model}'")
    else:
        # Direct supervised mode (no pretraining)
        model = MODEL_BUILDERS[args.model](args)

        config.num_epochs = args.epochs
        if args.model == "sigreg" and args.hpo:
            config.learning_rate = getattr(args, "lr", config.learning_rate)
            config.use_mixup = False
            config.label_smoothing = 0.0
            config.l1_lambda = 0.0
            config.weight_decay = 0.0
            config.dropout = 0.0
        run_name = f"{args.model}_seed{args.seed}"
        save_path = args.models_dir / f"{run_name}.pt"

        trainer = Trainer(config, run_name=run_name)
        result = trainer.fit(model, train_loader, val_loader, save_path=save_path)
        metrics = evaluate_model_on_test(
            model, test_loader, config, classes, run_name, log_wandb=config.use_wandb
        )

        print(f"Training done.")
        print(f"  Best val MCC: {result.best_val_mcc:.4f}")
        metrics["best_val_mcc"] = result.best_val_mcc
        print(f"  Test MCC:     {metrics.get('mcc', 'N/A')}")

    # ── Save results CSV ─────────────────────────────────────────────
    import csv
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
            "test_mcc": metrics.get("mcc", ""),
            "test_acc": metrics.get("accuracy", ""),
            "best_val_mcc": metrics.get("best_val_mcc", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        })
    print(f"\nResults appended to {results_file}")


if __name__ == "__main__":
    main()
