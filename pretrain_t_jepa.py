"""T-JEPA pretraining + fine-tuning pipeline for inMotion RSSI classification.

Two-phase workflow:
  1. Self-supervised pretraining:  T-JEPA learns temporal structure from
     unlabeled RSSI sequences by predicting masked-timestep latents.
  2. Supervised fine-tuning:  the pretrained context encoder is frozen
     (or fine-tuned) with a classification head on labeled data.

Usage:
    # Phase 1: Pretrain only
    python pretrain_t_jepa.py --data dataset.csv --epochs 300

    # Phase 2: Fine-tune from checkpoint
    python pretrain_t_jepa.py --data dataset.csv --finetune --checkpoint models/dl/t_jepa_best.pt --epochs 50

    # Both phases in one run
    python pretrain_t_jepa.py --data dataset.csv --pretrain-epochs 300 --finetune-epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from dl.config import DLConfig
from dl.data_loader import DLDataLoader
from dl.jepa_training import JEPATrainer, prepare_jepa_data
from dl.models.t_jepa import TJEPAModel, TJEPAClassifier
from dl.training import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-JEPA pretraining + fine-tuning")
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)

    # Pretraining
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--pred-dim", type=int, default=64)
    p.add_argument("--pred-layers", type=int, default=2)
    p.add_argument("--n-reg-tokens", type=int, default=2)
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=1.0)
    p.add_argument("--mask-min-ctx", type=float, default=0.5)
    p.add_argument("--mask-max-ctx", type=float, default=0.8)
    p.add_argument("--mask-min-tgt", type=float, default=0.15)
    p.add_argument("--mask-max-tgt", type=float, default=0.35)

    # Fine-tuning
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--unfreeze-encoder", action="store_true", help="Unfreeze encoder during fine-tuning")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])

    # Misc
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--models-dir", type=Path, default=Path("models/dl"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pretrain(args: argparse.Namespace) -> TJEPAModel:
    """Phase 1: Self-supervised pretraining."""
    print("=" * 60)
    print("Phase 1: T-JEPA Self-Supervised Pretraining")
    print("=" * 60)

    config = DLConfig()
    config.data_path = args.data
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.use_wandb = not args.no_wandb
    config.num_epochs = args.pretrain_epochs

    train_loader, val_loader = prepare_jepa_data(config)

    model = TJEPAModel(
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"T-JEPA model: {n_params:,} parameters")
    print(f"  d_model={args.d_model}, layers={args.num_layers}, heads={args.nhead}")
    print(f"  [REG] tokens: {args.n_reg_tokens}")
    print(f"  Masking: ctx={args.mask_min_ctx}-{args.mask_max_ctx}, tgt={args.mask_min_tgt}-{args.mask_max_tgt}")

    trainer = JEPATrainer(
        config=config,
        model=model,
        wandb_project="inMotion-tjepa" if config.use_wandb else None,
        wandb_name=f"tjepa_d{args.d_model}_l{args.num_layers}_reg{args.n_reg_tokens}",
    )

    result = trainer.pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.pretrain_epochs,
        lr=args.lr,
        weight_decay=args.wd,
        checkpoint_dir=args.models_dir,
    )

    print(f"\nPretraining complete:")
    print(f"  Best val loss: {result.best_val_loss:.4f} (epoch {result.best_epoch + 1})")
    if result.collapsed:
        print("  WARNING: Representation collapse detected. Try more [REG] tokens.")
    else:
        print("  No collapse detected — representations are healthy.")

    return model


def run_finetune(args: argparse.Namespace, model: TJEPAModel) -> dict:
    """Phase 2: Supervised fine-tuning."""
    print("\n" + "=" * 60)
    print("Phase 2: Supervised Fine-Tuning")
    print("=" * 60)

    config = DLConfig()
    config.data_path = args.data
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.num_epochs = args.finetune_epochs
    config.use_wandb = not args.no_wandb
    config.learning_rate = args.finetune_lr
    config.use_mixup = True
    config.mixup_alpha = 0.2
    config.label_smoothing = 0.1
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build classifier from pretrained encoder
    classifier = TJEPAClassifier(
        pretrained=model,
        num_classes=4,
        pooling=args.pooling,
        dropout=config.dropout,
    )

    if not args.unfreeze_encoder:
        # Freeze encoder, train only the head
        for p in classifier.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — training classification head only")
    else:
        print("Encoder unfrozen — full fine-tuning")

    # Load full dataset with labels
    data_loader = DLDataLoader(config)
    X, y = data_loader.load_and_preprocess()

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    from dl.data_loader import RSSIDataset

    train_ds = RSSIDataset(X_train, y_train)
    test_ds = RSSIDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    # Train with existing Trainer
    # The classifier expects (B, 10) but the data loader gives (B, 10, 4).
    # We need a wrapper that extracts raw RSSI.
    class TJEPAWrapper(nn.Module):
        def __init__(self, clf: TJEPAClassifier):
            super().__init__()
            self.clf = clf

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 10, 4) → (B, 10) raw RSSI
            return self.clf(x[:, :, 0])

    wrapped = TJEPAWrapper(classifier)
    # Move to GPU
    device = config.resolve_device()
    wrapped = wrapped.to(device)

    trainer = Trainer(
        config=config,
        run_name=f"tjepa_ft_d{args.d_model}_l{args.num_layers}",
        extra_wandb_config={
            "pretrained": True,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "pooling": args.pooling,
            "unfreeze": args.unfreeze_encoder,
        },
    )

    result = trainer.fit(
        model=wrapped,
        train_loader=train_loader,
        val_loader=test_loader,
    )

    # Final evaluation
    wrapped.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = wrapped(X_b)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_targets.extend(y_b.cpu().numpy().tolist())

    mcc = matthews_corrcoef(all_targets, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))

    print(f"\nFine-tuning results:")
    print(f"  Test MCC: {mcc:.4f}")
    print(f"  Test Acc: {acc:.4f}")

    return {"mcc": mcc, "acc": acc, "best_val_mcc": result.best_val_mcc}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if provided
    model: TJEPAModel | None = None
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
        model = TJEPAModel(
            n_features=10,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            n_reg_tokens=args.n_reg_tokens,
            pred_dim=args.pred_dim,
            pred_num_layers=args.pred_layers,
        )
        jepa_trainer = JEPATrainer(DLConfig(), model)
        jepa_trainer.load_checkpoint(args.checkpoint)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Phase 1: Pretrain (if not loading checkpoint or explicitly requested)
    if model is None and args.pretrain_epochs > 0:
        model = run_pretrain(args)

    # Phase 2: Fine-tune
    if args.finetune and model is not None:
        run_finetune(args, model)


if __name__ == "__main__":
    main()
