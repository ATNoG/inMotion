"""Stage-1 pretraining, probing and stage-2 fine-tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sklearn.metrics import matthews_corrcoef
from torch import Tensor, nn
from torch.utils.data import DataLoader

from inmotion.dl.config import DLConfig
from inmotion.dl.evaluation import evaluate_model_on_test
from inmotion.dl.training import Trainer
from inmotion.pipelines.exotic.config import RunConfig

if TYPE_CHECKING:
    from inmotion.dl.jepa_viz import JEPAVisualizer


def _masked_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute pretrain val loss ONLY at supervised (target) positions.

    Handles model interfaces:
      - T-JEPA:  forward → (tgt (B,total,D), preds LIST, ctx_mask, tgt_masks list, ctx_latents)
      - TS-JEPA: forward → (tgt_masked (B,M,D), preds TENSOR (B,M,D), mask_idx, non_mask_idx)
      - CF-JEPA: forward → (targets, predictions, ctx_latents, horizon_mask)
      - LeJEPA:  forward → (z, z_hat)
    """
    is_cf = hasattr(model, "horizon_start") or type(model).__name__.startswith("CFJEPA")
    model.eval()
    val_sum, val_n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            out = model.forward(x)
            tgt, preds = out[0], out[1]
            if is_cf:
                # CF-JEPA: (targets, predictions, ctx_latents, horizon_mask)
                horizon_mask = out[3]
                p = torch.nn.functional.normalize(preds, dim=-1)
                tgt_n = torch.nn.functional.normalize(tgt, dim=-1)
                diff = (p - tgt_n).abs().mean(dim=-1)  # (B, num_future)
                val_sum += (diff * horizon_mask).sum().item()
                val_n += max(horizon_mask.sum().item(), 1)
            elif getattr(model, "sigreg", None) is not None and not hasattr(
                model, "context_encoder"
            ):
                # LeJEPA: (z, z_hat) — next-step prediction MSE
                z, z_hat = tgt, preds
                val_sum += torch.nn.functional.mse_loss(z_hat[:, :-1], z[:, 1:]).item() * x.size(0)
                val_n += x.size(0)
            elif isinstance(preds, (list, tuple)):
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

    d = getattr(model, "d_model", None) or getattr(model, "embed_dim")
    encoder = getattr(model, "encoder", None) or getattr(model, "context_encoder")
    n_reg = getattr(encoder, "n_reg_tokens", 0)

    probe = nn.Linear(d, 4).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def pool(x: Tensor) -> Tensor:
        z = encoder(x)  # (B, N, D)
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
    args: RunConfig,
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

    encoder = getattr(model, "context_encoder", None) or getattr(model, "encoder")
    params = list(encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.pretrain_epochs, eta_min=args.pretrain_lr * 0.01
    )

    # WandB
    pretrain_wandb = None
    if not args.no_wandb:
        try:
            import wandb

            pretrain_wandb = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_pretrain_seed{args.seed}",
                config={
                    "model": args.model,
                    "phase": "pretrain",
                    "pretrain_epochs": args.pretrain_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.pretrain_lr,
                    "model_size_M": sum(p.numel() for p in model.parameters()) / 1e6,
                },
                reinit=True,
            )
        except Exception:
            pass

    best_path = save_dir / f"{args.model}_pretrain_best.pt"
    probe_cadence = getattr(args, "probe_cadence", 10)
    best_probe = -1.0
    best_val_loss = float("inf")
    val_patience_counter = 0
    probe_patience_counter = 0

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
            if hasattr(model, "_update_target_encoder"):
                model._update_target_encoder()
            train_loss += loss.item()
        sch.step()
        train_loss /= max(len(train_loader), 1)

        # ── Validation (masked, target positions only) ──
        val_loss = (
            _masked_val_loss(model, val_loader, device) if val_loader is not None else float("inf")
        )

        # ── Probe every N epochs (logging only, not used for stopping) ──
        probe_mcc = None
        if (epoch + 1) % probe_cadence == 0:
            probe_mcc = _probe_mcc(model, probe_train_loader, probe_val_loader, device)
            if probe_mcc > best_probe:
                best_probe = probe_mcc
                ckpt = {
                    "epoch": epoch,
                    "probe_mcc": probe_mcc,
                    "predictor": model.predictor.state_dict(),
                }
                if hasattr(model, "context_encoder"):
                    ckpt["context_encoder"] = model.context_encoder.state_dict()
                    ckpt["target_encoder"] = model.target_encoder.state_dict()
                else:
                    ckpt["encoder"] = model.encoder.state_dict()
                torch.save(ckpt, best_path)

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
                    epoch,
                    latents,
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
            print(
                f"  epoch {epoch + 1:3d}/{args.pretrain_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | ema={ema:.4f}{probe_str}"
            )

        # ── Early stopping (on PROBE MCC — the representation-health metric) ──
        # Val loss plateaus quickly for JEPA models while the probe MCC keeps
        # climbing; early-stopping on val loss kills pretraining prematurely.
        probe_patience = getattr(args, "probe_patience", 5)
        if probe_mcc is not None:
            if probe_mcc > best_probe + 1e-4:
                best_probe = probe_mcc
                probe_patience_counter = 0
            else:
                probe_patience_counter += 1
                if probe_patience_counter >= probe_patience and epoch > 50:
                    print(
                        f"  Early stopping at epoch {epoch + 1} "
                        f"(probe MCC not improving, best={best_probe:.4f})"
                    )
                    break
        # Keep the val-loss tracking for logging/checkpoint (best ckpt is by probe)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_patience_counter = 0
        else:
            val_patience_counter += 1

    elapsed = time.time() - t0
    print(f"Pretraining done in {elapsed:.0f}s. Best probe MCC: {best_probe:.4f}")
    print(f"Checkpoint: {best_path}")

    if pretrain_wandb is not None:
        import wandb

        wandb.log({"pretrain/best_probe_mcc": best_probe})
        wandb.finish()

    return best_path


def finetune_jepa(
    args: RunConfig,
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
    if args.model == "t_jepa":
        from inmotion.dl.models.t_jepa import TJEPAClassifier

        clf = TJEPAClassifier(
            pretrained=pretrained_model,
            num_classes=4,
            hidden_dim=128,
            dropout=0.3,
        )
    elif args.model == "lejepa":
        from inmotion.dl.models.lejepa import LeJEPAClassifier

        clf = LeJEPAClassifier(
            pretrained=pretrained_model,
            num_classes=4,
            hidden_dim=128,
            dropout=0.3,
        )
    elif args.model == "cf_jepa":
        from inmotion.dl.models.cf_jepa import CFJEPAClassifier

        clf = CFJEPAClassifier(
            pretrained=pretrained_model,
            num_classes=4,
            hidden_dim=128,
            dropout=0.3,
        )
    else:
        from inmotion.dl.models.ts_jepa import TSJEPAClassifier

        clf = TSJEPAClassifier(
            pretrained=pretrained_model,
            num_classes=4,
            hidden_dim=128,
            dropout=0.3,
        )
    clf.to(device)

    # Wrap so Trainer can call model(x) → logits
    class W(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x):
            return self.c(x)

    wrapped = W(clf)

    config.use_wandb = not args.no_wandb
    config.mixup_alpha = args.mixup_alpha
    config.use_mixup = args.mixup_alpha > 0
    config.label_smoothing = args.label_smoothing
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
    ft_params = sum(p.numel() for p in clf.parameters()) / 1e6
    ft_size_cfg = {
        "stage": 1,
        "model": args.model,
        "pretrained": True,
        "model_size_M": ft_params,
        "n_params": int(ft_params * 1e6),
    }
    r1 = Trainer(config, run_name=f"{run_name}_s1", extra_wandb_config=ft_size_cfg).fit(
        wrapped, train_loader, val_loader
    )
    print(f"  Stage 1 best val MCC: {r1.best_val_mcc:.4f} ({time.time() - t0:.0f}s)")

    # Stage 2: Unfreeze encoder, lower LR, use the remaining budget
    for p in clf.encoder.parameters():
        p.requires_grad = True
    config.num_epochs = args.finetune_epochs - stage1_epochs
    config.learning_rate = args.finetune_lr * 0.1
    print("Stage 2: Full fine-tuning (encoder unfrozen)...")
    ft_size_cfg["stage"] = 2
    t0 = time.time()
    r2 = Trainer(config, run_name=f"{run_name}_s2", extra_wandb_config=ft_size_cfg).fit(
        wrapped, train_loader, val_loader, save_path=save_path
    )

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
                    9999,
                    pre_latents,
                    all_labels,
                    title=f"{args.model.upper()} Pretrain Embeddings",
                )
                viz.save_checkpoint_snapshot(
                    10000,
                    post_latents,
                    all_labels,
                    title=f"{args.model.upper()} Fine-tuned Embeddings",
                )

    return metrics
