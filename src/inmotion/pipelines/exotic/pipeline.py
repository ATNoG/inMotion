"""World-model pipeline orchestration."""

from __future__ import annotations

import csv
import time
from typing import TYPE_CHECKING

import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn

from inmotion.dl.config import DLConfig
from inmotion.dl.evaluation import evaluate_model_on_test
from inmotion.dl.training import Trainer

if TYPE_CHECKING:
    pass

from inmotion.pipelines.exotic.builders import MODEL_BUILDERS
from inmotion.pipelines.exotic.config import RunConfig
from inmotion.pipelines.exotic.data import load_pretrain_data, load_supervised_data
from inmotion.pipelines.exotic.devices import resolve_device, set_seed
from inmotion.pipelines.exotic.hpo import _run_hpo
from inmotion.pipelines.exotic.training import finetune_jepa, pretrain_jepa


def run(args: RunConfig) -> int:
    """Execute the exotic (world-model) pipeline.

    ``args`` carries the same field names the argparse namespace used to, so
    the body below is unchanged; the CLI builds it from typed options.
    """
    set_seed(args.seed)

    # Anti-collapse regularizer mode (SigReg vs RDMReg)
    if args.sigreg_mode == "rdm":
        from inmotion.dl.sigreg import set_default_sigreg_mode

        set_default_sigreg_mode("rdm", p=args.rdm_p)
        print(f"Using RDMReg (p={args.rdm_p}) for anti-collapse regularization")

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
    config.rich_features = args.rich_features
    if args.rich_features:
        config.in_features = 18
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

    is_jepa = args.model in ("t_jepa", "ts_jepa", "lejepa", "cf_jepa")

    if is_jepa and not args.finetune_only and args.checkpoint is None:
        print(f"\n{'=' * 60}")
        print(f"Phase 1: SSL Pretraining ({args.model})")
        print(f"{'=' * 60}")

        # Setup visualization
        from inmotion.dl.jepa_viz import JEPAVisualizer

        viz = JEPAVisualizer(
            save_dir=args.viz_dir,
            model_name=args.model,
            seed=args.seed,
        )

        pt_config = DLConfig()
        pt_config.data_path = args.data
        pt_config.seed = args.seed
        pt_config.batch_size = args.batch_size
        pt_config.device = str(device)
        pt_config.rich_features = args.rich_features
        if args.rich_features:
            pt_config.in_features = 18

        (pt_train_loader, pt_val_loader, probe_train_loader, probe_val_loader) = load_pretrain_data(
            pt_config
        )
        pretrained_model = MODEL_BUILDERS[args.model](args)

        ckpt_path = pretrain_jepa(
            args,
            pretrained_model,
            pt_train_loader,
            pt_val_loader,
            probe_train_loader,
            probe_val_loader,
            device,
            args.models_dir,
            viz,
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
        if hasattr(pretrained_model, "context_encoder"):
            pretrained_model.context_encoder.load_state_dict(ckpt["context_encoder"])
        else:
            pretrained_model.encoder.load_state_dict(ckpt["encoder"])
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
            args,
            pretrained_model,
            config,
            train_loader,
            val_loader,
            test_loader,
            classes,
            device,
            args.models_dir,
            viz,
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
                print(
                    f"  Early stopping at epoch {epoch + 1} "
                    f"(patience={args.patience}, best={best_val_mcc:.4f})"
                )
                break
        # Load best and evaluate — wrap to return logits only
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))

        class _EvalWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)[0]

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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "data",
                "seed",
                "pretrained",
                "test_mcc",
                "test_acc",
                "best_val_mcc",
                "timestamp",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": args.model,
                "data": str(args.data),
                "seed": str(args.seed),
                "pretrained": str(pretrained_model is not None),
                "test_mcc": metrics.get("mcc", ""),
                "test_acc": metrics.get("accuracy", ""),
                "best_val_mcc": metrics.get("best_val_mcc", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
    print(f"\nResults appended to {results_file}")
    return 0
