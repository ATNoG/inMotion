"""Hyperparameter search for the JEPA models."""

from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import RSSIDataset
from inmotion.pipelines.exotic.builders import MODEL_BUILDERS
from inmotion.pipelines.exotic.data import (
    load_pretrain_data,
    load_supervised_data,
)
from inmotion.pipelines.exotic.hpo import (
    _apply_trial_params,
    _count_params,
    _efficiency,
    _hpo_objective_value,
    _make_hpo_study,
    _params_str,
    _trial_wandb_finish,
    _trial_wandb_start,
)
from inmotion.pipelines.exotic.training import finetune_jepa, pretrain_jepa


def _hpo_jepa(args, device, Xt, yt, Xv, yv) -> None:
    from torch.utils.data import TensorDataset

    is_ts = args.model == "ts_jepa"
    is_le = args.model == "lejepa"
    is_cf = args.model == "cf_jepa"
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
            from inmotion.dl.models.ts_jepa import TSJEPAClassifier, TSJEPAModel

            m = TSJEPAModel(
                seq_len=10,
                patch_size=2,
                in_channels=4,
                embed_dim=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                pred_dim=pred_dim,
                pred_num_layers=pred_layers,
                ema_start=ema_start,
            ).to(device)
        elif is_le:
            from inmotion.dl.models.lejepa import LeJEPAClassifier, LeJEPAModel

            m = LeJEPAModel(
                seq_len=10,
                in_channels=4,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                pred_num_layers=pred_layers,
                sigreg_lambda=getattr(args, "jepa_sigreg", 0.0),
            ).to(device)
        elif is_cf:
            from inmotion.dl.models.cf_jepa import CFJEPAClassifier, CFJEPAModel

            m = CFJEPAModel(
                seq_len=10,
                patch_size=2,
                in_channels=4,
                embed_dim=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                pred_dim=pred_dim,
                pred_num_layers=pred_layers,
                ema_start=ema_start,
                sigreg_lambda=getattr(args, "jepa_sigreg", 0.0),
            ).to(device)
        else:
            from inmotion.dl.models.t_jepa import TJEPAClassifier, TJEPAModel

            m = TJEPAModel(
                n_timesteps=10,
                n_channels=4,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                pred_dim=pred_dim,
                pred_num_layers=pred_layers,
                ema_start=ema_start,
            ).to(device)

        n_params = _count_params(m) + _count_params(
            m.context_encoder if hasattr(m, "context_encoder") else m.encoder
        )

        # ── WandB: start trial run ────────────────────────────────────
        run = _trial_wandb_start(
            trial, args, {"n_params": n_params, "params_str": _params_str(n_params)}
        )

        # ── Pretrain (50 epochs) ──────────────────────────────────────
        Xp = Xt.astype(np.float32)
        pm, ps = Xp.mean(axis=(0, 1), keepdims=True), Xp.std(axis=(0, 1), keepdims=True) + 1e-8
        Xp = (Xp - pm) / ps
        ptds = TensorDataset(torch.tensor(Xp, dtype=torch.float32))
        ptl = DataLoader(ptds, batch_size=args.batch_size, shuffle=True)

        params_list = list(
            (m.context_encoder if hasattr(m, "context_encoder") else m.encoder).parameters()
        ) + list(m.predictor.parameters())
        opt = torch.optim.AdamW(params_list, lr=pt_lr)
        for ep in range(pt_epochs):
            m.train()
            pt_loss_sum = 0.0
            pt_n = 0
            for (xb,) in ptl:
                loss, _ = m.pretrain_step(xb.to(device), ep, pt_epochs)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_list, 1.0)
                opt.step()
                if hasattr(m, "_update_target_encoder"):
                    m._update_target_encoder()
                pt_loss_sum += loss.item()
                pt_n += 1
            avg_pt_loss = pt_loss_sum / max(pt_n, 1)
            if run is not None:
                import wandb

                wandb.log(
                    {
                        "pretrain/epoch": ep,
                        "pretrain/loss": avg_pt_loss,
                        "pretrain/ema": getattr(m, "ema_momentum", 0.0),
                    }
                )

        # ── Fine-tune (stage 1: head only, stage 2: full) ────────────
        if is_ts:
            clf = TSJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)
        elif is_le:
            from inmotion.dl.models.lejepa import LeJEPAClassifier

            clf = LeJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)
        elif is_cf:
            from inmotion.dl.models.cf_jepa import CFJEPAClassifier

            clf = CFJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)
        else:
            clf = TJEPAClassifier(pretrained=m, num_classes=4, hidden_dim=128).to(device)

        class W(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c

            def forward(self, x):
                return self.c(x)

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
                opt1.zero_grad()
                loss.backward()
                opt1.step()
            # Val MCC
            clf.eval()
            vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if run is not None:
                import wandb

                wandb.log({"ft/stage1_epoch": ep, "ft/stage1_val_mcc": vmcc})

        # Stage 2: unfreeze, lower LR
        for p in clf.encoder.parameters():
            p.requires_grad = True
        opt2 = torch.optim.AdamW(clf.parameters(), lr=ft_lr * 0.1)
        best_mcc = -1.0
        for ep in range(ft_epochs - ft_epochs // 2):
            clf.train()
            for xb, yb in trl:
                loss = ce(wr(xb.to(device)), yb.to(device))
                opt2.zero_grad()
                loss.backward()
                opt2.step()
            clf.eval()
            vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    vap.extend(wr(xb.to(device)).argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if vmcc > best_mcc:
                best_mcc = vmcc
            if run is not None:
                import wandb

                wandb.log({"ft/stage2_epoch": ep, "ft/stage2_val_mcc": vmcc})

        # ── Final metrics ─────────────────────────────────────────────
        eff = _efficiency(best_mcc, n_params)
        final_metrics = {
            "trial_mcc": best_mcc,
            "trial_n_params": n_params,
            "trial_params_M": n_params / 1e6,
            "trial_efficiency": eff,
        }
        _trial_wandb_finish(run, final_metrics)
        obj = getattr(args, "hpo_objective", "mcc")
        if obj == "pareto":
            return best_mcc, float(n_params)
        return _hpo_objective_value(best_mcc, n_params, obj)

    from inmotion.dl.optimization import _log_trial_callback

    obj = getattr(args, "hpo_objective", "mcc")
    study = _make_hpo_study(f"exotic_{args.model}_{obj}", obj)
    study.optimize(
        objective, n_trials=args.hpo_trials, show_progress_bar=True, callbacks=[_log_trial_callback]
    )

    if obj == "pareto":
        print(f"\n  Pareto frontier ({len(study.best_trials)} points):")
        for t in sorted(study.best_trials, key=lambda t: t.values[0], reverse=True):
            print(f"    MCC={t.values[0]:.4f}  params={t.values[1] / 1e6:.2f}M  {t.params}")

        n_train = getattr(args, "hpo_pareto_train", 0)
        if n_train > 0:
            pts = _select_pareto_points(
                study, n_train, getattr(args, "hpo_pareto_select", "spread")
            )
            print(
                f"\n  Training {len(pts)} selected frontier point(s) "
                f"({getattr(args, 'hpo_pareto_select', 'spread')} selection)..."
            )
            manifest_path = args.results_dir / "pareto_runs.csv"
            rows = []
            for rank, trial in enumerate(pts):
                rows.append(_train_pareto_point(args, device, trial, rank))
                # reset checkpoint arg so the next point pretrains fresh
                args.checkpoint = None
            with open(manifest_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nManifest → {manifest_path}")
        else:
            # No pareto-train: apply best-MCC frontier point to args so the
            # fall-through full pretrain+finetune uses the winner's params.
            best_pt = max(study.best_trials, key=lambda t: t.values[0])
            print(f"\n  Applying best frontier point for full training: {best_pt.params}")
            _apply_trial_params(args, best_pt.params)
    else:
        _apply_best_jepa_params(args, study)


def _apply_best_jepa_params(args, study) -> None:
    """Copy best JEPA trial params to args."""
    bp = study.best_params
    print(f"\n  Best trial MCC: {study.best_value:.4f}")
    print(f"  Params: {bp}")
    _apply_trial_params(args, bp)
    # Map study params to CLI arg names
    _map = {
        "d_model": "d_model",
        "nhead": "nhead",
        "num_layers": "num_layers",
        "dim_ff": "dim_ff",
        "pred_dim": "pred_dim",
        "pred_layers": "pred_layers",
        "pretrain_lr": "pretrain_lr",
        "finetune_lr": "finetune_lr",
        "ema_start": "ema_start",
    }
    for sk, ak in _map.items():
        if sk in bp:
            setattr(args, ak, bp[sk])


def _select_pareto_points(study, n: int, mode: str) -> list:
    """Pick N trials from a pareto study's frontier.

    spread: sort by params ascending, take evenly spaced points
            (smallest, knee-ish, largest) for max coverage of the tradeoff.
    top:    best-MCC first.
    """
    pts = [t for t in study.best_trials if t.values and len(t.values) == 2]
    if not pts:
        return []
    if mode == "top":
        pts = sorted(pts, key=lambda t: t.values[0], reverse=True)
    else:  # spread by size
        pts = sorted(pts, key=lambda t: t.values[1])
        if len(pts) > n:
            idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
            pts = [pts[i] for i in dict.fromkeys(idx.tolist())]
    return pts[:n]


def _train_pareto_point(args, device, trial, rank: int) -> dict:
    """Full pretrain + finetune at one frontier point's hyperparameters.

    Checkpoint name encodes HPO MCC + param count so deployment tradeoffs are
    visible without opening the manifest. Returns a manifest row.
    """
    bp = trial.params
    mcc_hpo, n_params_hpo = trial.values[0], int(trial.values[1])
    tag = f"mcc{mcc_hpo:.3f}_p{n_params_hpo / 1e6:.2f}M"

    # Apply this point's params to args (same mapping as _apply_best_jepa_params)
    for k in (
        "d_model",
        "nhead",
        "num_layers",
        "dim_ff",
        "pred_dim",
        "pred_layers",
        "pretrain_lr",
        "finetune_lr",
        "ema_start",
    ):
        if k in bp and hasattr(args, k):
            setattr(args, k, bp[k])

    print(
        f"\n{'=' * 60}\n  Pareto point {rank}: MCC(hpo)={mcc_hpo:.4f} "
        f"params={n_params_hpo / 1e6:.2f}M\n  {bp}\n{'=' * 60}"
    )

    # ── Full SSL pretraining ──────────────────────────────────────────────
    pt_config = DLConfig()
    pt_config.data_path = args.data
    pt_config.seed = args.seed
    pt_config.batch_size = args.batch_size
    pt_config.device = str(device)
    pt_config.rich_features = args.rich_features
    if args.rich_features:
        pt_config.in_features = 18

    (pt_train, pt_val, probe_tr, probe_va) = load_pretrain_data(pt_config)
    model = MODEL_BUILDERS[args.model](args)

    ckpt_path = pretrain_jepa(
        args, model, pt_train, pt_val, probe_tr, probe_va, device, args.models_dir
    )
    args.checkpoint = ckpt_path

    # ── Full fine-tuning (reuses main()'s path via a fresh load) ─────────
    config = DLConfig()
    config.data_path = args.data
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.device = str(device)
    config.use_wandb = False
    config.rich_features = args.rich_features
    if args.rich_features:
        config.in_features = 18
    if args.wandb_project:
        config.wandb_project = args.wandb_project

    train_loader, val_loader, test_loader, classes = load_supervised_data(config)
    metrics = finetune_jepa(
        args, model, config, train_loader, val_loader, test_loader, classes, device, args.models_dir
    )

    # Rename the ft checkpoint to encode the tradeoff point
    run_name = f"{args.model}_pareto-{tag}_seed{args.seed}"
    src = args.models_dir / f"{args.model}_ft_seed{args.seed}.pt"
    dst = args.models_dir / f"{run_name}.pt"
    if src.exists():
        src.rename(dst)

    row = {
        "model": args.model,
        "point": rank,
        "mcc_hpo": round(mcc_hpo, 4),
        "params_M": round(n_params_hpo / 1e6, 3),
        "test_mcc": metrics.get("mcc"),
        "test_acc": metrics.get("accuracy"),
        "checkpoint": str(dst),
        "params": json.dumps(bp),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(f"  → {dst}  test_mcc={row['test_mcc']}")
    return row
