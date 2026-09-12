"""Hyperparameter search for SIGReg."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader

from inmotion.dl.data_loader import RSSIDataset

if TYPE_CHECKING:
    pass

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


def _hpo_sigreg(args, device, Xt, yt, Xv, yv) -> None:
    from inmotion.dl.models.sigreg_classifier import SIGRegClassifier

    def objective(trial):
        # Named after what they are for sigreg, so the winner applied below
        # lands on --num-filters/--num-blocks rather than on the transformer
        # flags whose names mean something else.
        num_filters = trial.suggest_categorical("num_filters", [128, 192, 256])
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
        sigreg_lambda = trial.suggest_float("sigreg_lambda", 0.001, 0.5, log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        m = SIGRegClassifier(
            in_features=4,
            num_filters=num_filters,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            num_classes=4,
            sigreg_lambda=sigreg_lambda,
        ).to(device)
        n_params = _count_params(m)
        run = _trial_wandb_start(
            trial, args, {"n_params": n_params, "params_str": _params_str(n_params)}
        )

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
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            vap, vat = [], []
            with torch.no_grad():
                for xb, yb in val:
                    logits, _ = m(xb.to(device))
                    vap.extend(logits.argmax(1).cpu().tolist())
                    vat.extend(yb.tolist())
            vmcc = float(matthews_corrcoef(vat, vap))
            if vmcc > best_mcc:
                best_mcc = vmcc
            if run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": ep,
                        "val_mcc": vmcc,
                        "ce_loss": loss_dict.get("ce_loss", 0),
                        "sigreg_loss": loss_dict.get("sigreg_loss", 0),
                    }
                )

        eff = _efficiency(best_mcc, n_params)
        _trial_wandb_finish(
            run,
            {
                "trial_mcc": best_mcc,
                "trial_n_params": n_params,
                "trial_params_M": n_params / 1e6,
                "trial_efficiency": eff,
            },
        )
        obj = getattr(args, "hpo_objective", "mcc")
        if obj == "pareto":
            return best_mcc, float(n_params)
        return _hpo_objective_value(best_mcc, n_params, obj)

    obj = getattr(args, "hpo_objective", "mcc")
    study = _make_hpo_study(f"exotic_{args.model}_{obj}", obj)
    study.optimize(objective, n_trials=args.hpo_trials, show_progress_bar=True)
    if obj == "pareto":
        print(f"\n  Pareto frontier ({len(study.best_trials)} points):")
        for t in sorted(study.best_trials, key=lambda t: t.values[0], reverse=True):
            print(f"    MCC={t.values[0]:.4f}  params={t.values[1] / 1e6:.2f}M  {t.params}")
        # Apply the best-MCC frontier point so the fall-through training
        # run after HPO uses the winner's hyperparameters.
        best_pt = max(study.best_trials, key=lambda t: t.values[0])
        print(f"\n  Applying best frontier point for full training: {best_pt.params}")
        _apply_trial_params(args, best_pt.params)
    else:
        bp = study.best_params
        print(f"\n  Best trial MCC: {study.best_value:.4f}")
        print(f"  Params: {bp}")
        _apply_trial_params(args, bp)
