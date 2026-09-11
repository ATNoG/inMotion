"""Hyperparameter search for the Mamba-3 hybrids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import RSSIDataset
from inmotion.dl.training import Trainer
from inmotion.pipelines.exotic.config import RunConfig

if TYPE_CHECKING:
    pass

from inmotion.pipelines.exotic.builders import _build_mamba3
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


def _hpo_mamba3(args, device, Xt, yt, Xv, yv) -> None:
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128, 192])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        d_state = trial.suggest_categorical("d_state", [8, 16, 32])
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        mimo_rank = trial.suggest_categorical("mimo_rank", [1, 2, 4])

        m = _build_mamba3(
            args.model.replace("mamba3_", ""),
            RunConfig(
                d_model=d_model,
                num_layers=num_layers,
                nhead=4,
                d_state=d_state,
                dropout=dropout,
                mimo_rank=mimo_rank,
            ),
        ).to(device)
        n_params = _count_params(m)
        run = _trial_wandb_start(
            trial, args, {"n_params": n_params, "params_str": _params_str(n_params)}
        )

        cfg = DLConfig()
        cfg.seed = args.seed
        cfg.batch_size = args.batch_size
        cfg.device = str(device)
        cfg.num_epochs = 30
        cfg.learning_rate = lr
        cfg.use_wandb = False
        trl = DataLoader(RSSIDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)
        val = DataLoader(RSSIDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)

        result = Trainer(cfg).fit(m, trl, val)
        mcc = float(result.best_val_mcc)

        if run is not None:
            import wandb

            for ep, (tl, vl, vm) in enumerate(
                zip(result.train_losses, result.val_losses, result.val_mccs), start=1
            ):
                wandb.log({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_mcc": vm})

        eff = _efficiency(mcc, n_params)
        _trial_wandb_finish(
            run,
            {
                "trial_mcc": mcc,
                "trial_n_params": n_params,
                "trial_params_M": n_params / 1e6,
                "trial_efficiency": eff,
            },
        )
        obj = getattr(args, "hpo_objective", "mcc")
        if obj == "pareto":
            return mcc, float(n_params)
        return _hpo_objective_value(mcc, n_params, obj)

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
        best_pt = max(study.best_trials, key=lambda t: t.values[0])
        print(f"\n  Applying best frontier point for full training: {best_pt.params}")
        _apply_trial_params(args, best_pt.params)
    else:
        bp = study.best_params
        print(f"\n  Best trial MCC: {study.best_value:.4f}")
        print(f"  Params: {bp}")
        _apply_trial_params(args, bp)
