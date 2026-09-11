"""Shared Optuna helpers and per-trial logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
import torch
from torch import Tensor, nn

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import DLDataLoader
from inmotion.pipelines.exotic.config import RunConfig

if TYPE_CHECKING:
    pass


class SIGRegWrapper(nn.Module):
    """Wraps SIGRegClassifier so Trainer can use it with compute_loss."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

    def compute_loss(self, logits, latents, targets):
        return self.model.compute_loss(logits, latents, targets)


def _count_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _params_str(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _efficiency(mcc: float, n_params: int) -> float:
    """Efficiency score: MCC per million parameters × 1000 (higher = more efficient)."""
    return (mcc * 1000.0) / max(n_params / 1e6, 0.001)


def _hpo_objective_value(mcc: float, n_params: int, objective: str) -> float:
    """Map a trial's (MCC, params) to the scalar the study maximizes.

    objective == "mcc"        → maximize MCC only (legacy behaviour)
    objective == "efficiency" → maximize MCC per million params (single scalar)
    objective == "pareto"     → handled separately via NSGA-II (2 objectives)
    """
    if objective == "efficiency":
        return _efficiency(mcc, n_params)
    return mcc


def _make_hpo_study(study_name: str, objective: str) -> optuna.Study:
    """Create a single-objective study, or a NSGA-II Pareto study."""
    from inmotion.dl.optimization import _get_optuna_storage

    storage = _get_optuna_storage("sqlite:///optuna_exotic.db")
    if objective == "pareto":
        return optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )
    return optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )


def _run_hpo(args: RunConfig, device: torch.device) -> None:
    """Dispatch to the search for the selected model family.

    The three searches import this module, so they are imported inside the
    function to keep the dependency acyclic.
    """
    """Run Optuna HPO, update args with best hyperparams."""

    cfg = DLConfig()
    cfg.data_path = args.data
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.device = str(device)

    dl = DLDataLoader(cfg)
    X, y = dl.load_and_preprocess()
    hpo_seed = args.seed + 0x5EED
    from sklearn.model_selection import train_test_split

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=hpo_seed, stratify=y)

    print(f"\n{'=' * 60}")
    print(f"HPO: {args.model} — {args.hpo_trials} trials")
    print(f"  Train: {len(Xt)} samples, Val: {len(Xv)} samples")
    print(f"{'=' * 60}")

    if args.model in ("t_jepa", "ts_jepa", "lejepa", "cf_jepa"):
        from inmotion.pipelines.exotic.hpo_jepa import _hpo_jepa

        _hpo_jepa(args, device, Xt, yt, Xv, yv)
    elif args.model.startswith("mamba3"):
        from inmotion.pipelines.exotic.hpo_mamba3 import _hpo_mamba3

        _hpo_mamba3(args, device, Xt, yt, Xv, yv)
    elif args.model == "sigreg":
        from inmotion.pipelines.exotic.hpo_sigreg import _hpo_sigreg

        _hpo_sigreg(args, device, Xt, yt, Xv, yv)


def _trial_wandb_start(trial, args: RunConfig, extra_config: dict | None = None) -> object | None:
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


def _apply_trial_params(args, params: dict) -> None:
    """Copy a trial's hyperparams onto args (keys matching CLI arg names)."""
    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)
