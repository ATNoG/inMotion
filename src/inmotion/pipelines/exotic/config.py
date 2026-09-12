"""Options for the world-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inmotion.data.paths import processed


@dataclass
class RunConfig:
    """Typed options for the exotic pipeline.

    Field names match the argparse namespace this replaced, so the pipeline body
    is unchanged. Built by the CLI (`inmotion train exotic`).
    """

    model: str = "t_jepa"
    data: Path = field(default_factory=lambda: processed("dataset.csv"))
    rich_features: bool = False
    seed: int = 42
    batch_size: int = 128
    gpu: int = 0
    pretrain_epochs: int = 300
    pretrain_lr: float = 0.0003
    pretrain_wd: float = 0.0001
    probe_cadence: int = 10
    probe_patience: int = 15
    finetune_epochs: int = 50
    epochs: int = 150
    patience: int = 25
    finetune_lr: float = 0.001
    mixup_alpha: float = 0.4
    label_smoothing: float = 0.1
    lr: float = 0.001
    finetune_only: bool = False
    checkpoint: Path = None
    unfreeze_encoder: bool = False
    d_model: int = 256
    d_state: int = 16
    dropout: float = 0.2
    mimo_rank: int = 4
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 512
    n_reg_tokens: int = 2
    pred_dim: int = 128
    pred_layers: int = 2
    ema_start: float = 0.99
    ema_end: float = 0.996
    n_preds: int = 1
    jepa_sigreg: float = 0.05
    sigreg_mode: str = "sigreg"
    rdm_p: float = 1.5
    pooling: str = "mean"
    cf_horizon_start: int = 1
    cf_horizon_end: int = 3
    cf_ctx_min: int = 1
    cf_ctx_max: int = 3
    #: SIGReg's architecture, stated rather than inferred. ``d_model`` and
    #: ``num_layers`` mean width and depth for the transformer family, so
    #: reading sigreg's filter count out of them let a documented transformer
    #: flag silently build the wrong model.
    num_filters: int = 192
    num_blocks: int = 3
    latent_dim: int = 128
    sigreg_lambda: float = 0.01
    hpo: bool = False
    hpo_trials: int = 30
    hpo_objective: str = "mcc"
    hpo_pareto_train: int = 0
    hpo_pareto_select: str = "spread"
    models_dir: Path = field(default_factory=lambda: processed("models/exotic"))
    results_dir: Path = field(default_factory=lambda: processed("results/exotic"))
    viz_dir: Path = field(default_factory=lambda: processed("models/exotic/viz"))
    no_wandb: bool = False
    wandb_project: str = "inMotion-exotic-hpo"
