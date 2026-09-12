"""Model construction for the world-model family."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from inmotion.pipelines.exotic.config import RunConfig

if TYPE_CHECKING:
    pass


def _build_t_jepa(args: RunConfig) -> nn.Module:
    from inmotion.dl.models.t_jepa import TJEPAModel

    n_channels = 18 if args.rich_features else 4
    return TJEPAModel(
        n_timesteps=10,
        n_channels=n_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        n_reg_tokens=args.n_reg_tokens,
        pred_dim=args.pred_dim,
        pred_num_layers=args.pred_layers,
        mask_ratio=0.3,
        n_target_masks=getattr(args, "n_preds", 1),
        sigreg_lambda=getattr(args, "jepa_sigreg", 0.0),
        ema_start=args.ema_start,
        ema_end=args.ema_end,
    )


def _build_ts_jepa(args: RunConfig) -> nn.Module:
    from inmotion.dl.models.ts_jepa import TSJEPAModel

    in_channels = 18 if args.rich_features else 4
    return TSJEPAModel(
        seq_len=10,
        patch_size=2,
        in_channels=in_channels,
        embed_dim=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        pred_dim=args.pred_dim,
        pred_num_layers=args.pred_layers,
        mask_ratio_start=0.4,
        mask_ratio_end=0.7,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
    )


def _build_lejepa(args: RunConfig) -> nn.Module:
    from inmotion.dl.models.lejepa import LeJEPAModel

    in_channels = 18 if args.rich_features else 4
    return LeJEPAModel(
        seq_len=10,
        in_channels=in_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        pred_num_layers=args.pred_layers,
        sigreg_lambda=args.jepa_sigreg,
    )


def _build_cf_jepa(args: RunConfig) -> nn.Module:
    from inmotion.dl.models.cf_jepa import CFJEPAModel

    in_channels = 18 if args.rich_features else 4
    return CFJEPAModel(
        seq_len=10,
        patch_size=2,
        in_channels=in_channels,
        embed_dim=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        pred_dim=args.pred_dim,
        pred_num_layers=args.pred_layers,
        horizon_start=getattr(args, "cf_horizon_start", 1),
        horizon_end=getattr(args, "cf_horizon_end", 3),
        ctx_min=getattr(args, "cf_ctx_min", 1),
        ctx_max=getattr(args, "cf_ctx_max", 3),
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        sigreg_lambda=args.jepa_sigreg,
    )


def _build_sigreg(args: RunConfig) -> nn.Module:
    """Build SIGReg from its own flags.

    The defaults (192 filters, 3 blocks, latent 128) are the architecture of the
    registered ``sigreg`` checkpoint. ``args.d_model``/``args.num_layers`` are
    deliberately *not* consulted: they are width and depth for the transformer
    family, and reusing them here built a 256-filter model from the documented
    retrain command while the registry expected 192.
    """
    from inmotion.dl.models.sigreg_classifier import SIGRegClassifier

    return SIGRegClassifier(
        in_features=4,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        latent_dim=args.latent_dim,
        num_classes=4,
        sigreg_lambda=args.sigreg_lambda,
    )


def _build_mamba3(variant: str, args: RunConfig) -> nn.Module:
    """Build a Mamba-3 hybrid from ``args``.

    ``d_state``, ``dropout`` and ``mimo_rank`` are read from ``args`` and fall
    back to the previous fixed defaults. They used to be hardcoded here while
    ``_hpo_mamba3`` suggested them, so the Optuna study reported three search
    dimensions that could not affect the model at all.
    """
    kw = {
        "in_features": 4,
        "d_model": args.d_model,
        "d_state": getattr(args, "d_state", 16),
        "num_classes": 4,
        "dropout": getattr(args, "dropout", 0.2),
        "mimo_rank": getattr(args, "mimo_rank", 4),
    }
    if variant == "cnn":
        from inmotion.dl.models.mamba3_cnn import Mamba3CNN

        return Mamba3CNN(cnn_channels=args.d_model, n_mamba_layers=args.num_layers, **kw)
    elif variant == "tcn":
        from inmotion.dl.models.mamba3_tcn import Mamba3TCN

        return Mamba3TCN(tcn_channels=args.d_model, n_mamba_layers=args.num_layers, **kw)
    elif variant == "transformer":
        from inmotion.dl.models.mamba3_transformer import Mamba3Transformer

        return Mamba3Transformer(nhead=args.nhead, num_blocks=args.num_layers, **kw)
    elif variant == "multiview":
        from inmotion.dl.models.mamba3_multiview import Mamba3MultiView

        return Mamba3MultiView(n_mamba_layers=args.num_layers, **kw)
    raise ValueError(f"Unknown Mamba-3 variant: {variant}")


MODEL_BUILDERS: dict[str, callable] = {
    "t_jepa": _build_t_jepa,
    "ts_jepa": _build_ts_jepa,
    "lejepa": _build_lejepa,
    "cf_jepa": _build_cf_jepa,
    "sigreg": _build_sigreg,
    "mamba3_cnn": lambda a: _build_mamba3("cnn", a),
    "mamba3_tcn": lambda a: _build_mamba3("tcn", a),
    "mamba3_transformer": lambda a: _build_mamba3("transformer", a),
    "mamba3_multiview": lambda a: _build_mamba3("multiview", a),
}
