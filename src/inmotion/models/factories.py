"""Architecture factories.

Each factory takes only keyword arguments and is registered in ``FACTORIES``
under a stable name. A checkpoint's architecture is therefore *data* (an
``arch`` dict in the registry or a checkpoint sidecar), not code that has to be
reverse-engineered from tensor shapes.

The configurations here were originally recovered by matching saved weight
shapes; they are pinned by ``tests/test_checkpoint_contract.py``, which asserts
every documented checkpoint still loads with ``strict=True`` and the parameter
count published in ``docs/isac/results/MASTER_RESULTS.md``.
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from .wrappers import TupleWrap

# ---------------------------------------------------------------------------
# JEPA family
# ---------------------------------------------------------------------------


def lejepa(
    seq_len: int = 10,
    in_channels: int = 4,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    pred_num_layers: int = 3,
    sigreg_lambda: float = 0.05,
    hidden_dim: int = 128,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """LeJEPA classifier (paper after-HPO winner: d_model=128, ff=256, layers=4)."""
    from inmotion.dl.models.lejepa import LeJEPAClassifier, LeJEPAModel

    pretrained = LeJEPAModel(
        seq_len=seq_len,
        in_channels=in_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        pred_num_layers=pred_num_layers,
        sigreg_lambda=sigreg_lambda,
    )
    return LeJEPAClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def t_jepa(
    n_timesteps: int = 10,
    n_channels: int = 18,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    n_reg_tokens: int = 2,
    pred_dim: int = 64,
    pred_num_layers: int = 3,
    hidden_dim: int = 128,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """T-JEPA classifier (paper after-HPO winner: d_model=128, ff=256)."""
    from inmotion.dl.models.t_jepa import TJEPAClassifier, TJEPAModel

    pretrained = TJEPAModel(
        n_timesteps=n_timesteps,
        n_channels=n_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        n_reg_tokens=n_reg_tokens,
        pred_dim=pred_dim,
        pred_num_layers=pred_num_layers,
    )
    return TJEPAClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def ts_jepa(
    seq_len: int = 10,
    patch_size: int = 2,
    in_channels: int = 18,
    embed_dim: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    pred_dim: int = 128,
    pred_num_layers: int = 2,
    hidden_dim: int = 128,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """TS-JEPA classifier (paper: embed_dim=256, layers=4, ff=512)."""
    from inmotion.dl.models.ts_jepa import TSJEPAClassifier, TSJEPAModel

    pretrained = TSJEPAModel(
        seq_len=seq_len,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        pred_dim=pred_dim,
        pred_num_layers=pred_num_layers,
    )
    return TSJEPAClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def cf_jepa(
    seq_len: int = 10,
    patch_size: int = 2,
    in_channels: int = 18,
    embed_dim: int = 256,
    nhead: int = 8,
    num_layers: int = 3,
    dim_feedforward: int = 256,
    pred_dim: int = 128,
    pred_num_layers: int = 1,
    horizon_start: int = 1,
    horizon_end: int = 3,
    ctx_min: int = 1,
    ctx_max: int = 3,
    ema_start: float = 0.996,
    ema_end: float = 0.999,
    sigreg_lambda: float = 0.05,
    hidden_dim: int = 128,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """CF-JEPA classifier (paper after-HPO: embed 256, layers 3, ff 256)."""
    from inmotion.dl.models.cf_jepa import CFJEPAClassifier, CFJEPAModel

    pretrained = CFJEPAModel(
        seq_len=seq_len,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        pred_dim=pred_dim,
        pred_num_layers=pred_num_layers,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        ctx_min=ctx_min,
        ctx_max=ctx_max,
        ema_start=ema_start,
        ema_end=ema_end,
        sigreg_lambda=sigreg_lambda,
    )
    return CFJEPAClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# SIGReg — two published widths exist, so the width is a parameter, not a
# second near-duplicate function.
# ---------------------------------------------------------------------------


def sigreg(
    in_features: int = 4,
    num_filters: int = 192,
    num_blocks: int = 3,
    latent_dim: int = 128,
    num_classes: int = 4,
    sigreg_lambda: float = 0.01,
) -> nn.Module:
    """SIGReg classifier wrapped so it returns logits only.

    ``num_filters=192`` reproduces ``backup/sigreg_seed*.pt`` (the checkpoints
    cited in the paper); ``num_filters=128`` reproduces the ``full-after-hpo``
    retrain.
    """
    from inmotion.dl.models.sigreg_classifier import SIGRegClassifier

    return TupleWrap(
        SIGRegClassifier(
            in_features=in_features,
            num_filters=num_filters,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            num_classes=num_classes,
            sigreg_lambda=sigreg_lambda,
        )
    )


# ---------------------------------------------------------------------------
# Mamba-3 hybrids
# ---------------------------------------------------------------------------


def mamba3_cnn(
    in_features: int = 4,
    cnn_channels: int = 192,
    d_model: int = 192,
    d_state: int = 16,
    n_mamba_layers: int = 3,
    num_classes: int = 4,
    dropout: float = 0.2,
    mimo_rank: int = 4,
) -> nn.Module:
    """Mamba-3 CNN hybrid."""
    from inmotion.dl.models.mamba3_cnn import Mamba3CNN

    return Mamba3CNN(
        in_features=in_features,
        cnn_channels=cnn_channels,
        d_model=d_model,
        d_state=d_state,
        n_mamba_layers=n_mamba_layers,
        num_classes=num_classes,
        dropout=dropout,
        mimo_rank=mimo_rank,
    )


def mamba3_tcn(
    in_features: int = 4,
    tcn_channels: int = 128,
    tcn_depth: int = 4,
    d_model: int = 128,
    d_state: int = 16,
    n_mamba_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.2,
    mimo_rank: int = 4,
) -> nn.Module:
    """Mamba-3 TCN hybrid.

    ``tcn_depth`` is the number of TCN residual blocks; ``n_mamba_layers`` the
    number of Mamba blocks. Both vary between saved variants, so neither is
    defaulted to a value that silently mis-loads.
    """
    from inmotion.dl.models.mamba3_tcn import Mamba3TCN

    return Mamba3TCN(
        in_features=in_features,
        tcn_channels=tcn_channels,
        tcn_depth=tcn_depth,
        d_model=d_model,
        d_state=d_state,
        n_mamba_layers=n_mamba_layers,
        num_classes=num_classes,
        dropout=dropout,
        mimo_rank=mimo_rank,
    )


def mamba3_transformer(
    in_features: int = 4,
    d_model: int = 128,
    d_state: int = 16,
    nhead: int = 4,
    num_blocks: int = 2,
    num_classes: int = 4,
    dropout: float = 0.2,
    mimo_rank: int = 4,
) -> nn.Module:
    """Mamba-3 transformer hybrid.

    ``num_blocks`` differs between saved variants (2 vs 3), so it is explicit.
    """
    from inmotion.dl.models.mamba3_transformer import Mamba3Transformer

    return Mamba3Transformer(
        in_features=in_features,
        d_model=d_model,
        d_state=d_state,
        nhead=nhead,
        num_blocks=num_blocks,
        num_classes=num_classes,
        dropout=dropout,
        mimo_rank=mimo_rank,
    )


def mamba3_multiview(
    in_features: int = 4,
    d_model: int = 256,
    d_state: int = 16,
    n_mamba_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.2,
    mimo_rank: int = 4,
) -> nn.Module:
    """Mamba-3 multi-view hybrid (checkpoint uses d_model=256)."""
    from inmotion.dl.models.mamba3_multiview import Mamba3MultiView

    return Mamba3MultiView(
        in_features=in_features,
        d_model=d_model,
        d_state=d_state,
        n_mamba_layers=n_mamba_layers,
        num_classes=num_classes,
        dropout=dropout,
        mimo_rank=mimo_rank,
    )


# ---------------------------------------------------------------------------
# Deep stacking ensemble (10 bases -> 4 Level2Nets -> meta MLP)
# ---------------------------------------------------------------------------


def deep_stack(
    num_classes: int = 4,
    dropout: float = 0.3,
    hidden: int = 40,
    num_level2: int = 4,
) -> nn.Module:
    """Deep stack ensemble, 10 heterogeneous bases.

    Base order matters: ``DeepStackEnsemble`` stores them as ``base_models.i``,
    so the checkpoint is only loadable with this exact sequence.
    """
    from inmotion.dl.models.cnn import CNNClassifier
    from inmotion.dl.models.cnn2d_rnn import CNN2DRNNClassifier
    from inmotion.dl.models.deep_stack import DeepStackEnsemble, Level2Net
    from inmotion.dl.models.gru import GRUClassifier
    from inmotion.dl.models.mamba import MambaClassifier
    from inmotion.dl.models.rnn import RNNClassifier
    from inmotion.dl.models.tcn import TCNClassifier

    bases: list[nn.Module] = [
        CNN2DRNNClassifier(
            in_features=4,
            num_filters=128,
            cnn_depth=3,
            hidden_size=256,
            num_rnn_layers=3,
            num_classes=4,
            dropout=0.3,
            rnn_type="lstm",
            bidirectional=True,
        ),
        CNNClassifier(4, 64, 2, 4, 0.2, [3, 5]),
        CNNClassifier(4, 256, 4, 4, 0.3, [3, 5, 7, 9]),
        GRUClassifier(4, 64, 2, 4, 0.2),
        MambaClassifier(4, 64, 16, 2, 2, 4, 0.2, mimo_rank=4),
        MambaClassifier(4, 256, 32, 4, 2, 4, 0.3, mimo_rank=8),
        RNNClassifier(4, 64, 2, 4, 0.2),
        RNNClassifier(4, 256, 4, 4, 0.3, bidirectional=True),
        TCNClassifier(4, 128, 3, 4, 4, 0.2),
        TCNClassifier(4, 512, 7, 8, 4, 0.3),
    ]
    return DeepStackEnsemble(
        bases,
        [Level2Net(hidden, num_classes) for _ in range(num_level2)],
        num_classes=num_classes,
        dropout=dropout,
    )


FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "lejepa": lejepa,
    "t_jepa": t_jepa,
    "ts_jepa": ts_jepa,
    "cf_jepa": cf_jepa,
    "sigreg": sigreg,
    "mamba3_cnn": mamba3_cnn,
    "mamba3_tcn": mamba3_tcn,
    "mamba3_transformer": mamba3_transformer,
    "mamba3_multiview": mamba3_multiview,
    "deep_stack": deep_stack,
}


def build(factory: str, arch: dict[str, Any] | None = None) -> nn.Module:
    """Instantiate ``factory`` with ``arch`` keyword arguments."""
    if factory not in FACTORIES:
        raise KeyError(f"unknown factory {factory!r}; known: {sorted(FACTORIES)}")
    return FACTORIES[factory](**(arch or {}))


# ---------------------------------------------------------------------------
# Classical DL classifiers (the `inmotion/dl/models` families). These were
# trained by the DL pipeline with per-model configs that were only ever recorded
# in a now-deleted
# Optuna database, so `adopt` recovers them from checkpoint shapes.
# ---------------------------------------------------------------------------


def rnn(
    in_features: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.3,
    bidirectional: bool = False,
) -> nn.Module:
    """Plain RNN classifier."""
    from inmotion.dl.models.rnn import RNNClassifier

    return RNNClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def gru(
    in_features: int = 4,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_classes: int = 4,
    dropout: float = 0.3,
    bidirectional: bool = False,
    use_attention: bool = False,
) -> nn.Module:
    """GRU classifier, optionally with attention pooling."""
    from inmotion.dl.models.gru import GRUClassifier

    return GRUClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
    )


def lstm(
    in_features: int = 4,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_classes: int = 4,
    dropout: float = 0.3,
    bidirectional: bool = False,
    use_attention: bool = True,
) -> nn.Module:
    """LSTM classifier, optionally with attention pooling."""
    from inmotion.dl.models.lstm import LSTMClassifier

    return LSTMClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
    )


def bilstm(
    in_features: int = 4,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_classes: int = 4,
    dropout: float = 0.3,
    use_attention: bool = True,
) -> nn.Module:
    """Bidirectional LSTM classifier."""
    from inmotion.dl.models.bilstm import BiLSTMClassifier

    return BiLSTMClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        use_attention=use_attention,
    )


def tcn(
    in_features: int = 4,
    num_channels: int = 128,
    kernel_size: int = 3,
    depth: int = 4,
    num_classes: int = 4,
    dropout: float = 0.2,
) -> nn.Module:
    """Temporal convolutional network classifier."""
    from inmotion.dl.models.tcn import TCNClassifier

    return TCNClassifier(
        in_features=in_features,
        num_channels=num_channels,
        kernel_size=kernel_size,
        depth=depth,
        num_classes=num_classes,
        dropout=dropout,
    )


def mamba(
    in_features: int = 4,
    d_model: int = 64,
    d_state: int = 16,
    num_layers: int = 2,
    expand: int = 2,
    num_classes: int = 4,
    dropout: float = 0.3,
    mimo_rank: int = 4,
) -> nn.Module:
    """Mamba (selective SSM) classifier."""
    from inmotion.dl.models.mamba import MambaClassifier

    return MambaClassifier(
        in_features=in_features,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        expand=expand,
        num_classes=num_classes,
        dropout=dropout,
        mimo_rank=mimo_rank,
    )


def transformer(
    in_features: int = 4,
    d_model: int = 64,
    n_heads: int = 4,
    num_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.1,
    max_seq_len: int = 512,
) -> nn.Module:
    """Transformer encoder classifier."""
    from inmotion.dl.models.transformer import TransformerClassifier

    return TransformerClassifier(
        in_features=in_features,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )


def cnn(
    in_features: int = 4,
    num_filters: int = 128,
    num_blocks: int = 3,
    num_classes: int = 4,
    dropout: float = 0.3,
    kernel_sizes: list[int] | None = None,
) -> nn.Module:
    """Multi-branch residual CNN classifier."""
    from inmotion.dl.models.cnn import CNNClassifier

    return CNNClassifier(
        in_features=in_features,
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_classes=num_classes,
        dropout=dropout,
        kernel_sizes=kernel_sizes,
    )


def cnn2d_rnn(
    in_features: int = 4,
    num_filters: int = 32,
    cnn_depth: int = 2,
    hidden_size: int = 64,
    num_rnn_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.3,
    rnn_type: str = "lstm",
    bidirectional: bool = False,
) -> nn.Module:
    """2-D CNN front-end feeding an RNN."""
    from inmotion.dl.models.cnn2d_rnn import CNN2DRNNClassifier

    return CNN2DRNNClassifier(
        in_features=in_features,
        num_filters=num_filters,
        cnn_depth=cnn_depth,
        hidden_size=hidden_size,
        num_rnn_layers=num_rnn_layers,
        num_classes=num_classes,
        dropout=dropout,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
    )


def meta_fusion(
    in_features: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.3,
    meta_embed_dim: int = 8,
    rnn_type: str = "lstm",
    bidirectional: bool = False,
) -> nn.Module:
    """RNN plus metadata (noise flag, concurrent path) fusion classifier."""
    from inmotion.dl.models.meta_fusion import MetaFusionClassifier

    return MetaFusionClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        meta_embed_dim=meta_embed_dim,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
    )


FACTORIES.update(
    {
        "rnn": rnn,
        "gru": gru,
        "lstm": lstm,
        "bilstm": bilstm,
        "tcn": tcn,
        "mamba": mamba,
        "transformer": transformer,
        "cnn": cnn,
        "cnn2d_rnn": cnn2d_rnn,
        "meta_fusion": meta_fusion,
    }
)


def voting_ensemble(
    models: list[dict[str, Any]],
    weights: list[float] | None = None,
    num_classes: int = 4,
) -> nn.Module:
    """Averaging ensemble over nested member specs.

    ``models`` is a list of ``{"factory": ..., "arch": {...}}``. Storing members
    as data keeps an ensemble rebuildable from its sidecar alone.
    """
    from inmotion.dl.models.ensemble import VotingEnsemble

    members = [build(m["factory"], m.get("arch")) for m in models]
    if weights is None:
        weights = [1.0] * len(members)
    return VotingEnsemble(members, list(weights))


def stacking_ensemble(
    base_models: list[dict[str, Any]],
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """Stacking ensemble: base members feed a learned meta-learner."""
    from inmotion.dl.models.ensemble import StackingEnsemble

    members = [build(m["factory"], m.get("arch")) for m in base_models]
    return StackingEnsemble(members, num_classes=num_classes, dropout=dropout)


FACTORIES.update(
    {
        "voting_ensemble": voting_ensemble,
        "stacking_ensemble": stacking_ensemble,
    }
)


# ---------------------------------------------------------------------------
# JEPA pre-training checkpoints. These save the *base* world model (plus its
# predictor) rather than a classifier, so they need the model classes directly
# instead of the `<X>Classifier` wrappers.
# ---------------------------------------------------------------------------


def lejepa_pretrain(**arch: Any) -> nn.Module:
    """Base LeJEPA world model (no classification head)."""
    from inmotion.dl.models.lejepa import LeJEPAModel

    arch.pop("hidden_dim", None)
    arch.pop("num_classes", None)
    arch.pop("dropout", None)
    return LeJEPAModel(**arch)


def t_jepa_pretrain(**arch: Any) -> nn.Module:
    """Base T-JEPA world model."""
    from inmotion.dl.models.t_jepa import TJEPAModel

    arch.pop("hidden_dim", None)
    arch.pop("num_classes", None)
    arch.pop("dropout", None)
    return TJEPAModel(**arch)


def ts_jepa_pretrain(**arch: Any) -> nn.Module:
    """Base TS-JEPA world model."""
    from inmotion.dl.models.ts_jepa import TSJEPAModel

    arch.pop("hidden_dim", None)
    arch.pop("num_classes", None)
    arch.pop("dropout", None)
    return TSJEPAModel(**arch)


def cf_jepa_pretrain(**arch: Any) -> nn.Module:
    """Base CF-JEPA world model."""
    from inmotion.dl.models.cf_jepa import CFJEPAModel

    arch.pop("hidden_dim", None)
    arch.pop("num_classes", None)
    arch.pop("dropout", None)
    return CFJEPAModel(**arch)


FACTORIES.update(
    {
        "lejepa_pretrain": lejepa_pretrain,
        "t_jepa_pretrain": t_jepa_pretrain,
        "ts_jepa_pretrain": ts_jepa_pretrain,
        "cf_jepa_pretrain": cf_jepa_pretrain,
    }
)
