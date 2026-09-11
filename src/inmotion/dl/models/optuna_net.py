"""Optuna NAS: build model architecture from trial suggestions."""

from optuna import Trial
from torch import nn

from .autoformer import AutoformerClassifier
from .cnn import CNNClassifier
from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .rnn import RNNClassifier

_ARCH_NAMES = ["rnn", "gru", "lstm", "lstm_attn", "cnn", "autoformer"]


def build_optuna_model(trial: Trial, in_features: int, num_classes: int) -> nn.Module:
    """Let Optuna choose and parameterise the full network architecture."""
    arch: str = trial.suggest_categorical("arch", _ARCH_NAMES)  # type: ignore[assignment]
    dropout: float = trial.suggest_float("dropout", 0.1, 0.6)

    if arch in ("rnn", "gru", "lstm", "lstm_attn"):
        hidden_size: int = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers: int = trial.suggest_int("num_layers", 1, 4)
        bidir: bool = trial.suggest_categorical("bidirectional", [True, False])  # type: ignore[assignment]

        if arch == "rnn":
            return RNNClassifier(in_features, hidden_size, num_layers, num_classes, dropout, bidir)
        if arch == "gru":
            return GRUClassifier(in_features, hidden_size, num_layers, num_classes, dropout, bidir)
        # lstm / lstm_attn
        use_attn = arch == "lstm_attn"
        return LSTMClassifier(
            in_features, hidden_size, num_layers, num_classes, dropout, bidir, use_attn
        )

    if arch == "cnn":
        num_filters: int = trial.suggest_int("num_filters", 32, 256, log=True)
        num_blocks: int = trial.suggest_int("num_blocks", 1, 4)
        ks_choice: int = trial.suggest_categorical("kernel_size_set", [0, 1, 2])  # type: ignore[assignment]
        kernel_sets: list[list[int]] = [[3], [3, 5], [3, 5, 7]]
        return CNNClassifier(
            in_features, num_filters, num_blocks, num_classes, dropout, kernel_sets[ks_choice]
        )

    # autoformer — d_model always mult of 8 so all head choices valid every trial
    d_model_mult: int = trial.suggest_int("d_model_mult", 4, 16)
    d_model = d_model_mult * 8  # 32–128
    n_heads_raw: int = trial.suggest_categorical("n_heads", [1, 2, 4, 8])  # type: ignore[assignment]
    n_heads = n_heads_raw
    while d_model % n_heads != 0:
        n_heads //= 2
    num_enc_layers: int = trial.suggest_int("num_enc_layers", 1, 4)
    factor: int = trial.suggest_int("factor", 1, 3)
    return AutoformerClassifier(
        in_features, d_model, n_heads, num_enc_layers, num_classes, dropout, factor
    )
