"""Mixture of Experts: binary one-vs-rest DL classifiers + meta learner.

Supports GRU, LSTM, RNN and CNN binary experts in any combination.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cnn import CNNClassifier
from .ensemble import MetaLearner
from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .rnn import RNNClassifier


class MixtureOfExperts(nn.Module):
    """N binary experts, each predicts P(class_k vs rest). Meta maps N probs → num_classes."""

    def __init__(
        self,
        experts: list[nn.Module],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(experts) != num_classes:
            raise ValueError(f"Need {num_classes} experts, got {len(experts)}")
        self.experts = nn.ModuleList(experts)
        self.meta = MetaLearner(num_classes, num_classes, dropout)

    def get_expert_features(self, x: Tensor) -> Tensor:
        """Return (B, num_classes): P(positive | expert_k) for each binary expert."""
        probs = [F.softmax(e(x), dim=-1)[:, 1] for e in self.experts]
        return torch.stack(probs, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.meta(self.get_expert_features(x))


def build_moe_expert(
    in_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Binary GRU expert (bidirectional). Kept for backward compatibility."""
    return GRUClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        bidirectional=True,
    )


# Arch types supported as MoE experts
MOE_ARCH_TYPES: list[str] = ["gru", "lstm", "rnn", "cnn"]

# Named MoE combinations: maps combo name → list of arch type per class slot.
# All lists must have length == num_classes (4).
MOE_COMBOS: dict[str, list[str]] = {
    "MoE_4GRU": ["gru", "gru", "gru", "gru"],
    "MoE_4LSTM": ["lstm", "lstm", "lstm", "lstm"],
    "MoE_4RNN": ["rnn", "rnn", "rnn", "rnn"],
    "MoE_4CNN": ["cnn", "cnn", "cnn", "cnn"],
    "MoE_Mixed": ["gru", "lstm", "rnn", "cnn"],
    "MoE_2GRU2LSTM": ["gru", "gru", "lstm", "lstm"],
    "MoE_2RNN2CNN": ["rnn", "rnn", "cnn", "cnn"],
    "MoE_2GRU1LSTM1CNN": ["gru", "gru", "lstm", "cnn"],
}

# Kernel sets indexed by kernel_set param
_KERNEL_SETS: list[list[int]] = [[3], [3, 5], [3, 5, 7]]


def build_moe_expert_typed(
    arch_type: str,
    in_features: int,
    params: dict,
) -> nn.Module:
    """Build a binary (num_classes=2) expert of the given arch type using a params dict.

    params keys (all optional, fall back to sensible defaults):
      gru/lstm/rnn: hidden_size, num_layers, bidirectional
      lstm:         use_attention
      cnn:          num_filters, num_blocks, kernel_set (0/1/2)
      all:          dropout
    """
    dropout: float = float(params.get("dropout", 0.3))

    match arch_type:
        case "gru":
            return GRUClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=2,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
            )
        case "lstm":
            return LSTMClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=2,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
                use_attention=bool(params.get("use_attention", False)),
            )
        case "rnn":
            return RNNClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=2,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
            )
        case "cnn":
            ks_idx = int(params.get("kernel_set", 1))
            return CNNClassifier(
                in_features=in_features,
                num_filters=int(params.get("num_filters", 64)),
                num_blocks=int(params.get("num_blocks", 2)),
                num_classes=2,
                dropout=dropout,
                kernel_sizes=_KERNEL_SETS[ks_idx],
            )
        case _:
            raise ValueError(f"Unknown MoE expert arch: {arch_type!r}")
