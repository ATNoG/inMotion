"""Mixture of Experts: 4 binary one-vs-rest DL classifiers + meta learner."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ensemble import MetaLearner
from .gru import GRUClassifier


class MixtureOfExperts(nn.Module):
    """4 binary experts, each predicts P(class_k vs rest). Meta maps 4 probs → 4-class."""

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
    """Binary GRU expert (bidirectional, two classes: positive vs rest)."""
    return GRUClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        bidirectional=True,
    )
