"""Voting and Stacking ensemble wrappers over trained nn.Module classifiers."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VotingEnsemble(nn.Module):
    """Soft-voting ensemble: average softmax probabilities."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        n = len(models)
        if weights is None:
            w = torch.ones(n) / n
        else:
            t = torch.tensor(weights, dtype=torch.float32)
            w = t / t.sum()
        self.register_buffer("weights", w)

    def forward(self, x: Tensor) -> Tensor:
        probs = torch.stack(
            [F.softmax(m(x), dim=-1) for m in self.models], dim=0
        )  # (n_models, B, C)
        w = self.weights.view(-1, 1, 1)  # type: ignore[attr-defined]
        return (probs * w).sum(dim=0)  # (B, C) — already probabilities


class MetaLearner(nn.Module):
    """Small MLP meta-learner for stacking."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        hidden = max(64, in_features * 2)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StackingEnsemble(nn.Module):
    """Stacking ensemble: base models → meta-learner."""

    def __init__(
        self,
        base_models: list[nn.Module],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        meta_in = len(base_models) * num_classes
        self.meta = MetaLearner(meta_in, num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        probs = [F.softmax(m(x), dim=-1) for m in self.base_models]  # each (B, C)
        stacked = torch.cat(probs, dim=-1)  # (B, n*C)
        return self.meta(stacked)  # (B, C) logits

    def freeze_base_models(self) -> None:
        for m in self.base_models:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_base_models(self) -> None:
        for m in self.base_models:
            for p in m.parameters():
                p.requires_grad = True
