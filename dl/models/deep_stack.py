"""Deep stacking: 8 DL base models → 4 label-specific Level2 nets → meta classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ensemble import MetaLearner


class Level2Net(nn.Module):
    """MLP: concatenated base logits → binary P(class_k vs rest)."""

    def __init__(self, in_features: int, dropout: float = 0.3) -> None:
        super().__init__()
        hidden = max(64, in_features)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeepStackEnsemble(nn.Module):
    """8 base models → concat logits (B, 8*C) → 4 Level2 binary nets → meta (B,4) → logits."""

    def __init__(
        self,
        base_models: list[nn.Module],
        level2_nets: list[nn.Module],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.level2 = nn.ModuleList(level2_nets)
        self.meta = MetaLearner(num_classes, num_classes, dropout)

    def get_base_features(self, x: Tensor) -> Tensor:
        """Concatenate logits from all base models → (B, n_base * num_classes)."""
        return torch.cat([m(x) for m in self.base_models], dim=-1)

    def get_level2_features(self, base_feats: Tensor) -> Tensor:
        """P(class_k) from each Level2 net → (B, num_classes)."""
        probs = [F.softmax(l2(base_feats), dim=-1)[:, 1] for l2 in self.level2]
        return torch.stack(probs, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        base_feats = self.get_base_features(x)
        l2_feats = self.get_level2_features(base_feats)
        return self.meta(l2_feats)

    def freeze_base_models(self) -> None:
        for m in self.base_models:
            for p in m.parameters():
                p.requires_grad_(False)

    def freeze_level2(self) -> None:
        for m in self.level2:
            for p in m.parameters():
                p.requires_grad_(False)
