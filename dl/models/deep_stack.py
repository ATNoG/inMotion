"""Deep stacking: 8 DL base models → concat logits → deep meta MLP → logits.

Architecture (improved):
  Stage 1 (frozen base): base_feat = concat_logits(base_models, x)  (B, n_base * C)
           → Level2Net[k](base_feat) → full C-class logits per Level2 net
           → concat softmax outputs → DeepMetaMLP → logits
  Stage 2 (joint fine-tune): unfreeze base models, continue training with small LR.

Key fixes over legacy:
  - Level2Net outputs full multiclass (C classes), not binary — no information collapse.
  - Meta receives concat of all Level2 softmax outputs: (B, n_L2 * C) rich signal.
  - DeepMetaMLP is a 3-layer residual MLP with skip connection.
  - DeepStackEnsemble supports joint fine-tuning via unfreeze/freeze helpers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Level2Net(nn.Module):
    """MLP: concatenated base logits (B, n_base*C) → full multiclass logits (B, C).

    Multiclass output (instead of legacy binary) preserves all inter-class signal.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        hidden = max(128, in_features)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeepMetaMLP(nn.Module):
    """3-layer residual MLP meta-learner with skip connection.

    Input: concat of all Level2 softmax outputs → (B, n_l2 * num_classes).
    Output: (B, num_classes) logits.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        hidden = max(128, in_features * 2)
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.skip = nn.Linear(in_features, hidden)  # projection for residual
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip(x)
        h = self.act(self.norm1(self.fc1(x)))
        h = self.drop(h)
        h = self.act(self.norm2(self.fc2(h) + skip))
        h = self.drop(h)
        return self.fc3(h)


class DeepStackEnsemble(nn.Module):
    """8 base models → concat logits (B, n_base*C) → n_l2 Level2 nets → DeepMetaMLP → logits.

    Supports two-stage training:
      Stage 1: freeze base models, train Level2 nets + meta.
      Stage 2: unfreeze all, joint fine-tune end-to-end with lower LR.
    """

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
        # Meta receives concat of all Level2 softmax: (n_l2 * num_classes)
        meta_in = len(level2_nets) * num_classes
        self.meta = DeepMetaMLP(meta_in, num_classes, dropout)
        self.num_classes = num_classes

    def get_base_features(self, x: Tensor) -> Tensor:
        """Concatenate logits from all base models → (B, n_base * num_classes)."""
        return torch.cat([m(x) for m in self.base_models], dim=-1)

    def get_level2_features(self, base_feats: Tensor) -> Tensor:
        """Concat softmax outputs from all Level2 nets → (B, n_l2 * num_classes)."""
        probs = [F.softmax(l2(base_feats), dim=-1) for l2 in self.level2]  # each (B, C)
        return torch.cat(probs, dim=-1)  # (B, n_l2 * C)

    def forward(self, x: Tensor) -> Tensor:
        base_feats = self.get_base_features(x)
        l2_feats = self.get_level2_features(base_feats)
        return self.meta(l2_feats)

    def freeze_base_models(self) -> None:
        for m in self.base_models:
            for p in m.parameters():
                p.requires_grad_(False)

    def unfreeze_base_models(self) -> None:
        for m in self.base_models:
            for p in m.parameters():
                p.requires_grad_(True)

    def freeze_level2(self) -> None:
        for m in self.level2:
            for p in m.parameters():
                p.requires_grad_(False)

    def unfreeze_level2(self) -> None:
        for m in self.level2:
            for p in m.parameters():
                p.requires_grad_(True)
