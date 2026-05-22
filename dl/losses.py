"""Custom loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multi-class focal loss. Down-weights easy examples so model focuses on hard ones."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing.

    Smooth target: (1 - ε) on the true class + ε / (C-1) on all other classes.
    Reduces overconfident predictions and acts as a calibration regulariser.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_val = self.smoothing / max(self.num_classes - 1, 1)
            smooth = torch.full_like(log_probs, smooth_val)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_probs).sum(dim=-1).mean()
