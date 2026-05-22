"""Temporal Convolutional Network (TCN) classifier for RSSI time-series.

Architecture:
  Stack of TemporalBlocks — each block has two dilated causal Conv1d layers
  with weight-norm, BatchNorm, GELU activation, dropout, and a residual skip.
  Dilation doubles at each block: 1, 2, 4, 8, …

  Input:  (B, T, in_features)
  Output: (B, num_classes) logits

Reference: Bai et al. "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" (2018).
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class _CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) zero-padding and weight-normalisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,  # manual causal padding
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.causal_pad, 0))
        return self.conv(x)


class _TemporalBlock(nn.Module):
    """Two dilated-causal convs + BatchNorm + GELU + dropout + residual skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        # Skip connection for channel dimension mismatch
        self.skip: nn.Module
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.drop(out)
        return self.act(out + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network classifier.

    Args:
        in_features:  Number of input channels per timestep.
        num_channels: Number of conv filters in each temporal block.
        kernel_size:  Convolution kernel size (3, 5, or 7 recommended).
        depth:        Number of temporal blocks (dilation doubles each block).
        num_classes:  Number of output classes.
        dropout:      Dropout probability (keep ≤ 0.4 for stability).
    """

    def __init__(
        self,
        in_features: int,
        num_channels: int = 128,
        kernel_size: int = 3,
        depth: int = 4,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            in_ch = in_features if i == 0 else num_channels
            dilation = 2**i
            layers.append(_TemporalBlock(in_ch, num_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, in_features) → (B, num_classes)."""
        x = x.permute(0, 2, 1)  # → (B, in_features, T)
        x = self.network(x)  # → (B, num_channels, T)
        x = self.pool(x).squeeze(-1)  # → (B, num_channels)
        x = self.drop(x)
        return self.classifier(x)  # → (B, num_classes)
