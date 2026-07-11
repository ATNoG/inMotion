"""Mamba-3 + CNN hybrid classifier.

CNN frontend extracts local temporal features (multi-scale conv, residual),
Mamba-3 backend models the sequence of CNN features with selective SSM.

Variant 4a from the research plan. Expected: MCC 0.85–0.86.

Reference:
    Lahoti et al. (2026) "Mamba-3" — arxiv.org/abs/2603.15569
    Boukhari (2025) "Mamba-CNN Hybrid" — arxiv.org/abs/2509.01431
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl.models.mamba import _MambaBlock


class _MultiScaleCNN(nn.Module):
    """Lightweight multi-scale CNN frontend (1D conv over engineered channels)."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5]
        self.branches = nn.ModuleList([
            nn.Conv1d(in_features, out_channels // len(kernel_sizes), k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.proj = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x.permute(0, 2, 1)  # (B, L, out_channels)


class Mamba3CNN(nn.Module):
    """Mamba-3 + Multi-Scale CNN for RSSI sequence classification."""

    def __init__(
        self,
        in_features: int = 4,
        cnn_channels: int = 128,
        cnn_kernels: list[int] | None = None,
        d_model: int = 128,
        d_state: int = 16,
        n_mamba_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.cnn = _MultiScaleCNN(
            in_features=in_features,
            out_channels=cnn_channels,
            kernel_sizes=cnn_kernels,
            dropout=dropout,
        )
        # Project CNN output to d_model if different
        self.cnn_proj = (
            nn.Linear(cnn_channels, d_model) if cnn_channels != d_model else nn.Identity()
        )
        self.mamba_blocks = nn.Sequential(*[
            _MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout, mimo_rank=mimo_rank)
            for _ in range(n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        x = self.cnn(x)  # (B, L, cnn_channels)
        x = self.cnn_proj(x)  # (B, L, d_model)
        x = self.mamba_blocks(x)  # (B, L, d_model)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.head(x)
