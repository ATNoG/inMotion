"""Mamba-3 + TCN parallel hybrid.

TCN dilated convolutions and Mamba-3 selective SSM run in parallel,
outputs concatenated and classified.

Reference:
    SST (Xu et al., 2024) "Multi-Scale Hybrid Mamba-Transformer Experts"
        arxiv.org/abs/2404.14757  — CIKM 2025
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl.models.mamba import _MambaBlock
from dl.models.tcn import TCNClassifier


class _TCNEncoder(nn.Module):
    """TCN conv stack (from TCNClassifier) without pooling/classifier head."""

    def __init__(
        self,
        in_features: int,
        num_channels: int = 128,
        kernel_size: int = 3,
        depth: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        tcn = TCNClassifier(
            in_features=in_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            depth=depth,
            num_classes=4,
            dropout=dropout,
        )
        self.network = tcn.network  # Sequential of _TemporalBlock

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C) → TCN expects (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return x.permute(0, 2, 1)  # (B, L, num_channels)


class Mamba3TCN(nn.Module):
    """Mamba-3 + TCN parallel hybrid classifier."""

    def __init__(
        self,
        in_features: int = 4,
        tcn_channels: int = 128,
        tcn_depth: int = 4,
        d_model: int = 128,
        d_state: int = 16,
        n_mamba_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.tcn_enc = _TCNEncoder(in_features, tcn_channels, depth=tcn_depth, dropout=dropout)
        self.tcn_proj = nn.Linear(tcn_channels, d_model)

        self.mamba_input_proj = nn.Linear(in_features, d_model)
        self.mamba = nn.Sequential(*[
            _MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout, mimo_rank=mimo_rank)
            for _ in range(n_mamba_layers)
        ])

        self.combined_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        tcn_out = self.tcn_enc(x)  # (B, L, tcn_channels)
        tcn_out = self.tcn_proj(tcn_out)  # (B, L, d_model)

        mamba_out = self.mamba(self.mamba_input_proj(x))  # (B, L, d_model)

        combined = torch.cat([tcn_out, mamba_out], dim=-1)  # (B, L, 2*d_model)
        combined = self.combined_proj(combined)

        combined = combined.permute(0, 2, 1)
        pooled = self.pool(combined).squeeze(-1)
        return self.head(pooled)
