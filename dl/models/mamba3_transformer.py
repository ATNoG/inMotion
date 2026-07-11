"""Mamba-3 + Transformer hybrid (State Space Transformer).

Mamba-3 handles full-sequence patterns, Transformer with small window
attention refines local context.  Mamba → Transformer ordering avoids
the information interference found in naive interleaving (SST, CIKM 2025).

Variant 4c from the research plan.

Reference:
    SST (Xu et al., 2024) "Multi-Scale Hybrid Mamba-Transformer Experts"
        arxiv.org/abs/2404.14757  — CIKM 2025
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dl.models.mamba import _MambaBlock


class _HybridBlock(nn.Module):
    """Mamba → Transformer block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        nhead: int = 4,
        dropout: float = 0.1,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.mamba = _MambaBlock(
            d_model=d_model, d_state=d_state, dropout=dropout, mimo_rank=mimo_rank
        )
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # Mamba: full-sequence selective scan
        x = x + self.mamba(self.norm(x))

        # Transformer: window-attention for local refinement
        residual = x
        x_normed = self.attn_norm(x)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = residual + attn_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Mamba3Transformer(nn.Module):
    """Mamba-3 + Transformer hybrid classifier.

    Adds sin/cos positional encoding before the hybrid blocks.
    """

    def __init__(
        self,
        in_features: int = 4,
        d_model: int = 128,
        d_state: int = 16,
        nhead: int = 4,
        num_blocks: int = 3,
        num_classes: int = 4,
        dropout: float = 0.1,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)

        # Sin/cos positional encoding
        pe = torch.zeros(10, d_model)
        pos = torch.arange(10, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_embed", pe.unsqueeze(0))

        self.blocks = nn.ModuleList([
            _HybridBlock(d_model, d_state, nhead, dropout, mimo_rank)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        x = self.input_proj(x)  # (B, L, d_model)
        x = x + self.pos_embed  # add positional encoding

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.head(x)
