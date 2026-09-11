"""Lightweight Transformer classifier for short RSSI time-series (seq_len ≤ 512).

Architecture:
  input_proj: linear projection in_features → d_model
  pos_emb:    learnable positional embeddings (no sinusoidal bias)
  encoder:    N × TransformerEncoderLayer (pre-LayerNorm for stability)
  head:       global average-pool → dropout → linear → logits

Pre-LayerNorm ("norm_first=True") is consistently more stable than post-LN,
especially at small datasets. The model is lightweight enough to work well on
only 10 timesteps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        # Ensure d_model is divisible by n_heads
        while d_model % n_heads != 0:
            n_heads = max(1, n_heads // 2)
        self.input_proj = nn.Linear(in_features, d_model)
        # Learnable absolute positional embeddings (robust for short sequences)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoid deprecation warning
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, in_features) → (B, num_classes)."""
        B, T, _ = x.shape
        x = self.input_proj(x) + self.pos_emb[:, :T, :]  # (B, T, d_model)
        x = self.encoder(x)  # (B, T, d_model)
        x = self.norm(x.mean(dim=1))  # global avg-pool → (B, d_model)
        x = self.drop(x)
        return self.head(x)  # (B, num_classes)
