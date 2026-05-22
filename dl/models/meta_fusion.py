"""Metadata-fusion classifier: RSSI backbone + noise / path context embeddings.

Architecture
────────────
  RSSI encoder (LSTM or GRU)  →  rssi_feat  (B, rssi_dim)
  noise embedding              →  e_noise    (B, meta_dim)
  concurrent_noise_path embed  →  e_path     (B, meta_dim)
  ──────────────────────────────────────────────────────
  concat([rssi_feat, e_noise, e_path])       (B, rssi_dim + 2*meta_dim)
       ↓  MLP head  →  logits                (B, num_classes)

This lets the model say:
  "I see a signal drop (AB pattern) but noise=True and path=BA →
   it's interference, not real movement."

Metadata encoding
─────────────────
  meta[:, 0]  = noise_int       (int64)  0 = False, 1 = True
  meta[:, 1]  = noise_path_idx  (int64)  0..3 = AA/AB/BA/BB, 4 = unknown/none
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MetaFusionClassifier(nn.Module):
    """RSSI sequence encoder fused with metadata context embeddings."""

    # Noise-path vocabulary: AA=0, AB=1, BA=2, BB=3, unknown=4
    NOISE_PATH_VOCAB = 5
    NOISE_VOCAB = 2

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        meta_embed_dim: int = 8,
        rnn_type: str = "lstm",  # "lstm" | "gru"
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        assert self.rnn_type in ("lstm", "gru")

        # ── RSSI backbone ─────────────────────────────────────────────────────
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rssi_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Linear(rssi_dim, 1)
        self.rssi_norm = nn.LayerNorm(rssi_dim)

        # ── Metadata embeddings ───────────────────────────────────────────────
        self.noise_embed = nn.Embedding(self.NOISE_VOCAB, meta_embed_dim)
        self.path_embed = nn.Embedding(self.NOISE_PATH_VOCAB, meta_embed_dim)

        # ── Fusion head ───────────────────────────────────────────────────────
        fused_dim = rssi_dim + 2 * meta_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.noise_embed.weight, std=0.01)
        nn.init.normal_(self.path_embed.weight, std=0.01)

    def forward(self, x: Tensor, meta: Tensor) -> Tensor:
        """Args:
            x:    (B, T, in_features)   — RSSI sequence
            meta: (B, 2)  int64          — [noise_int, noise_path_idx]
        Returns:
            logits: (B, num_classes)
        """
        # ── RSSI branch ───────────────────────────────────────────────────────
        rnn_out, _ = self.rnn(x)  # (B, T, rssi_dim)
        scores = self.attn(rnn_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)
        rssi_feat = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, rssi_dim)
        rssi_feat = self.rssi_norm(rssi_feat)

        # ── Metadata branch ───────────────────────────────────────────────────
        noise_idx = meta[:, 0].clamp(0, self.NOISE_VOCAB - 1)
        path_idx = meta[:, 1].clamp(0, self.NOISE_PATH_VOCAB - 1)
        e_noise = self.noise_embed(noise_idx)  # (B, meta_embed_dim)
        e_path = self.path_embed(path_idx)  # (B, meta_embed_dim)

        # ── Fusion & classification ───────────────────────────────────────────
        fused = torch.cat([rssi_feat, e_noise, e_path], dim=-1)  # (B, fused_dim)
        return self.head(fused)
