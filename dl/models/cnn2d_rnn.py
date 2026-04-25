"""2D-CNN + RNN classifier.

Architecture:
  Input  (B, T, F)  →  "fake 2D image"  (B, 1, T, F)
       ↓  Conv2d blocks (capture joint time × feature patterns)
       ↓  Reshape to sequence  (B, T', C)
       ↓  LSTM or GRU  (model temporal dynamics after CNN)
       ↓  Attention pooling  →  classifier

The 2D view lets the convolutional filters learn receptive fields that span
both the time axis (x = 1..10) and the feature/value axis (y = RSSI channels),
catching diagonal patterns that 1D-CNN misses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _Conv2dBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: tuple[int, int], dropout: float
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.shortcut(x)


class CNN2DRNNClassifier(nn.Module):
    """2D-CNN followed by LSTM or GRU for RSSI sequence classification."""

    def __init__(
        self,
        in_features: int,
        num_filters: int = 32,
        cnn_depth: int = 2,
        hidden_size: int = 64,
        num_rnn_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        rnn_type: str = "lstm",  # "lstm" | "gru"
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        assert self.rnn_type in ("lstm", "gru"), f"rnn_type must be 'lstm' or 'gru', got {rnn_type}"

        # ── 2D-CNN ────────────────────────────────────────────────────────────
        # Input image: (B, 1, T, F)  where T=seq_len, F=in_features
        cnn_layers: list[nn.Module] = []
        ch = 1
        for i in range(cnn_depth):
            out_ch = num_filters * (2 ** min(i, 2))
            cnn_layers.append(_Conv2dBlock(ch, out_ch, (3, 3), dropout))
            ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = ch  # final channel count after CNN

        # ── RNN ───────────────────────────────────────────────────────────────
        # After CNN: (B, C, T, F) → reshape to (B, T, C*F) — treat T as seq dim
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        rnn_in = self.cnn_out_channels * in_features  # C * F features per timestep
        self.rnn = rnn_cls(
            input_size=rnn_in,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)

        # ── Attention + classifier ─────────────────────────────────────────────
        self.attn = nn.Linear(out_size, 1)
        self.norm = nn.LayerNorm(out_size)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_size, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F)
        B, T, F = x.shape

        # ── 2D image: (B, 1, T, F) ───────────────────────────────────────────
        img = x.unsqueeze(1)  # (B, 1, T, F)
        feat = self.cnn(img)  # (B, C, T, F)  — same spatial size (padded)

        # ── Reshape to RNN sequence: (B, T, C*F) ─────────────────────────────
        C = feat.size(1)
        seq = feat.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C*F)

        # ── RNN ───────────────────────────────────────────────────────────────
        rnn_out, _ = self.rnn(seq)  # (B, T, out_size)

        # ── Attention pooling ─────────────────────────────────────────────────
        scores = self.attn(rnn_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, out_size)

        context = self.norm(context)
        context = self.drop(context)
        return self.classifier(context)
