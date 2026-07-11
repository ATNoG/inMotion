"""Multi-View Mamba-3 with Tango Scanning.

Three parallel feature views feed a Mamba-3 backbone with tango scanning
(forward + reverse sequence).  A learned switch gate weights the views.

Reference:
    Ahamed & Cheng (2024) "TSCMamba" — arxiv.org/abs/2406.04419
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl.models.mamba import _MambaBlock


class _LocalView(nn.Module):
    def __init__(self, in_features: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, out_dim, 3, padding=1),
            nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, 5, padding=2, groups=out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


class _GlobalView(nn.Module):
    def __init__(self, seq_len: int, in_features: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * in_features, out_dim * 2),
            nn.GELU(), nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x).unsqueeze(1).expand(-1, x.size(1), -1)


class _SwitchGate(nn.Module):
    def __init__(self, n_views: int, d_model: int) -> None:
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1),
        )

    def forward(self, views: list[Tensor]) -> Tensor:
        scores = [self.score_net(v.mean(dim=1)).squeeze(-1) for v in views]
        weights = torch.stack(scores, dim=-1).softmax(dim=-1)  # (B, N)
        return sum(w.unsqueeze(-1).unsqueeze(-1) * v for w, v in zip(weights.unbind(1), views))


class _TangoMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1, mimo_rank: int = 4) -> None:
        super().__init__()
        self.mamba = _MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout, mimo_rank=mimo_rank)

    def forward(self, x: Tensor) -> Tensor:
        fwd = self.mamba(x)
        rev = self.mamba(x.flip(dims=[1]))
        return fwd + rev.flip(dims=[1])


class Mamba3MultiView(nn.Module):
    def __init__(
        self, in_features: int = 4, d_model: int = 128, d_state: int = 16,
        n_mamba_layers: int = 2, num_classes: int = 4, dropout: float = 0.2,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.local_view = _LocalView(in_features, d_model, dropout)
        self.global_view = _GlobalView(10, in_features, d_model)
        self.raw_view = nn.Linear(in_features, d_model)
        self.gate = _SwitchGate(3, d_model)
        self.mamba_blocks = nn.Sequential(*[
            _TangoMamba(d_model, d_state, dropout, mimo_rank) for _ in range(n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        v1 = self.local_view(x)
        v2 = self.global_view(x)
        v3 = self.raw_view(x)
        fused = self.gate([v1, v2, v3])
        out = self.mamba_blocks(fused)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        out = self.pool(out).squeeze(-1)
        return self.head(out)
