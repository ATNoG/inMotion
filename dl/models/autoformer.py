"""Autoformer classifier with Auto-Correlation and series decomposition."""

import math

import torch
from torch import Tensor, nn


class SeriesDecomposition(nn.Module):
    """Decompose input into trend (moving avg) and seasonal (remainder)."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, seq_len, d_model)
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Trim if avg pool added extra length
        trend = trend[:, : x.shape[1], :]
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """FFT-based auto-correlation attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, factor: int = 1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)  # (B, H, L, D)
        k = self.k_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)

        # FFT cross-correlation
        q_f = torch.fft.rfft(q, dim=2)
        k_f = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(q_f * torch.conj(k_f), n=L, dim=2)  # (B, H, L, D)

        # Top-k lag selection
        k_top = max(1, int(self.factor * math.log(L + 1)))
        k_top = min(k_top, L)
        corr_mean = corr.mean(dim=-1)  # (B, H, L)
        top_vals, top_idx = corr_mean.topk(k_top, dim=-1)  # (B, H, k_top)
        weights = torch.softmax(top_vals, dim=-1)  # (B, H, k_top)

        # Vectorised lag-aggregation: gather shifted V
        t = torch.arange(L, device=x.device).view(1, 1, 1, L)  # (1,1,1,L)
        lags = top_idx.unsqueeze(-1)  # (B, H, k_top, 1)
        idx = (t + lags) % L  # (B, H, k_top, L)
        idx_exp = idx.unsqueeze(-1).expand(B, H, k_top, L, D)
        v_exp = v.unsqueeze(2).expand(B, H, k_top, L, D)
        gathered = v_exp.gather(3, idx_exp)  # (B, H, k_top, L, D)
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, H, k_top, 1, 1)
        out = (gathered * w).sum(dim=2)  # (B, H, L, D)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)  # (B, L, d_model)
        return self.drop(self.out_proj(out))


class AutoformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        factor: int = 1,
        decomp_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.autocorr = AutoCorrelation(d_model, n_heads, dropout, factor)
        self.decomp1 = SeriesDecomposition(decomp_kernel)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.decomp2 = SeriesDecomposition(decomp_kernel)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.autocorr(x)
        x, _ = self.decomp1(x)
        x = x + self.ff(self.norm(x))
        x, _ = self.decomp2(x)
        return x


class AutoformerClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        factor: int = 1,
        decomp_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_features, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                AutoformerEncoderLayer(d_model, n_heads, dropout, factor, decomp_kernel)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, in_features)
        x = self.pos_drop(self.embedding(x))
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x).mean(dim=1)  # global avg pool over time
        x = self.drop(x)
        return self.classifier(x)
