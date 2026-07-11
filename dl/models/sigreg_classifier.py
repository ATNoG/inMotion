"""SIGReg classifier — CNN with Gaussian latent regularizer + dynamic augmentation.

Replaces L1+L2+dropout+mixup+label_smoothing with a single Gaussian
regularizer (SIGReg from LeWorldModel/LeJEPA) on a latent bottleneck.
Uses dropout-style dynamic augmentation to create infinite samples.

Reference:
    Maes et al. (2026) "LeWorldModel" — arxiv.org/abs/2603.19312
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl.sigreg import LeJEPASIGReg

class DynamicAugment(nn.Module):
    """On-the-fly augmentation: drops timesteps, channels, or blocks."""

    def __init__(self, drop_timestep_p: float = 0.1, drop_channel_p: float = 0.1,
                 block_mask_p: float = 0.2, block_max_len: int = 4, noise_std: float = 0.5) -> None:
        super().__init__()
        self.drop_t_p = drop_timestep_p; self.drop_c_p = drop_channel_p
        self.block_p = block_mask_p; self.block_max = block_max_len
        self.noise_std = noise_std

    def forward(self, x: Tensor) -> Tensor:
        if not self.training: return x
        B, L, C = x.shape; device = x.device
        if self.drop_t_p > 0:
            x = x * (torch.rand(B, L, 1, device=device) > self.drop_t_p)
        if self.drop_c_p > 0:
            x = x * (torch.rand(B, 1, C, device=device) > self.drop_c_p)
        if self.block_p > 0 and torch.rand(1, device=device).item() < self.block_p:
            bl = torch.randint(1, self.block_max + 1, (1,), device=device).item()
            if bl < L:
                for b in range(B):
                    s = torch.randint(0, L - bl + 1, (1,), device=device).item()
                    x[b, s:s + bl, :] = 0.0
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * (x != 0).float()
        return x


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        p = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class _CNNEncoder(nn.Module):
    def __init__(self, in_features: int, num_filters: int = 128, num_blocks: int = 3,
                 latent_dim: int = 128) -> None:
        super().__init__()
        ks = [3, 5, 7]; pb = num_filters // len(ks)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_features, pb, k, padding=k//2),
                          nn.BatchNorm1d(pb), nn.GELU()) for k in ks
        ])
        self.proj = nn.Sequential(nn.Conv1d(pb * len(ks), num_filters, 1),
                                  nn.BatchNorm1d(num_filters), nn.GELU())
        self.res_blocks = nn.Sequential(*[_ResBlock(num_filters, 3) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.latent_proj = nn.Linear(num_filters, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = torch.cat([b(x) for b in self.branches], dim=1)
        x = self.proj(x); x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.latent_proj(x)


class SIGRegClassifier(nn.Module):
    """CNN + latent bottleneck + SIGReg + dynamic augmentation.

    Trainer auto-detects `last_aux_loss` (same pattern as MoE) and adds
    it to the cross-entropy loss. No changes needed to Trainer.
    """

    def __init__(self, in_features: int = 4, num_filters: int = 256, num_blocks: int = 4,
                 latent_dim: int = 256, num_classes: int = 4, sigreg_lambda: float = 0.01,
                 aug_drop_t: float = 0.15, aug_drop_c: float = 0.1, aug_block_p: float = 0.3,
                 aug_block_len: int = 4, aug_noise: float = 1.0) -> None:
        super().__init__()
        self.augment = DynamicAugment(aug_drop_t, aug_drop_c, aug_block_p, aug_block_len, aug_noise)
        self.encoder = _CNNEncoder(in_features, num_filters, num_blocks, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.sigreg = LeJEPASIGReg()
        self.sigreg_lambda = sigreg_lambda
        self.last_aux_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.augment(x)
        z = self.encoder(x)
        self.last_aux_loss = self.sigreg_lambda * self.sigreg(z)
        return self.classifier(z)
