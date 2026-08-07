"""SIGReg classifier — CNN encoder with Gaussian latent regularizer.

Replaces the standard L1+L2+dropout+mixup+label_smoothing stack with a single
Gaussian regularizer (SIGReg from LeWorldModel/LeJEPA) on a latent bottleneck.

Principle: enforce N(0,I) distribution on latent vectors. This prevents
overfitting by penalizing any deviation from the Gaussian prior, naturally
limiting the information the model can store in the latent space.

Reference:
    Maes et al. (2026) "LeWorldModel" — arXiv:2603.19312
    Balestriero & LeCun (2025) "LeJEPA" — arXiv:2511.08544
"""

from __future__ import annotations

import torch
from torch import Tensor, nn




# ═══════════════════════════════════════════════════════════════════════════════
# Residual CNN encoder
# ═══════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    """1D residual conv block with GELU activation."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class _CNNEncoder(nn.Module):
    """Multi-scale CNN encoder for RSSI sequences.

    Input:  (B, seq_len, in_features)
    Output: (B, num_filters) — global pooled features
    """

    def __init__(
        self,
        in_features: int = 4,
        num_filters: int = 128,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        # Multi-scale initial convolution
        self.conv3 = nn.Conv1d(in_features, num_filters // 3, 3, padding=1)
        self.conv5 = nn.Conv1d(in_features, num_filters // 3, 5, padding=2)
        self.conv7 = nn.Conv1d(in_features, num_filters - 2 * (num_filters // 3), 7, padding=3)
        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.GELU()

        # Residual blocks
        self.blocks = nn.Sequential(*[
            _ResBlock(num_filters, 3) for _ in range(num_blocks)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_features) → (B, num_filters)."""
        x = x.permute(0, 2, 1)  # (B, in_features, seq_len)

        # Multi-scale conv
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        x = torch.cat([c3, c5, c7], dim=1)
        x = self.act(self.bn(x))

        # Residual blocks
        x = self.blocks(x)

        # Global pool
        return self.pool(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGReg loss — Gaussian regularizer
# ═══════════════════════════════════════════════════════════════════════════════

class _SIGRegLoss(nn.Module):
    """Gaussian latent regularizer using ECF (Empirical Characteristic Function).

    From LeJEPA paper: compares the empirical CF of latent vectors to the
    CF of N(0,I) = exp(-t^2/2) via weighted L2 integration.

    This is more principled than simple moment-matching — it captures ALL
    moments of the distribution through the characteristic function.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17, n_slices: int = 256) -> None:
        super().__init__()
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.n_slices = n_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: Tensor) -> Tensor:
        """Compute SIGReg loss on latent vectors z: (B, D)."""
        # Random projection to 1D slices
        A = torch.randn(z.size(-1), self.n_slices, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)
        x = z @ A  # (B, S)

        # ECF: (1/B) * Σ_b cos(t * x_b) ≈ Re[CF(t)]
        x_t = x.unsqueeze(-1) * self.t  # (B, S, K)
        cos_mean = x_t.cos().mean(0)    # (S, K)
        sin_mean = x_t.sin().mean(0)    # (S, K)

        # Compare to N(0,1) CF
        err = (cos_mean - self.phi).square() + sin_mean.square()  # (S, K)
        statistic = (err @ self.weights) * z.size(0)  # (S,)
        return statistic.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Full SIGReg classifier
# ═══════════════════════════════════════════════════════════════════════════════

class SIGRegClassifier(nn.Module):
    """CNN + Gaussian latent bottleneck + SIGReg.

    Total loss = cross_entropy(logits, labels) + λ * sigreg(latent_vectors)

    The latent bottleneck sits between the CNN encoder and the classifier head.
    SIGReg on the latent vectors replaces L1, L2, dropout, and Mixup.
    """

    def __init__(
        self,
        in_features: int = 4,
        num_filters: int = 128,
        num_blocks: int = 3,
        latent_dim: int = 128,
        num_classes: int = 4,
        sigreg_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.sigreg_lambda = sigreg_lambda

        # Encoder: CNN → global pooled features → latent projection
        self.encoder = _CNNEncoder(in_features, num_filters, num_blocks)
        self.latent_proj = nn.Linear(num_filters, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)

        # SIGReg loss module
        self.sigreg = _SIGRegLoss()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning logits and latent vectors.

        Args:
            x: (B, seq_len, in_features)

        Returns:
            logits: (B, num_classes)
            latents: (B, latent_dim)
        """
        features = self.encoder(x)                     # (B, num_filters)
        latents = self.latent_norm(self.latent_proj(features))  # (B, latent_dim)
        logits = self.classifier(latents)               # (B, num_classes)
        return logits, latents

    def compute_loss(
        self, logits: Tensor, latents: Tensor, targets: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total loss = CE + λ * SIGReg.

        Args:
            logits: (B, num_classes)
            latents: (B, latent_dim)
            targets: (B,)

        Returns:
            total_loss, metrics dict
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
        sigreg_loss = self.sigreg(latents)
        total = ce_loss + self.sigreg_lambda * sigreg_loss
        return total, {
            "ce_loss": ce_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "total_loss": total.item(),
        }
