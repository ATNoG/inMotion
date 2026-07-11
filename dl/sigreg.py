"""SIGReg — LeJEPA-style Gaussian regularizer via Empirical Characteristic Function.

Implements the ECF-based Gaussianity test from LeJEPA (Balestriero & LeCun, 2025).
Much more principled than moment-matching — captures ALL moments through the CF.

Reference:
    Balestriero & LeCun (2025) "LeJEPA" — arxiv.org/abs/2511.08544
    github.com/galilai-group/lejepa
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LeJEPASIGReg(nn.Module):
    """ECF-based Gaussian regularizer from the LeJEPA paper.

    Compares the empirical characteristic function of latent vectors
    to exp(-t²/2) (the CF of N(0,I)) via weighted L2 integration.

    This is the "SIGReg" from the LeJEPA minimal example (~20 lines).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: Tensor) -> Tensor:
        """Compute SIGReg loss on latent vectors.

        Args:
            z: (B, D) latent vectors

        Returns:
            scalar loss
        """
        # Random projection to 1D slices (LeJEPA paper: 256 slices)
        A = torch.randn(z.size(-1), 256, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project: (B, D) @ (D, 256) → (B, 256)
        x = z @ A  # (B, S)

        # ECF computation
        x_t = x.unsqueeze(-1) * self.t  # (B, S, K)
        cos_mean = x_t.cos().mean(0)  # (S, K) — mean over batch
        sin_mean = x_t.sin().mean(0)  # (S, K)

        # Compare to N(0,1) CF = exp(-t²/2)
        err = (cos_mean - self.phi).square() + sin_mean.square()  # (S, K)
        statistic = (err @ self.weights) * z.size(0)  # (S,)
        return statistic.mean()


# ═══════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════

_sigreg_default: LeJEPASIGReg | None = None


def lejepa_sigreg(z: Tensor) -> Tensor:
    """Module-level convenience: compute LeJEPA SIGReg on latent vectors."""
    global _sigreg_default
    if _sigreg_default is None or _sigreg_default.t.device != z.device:
        _sigreg_default = LeJEPASIGReg().to(z.device)
    return _sigreg_default(z)


# Keep old moment-matching for comparison
def sigreg_moments(z: Tensor) -> Tensor:
    mean = z.mean(dim=0)
    std = z.std(dim=0, unbiased=False)
    return mean.pow(2).sum() + (std - 1.0).pow(2).sum()
