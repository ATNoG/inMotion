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


# ═══════════════════════════════════════════════════════════════════
# Default regularizer mode (SigReg vs RDMReg)
# ═══════════════════════════════════════════════════════════════════
# The JEPA model files instantiate `LeJEPASIGReg` directly. To swap in
# RDMReg without editing every model, they can call this factory which
# respects a module-level mode switch set via set_default_sigreg_mode().
_USE_RDM = False
_RDM_P = 1.5


def make_sigreg(p: float | None = None) -> nn.Module:
    """Factory: returns LeJEPASIGReg or RDMReg depending on the module mode."""
    if _USE_RDM:
        return RDMReg(p=p if p is not None else _RDM_P)
    return LeJEPASIGReg()


def set_default_sigreg_mode(mode: str = "sigreg", p: float = 1.5) -> None:
    """Switch the default regularizer: 'sigreg' (Gaussian) or 'rdm' (sparse RGG)."""
    global _USE_RDM, _RDM_P
    _USE_RDM = (mode.lower() == "rdm")
    _RDM_P = p


# Keep old moment-matching for comparison
def sigreg_moments(z: Tensor) -> Tensor:
    mean = z.mean(dim=0)
    std = z.std(dim=0, unbiased=False)
    return mean.pow(2).sum() + (std - 1.0).pow(2).sum()


# ═══════════════════════════════════════════════════════════════════
# RDMReg — Rectified Distribution Matching Regularization
# ═══════════════════════════════════════════════════════════════════
# From "Rectified LpJEPA" (arXiv:2602.01456). Replaces SigReg's isotropic
# Gaussian target with a Rectified Generalized Gaussian (RGG) target:
#   features = ReLU(f_θ(x))           → sparse, non-negative
#   target   = ReLU(μ + σ·S·(pG)^{1/p})  with G ~ Gamma(1/p, 1)
# matched via a sliced 2-Wasserstein loss.
#
# Why it helps here: sparse non-negative embeddings are ideal inputs for
# GBDT/XGBoost/CatBoost (they already threshold), and sparsity regularizes
# the small dataset. p < 2 → sparser than Gaussian (p=2 recovers LeJEPA).

class RDMReg(nn.Module):
    """Rectified Distribution Matching regularizer (sparse, non-negative).

    Usage (drop-in for LeJEPASIGReg):
        reg = RDMReg(p=1.5, n_projections=256)
        loss += lam * reg(z)   # z: (B, D) latent vectors

    The features should be ReLU'd by the caller OR passed raw and rectified
    here (we apply ReLU internally for a drop-in experience).
    """

    def __init__(
        self,
        p: float = 1.5,          # RGG shape; p<2 → sparser than Gaussian
        sigma: float = 1.0,      # pre-rectification scale
        n_projections: int = 64,  # a handful of projections suffices (paper)
        n_samples: int = 512,    # target samples per projection
    ) -> None:
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.n_projections = n_projections
        self.n_samples = n_samples

    def _sample_rgg(self, batch: int, dim: int, device: torch.device) -> Tensor:
        """Sample from the Rectified Generalized Gaussian:
        x = ReLU(σ · S · (p·G)^{1/p})  where G ~ Gamma(1/p, 1), S ~ ±1.
        Samples directly on device (no CPU→GPU copy).
        """
        shape = (batch, dim, self.n_projections)
        g = torch.distributions.Gamma(1.0 / self.p, 1.0).sample(shape).to(device)
        s = torch.randint(0, 2, shape, device=device).float() * 2 - 1
        raw = self.sigma * s * (self.p * g) ** (1.0 / self.p)
        return torch.relu(raw)  # (batch, dim, n_proj)

    def _sliced_w2(self, a: Tensor, b: Tensor) -> Tensor:
        """Sliced 2-Wasserstein: sort projections, mean squared difference.

        a, b: (batch, dim, n_proj) — sorted along the projection axis.
        """
        a_sorted, _ = torch.sort(a, dim=0)
        b_sorted, _ = torch.sort(b, dim=0)
        return (a_sorted - b_sorted).pow(2).mean()

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, D) latent vectors → scalar RDMReg loss."""
        z = torch.relu(z)  # rectified features (sparse, non-negative)
        B, D = z.shape

        # Project to 1D slices: (B, D) → (B, D, n_projections) via random proj
        A = torch.randn(D, self.n_projections, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)
        slices = z @ A  # (B, n_projections)

        # Target RGG slices (same shape): (B, D, n_proj) → reduce D by pooling
        # We match the empirical marginal of the projections to the RGG target
        # marginal via a 1D sliced-W2 on the (B, n_proj) projections.
        target = self._sample_rgg(B, 1, z.device).squeeze(1)  # (B, n_proj)
        return self._sliced_w2(slices, target)


class RDMSigReg(nn.Module):
    """RDMReg + invariance: match rectified latents to RGG AND pull same-class
    latents together (VICReg-style invariance on the non-rectified space).

    This is the full LpJEPA recipe: invariance ℓ2 + RDMReg on both views.
    For single-branch usage (our case) the invariance term is a no-op unless
    two views are provided; we keep it for API compatibility and let the
    caller pass a single view (invariance weight 0).
    """

    def __init__(self, p: float = 1.5, sigma: float = 1.0, inv_weight: float = 0.0) -> None:
        super().__init__()
        self.rdm = RDMReg(p=p, sigma=sigma)
        self.inv_weight = inv_weight

    def forward(self, z: Tensor, z2: Tensor | None = None) -> Tensor:
        loss = self.rdm(z)
        if z2 is not None and self.inv_weight > 0:
            loss = loss + self.inv_weight * (z - z2).pow(2).mean()
        return loss
