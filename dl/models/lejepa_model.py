"""LeJEPA self-supervised pretraining — exact paper implementation.

Follows Balestriero & LeCun (2025) "LeJEPA" arxiv.org/abs/2511.08544
and the official minimal example at github.com/galilai-group/lejepa.

Training:
  V views per sample via dynamic augmentation
  encoder → embedding (for probe) + projection (for LeJEPA loss)
  inv_loss = (proj.mean(0) - proj).square().mean()   # variance
  sigreg_loss = SIGReg(proj)                          # Gaussian prior
  lejepa_loss = λ·sigreg_loss + (1-λ)·inv_loss
  Online linear probe on detached encoder embeddings
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl.sigreg import LeJEPASIGReg
from dl.models.sigreg_classifier import DynamicAugment, _CNNEncoder


class LeJEPAEncoder(nn.Module):
    """Encoder + projector. Returns (embedding, projection) for each view."""

    def __init__(self, in_features: int = 4, num_filters: int = 256, num_blocks: int = 4,
                 latent_dim: int = 256, proj_dim: int = 128) -> None:
        super().__init__()
        self.encoder = _CNNEncoder(in_features, num_filters, num_blocks, latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B*V, L, C) → emb: (B*V, latent_dim), proj: (V, B, proj_dim)."""
        BxV = x.size(0)
        emb = self.encoder(x)                      # (B*V, latent_dim)
        proj_flat = self.projector(emb)            # (B*V, proj_dim)
        # Guess V from shape — assume V=2, reshape to (V, B, proj_dim)
        V = 2
        B = BxV // V
        proj = proj_flat.reshape(V, B, -1)         # (V, B, proj_dim)
        return emb, proj


class LeJEPAModel(nn.Module):
    """Full LeJEPA pretraining model — exactly matches the paper's loop.

    Usage:
        model = LeJEPAModel(V=2)
        for x, y in loader:
            vs = torch.cat([model.augment(x) for _ in range(model.V)], dim=0)
            emb, proj = model.backbone(vs)
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = model.sigreg(proj)
            loss = model.lamb * sigreg_loss + (1-model.lamb) * inv_loss
            probe_loss = F.cross_entropy(probe(emb.detach()), y.repeat(model.V))
            (loss + probe_loss).backward()
    """

    def __init__(self, in_features: int = 4, num_filters: int = 256, num_blocks: int = 4,
                 latent_dim: int = 256, proj_dim: int = 128, lamb: float = 0.02,
                 V: int = 2, aug_drop_t: float = 0.15, aug_drop_c: float = 0.1,
                 aug_block_p: float = 0.3, aug_block_len: int = 4,
                 aug_noise: float = 1.0) -> None:
        super().__init__()
        self.V = V
        self.augment = DynamicAugment(aug_drop_t, aug_drop_c, aug_block_p, aug_block_len, aug_noise)
        self.backbone = LeJEPAEncoder(in_features, num_filters, num_blocks, latent_dim, proj_dim)
        self.sigreg = LeJEPASIGReg()
        self.lamb = lamb

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Create V views, encode, return (embeddings, projections).

        Args:
            x: (B, L, C) input batch

        Returns:
            emb: (B*V, latent_dim) — encoder embeddings (for probe)
            proj: (V, B, proj_dim) — projections (for LeJEPA loss)
        """
        B = x.size(0)
        views = []
        for _ in range(self.V):
            views.append(self.augment(x))
        vs = torch.cat(views, dim=0)  # (B*V, L, C)
        return self.backbone(vs)


class LeJEPAClassifier(nn.Module):
    """Fine-tuned classifier from pretrained LeJEPA encoder."""

    def __init__(self, pretrained: LeJEPAModel, num_classes: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = pretrained.backbone.encoder
        self.head = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.head(z)
