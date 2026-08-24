"""LeJEPA: end-to-end latent next-step prediction with SIGReg.

Adaptation of LeWorldModel (Maes et al., 2026) to tabular RSSI sequences.
Unlike T-JEPA/TS-JEPA, there is no EMA, no stop-gradient, no target encoder,
and no [REG] tokens. The encoder and predictor are trained jointly with
gradients flowing through both, using only two terms:

    loss = ||pred(z_t) - z_{t+1}||^2 + lambda * SIGReg(z)

SIGReg pushes the latent embeddings toward N(0, I) to prevent collapse.

Reference:
    Maes et al. (2026) "LeWorldModel" — arXiv:2603.19312
    Balestriero & LeCun (2025) "LeJEPA" — arXiv:2511.08544
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LeJEPAEncoder(nn.Module):
    """Tokenize timesteps, encode with a transformer, project with BN.

    The projection head (Linear + BatchNorm) bypasses the transformer's final
    LayerNorm, which the LeWM paper notes is required for SIGReg to optimize.
    """

    def __init__(
        self,
        seq_len: int = 10,
        in_channels: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.tokenizer = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Projection head with BatchNorm (paper's anti-collapse requirement)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, seq_len, d_model)."""
        B, T, _ = x.shape
        z = self.tokenizer(x) + self.pos_embed[:, :T, :]
        z = self.transformer(z)
        z = self.proj(z)
        z = z.transpose(1, 2)  # (B, d_model, T)
        z = self.bn(z)
        z = z.transpose(1, 2)  # (B, T, d_model)
        return z


class LeJEPAPredictor(nn.Module):
    """Causal transformer predicting the next-step latent embedding."""

    def __init__(
        self,
        seq_len: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, T, d_model) → (B, T, d_model), causal next-step predictions."""
        B, T, _ = z.shape
        causal = torch.triu(torch.ones(T, T, device=z.device), diagonal=1).bool()
        h = self.input_proj(z)
        h = self.transformer(h, mask=causal)
        return self.output_proj(h)


class LeJEPAModel(nn.Module):
    """End-to-end next-step latent prediction. No EMA, stop-grad, or REG tokens."""

    def __init__(
        self,
        seq_len: int = 10,
        in_channels: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        pred_num_layers: int = 2,
        sigreg_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.sigreg_lambda = sigreg_lambda

        from dl.sigreg import LeJEPASIGReg
        self.sigreg = LeJEPASIGReg()

        self.encoder = LeJEPAEncoder(
            seq_len, in_channels, d_model, nhead,
            num_layers, dim_feedforward, dropout,
        )
        self.predictor = LeJEPAPredictor(
            seq_len, d_model, nhead,
            pred_num_layers, dim_feedforward, dropout=0.1,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B, T, C) → (z, z_hat), both (B, T, d_model)."""
        z = self.encoder(x)
        z_hat = self.predictor(z)
        return z, z_hat

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        z, z_hat = self.forward(x)

        # Next-step prediction: z_hat[:, t] should match z[:, t+1]
        pred_loss = F.mse_loss(z_hat[:, :-1], z[:, 1:])

        # SIGReg on all embeddings → N(0, I)
        B, T, D = z.shape
        sigreg_loss = self.sigreg(z.reshape(B * T, D))

        loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return loss, {
            "pretrain_loss": loss.item(),
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
        }

    @torch.no_grad()
    def get_latents(self, x: Tensor) -> Tensor:
        self.eval()
        z = self.encoder(x)
        return z.reshape(-1, self.d_model)


class LeJEPAClassifier(nn.Module):
    """Classifier built on the pretrained LeJEPA encoder."""

    def __init__(
        self,
        pretrained: LeJEPAModel,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.encoder
        self.attn = nn.Linear(pretrained.d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(pretrained.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C) → (B, num_classes)."""
        latents = self.encoder(x)  # (B, T, d_model)
        w = self.attn(latents).softmax(dim=1)
        pooled = (latents * w).sum(dim=1)
        return self.head(pooled)
