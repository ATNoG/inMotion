"""CF-JEPA: Mask-free multi-horizon forward prediction for time series.

Adaptation of CF-JEPA (arXiv:2606.07031) to short RSSI sequences.
Replaces masking with multi-horizon *forward* prediction: a random crop of
the series serves as context, and the model predicts the latent embeddings of
the *future* at multiple horizons. This exploits temporal ordering directly
instead of destroying it with masks — critical when the sequence is only
10 steps long (masking 30-70% leaves almost no context).

Key design (from the paper):
  - Random context crop of the encoded patch sequence
  - Predictor maps context latents → future latents at short/mid/long horizons
  - Horizon annealing: predictions start short, horizon lengthens during training
  - EMA target encoder; online encoder is best for classification (we use the
    online encoder in the fine-tuned head)
  - Optional SIGReg anti-collapse on the pooled context latents

Reference:
    "CF-JEPA: Mask-free forward prediction" — arXiv:2606.07031
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CFJEPAPatchTokenizer(nn.Module):
    """Conv1D patch tokenizer for multi-channel time series."""

    def __init__(
        self,
        seq_len: int = 10,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, num_patches, embed_dim)."""
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class CFJEPAEncoder(nn.Module):
    """Transformer encoder over Conv1D patches (shared online/target)."""

    def __init__(
        self,
        seq_len: int = 10,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.tokenizer = CFJEPAPatchTokenizer(seq_len, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Tensor, keep_indices: Tensor | None = None
    ) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, n_kept, embed_dim).

        When keep_indices is not None, only those patches are encoded
        (context crop path). When None, all patches (target path).
        """
        patches = self.tokenizer(x)
        patches = patches + self.pos_embed[:, : patches.size(1), :]

        if keep_indices is not None:
            patches = torch.stack(
                [patches[b, keep_indices[b], :] for b in range(patches.size(0))], dim=0
            )

        out = self.transformer(patches)
        return self.norm(out)


class CFJEPAPredictor(nn.Module):
    """Predicts future latent embeddings from context latents at multiple horizons.

    The predictor attends over the context crop and outputs predictions for
    each future patch position (one output per future position). Horizon
    annealing is implemented at the loss level (which outputs get masked out).
    """

    def __init__(
        self,
        num_patches: int = 5,
        encoder_embed_dim: int = 256,
        predictor_embed_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.predictor_embed_dim = predictor_embed_dim

        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Query tokens: one per future position to predict.
        # The predictor outputs predictions for ALL positions ≥ 1; the loss
        # masks out positions beyond the current horizon (annealing).
        self.query_tokens = nn.Parameter(torch.randn(1, num_patches - 1, predictor_embed_dim) * 0.02)

        # Future-position embeddings — each query knows which future step it predicts
        self.future_pos_embed = nn.Parameter(
            torch.randn(1, num_patches - 1, predictor_embed_dim) * 0.02
        )

        # Context position embedding for the input projection
        self.ctx_pos_embed = nn.Parameter(torch.randn(1, num_patches, predictor_embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
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

        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(self, ctx_latents: Tensor) -> Tensor:
        """Predict future latents from a context crop.

        Args:
            ctx_latents: (B, K, embed_dim) — online encoder output for the crop

        Returns:
            predictions: (B, num_patches - 1, embed_dim) — predicted latents
            for future positions (output i predicts patch position ctx_end + i)
        """
        B, K, _ = ctx_latents.shape
        pred_dim = self.predictor_embed_dim
        num_future = self.num_patches - 1

        # Project context latents and add context-position embedding
        ctx_proj = self.input_proj(ctx_latents) + self.ctx_pos_embed[:, :K, :]

        # Query tokens for future positions
        queries = self.query_tokens.expand(B, -1, -1) + self.future_pos_embed

        # Concatenate: context then queries → transformer
        full = torch.cat([ctx_proj, queries], dim=1)  # (B, K + num_future, pred_dim)
        out = self.transformer(full)

        # Extract future predictions (the last num_future positions)
        future_out = out[:, K:, :]  # (B, num_future, pred_dim)
        return self.output_proj(future_out)  # (B, num_future, embed_dim)


class CFJEPAModel(nn.Module):
    """CF-JEPA architecture for self-supervised pretraining.

    Usage:
        model = CFJEPAModel()
        for x_batch in loader:  # x: (B, 10, 4)
            loss, metrics = model.pretrain_step(x_batch, epoch, total_epochs)
    """

    def __init__(
        self,
        seq_len: int = 10,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        pred_dim: int = 128,
        pred_num_layers: int = 2,
        horizon_start: int = 1,
        horizon_end: int = 3,
        ctx_min: int = 1,
        ctx_max: int = 3,
        ema_start: float = 0.996,
        ema_end: float = 0.999,
        sigreg_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.horizon_start = horizon_start
        self.horizon_end = horizon_end
        self.ctx_min = min(ctx_min, self.num_patches - 1)
        self.ctx_max = min(ctx_max, self.num_patches - 1)
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.ema_momentum = ema_start
        self.sigreg_lambda = sigreg_lambda

        if sigreg_lambda > 0:
            from dl.sigreg import make_sigreg
            self.sigreg = make_sigreg()

        self.context_encoder = CFJEPAEncoder(
            seq_len, patch_size, in_channels, embed_dim, nhead,
            num_layers, dim_feedforward, dropout,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = CFJEPAPredictor(
            num_patches=self.num_patches,
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=pred_dim,
            nhead=nhead,
            num_layers=pred_num_layers,
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
        )

    def _horizon(self, epoch: int, total_epochs: int) -> int:
        """Horizon annealing: linear from horizon_start → horizon_end."""
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        h = self.horizon_start + (self.horizon_end - self.horizon_start) * progress
        return max(1, int(round(h)))

    def _sample_context(self, B: int, device: torch.device) -> Tensor:
        """Sample a random context crop length per sample."""
        # Uniform over [ctx_min, ctx_max]; longer crops = easier task
        ctx_len = torch.randint(self.ctx_min, self.ctx_max + 1, (B,), device=device)
        # Context always starts at patch 0 (causal: predict the future from the past)
        return ctx_len

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def forward(
        self, x: Tensor, epoch: int = 0, total_epochs: int = 100
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pretraining forward pass.

        Returns:
            tgt_future: (B, num_future, embed_dim) — target latents for future
            predictions: (B, num_future, embed_dim) — predicted future latents
            ctx_latents: (B, K, embed_dim) — online encoder crop latents
            horizon: int — the current (annealed) horizon
        """
        B, device = x.size(0), x.device
        num_patches = self.num_patches
        horizon = self._horizon(epoch, total_epochs)

        ctx_len = self._sample_context(B, device)  # (B,)

        # Online encoder: encode all patches, keep the first ctx_len per sample
        all_latents = self.context_encoder(x, keep_indices=None)  # (B, N, D)
        max_ctx = int(ctx_len.max().item())
        ctx_latents = torch.zeros(B, max_ctx, self.embed_dim, device=device)
        for b in range(B):
            cl = int(ctx_len[b])
            ctx_latents[b, :cl, :] = all_latents[b, :cl, :]
        # (B, max_ctx, embed_dim) — shorter contexts are zero-padded (harmless:
        # the predictor's queries attend to all context tokens, and zeros carry
        # no signal)

        # Target encoder: encode all patches, extract the future positions
        with torch.no_grad():
            tgt_all = self.target_encoder(x, keep_indices=None)  # (B, N, D)

        # Predict future positions 1..num_patches-1 beyond the context end
        num_future = self.num_patches - 1
        predictions = self.predictor(ctx_latents)  # (B, num_future, D)

        # Build per-sample target: tgt_future[b, i] = tgt_all[b, ctx_len[b] + i]
        # for i in 0..horizon-1 (and beyond → masked by horizon mask).
        targets = torch.zeros(B, num_future, self.embed_dim, device=device)
        for b in range(B):
            cl = int(ctx_len[b])
            avail = min(num_future, num_patches - cl)
            for i in range(avail):
                targets[b, i, :] = tgt_all[b, cl + i, :]

        # Horizon mask: positions i < horizon and within available future
        horizon_mask = torch.zeros(B, num_future, device=device)
        for b in range(B):
            cl = int(ctx_len[b])
            avail = min(horizon, num_patches - cl)
            horizon_mask[b, :avail] = 1.0

        return targets, predictions, ctx_latents, horizon_mask

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Single pretraining step: multi-horizon forward prediction loss."""
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        targets, predictions, ctx_latents, horizon_mask = self.forward(x, epoch, total_epochs)

        # L1 loss on normalized latents at predicted positions (masked by horizon)
        tgt_n = F.normalize(targets, dim=-1)
        pred_n = F.normalize(predictions, dim=-1)
        diff = (pred_n - tgt_n).abs().mean(dim=-1)  # (B, num_future)
        diff = (diff * horizon_mask).sum() / max(horizon_mask.sum(), 1)

        loss = diff
        if self.sigreg_lambda > 0:
            pooled = ctx_latents.mean(dim=1)
            loss = loss + self.sigreg_lambda * self.sigreg(pooled)

        return loss, {"pretrain_loss": loss.item()}

    @torch.no_grad()
    def get_latents(self, x: Tensor) -> Tensor:
        """Extract online encoder latents for visualization."""
        self.eval()
        latents = self.context_encoder(x, keep_indices=None)
        return latents.reshape(-1, self.embed_dim)


class CFJEPAClassifier(nn.Module):
    """Classifier built on the CF-JEPA online encoder.

    Uses the *online* encoder (the paper's finding: it is best for
    classification) with learned attention pooling over patches.
    """

    def __init__(
        self,
        pretrained: CFJEPAModel,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.attn = nn.Linear(pretrained.embed_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(pretrained.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, num_classes)."""
        latents = self.encoder(x, keep_indices=None)  # (B, num_patches, embed_dim)
        w = self.attn(latents).softmax(dim=1)
        pooled = (latents * w).sum(dim=1)
        return self.head(pooled)
