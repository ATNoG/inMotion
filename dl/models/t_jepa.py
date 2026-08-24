"""T-JEPA: Tabular Joint-Embedding Predictive Architecture for RSSI sequences.

Treats each (timestep, channel) pair as a "feature" in a tabular dataset.
Masks a random subset of features, encodes the unmasked ones (context),
and predicts the latent representations of masked ones (target).

Key innovations from the paper:
  - [REG] tokens: learnable parameters that prevent representation collapse
  - EMA target encoder: no gradients through the target branch
  - Predictor bottleneck: shallower + narrower than encoder

Reference:
    Thimonier et al. (2024) "T-JEPA" — arXiv:2410.05016
    https://github.com/jose-melo/t-jepa
"""

from __future__ import annotations

import copy, math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════════════════
# Mask generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_masks(
    batch_size: int,
    n_timesteps: int,
    n_channels: int,
    mask_ratio: float,
    n_target_masks: int,
    device: torch.device,
) -> tuple[Tensor, list[Tensor]]:
    """Generate context + K target masks.

    Masks contiguous temporal blocks per channel — the model must infer
    missing time segments from surrounding timesteps and channels.
    (Contiguous beats random per-feature masking on this time-series data.)

    Returns:
        ctx_mask: (B, total) — 1=context
        tgt_masks: list of K (B, total) — 1=target
    """
    total = n_timesteps * n_channels
    ctx_mask = torch.zeros(batch_size, total, device=device)
    tgt_masks = [torch.zeros(batch_size, total, device=device)
                 for _ in range(n_target_masks)]

    n_mask = max(1, int(n_timesteps * mask_ratio))
    for b in range(batch_size):
        start = torch.randint(0, n_timesteps - n_mask + 1, (1,)).item()
        tgt = torch.zeros(n_timesteps, n_channels, device=device)
        tgt[start : start + n_mask, :] = 1.0
        tgt_flat = tgt.reshape(-1)
        ctx_mask[b] = 1.0 - tgt_flat
        tgt_masks[0][b] = tgt_flat

        # Extra target masks (if n_target_masks > 1): different block
        for k in range(1, n_target_masks):
            start = torch.randint(0, n_timesteps - n_mask + 1, (1,)).item()
            tgt_k = torch.zeros(n_timesteps, n_channels, device=device)
            tgt_k[start : start + n_mask, :] = 1.0
            tgt_masks[k][b] = tgt_k.reshape(-1)

    return ctx_mask, tgt_masks


# ═══════════════════════════════════════════════════════════════════════════════
# Learnable [REG] tokens — the paper's key collapse-prevention mechanism
# ═══════════════════════════════════════════════════════════════════════════════

class RegularizationTokens(nn.Module):
    """Learnable [REG] tokens.

    Appended to the context input. Not masked, not predicted. They act as a
    "pressure release valve": if the encoder tries to collapse, information
    leaks into [REG] tokens instead, and the loss penalizes it.

    Reference: T-JEPA paper, Section 3.3.
    """

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature tokenizer — maps each (timestep, channel) to a d_model embedding
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Per-feature linear projection + feature-index embedding.

    Input:  (B, n_timesteps, n_channels)
    Output: (B, n_reg + n_timesteps*n_channels_kept, d_model)
    """

    def __init__(
        self,
        n_timesteps: int,
        n_channels: int,
        d_model: int,
        n_reg_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.total_features = n_timesteps * n_channels
        self.d_model = d_model
        self.n_reg_tokens = n_reg_tokens

        # Per-feature projection weights
        self.weight = nn.Parameter(
            torch.empty(n_reg_tokens + self.total_features, d_model)
        )
        self.bias = nn.Parameter(torch.zeros(self.total_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Feature-index embedding (which timestep? which channel?)
        self.feat_idx_embed = nn.Embedding(self.total_features, d_model)

        # [REG] tokens — LEARNABLE (not constant ones)
        self.reg_tokens = RegularizationTokens(n_reg_tokens, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, n_timesteps, n_channels) → (B, n_kept, d_model).

        When mask is provided, only unmasked features are kept.
        [REG] tokens always prepended and never masked.
        """
        B, T, C = x.shape
        x_flat = x.reshape(B, T * C)  # (B, total_features)

        # Prepend [REG] token values (ones as input value, weight matrix projects)
        reg_input = torch.ones(B, self.n_reg_tokens, device=x.device, dtype=x.dtype)
        x_with_reg = torch.cat([reg_input, x_flat], dim=1)  # (B, n_reg + total)

        # Linear projection (each position has its own weight row)
        out = self.weight[None] * x_with_reg[:, :, None]  # (B, N, d_model)
        bias_pad = torch.cat([
            torch.zeros(self.n_reg_tokens, self.d_model, device=x.device),
            self.bias,
        ], dim=0)
        out = out + bias_pad[None]

        # Add feature-index embeddings
        feat_idx_emb = self.feat_idx_embed(
            torch.arange(self.total_features, device=x.device)
        )
        feat_idx_emb = torch.cat([
            torch.zeros(self.n_reg_tokens, self.d_model, device=x.device),
            feat_idx_emb,
        ], dim=0)
        out = out + feat_idx_emb[None]

        # Apply mask (if provided) — vectorized: zero masked tokens and
        # mark them as padding for the transformer (no per-sample gather)
        if mask is not None:
            reg_mask = torch.ones(B, self.n_reg_tokens, dtype=mask.dtype, device=mask.device)
            full_mask = torch.cat([reg_mask, mask], dim=1)  # (B, n_reg + total)
            out = out * full_mask.unsqueeze(-1)
            self._last_key_padding = (full_mask == 0)  # True = padded/masked

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder — shared architecture for context and target branches
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAEncoder(nn.Module):
    """Transformer encoder for T-JEPA context/target branch."""

    def __init__(
        self,
        n_timesteps: int = 10,
        n_channels: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        n_reg_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_reg_tokens = n_reg_tokens

        self.tokenizer = FeatureTokenizer(n_timesteps, n_channels, d_model, n_reg_tokens)

        # Simple linear projection as positional encoding replacement
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, n_timesteps, n_channels) → (B, n_kept, d_model)."""
        out = self.tokenizer(x, mask=mask)
        out = self.input_proj(out)
        # Only use key_padding when mask was applied (otherwise it's stale)
        key_pad = getattr(self.tokenizer, "_last_key_padding", None) if mask is not None else None
        out = self.transformer(out, src_key_padding_mask=key_pad)
        return self.norm(out)

# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAPredictor(nn.Module):
    """Predicts target latents from context latents.

    Critical: MUST be shallower and narrower than the encoder.
    This bottleneck forces the predictor to learn a compressed mapping,
    which is the key insight of JEPA architectures.
    """

    def __init__(
        self,
        total_features: int,
        d_model: int = 256,
        pred_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        n_reg_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.n_reg_tokens = n_reg_tokens

        # Input projection: encoder dim → predictor dim
        self.input_proj = nn.Linear(d_model, pred_dim)

        # Mask tokens: learned embeddings for positions to be predicted
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)

        # Positional embedding (fixed sincos, as in official T-JEPA).
        # CRITICAL: mask tokens are identical across positions; without
        # position info the predictor cannot tell WHICH feature it predicts
        # and degenerates to predicting the mean latent everywhere.
        self.pos_embed = nn.Parameter(
            self._sincos_pos_embed(n_reg_tokens + total_features, pred_dim),
            requires_grad=False,
        )

        # Predictor transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
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

        # Output projection: predictor dim → encoder dim
        self.output_proj = nn.Linear(pred_dim, d_model)

    @staticmethod
    def _sincos_pos_embed(num_positions: int, dim: int) -> Tensor:
        """Fixed 1D sin/cos positional embedding (same as official T-JEPA)."""
        pos = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        denom = 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(pos / denom)
        pe[:, 1::2] = torch.cos(pos / denom)
        return pe.unsqueeze(0)  # (1, num_positions, dim)

    def forward(
        self,
        ctx_latents: Tensor,
        ctx_mask: Tensor,
        tgt_mask: Tensor,
    ) -> list[Tensor]:
        """Predict target latents from context.

        Args:
            ctx_latents: (B, n_ctx, d_model) — context encoder output
            ctx_mask: (B, total_features) — which features were context
            tgt_mask: (B, total_features) — which features to predict

        Returns:
            List of (B, total_features, d_model) predictions
        """
        B = ctx_latents.size(0)
        reg = self.n_reg_tokens
        pred_dim = self.mask_token.size(-1)

        # Context latents are already at absolute positions (encoder keeps
        # full sequence, masked positions zeroed). Project to predictor dim.
        ctx_proj = self.input_proj(ctx_latents)  # (B, total_positions, pred_dim)

        # Build full sequence: mask tokens everywhere, context where available.
        # torch.where = vectorized, no in-place on leaf tensors.
        total_positions = reg + ctx_mask.size(1)
        full_seq = self.mask_token.expand(B, total_positions, -1) * 1.0
        ctx_flag = torch.cat(
            [torch.ones(B, reg, device=ctx_mask.device, dtype=ctx_mask.dtype), ctx_mask],
            dim=1,
        ).unsqueeze(-1)  # (B, total_positions, 1): 1=context
        full_seq = torch.where(ctx_flag > 0, ctx_proj, full_seq)

        # Add positional embedding — each mask token now knows which
        # feature it represents (critical for the predictor to work)
        full_seq = full_seq + self.pos_embed

        # Transformer forward
        out = self.transformer(full_seq)  # (B, total_positions, pred_dim)
        out = self.output_proj(out)       # (B, total_positions, d_model)

        # Extract target predictions (strip REG tokens)
        predictions = out[:, reg:, :]     # (B, total_features, d_model)
        return [predictions]


# ═══════════════════════════════════════════════════════════════════════════════
# Full T-JEPA pretraining model
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAModel(nn.Module):
    """T-JEPA architecture for self-supervised pretraining.

    Usage:
        model = TJEPAModel()
        for x_batch in loader:  # x: (B, 10, 4)
            loss, metrics = model.pretrain_step(x_batch, epoch, total_epochs)
    """

    def __init__(
        self,
        n_timesteps: int = 10,
        n_channels: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        n_reg_tokens: int = 2,
        pred_dim: int = 128,
        pred_num_layers: int = 2,
        mask_ratio: float = 0.3,
        n_target_masks: int = 1,
        sigreg_lambda: float = 0.0,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.total_features = n_timesteps * n_channels
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.n_target_masks = n_target_masks
        self.sigreg_lambda = sigreg_lambda
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.ema_momentum = ema_start

        # Anti-collapse regularizer (LeWM mechanism): pushes the context
        # latent distribution toward N(0,I), preventing the encoder from
        # collapsing to a low-rank cluster (the T-JEPA failure mode).
        if sigreg_lambda > 0:
            from dl.sigreg import make_sigreg
            self.sigreg = make_sigreg()

        # Context encoder — trained by gradient descent
        self.context_encoder = TJEPAEncoder(
            n_timesteps, n_channels, d_model, nhead, num_layers,
            dim_feedforward, dropout, n_reg_tokens,
        )

        # Target encoder — updated via EMA of context encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor — shallower, narrower
        self.predictor = TJEPAPredictor(
            total_features=n_timesteps * n_channels,
            d_model=d_model, pred_dim=pred_dim, nhead=nhead,
            num_layers=pred_num_layers, dim_feedforward=dim_feedforward // 2,
            dropout=dropout, n_reg_tokens=n_reg_tokens,
        )

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update of target encoder weights."""
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor], Tensor]:
        """Pretraining forward pass.

        Args:
            x: (B, n_timesteps, n_channels)

        Returns:
            tgt_latents: (B, total_features, d_model)
            predictions: list of (B, total_features, d_model) — one per target mask
            ctx_mask: (B, total_features) — 1=context
            tgt_masks: list of K (B, total_features) — 1=target
            ctx_latents: (B, n_kept, d_model) — context encoder output
        """
        B, device = x.size(0), x.device

        ctx_mask, tgt_masks = generate_masks(
            B, self.n_timesteps, self.n_channels,
            self.mask_ratio, self.n_target_masks, device,
        )

        # Target encoder: encode ALL features → (B, n_reg + total, d_model)
        with torch.no_grad():
            tgt_full = self.target_encoder(x, mask=None)
            tgt_latents = tgt_full[:, self.context_encoder.n_reg_tokens:, :]

        # Context encoder: encode only unmasked features
        ctx_latents = self.context_encoder(x, mask=ctx_mask)

        # Predictor: predict target latents for each target mask
        predictions = [
            self.predictor(ctx_latents, ctx_mask, tgt_mask)[0]
            for tgt_mask in tgt_masks
        ]

        return tgt_latents, predictions, ctx_mask, tgt_masks, ctx_latents

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Single pretraining step: forward + loss + EMA update."""
        # Cosine EMA schedule
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        tgt, preds, _, tgt_masks, ctx_latents = self.forward(x)

        # L2 loss on unit-norm latents (both sides) — keeps the objective
        # bounded instead of letting latent magnitude drift upward.
        loss = torch.tensor(0.0, device=x.device)
        n = 0
        for p, tgt_mask in zip(preds, tgt_masks):
            p = F.normalize(p, dim=-1)
            tgt_n = F.normalize(tgt, dim=-1)
            diff = (p - tgt_n).pow(2).sum(dim=-1)
            loss = loss + (diff * tgt_mask).sum()
            n += tgt_mask.sum()
        loss = loss / max(n, 1)

        # Anti-collapse: SIGReg on pooled context latents (if enabled)
        if self.sigreg_lambda > 0:
            pooled = ctx_latents.mean(dim=1)  # (B, d_model)
            loss = loss + self.sigreg_lambda * self.sigreg(pooled)

        return loss, {"pretrain_loss": loss.item()}

    @torch.no_grad()
    def get_latents(self, x: Tensor) -> Tensor:
        """Extract context encoder latents for visualization/monitoring."""
        self.eval()
        latents = self.context_encoder(x, mask=None)
        # Strip REG tokens, flatten to (B*total_features, d_model)
        latents = latents[:, self.context_encoder.n_reg_tokens:, :]
        return latents.reshape(-1, self.d_model)


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAClassifier(nn.Module):
    """Classifier built on pretrained T-JEPA context encoder.

    Input: (B, n_timesteps, n_channels) → logits: (B, num_classes)
    """

    def __init__(
        self,
        pretrained: TJEPAModel,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.attn = nn.Linear(pretrained.d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(pretrained.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, n_timesteps, n_channels) → (B, num_classes)."""
        latents = self.encoder(x, mask=None)  # (B, n_reg+total, d_model)
        # Pool all features except REG tokens with learned attention
        latents = latents[:, self.encoder.n_reg_tokens:, :]  # (B, total, d_model)
        w = self.attn(latents).softmax(dim=1)  # (B, total, 1)
        pooled = (latents * w).sum(dim=1)  # (B, d_model)
        return self.head(pooled)
