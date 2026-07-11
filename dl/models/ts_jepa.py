"""TS-JEPA: Time-Series Joint Embedding Predictive Architecture.

Patch-based self-supervised learning for time series.  Tokenizes the
sequence into contiguous patches via Conv1D, masks entire patches, and
predicts their latent representations from unmasked ones.

Key differences from T-JEPA (tabular):
  - Patch tokenization preserves local temporal structure
  - Block masking (whole patches) instead of per-feature masking
  - No [REG] tokens needed — masking is structural, not columnar
  - Predictor concatenates context + mask tokens (I-JEPA style)
  - EMA updated per batch, not per epoch

Reference:
    "Joint Embeddings Go Temporal" (TS-JEPA) — NeurIPS 2024 Workshop
    arxiv.org/abs/2509.25449
    Official code: github.com/Sennadir/TS_JEPA

"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════════════════
# Patch tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class PatchTokenizer(nn.Module):
    """Conv1D-based patching: splits a sequence into non-overlapping patches.

    For a 10-step RSSI sequence with patch_size=2, stride=2 → 5 patches.
    Each scalar timestep becomes a patch of `patch_size` values.
    """

    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = seq_len // patch_size

        # Conv1D over raw sequence to create patch embeddings
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len) or (B, seq_len, 1)  →  (B, num_patches, embed_dim)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        # Conv1D expects (B, C, L)
        x = x.permute(0, 2, 1)  # (B, 1, L)
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, embed_dim)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Sin/cos positional encoding
# ═══════════════════════════════════════════════════════════════════════════════

class SinCosPosEmbed(nn.Module):
    """Fixed sin/cos positional embedding for patch positions."""

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        pe = torch.zeros(num_patches, embed_dim)
        pos = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        omega = torch.arange(embed_dim // 2, dtype=torch.float)
        omega = 1.0 / (10000.0 ** (omega / (embed_dim // 2)))
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, num_patches, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


# ═══════════════════════════════════════════════════════════════════════════════
# Mask utilities — patch-index based (faithful to TS-JEPA paper)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_mask_by_index(x: Tensor, indices: Tensor) -> Tensor:
    """Gather patches at given indices from x.

    Args:
        x: (B, N, D) — all patches
        indices: (B, K) — indices of patches to KEEP (not mask)

    Returns:
        (B, K, D)
    """
    B = x.size(0)
    out_list = [x[b, indices[b], :] for b in range(B)]
    return torch.stack(out_list, dim=0)


def generate_patch_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.5,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Generate mask and non-mask indices for TS-JEPA pretraining.

    Randomly selects `mask_ratio` fraction of patches to mask.

    Args:
        batch_size: B
        num_patches: N (total patches per sample)
        mask_ratio: fraction of patches to mask (0.0–1.0)

    Returns:
        mask_indices: (B, M) — indices of MASKED patches
        non_mask_indices: (B, K) — indices of UNMASKED (context) patches
        where M + K = N
    """
    n_masked = max(1, int(num_patches * mask_ratio))

    mask_list: list[Tensor] = []
    non_mask_list: list[Tensor] = []

    for _ in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        mask_idx = perm[:n_masked].sort().values
        non_mask_idx = perm[n_masked:].sort().values
        mask_list.append(mask_idx)
        non_mask_list.append(non_mask_idx)

    mask_indices = torch.stack(mask_list)  # (B, M)
    non_mask_indices = torch.stack(non_mask_list)  # (B, K)
    return mask_indices, non_mask_indices


# ═══════════════════════════════════════════════════════════════════════════════
# TS-JEPA Encoder (context and target share this architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAEncoder(nn.Module):
    """Transformer encoder for TS-JEPA. Used for both context and target.

    Context mode: receives unmasked patches only (mask via index).
    Target mode: receives ALL patches (no mask).
    """

    def __init__(
        self,
        seq_len: int = 10,
        patch_size: int = 2,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        num_patches = seq_len // patch_size

        self.tokenizer = PatchTokenizer(
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=in_channels,
        )
        self.pos_embed = SinCosPosEmbed(num_patches, embed_dim)

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
        self,
        x: Tensor,
        keep_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode the sequence.

        Args:
            x: (B, seq_len) or (B, seq_len, C) — raw sequence
            keep_indices: (B, K) — indices of patches to keep.
                          If None, all patches are kept (target mode).

        Returns:
            (B, K, embed_dim) or (B, N, embed_dim) — encoded patches
        """
        # Tokenize
        patches = self.tokenizer(x)  # (B, N, embed_dim)
        patches = self.pos_embed(patches)

        # Apply mask by index if provided
        if keep_indices is not None:
            patches = _apply_mask_by_index(patches, keep_indices)  # (B, K, D)

        # Transformer
        out = self.transformer(patches)
        out = self.norm(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# TS-JEPA Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAPredictor(nn.Module):
    """Predictor: takes context latents + mask tokens, predicts target latents.

    Concatenates context embeddings with learned mask tokens at the correct
    patch positions, runs through a Transformer, and extracts only the
    masked-position outputs.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int = 128,
        predictor_embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Project encoder output down to predictor dimension
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Sin/cos positional embedding for full patch grid
        self.pos_embed = SinCosPosEmbed(num_patches, predictor_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer
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
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder dimension for L2 loss
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(
        self,
        ctx_encoded: Tensor,
        mask_indices: Tensor,
        non_mask_indices: Tensor,
    ) -> Tensor:
        """Predict target latents at masked positions.

        Args:
            ctx_encoded: (B, K, encoder_embed_dim) — context encoder output
            mask_indices: (B, M) — indices of masked patches
            non_mask_indices: (B, K) — indices of unmasked patches

        Returns:
            (B, M, encoder_embed_dim) — predictions at masked positions
        """
        B = ctx_encoded.size(0)
        M = mask_indices.size(1)
        K = non_mask_indices.size(1)

        # Project to predictor dimension
        h = self.input_proj(ctx_encoded)  # (B, K, pred_dim)

        # Build positional embeddings for context tokens
        pos_all = self.pos_embed.pe.expand(B, -1, -1)  # (B, N, pred_dim)
        ctx_pos = _apply_mask_by_index(pos_all, non_mask_indices)  # (B, K, pred_dim)

        # Build positional embeddings for masked tokens
        mask_pos = _apply_mask_by_index(pos_all, mask_indices)  # (B, M, pred_dim)

        # Expand mask tokens and add positional embeddings
        pred_tokens = self.mask_token.expand(B, M, -1) + mask_pos

        # Add positional embeddings to context
        h = h + ctx_pos

        # Concatenate: context first, then mask tokens
        x = torch.cat([h, pred_tokens], dim=1)  # (B, K+M, pred_dim)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Extract only the mask-token positions (last M positions)
        x = x[:, K:, :]  # (B, M, pred_dim)

        # Project back to encoder dimension
        x = self.output_proj(x)  # (B, M, encoder_embed_dim)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Full TS-JEPA model
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAModel(nn.Module):
    """Full TS-JEPA architecture for pretraining.

    Usage (pretraining):
        model = TSJEPAModel(seq_len=10, patch_size=2)
        for x_batch in loader:
            loss = model.pretrain_step(x_batch, ema_momentum=0.996)

    Usage (fine-tuning):
        encoder = model.context_encoder
        # Add classification head: pool patch embeddings → Linear(embed_dim, 4)
    """

    def __init__(
        self,
        seq_len: int = 10,
        patch_size: int = 2,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        pred_dim: int = 64,
        pred_num_layers: int = 2,
        mask_ratio: float = 0.5,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_start

        # Context encoder (trained via gradients)
        self.context_encoder = TSJEPAEncoder(
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            in_channels=in_channels,
        )

        # Target encoder (EMA of context)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = TSJEPAPredictor(
            num_patches=self.num_patches,
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=pred_dim,
            nhead=nhead,
            num_layers=pred_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update of target encoder from context encoder."""
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pretraining forward pass.

        Args:
            x: (B, seq_len) raw RSSI sequence

        Returns:
            target_latents: (B, M, embed_dim) — target encoder output at masked positions
            predictions: (B, M, embed_dim) — predictor output
            mask_indices: (B, M) — indices of masked patches
            non_mask_indices: (B, K) — indices of context patches
        """
        B = x.size(0)
        device = x.device

        # Generate masks
        mask_indices, non_mask_indices = generate_patch_masks(
            B, self.num_patches, self.mask_ratio, device
        )

        # Target encoder: encode ALL patches (no mask), then extract masked
        with torch.no_grad():
            tgt_all = self.target_encoder(x, keep_indices=None)  # (B, N, D)
            # Layer-normalize targets (as in reference implementation)
            tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
            tgt_masked = _apply_mask_by_index(tgt_all, mask_indices)  # (B, M, D)

        # Context encoder: encode only unmasked patches
        ctx_encoded = self.context_encoder(x, keep_indices=non_mask_indices)  # (B, K, D)

        # Predictor: predict masked latents from context
        predictions = self.predictor(ctx_encoded, mask_indices, non_mask_indices)  # (B, M, D)

        return tgt_masked, predictions, mask_indices, non_mask_indices

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Single pretraining step.

        Args:
            x: (B, seq_len)
            epoch: current epoch
            total_epochs: total pretraining epochs

        Returns:
            loss, metrics dict
        """
        # Update EMA momentum (cosine schedule, per paper)
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        # Forward
        tgt_masked, predictions, _mask_idx, _non_mask_idx = self.forward(x)

        # L1 loss (as in reference code) — L2 also works
        loss = F.l1_loss(predictions, tgt_masked)

        # EMA update (per batch, as in reference code)
        self._update_target_encoder()

        return loss, {"pretrain_loss": loss.item()}


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAClassifier(nn.Module):
    """Fine-tuned TS-JEPA for supervised classification.

    Takes a pretrained TSJEPAModel, discards predictor and target encoder,
    keeps the context encoder, pools patch embeddings, and classifies.
    """

    def __init__(
        self,
        pretrained: TSJEPAModel,
        num_classes: int = 4,
        pooling: str = "mean",
        hidden_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.embed_dim = pretrained.embed_dim
        self.pooling = pooling

        in_dim = self.embed_dim
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len) → (B, num_classes)."""
        latents = self.encoder(x, keep_indices=None)  # (B, N, D)

        if self.pooling == "mean":
            pooled = latents.mean(dim=1)
        elif self.pooling == "max":
            pooled = latents.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(pooled)
