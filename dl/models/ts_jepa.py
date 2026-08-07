"""TS-JEPA: Time-Series Joint Embedding Predictive Architecture.

Masks contiguous temporal blocks in latent space (after Conv1D patching)
and trains a predictor to recover the masked latents from unmasked context.

Key design:
  - Conv1D patching over multi-channel input (all 4 engineered features)
  - Contiguous temporal block masking (forces inference of missing segments)
  - EMA target encoder (no gradients through target branch)
  - Predictor bottleneck (shallower than encoder)

Reference:
    TS-JEPA (2024) — arXiv:2509.25449
    I-JEPA (Assran et al., 2023) — arXiv:2301.08243
"""

from __future__ import annotations

import copy, math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════════════════
# Patch tokenizer — Conv1D over multi-channel input
# ═══════════════════════════════════════════════════════════════════════════════

class PatchTokenizer(nn.Module):
    """Conv1D patch tokenizer for multi-channel time series.

    Input:  (B, seq_len, in_channels)
    Output: (B, num_patches, embed_dim)
    """

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
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, num_patches, embed_dim)."""
        x = x.permute(0, 2, 1)          # (B, in_channels, seq_len)
        x = self.proj(x)                # (B, embed_dim, num_patches)
        return x.permute(0, 2, 1)      # (B, num_patches, embed_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Mask generation — contiguous temporal blocks
# ═══════════════════════════════════════════════════════════════════════════════

def generate_patch_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
    min_block: int = 1,
) -> tuple[Tensor, Tensor]:
    """Generate contiguous-block mask indices.

    Masks a contiguous block of patches. The model must infer the missing
    temporal segment from surrounding context — exactly the skill needed
    for interference-robust RSSI classification.

    Returns:
        mask_indices: (B, M) — indices of masked patches
        non_mask_indices: (B, K) — indices of context patches
    """
    n_mask = max(min_block, int(num_patches * mask_ratio))
    n_mask = min(n_mask, num_patches - 1)  # at least 1 context patch

    mask_list, non_mask_list = [], []
    for _ in range(batch_size):
        start = torch.randint(0, num_patches - n_mask + 1, (1,)).item()
        mask_idx = torch.arange(start, start + n_mask, device=device)

        all_idx = torch.arange(num_patches, device=device)
        non_mask_idx = all_idx[~torch.isin(all_idx, mask_idx)]

        mask_list.append(mask_idx)
        non_mask_list.append(non_mask_idx)

    return torch.stack(mask_list), torch.stack(non_mask_list)


def _gather_by_index(x: Tensor, indices: Tensor) -> Tensor:
    """Gather patches at given indices. x: (B, N, D), indices: (B, K)."""
    B = x.size(0)
    return torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder — shared architecture for context and target
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAEncoder(nn.Module):
    """Transformer encoder over Conv1D patches."""

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
        num_patches = seq_len // patch_size

        self.tokenizer = PatchTokenizer(seq_len, patch_size, in_channels, embed_dim)

        # Learnable positional embedding per patch
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

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
        self, x: Tensor, keep_indices: Optional[Tensor] = None
    ) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, n_kept, embed_dim).

        When keep_indices is not None, only those patches are encoded
        (context encoder path). When None, all patches (target encoder path).
        """
        patches = self.tokenizer(x)     # (B, num_patches, embed_dim)
        patches = patches + self.pos_embed[:, : patches.size(1), :]

        if keep_indices is not None:
            patches = _gather_by_index(patches, keep_indices)

        out = self.transformer(patches)
        return self.norm(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor — shallower than encoder
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAPredictor(nn.Module):
    """Predicts masked latents from context latents.

    Input: context latents + mask tokens at target positions
    Output: predicted latents for masked positions
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int = 256,
        predictor_embed_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        # Mask token — learned embedding for positions to predict
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_embed_dim) * 0.02)

        # Positional embedding for the predictor
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, predictor_embed_dim) * 0.02)

        # Input projection: encoder dim → predictor dim
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Predictor transformer (shallower)
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

        # Output projection: predictor dim → encoder dim
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(
        self,
        ctx_latents: Tensor,
        mask_indices: Tensor,
        non_mask_indices: Tensor,
    ) -> Tensor:
        """Predict masked latents.

        Args:
            ctx_latents: (B, K, embed_dim) — context encoder output
            mask_indices: (B, M) — which patches are masked
            non_mask_indices: (B, K) — which patches are context

        Returns:
            predictions: (B, M, embed_dim) — predicted latents at masked positions
        """
        B, _, D = ctx_latents.shape
        pred_dim = self.mask_token.size(-1)

        # Project context latents
        ctx_proj = self.input_proj(ctx_latents)  # (B, K, pred_dim)

        # Build full sequence: mask tokens everywhere, then fill context positions
        full_seq = self.mask_token.expand(B, self.num_patches, -1).clone()
        for b in range(B):
            for i, idx in enumerate(non_mask_indices[b]):
                full_seq[b, idx, :] = ctx_proj[b, i, :]

        full_seq = full_seq + self.pos_embed[:, : self.num_patches, :]

        # Transformer forward
        out = self.transformer(full_seq)    # (B, num_patches, pred_dim)
        out = self.output_proj(out)          # (B, num_patches, embed_dim)

        # Extract masked positions
        return _gather_by_index(out, mask_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# Full TS-JEPA pretraining model
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAModel(nn.Module):
    """TS-JEPA architecture for self-supervised pretraining.

    Usage:
        model = TSJEPAModel()
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
        mask_ratio_start: float = 0.4,
        mask_ratio_end: float = 0.7,
        ema_start: float = 0.996,
        ema_end: float = 0.999,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.ema_momentum = ema_start

        # Context encoder
        self.context_encoder = TSJEPAEncoder(
            seq_len, patch_size, in_channels, embed_dim, nhead,
            num_layers, dim_feedforward, dropout,
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
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
        )

    def _mask_ratio(self, epoch: int, total_epochs: int) -> float:
        """Curriculum: linear increase from start → end ratio."""
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        return self.mask_ratio_start + (self.mask_ratio_end - self.mask_ratio_start) * progress

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update."""
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def forward(
        self, x: Tensor, epoch: int = 0, total_epochs: int = 100
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pretraining forward pass.

        Args:
            x: (B, seq_len, in_channels)
            epoch, total_epochs: for curriculum

        Returns:
            target_latents: (B, M, embed_dim)
            predictions: (B, M, embed_dim)
            mask_indices, non_mask_indices
        """
        B, device = x.size(0), x.device
        mask_ratio = self._mask_ratio(epoch, total_epochs)

        mask_indices, non_mask_indices = generate_patch_masks(
            B, self.num_patches, mask_ratio, device
        )

        # Target encoder: encode ALL patches, extract masked → (B, M, D)
        with torch.no_grad():
            tgt_all = self.target_encoder(x, keep_indices=None)
            tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
            tgt_masked = _gather_by_index(tgt_all, mask_indices)

        # Context encoder: encode only unmasked patches → (B, K, D)
        ctx_encoded = self.context_encoder(x, keep_indices=non_mask_indices)

        # Predictor: predict masked latents → (B, M, D)
        predictions = self.predictor(ctx_encoded, mask_indices, non_mask_indices)

        return tgt_masked, predictions, mask_indices, non_mask_indices

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Single pretraining step."""
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        tgt_masked, predictions, _, _ = self.forward(x, epoch, total_epochs)
        loss = F.l1_loss(predictions, tgt_masked)

        self._update_target_encoder()
        return loss, {"pretrain_loss": loss.item()}

    @torch.no_grad()
    def get_latents(self, x: Tensor) -> Tensor:
        """Extract context encoder latents for visualization."""
        self.eval()
        latents = self.context_encoder(x, keep_indices=None)
        return latents.reshape(-1, self.embed_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TSJEPAClassifier(nn.Module):
    """Classifier built on pretrained TS-JEPA context encoder."""

    def __init__(
        self,
        pretrained: TSJEPAModel,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(pretrained.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, seq_len, in_channels) → (B, num_classes)."""
        latents = self.encoder(x, keep_indices=None)  # (B, num_patches, embed_dim)
        pooled = self.pool(latents.transpose(1, 2)).squeeze(-1)
        return self.head(pooled)
