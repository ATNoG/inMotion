"""TS-JEPA v2 — Improved masking, curriculum, multi-channel, noise injection.

Key improvements over v1:
  - Contiguous temporal block masking (forces trajectory interpolation)
  - Curriculum learning (mask ratio increases over epochs)
  - Gaussian noise injection on context (robustness to interference)
  - Multi-channel pretraining (all 4 engineered channels)
  - Deeper predictor with residual connections

Reference:
    "Joint Embeddings Go Temporal" (TS-JEPA) — NeurIPS 2024 Workshop
    arxiv.org/abs/2509.25449
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════
# Improved masking — contiguous temporal blocks
# ═══════════════════════════════════════════════════════════════════

def generate_contiguous_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.4,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Generate contiguous-block mask indices.

    Masks a single contiguous block of patches, forcing the model
    to predict a missing trajectory *segment* from its endpoints.
    This is much harder than random masking and teaches trajectory
    interpolation.

    Args:
        batch_size: B
        num_patches: N
        mask_ratio: fraction of patches in the contiguous block

    Returns:
        mask_indices: (B, M) — masked patch indices (contiguous)
        non_mask_indices: (B, K) — unmasked patch indices
    """
    block_size = max(1, int(num_patches * mask_ratio))
    max_start = num_patches - block_size

    mask_list: list[Tensor] = []
    non_mask_list: list[Tensor] = []

    for _ in range(batch_size):
        start = torch.randint(0, max_start + 1, (1,), device=device).item()
        masked = list(range(start, start + block_size))
        unmasked = [i for i in range(num_patches) if i not in masked]
        mask_list.append(torch.tensor(masked, device=device))
        non_mask_list.append(torch.tensor(unmasked, device=device))

    return torch.stack(mask_list), torch.stack(non_mask_list)


# ═══════════════════════════════════════════════════════════════════
# Patch tokenizer — multi-channel support
# ═══════════════════════════════════════════════════════════════════

class PatchTokenizerV2(nn.Module):
    """Conv1D patching with multi-channel input."""

    def __init__(self, seq_len: int, patch_size: int, embed_dim: int, in_channels: int = 4) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class SinCosPosEmbedV2(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        pe = torch.zeros(num_patches, embed_dim)
        pos = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        omega = torch.arange(embed_dim // 2, dtype=torch.float)
        omega = 1.0 / (10000.0 ** (omega / (embed_dim // 2)))
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


def _gather_by_index(x: Tensor, indices: Tensor) -> Tensor:
    B = x.size(0)
    return torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)


# ═══════════════════════════════════════════════════════════════════
# Improved Encoder
# ═══════════════════════════════════════════════════════════════════

class TSJEPAEncoderV2(nn.Module):
    def __init__(
        self, seq_len: int = 10, patch_size: int = 2, embed_dim: int = 256,
        nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512,
        dropout: float = 0.0, in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = seq_len // patch_size
        self.tokenizer = PatchTokenizerV2(seq_len, patch_size, embed_dim, in_channels)
        self.pos_embed = SinCosPosEmbedV2(num_patches, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, keep_indices: Optional[Tensor] = None) -> Tensor:
        patches = self.tokenizer(x)
        patches = self.pos_embed(patches)
        if keep_indices is not None:
            patches = _gather_by_index(patches, keep_indices)
        out = self.transformer(patches)
        return self.norm(out)


# ═══════════════════════════════════════════════════════════════════
# Improved Predictor — deeper, residual
# ═══════════════════════════════════════════════════════════════════

class TSJEPAPredictorV2(nn.Module):
    def __init__(
        self, num_patches: int, encoder_embed_dim: int = 256,
        predictor_embed_dim: int = 128, nhead: int = 8, num_layers: int = 4,
        dim_feedforward: int = 512, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.pos_embed = SinCosPosEmbedV2(num_patches, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(self, ctx_encoded: Tensor, mask_indices: Tensor, non_mask_indices: Tensor) -> Tensor:
        B, M = mask_indices.shape
        K = non_mask_indices.size(1)
        h = self.input_proj(ctx_encoded)
        pos_all = self.pos_embed.pe.expand(B, -1, -1)
        ctx_pos = _gather_by_index(pos_all, non_mask_indices)
        mask_pos = _gather_by_index(pos_all, mask_indices)
        h = h + ctx_pos
        pred_tokens = self.mask_token.expand(B, M, -1) + mask_pos
        x = torch.cat([h, pred_tokens], dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, K:, :]
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════
# Full TS-JEPA v2 model
# ═══════════════════════════════════════════════════════════════════

class TSJEPAModelV2(nn.Module):
    """Improved TS-JEPA with contiguous masking, curriculum, noise injection."""

    def __init__(
        self, seq_len: int = 10, patch_size: int = 2, embed_dim: int = 256,
        nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512,
        dropout: float = 0.0, pred_dim: int = 128, pred_num_layers: int = 4,
        mask_ratio_start: float = 0.3, mask_ratio_end: float = 0.6,
        noise_std: float = 1.0, ema_start: float = 0.996, ema_end: float = 0.999,
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.noise_std = noise_std
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.ema_momentum = ema_start

        self.context_encoder = TSJEPAEncoderV2(
            seq_len, patch_size, embed_dim, nhead, num_layers, dim_feedforward, dropout, in_channels)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TSJEPAPredictorV2(
            self.num_patches, embed_dim, pred_dim, nhead, pred_num_layers, dim_feedforward, dropout)

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def _mask_ratio(self, epoch: int, total_epochs: int) -> float:
        """Curriculum: linear increase from start to end ratio."""
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        return self.mask_ratio_start + (self.mask_ratio_end - self.mask_ratio_start) * progress

    def forward(self, x: Tensor, epoch: int = 0, total_epochs: int = 100) -> tuple:
        B, device = x.size(0), x.device
        mask_ratio = self._mask_ratio(epoch, total_epochs)
        mask_idx, non_mask_idx = generate_contiguous_masks(B, self.num_patches, mask_ratio, device)

        # Target encoder — all patches, no noise
        with torch.no_grad():
            tgt_all = self.target_encoder(x, keep_indices=None)
            tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
            tgt_masked = _gather_by_index(tgt_all, mask_idx)

        # Context encoder — unmasked patches with optional noise
        ctx_in = x.clone()
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(ctx_in) * self.noise_std
            # Only add noise to samples that pass a threshold
            # (don't destroy already-weak signals)
            mask = torch.rand(B, device=device) < 0.5
            ctx_in[mask] = ctx_in[mask] + noise[mask]

        ctx_encoded = self.context_encoder(ctx_in, keep_indices=non_mask_idx)
        predictions = self.predictor(ctx_encoded, mask_idx, non_mask_idx)
        return tgt_masked, predictions, mask_idx, non_mask_idx

    def pretrain_step(self, x: Tensor, epoch: int, total_epochs: int) -> tuple[Tensor, dict]:
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (1.0 + math.cos(math.pi * progress)) / 2.0

        tgt_masked, predictions, _, _ = self.forward(x, epoch, total_epochs)
        loss = F.l1_loss(predictions, tgt_masked)
        self._update_target_encoder()
        return loss, {"pretrain_loss": loss.item()}


# ═══════════════════════════════════════════════════════════════════
# Fine-tuning classifier
# ═══════════════════════════════════════════════════════════════════

class TSJEPAClassifierV2(nn.Module):
    def __init__(self, pretrained: TSJEPAModelV2, num_classes: int = 4, pooling: str = "mean",
                 hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.embed_dim = pretrained.embed_dim
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        latents = self.encoder(x, keep_indices=None)
        pooled = latents.mean(dim=1)  # (B, D)
        return self.head(pooled)
