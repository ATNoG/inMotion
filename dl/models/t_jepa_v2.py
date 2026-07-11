"""T-JEPA v2 — Improved feature-masking JEPA for RSSI sequences.

Key improvements over v1:
  - Contiguous temporal masking (mask blocks of consecutive timesteps)
  - Curriculum learning (mask ratio increases over epochs)
  - Gaussian noise injection on context (robustness to interference)
  - Multi-channel support (all 4 engineered channels, 2D positional encoding)
  - Deeper encoder + predictor
  - Progressive unfreezing during fine-tuning

Reference:
    Thimonier et al. (2024) "T-JEPA" — arXiv:2410.05016
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════
# Improved masking — contiguous temporal blocks with channel awareness
# ═══════════════════════════════════════════════════════════════════

def generate_contiguous_feature_masks(
    batch_size: int,
    n_timesteps: int,
    n_channels: int = 4,
    mask_ratio: float = 0.4,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Generate contiguous temporal masks across all channels.

    Masks a contiguous block of timesteps (all channels masked for those
    timesteps), forcing the model to predict a missing time segment from
    its temporal neighbors across all signal channels.

    Returns:
        mask_features: (B, n_timesteps * n_channels) — 1=keep for context
        target_features: (B, n_timesteps * n_channels) — 1=target to predict
    """
    total = n_timesteps * n_channels
    block_t = max(1, int(n_timesteps * mask_ratio))

    mask_list: list[Tensor] = []
    tgt_list: list[Tensor] = []

    for _ in range(batch_size):
        # Pick random contiguous block of timesteps
        start_t = torch.randint(0, n_timesteps - block_t + 1, (1,), device=device).item()
        masked_t = set(range(start_t, start_t + block_t))

        m = torch.zeros(total, device=device)
        t = torch.zeros(total, device=device)
        for ti in range(n_timesteps):
            for ci in range(n_channels):
                idx = ti * n_channels + ci
                if ti in masked_t:
                    t[idx] = 1.0  # target to predict
                else:
                    m[idx] = 1.0  # context (keep)

        mask_list.append(m)
        tgt_list.append(t)

    return torch.stack(mask_list), torch.stack(tgt_list)


# ═══════════════════════════════════════════════════════════════════
# 2D positional encoding (time + channel)
# ═══════════════════════════════════════════════════════════════════

class PosEmbed2D(nn.Module):
    """2D sin/cos positional encoding for (timestep, channel) grid."""

    def __init__(self, n_timesteps: int, n_channels: int, d_model: int) -> None:
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D PE"
        half = d_model // 2
        pe = torch.zeros(n_timesteps * n_channels, d_model)

        pos_t = torch.arange(n_timesteps, dtype=torch.float).repeat_interleave(n_channels)
        pos_c = torch.arange(n_channels, dtype=torch.float).repeat(n_timesteps)

        omega = torch.arange(half // 2, dtype=torch.float)
        omega = 1.0 / (10000.0 ** (omega / (half // 2)))

        # Time encoding in first half
        pe[:, 0::4] = torch.sin(pos_t.unsqueeze(1) * omega)
        pe[:, 1::4] = torch.cos(pos_t.unsqueeze(1) * omega)
        # Channel encoding in second half
        pe[:, 2::4] = torch.sin(pos_c.unsqueeze(1) * omega)
        pe[:, 3::4] = torch.cos(pos_c.unsqueeze(1) * omega)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


# ═══════════════════════════════════════════════════════════════════
# Feature tokenizer — multi-channel
# ═══════════════════════════════════════════════════════════════════

class FeatureTokenizerV2(nn.Module):
    def __init__(self, n_timesteps: int, n_channels: int, d_model: int, n_reg_tokens: int = 2) -> None:
        super().__init__()
        total = n_timesteps * n_channels
        self.n_reg_tokens = n_reg_tokens
        self.total = total
        self.weight = nn.Parameter(torch.empty(n_reg_tokens + total, d_model))
        self.bias = nn.Parameter(torch.empty(total, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_timesteps, n_channels) → flatten → (B, total)
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        reg = torch.ones(B, self.n_reg_tokens, device=x.device, dtype=x.dtype)
        x_in = torch.cat([reg, x_flat], dim=1)
        out = self.weight[None] * x_in[:, :, None]
        bias_pad = torch.cat([torch.zeros(self.n_reg_tokens, out.size(-1), device=x.device), self.bias])
        return out + bias_pad[None]


# ═══════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════

class TJEPAEncoderV2(nn.Module):
    def __init__(self, n_timesteps: int = 10, n_channels: int = 4, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.0, n_reg_tokens: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_reg_tokens = n_reg_tokens
        self.total_features = n_timesteps * n_channels
        self.tokenizer = FeatureTokenizerV2(n_timesteps, n_channels, d_model, n_reg_tokens)
        self.pos_embed = PosEmbed2D(n_timesteps, n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_timesteps, n_channels)
        h = self.tokenizer(x)  # (B, n_reg + total, d_model)
        # Add 2D pos embed to feature positions only (skip REG tokens)
        feat = h[:, self.n_reg_tokens:, :]
        feat = self.pos_embed(feat)
        h = torch.cat([h[:, :self.n_reg_tokens, :], feat], dim=1)
        h = self.transformer(h)
        return self.norm(h)


# ═══════════════════════════════════════════════════════════════════
# Predictor
# ═══════════════════════════════════════════════════════════════════

class TJEPAPredictorV2(nn.Module):
    def __init__(self, total_features: int, d_model: int = 256, pred_dim: int = 128,
                 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.0, n_reg_tokens: int = 2) -> None:
        super().__init__()
        self.total = total_features
        self.n_reg_tokens = n_reg_tokens
        self.input_proj = nn.Linear(d_model, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, d_model)

    def forward(self, ctx_encoded: Tensor, ctx_mask: Tensor, tgt_mask: Tensor) -> list[Tensor]:
        B = ctx_encoded.size(0)
        # ctx_encoded: (B, n_kept, d_model) — includes REG tokens + context features
        h = self.input_proj(ctx_encoded)
        # Build full input: for each position, use context embedding or mask token
        # Positions: [REG tokens][features 0..total-1]
        # ctx_mask/tgt_mask only cover feature positions; REG tokens always context
        total_positions = self.n_reg_tokens + self.total
        full = self.mask_token.expand(B, total_positions, -1).clone()
        # REG tokens: fill from context (first n_reg positions)
        full[:, :self.n_reg_tokens, :] = h[:, :self.n_reg_tokens, :]
        # Feature positions: scatter context embeddings
        for b in range(B):
            ctx_feat_start = self.n_reg_tokens
            kept_idx = ctx_mask[b].nonzero(as_tuple=True)[0]
            for j, fi in enumerate(kept_idx):
                pos = self.n_reg_tokens + fi.item()
                full[b, pos, :] = h[b, ctx_feat_start + j, :]
        full = self.transformer(full)
        full = self.norm(full)
        predictions = self.output_proj(full[:, self.n_reg_tokens:, :])
        return [predictions]


# ═══════════════════════════════════════════════════════════════════
# Full T-JEPA v2 model
# ═══════════════════════════════════════════════════════════════════

class TJEPAModelV2(nn.Module):
    def __init__(self, n_timesteps: int = 10, n_channels: int = 4, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.0, n_reg_tokens: int = 2, pred_dim: int = 128,
                 pred_num_layers: int = 4, mask_ratio_start: float = 0.3,
                 mask_ratio_end: float = 0.6, noise_std: float = 1.0,
                 ema_start: float = 0.996, ema_end: float = 0.999) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.total = n_timesteps * n_channels
        self.mask_start = mask_ratio_start
        self.mask_end = mask_ratio_end
        self.noise_std = noise_std
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.ema_momentum = ema_start

        self.context_encoder = TJEPAEncoderV2(
            n_timesteps, n_channels, d_model, nhead, num_layers, dim_feedforward, dropout, n_reg_tokens)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = TJEPAPredictorV2(
            self.total, d_model, pred_dim, nhead, pred_num_layers, dim_feedforward, dropout, n_reg_tokens)

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        m = self.ema_momentum
        for pc, pt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(m).add_(pc.data, alpha=1.0 - m)

    def _mask_ratio(self, epoch: int, total_epochs: int) -> float:
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        return self.mask_start + (self.mask_end - self.mask_start) * progress

    def forward(self, x: Tensor, epoch: int = 0, total_epochs: int = 100) -> tuple:
        B, device = x.size(0), x.device
        ratio = self._mask_ratio(epoch, total_epochs)
        ctx_mask, tgt_mask = generate_contiguous_feature_masks(
            B, self.n_timesteps, self.n_channels, ratio, device)

        with torch.no_grad():
            tgt_all = self.target_encoder(x)
            tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
            tgt_feat = tgt_all[:, self.context_encoder.n_reg_tokens:, :]

        ctx_in = x.clone()
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(ctx_in) * self.noise_std
            mask_noise = torch.rand(B, device=device) < 0.5
            ctx_in[mask_noise] = ctx_in[mask_noise] + noise[mask_noise]

        ctx_encoded = self.context_encoder(ctx_in)
        predictions = self.predictor(ctx_encoded, ctx_mask, tgt_mask)
        return tgt_feat, predictions, ctx_mask, tgt_mask

    def pretrain_step(self, x: Tensor, epoch: int, total_epochs: int) -> tuple[Tensor, dict]:
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (1.0 + math.cos(math.pi * progress)) / 2.0
        tgt, preds, _, tgt_m = self.forward(x, epoch, total_epochs)
        loss = torch.tensor(0.0, device=x.device)
        n = 0
        n_reg = self.context_encoder.n_reg_tokens
        for p in preds:
            diff = (p - tgt).pow(2).sum(dim=-1)
            loss = loss + (diff * tgt_m).sum()
            n += tgt_m.sum()
        loss = loss / max(n, 1)
        self._update_target_encoder()
        return loss, {"pretrain_loss": loss.item()}


# ═══════════════════════════════════════════════════════════════════
# Fine-tuning classifier
# ═══════════════════════════════════════════════════════════════════

class TJEPAClassifierV2(nn.Module):
    def __init__(self, pretrained: TJEPAModelV2, num_classes: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.d_model = pretrained.context_encoder.d_model
        self.head = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        latents = self.encoder(x)  # (B, n_reg+total, d_model)
        pooled = latents.mean(dim=1)
        return self.head(pooled)
