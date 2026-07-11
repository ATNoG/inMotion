"""T-JEPA: Tabular Joint-Embedding Predictive Architecture for RSSI sequences.

Treats each of the 10 timesteps as a "feature" (column) in a tabular dataset.
Predicts the latent representation of masked timesteps from unmasked ones,
without data augmentations.  Uses [REG] regularization tokens to prevent
representation collapse during pretraining.

References:
    Thimonier et al. (2024)  "T-JEPA: Augmentation-Free Self-Supervised
        Learning for Tabular Data"  —  arXiv:2410.05016
    Assran et al. (2023)  "Self-Supervised Learning from Images with a
        Joint-Embedding Predictive Architecture" (I-JEPA)  —  CVPR 2023
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


# ═══════════════════════════════════════════════════════════════════════════════
# Positional encoding (sin/cos, non-learnable)
# ═══════════════════════════════════════════════════════════════════════════════

class SinCosPositionalEncoding(nn.Module):
    """Fixed sin/cos positional encoding for feature positions 0..max_len-1."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, num_features, d_model)."""
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ═══════════════════════════════════════════════════════════════════════════════
# Feature tokenizer  (each timestep → d_model via learned linear projection)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Maps each scalar timestep value to a d_model embedding.

    Per-feature linear projections + optional feature-index embeddings
    preserve column identity through the transformer.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_reg_tokens: int = 1,
        use_feature_index: bool = True,
        use_feature_type: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_reg_tokens = n_reg_tokens

        # Per-timestep linear projection  (in_features=1 per timestep)
        self.weight = nn.Parameter(torch.empty(n_features + n_reg_tokens, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self._init_weights()

        # Feature-index embedding: learnable embedding per column position
        self.use_feature_index = use_feature_index
        if use_feature_index:
            self.feat_idx_embed = nn.Embedding(n_features, d_model)

        # Feature-type embedding (unused for purely-numerical RSSI; kept for extension)
        self.use_feature_type = use_feature_type
        if use_feature_type:
            self.feat_type_embed = nn.Embedding(2, d_model)  # 0=numerical, 1=categorical

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, n_features)  →  (B, n_reg_tokens + n_features_kept, d_model).

        If `mask` is provided, only unmasked features are kept.
        [REG] tokens are always prepended and never masked.
        """
        B = x.size(0)
        # Prepend [REG] tokens as ones  (the weight matrix handles projection)
        reg_tokens = torch.ones(B, self.n_reg_tokens, device=x.device, dtype=x.dtype)
        x_with_reg = torch.cat([reg_tokens, x], dim=1)  # (B, n_reg + n_feat)

        # Linear projection:  weight[None] * x_with_reg[:, :, None]  →  (B, N_tot, d_model)
        out = self.weight[None] * x_with_reg[:, :, None]  # (B, N_tot, d_model)
        # Bias is only for actual features (not REG tokens)
        bias_padded = torch.cat([
            torch.zeros(self.n_reg_tokens, self.d_model, device=x.device),
            self.bias,
        ], dim=0)
        out = out + bias_padded[None]  # (B, N_tot, d_model)

        # Feature-index embeddings for actual features
        if self.use_feature_index:
            feat_idx_emb = self.feat_idx_embed(
                torch.arange(self.n_features, device=x.device)
            )  # (n_features, d_model)
            feat_idx_emb = torch.cat([
                torch.zeros(self.n_reg_tokens, self.d_model, device=x.device),
                feat_idx_emb,
            ], dim=0)
            out = out + feat_idx_emb[None]

        # Apply mask: keep [REG] tokens + unmasked features
        if mask is not None:
            # mask shape: (B, n_features), 1=keep, 0=masked for each feature
            # Build full mask: [REG tokens always kept, then feature mask]
            reg_mask = torch.ones(B, self.n_reg_tokens, dtype=mask.dtype, device=mask.device)
            full_mask = torch.cat([reg_mask, mask], dim=1)  # (B, n_reg + n_feat)
            # Zero out masked positions, then gather kept tokens per sample
            out = out * full_mask.unsqueeze(-1)
            # Remove fully-zeroed tokens (they carry no information for the encoder)
            # Actually: the reference T-JEPA applies the mask by gathering indices.
            # We do the same for efficiency.
            out_list: list[Tensor] = []
            for b in range(B):
                keep_idx = full_mask[b].nonzero(as_tuple=True)[0]  # indices to keep
                out_list.append(out[b, keep_idx, :])
            # Pad to max kept length for batched processing
            max_keep = max(o.size(0) for o in out_list)
            out_padded = torch.zeros(B, max_keep, self.d_model, device=x.device, dtype=x.dtype)
            key_padding = torch.ones(B, max_keep, dtype=torch.bool, device=x.device)
            for b, o in enumerate(out_list):
                n = o.size(0)
                out_padded[b, :n, :] = o
                key_padding[b, :n] = False
            # We also return key_padding_mask for the transformer encoder
            self._last_key_padding = key_padding
            return out_padded

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# T-JEPA Context / Target Encoder  (Transformer)
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAEncoder(nn.Module):
    """Transformer encoder used for both context and target branches.

    The target encoder's weights are updated via EMA of the context encoder.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        n_reg_tokens: int = 1,
        use_feature_index: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.tokenizer = FeatureTokenizer(
            n_features=n_features,
            d_model=d_model,
            n_reg_tokens=n_reg_tokens,
            use_feature_index=use_feature_index,
        )
        self.pos_encoder = SinCosPositionalEncoding(d_model, max_len=n_features + n_reg_tokens)

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

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """x: (B, n_features) raw values.

        Returns: (B, n_kept, d_model) — tokenizer output after transformer.

        When `mask` is None, all features are encoded (target encoder path).
        When `mask` is provided, only unmasked features are kept (context encoder path).
        """
        out = self.tokenizer(x, mask=mask)  # (B, n_kept, d_model)
        out = self.pos_encoder(out)

        # Use stored key_padding_mask from tokenizer if available
        if mask is not None and hasattr(self.tokenizer, "_last_key_padding"):
            key_padding_mask = self.tokenizer._last_key_padding

        out = self.transformer(out, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# T-JEPA Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAPredictor(nn.Module):
    """Transformer predictor for T-JEPA.

    Takes context encoder output + mask tokens, predicts target encoder
    latents at masked positions.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        pred_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pred_dim = pred_dim
        self.n_features = n_features

        # Project encoder output down to predictor dimension (bottleneck)
        self.input_proj = nn.Linear(d_model, pred_dim)
        self.input_norm = nn.LayerNorm(pred_dim)

        # Learnable mask token  (replaces masked features in predictor input)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Sin/cos positional embedding for the predictor (full feature grid)
        pred_pos = torch.zeros(1, n_features, pred_dim)
        pos = torch.arange(n_features, dtype=torch.float).unsqueeze(1)
        div_term = (-math.log(10000.0) / pred_dim)
        div = torch.exp(torch.arange(0, pred_dim, 2, dtype=torch.float) * div_term)
        pred_pos[0, :, 0::2] = torch.sin(pos * div)
        pred_pos[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pred_pos_embed", pred_pos)

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
        self.norm = nn.LayerNorm(pred_dim)

        # Project back to encoder dimension for L2 loss with target latents
        self.output_proj = nn.Linear(pred_dim, d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        ctx_enc_out: Tensor,
        ctx_mask: Tensor,
        tgt_mask: Tensor,
    ) -> list[Tensor]:
        """Predict target latents at masked positions.

        Args:
            ctx_enc_out: Context encoder output  (B, n_kept_ctx, d_model)
            ctx_mask: Binary mask for context  (B, n_features), 1=kept
            tgt_mask: Binary mask for target  (B, n_features), 1=target to predict

        Returns:
            List of predicted tensors, one per target mask,
            each (B, n_features, d_model) — predictions at ALL positions
            (only masked positions receive loss).
        """
        B = ctx_enc_out.size(0)

        # Project context embeddings to predictor space
        h = self.input_proj(ctx_enc_out)  # (B, n_kept_ctx, pred_dim)
        h = self.input_norm(h)

        # Build full predictor input: for each feature position, either use
        # the context embedding (if unmasked) or the learnable mask token.
        pred_input = self.mask_token.expand(B, self.n_features, -1).clone()

        # Scatter context embeddings into their positions
        # ctx_mask has 1s where features are kept; we need to map context
        # output positions back to feature indices.
        for b in range(B):
            kept_idx = ctx_mask[b].nonzero(as_tuple=True)[0]
            for j, feat_idx in enumerate(kept_idx):
                if j < h.size(1):
                    pred_input[b, feat_idx, :] = h[b, j, :]

        # Add positional embeddings
        pred_input = pred_input + self.pred_pos_embed

        # Transformer
        pred_input = self.transformer(pred_input)
        pred_input = self.norm(pred_input)

        # Project back to d_model
        predictions = self.output_proj(pred_input)  # (B, n_features, d_model)

        return [predictions]


# ═══════════════════════════════════════════════════════════════════════════════
# Mask generation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def generate_masks(
    batch_size: int,
    n_features: int,
    n_context_masks: int = 1,
    n_target_masks: int = 4,
    min_ctx_share: float = 0.5,
    max_ctx_share: float = 0.8,
    min_tgt_share: float = 0.15,
    max_tgt_share: float = 0.35,
    ensure_disjoint: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[list[Tensor], list[Tensor]]:
    """Generate context and target binary masks for T-JEPA pretraining.

    Each mask is (B, n_features) with 1=keep (context) or 1=predict (target).

    Context and target masks for the same sample are disjoint:
    no feature appears in both context and target.

    Returns:
        ctx_masks: list of `n_context_masks` tensors, each (B, n_features)
        tgt_masks: list of `n_target_masks` tensors, each (B, n_features)
    """
    ctx_masks: list[Tensor] = []
    tgt_masks: list[Tensor] = []

    for _ in range(batch_size):
        # Shuffle feature indices
        perm = torch.randperm(n_features, device=device)

        # Context: keep a random share of features
        ctx_share = (
            min_ctx_share + torch.rand(1, device=device).item() * (max_ctx_share - min_ctx_share)
        )
        n_ctx = max(1, int(n_features * ctx_share))
        ctx_idx = perm[:n_ctx]

        # Target: predict a random share of the REMAINING features
        remaining = perm[n_ctx:]
        tgt_share = (
            min_tgt_share + torch.rand(1, device=device).item() * (max_tgt_share - min_tgt_share)
        )
        n_tgt = max(1, min(int(n_features * tgt_share), len(remaining)))
        tgt_idx = remaining[:n_tgt]

        # Build binary masks
        ctx = torch.zeros(n_features, dtype=torch.float32, device=device)
        ctx[ctx_idx] = 1.0
        tgt = torch.zeros(n_features, dtype=torch.float32, device=device)
        tgt[tgt_idx] = 1.0

        ctx_masks.append(ctx)
        tgt_masks.append(tgt)

    # Stack into batches: list of (n_features,) → list of (1, n_features)
    # We keep them per-sample for the training loop; stack when needed.
    ctx_batch = torch.stack(ctx_masks)  # (B, n_features)
    tgt_batch = torch.stack(tgt_masks)  # (B, n_features)

    return [ctx_batch], [tgt_batch]


# ═══════════════════════════════════════════════════════════════════════════════
# Full T-JEPA model  (pretraining wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAModel(nn.Module):
    """Full T-JEPA architecture for pretraining.

    Usage (pretraining):
        model = TJEPAModel(n_features=10, d_model=128)
        for x_batch in loader:
            loss = model.pretrain_step(x_batch)

    Usage (fine-tuning):
        encoder = model.context_encoder
        # Add classification head on top of encoder output
        classifier = nn.Linear(encoder.d_model, 4)
        # Train with standard cross-entropy
    """

    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        n_reg_tokens: int = 1,
        pred_dim: int = 64,
        pred_num_layers: int = 2,
        ema_momentum: float = 0.996,
        target_ema_start: float = 0.996,
        target_ema_end: float = 1.0,
        mask_min_ctx: float = 0.5,
        mask_max_ctx: float = 0.8,
        mask_min_tgt: float = 0.15,
        mask_max_tgt: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.ema_momentum = ema_momentum
        self.ema_start = target_ema_start
        self.ema_end = target_ema_end

        # Masking hyperparams
        self.mask_min_ctx = mask_min_ctx
        self.mask_max_ctx = mask_max_ctx
        self.mask_min_tgt = mask_min_tgt
        self.mask_max_tgt = mask_max_tgt

        # Context encoder (trained via gradients)
        self.context_encoder = TJEPAEncoder(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_reg_tokens=n_reg_tokens,
            use_feature_index=True,
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = TJEPAEncoder(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_reg_tokens=n_reg_tokens,
            use_feature_index=True,
        )
        # Initialize target encoder identical to context
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Freeze target encoder — updated via EMA only
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = TJEPAPredictor(
            n_features=n_features,
            d_model=d_model,
            pred_dim=pred_dim,
            nhead=nhead,
            num_layers=pred_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self._step_count: int = 0

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update of target encoder from context encoder."""
        # Cosine schedule for momentum
        # This is a simplified version; paper uses 0.996 → 1.0
        m = self.ema_momentum
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            tgt_p.data.mul_(m).add_(ctx_p.data, alpha=1.0 - m)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor], Tensor, Tensor]:
        """Pretraining forward pass.

        Args:
            x: (B, n_features) raw timestep values

        Returns:
            tgt_latents: (B, n_features, d_model) — REG tokens stripped
            predictions: list of (B, n_features, d_model)
            ctx_mask: (B, n_features) — 1=kept for context
            tgt_mask: (B, n_features) — 1=target to predict
        """
        B = x.size(0)
        device = x.device
        n_reg = self.context_encoder.tokenizer.n_reg_tokens

        # Generate masks
        ctx_masks, tgt_masks = generate_masks(
            batch_size=B,
            n_features=self.n_features,
            n_context_masks=1,
            n_target_masks=1,
            min_ctx_share=self.mask_min_ctx,
            max_ctx_share=self.mask_max_ctx,
            min_tgt_share=self.mask_min_tgt,
            max_tgt_share=self.mask_max_tgt,
            device=device,
        )
        ctx_mask = ctx_masks[0]  # (B, n_features)
        tgt_mask = tgt_masks[0]  # (B, n_features)

        # Target encoder: encode ALL features → (B, n_reg + n_features, d_model)
        with torch.no_grad():
            tgt_latents_full = self.target_encoder(x, mask=None)
            tgt_latents = tgt_latents_full[:, n_reg:, :]  # strip [REG] tokens

        # Context encoder: encode only unmasked features
        ctx_latents = self.context_encoder(x, mask=ctx_mask)

        # Predictor: predict target latents at all positions
        predictions = self.predictor(ctx_latents, ctx_mask, tgt_mask)

        return tgt_latents, predictions, ctx_mask, tgt_mask

    def pretrain_step(
        self, x: Tensor, epoch: int, total_epochs: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Single pretraining step with loss computation and EMA update."""
        progress = epoch / max(total_epochs - 1, 1)
        self.ema_momentum = self.ema_end - (self.ema_end - self.ema_start) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

        tgt_latents, predictions, _ctx_mask, tgt_mask = self.forward(x)

        loss = torch.tensor(0.0, device=x.device)
        n_targets = 0

        for pred in predictions:
            diff = (pred - tgt_latents).pow(2).sum(dim=-1)
            masked_diff = diff * tgt_mask
            loss = loss + masked_diff.sum()
            n_targets += tgt_mask.sum()

        if n_targets > 0:
            loss = loss / n_targets

        self._update_target_encoder()
        self._step_count += 1

        return loss, {"pretrain_loss": loss.item()}


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-tuning classifier  (T-JEPA encoder → linear head)
# ═══════════════════════════════════════════════════════════════════════════════

class TJEPAClassifier(nn.Module):
    """Fine-tuned T-JEPA for supervised classification.

    Takes a pretrained TJEPAModel, discards the predictor and target encoder,
    keeps the context encoder, and adds a classification head.
    """

    def __init__(
        self,
        pretrained: TJEPAModel,
        num_classes: int = 4,
        pooling: str = "mean",  # "mean" | "max" | "cls"
        hidden_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained.context_encoder
        self.d_model = pretrained.d_model
        self.pooling = pooling

        # Freeze or not? Start frozen, optionally unfreeze
        self.encoder.requires_grad_(True)

        in_dim = self.d_model
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
        """x: (B, n_features) → (B, num_classes)."""
        # Encode all features (no mask)
        latents = self.encoder(x, mask=None)  # (B, n_features, d_model)

        # Pool over feature dimension
        if self.pooling == "mean":
            pooled = latents.mean(dim=1)
        elif self.pooling == "max":
            pooled = latents.max(dim=1).values
        elif self.pooling == "cls":
            # First token is [REG] token
            pooled = latents[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(pooled)
