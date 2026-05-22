"""Mixture of Experts with soft-gating, load-balance regularisation, and optional top-k routing.

Architecture:
  - SoftMixtureOfExperts: N full-multiclass experts + learned gating network, trained end-to-end.
    Each expert sees the full input; the gating network produces per-sample routing weights.
    Auxiliary load-balance loss (Switch-Transformer style) prevents expert collapse.
  - MixtureOfExperts: legacy binary one-vs-rest version, kept for backward compat.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cnn import CNNClassifier
from .ensemble import MetaLearner
from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .rnn import RNNClassifier
from .tcn import TCNClassifier

# Kernel sets indexed by kernel_set param
_KERNEL_SETS: list[list[int]] = [[3], [3, 5], [3, 5, 7]]


# ── Gating network ─────────────────────────────────────────────────────────────


class GatingNetwork(nn.Module):
    """Lightweight router: global-avg-pool over time → MLP → softmax routing weights.

    Input: (B, T, F) or (B, F).
    Output: (B, num_experts) routing probabilities summing to 1.
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # global average pool over time: (B, T, F) → (B, F)
        return F.softmax(self.net(x), dim=-1)  # (B, num_experts)


# ── Auxiliary load-balance loss ────────────────────────────────────────────────


def load_balance_loss(routing_weights: Tensor) -> Tensor:
    """Switch-Transformer-style auxiliary load-balance loss.

    Penalises routing collapse (all tokens to one expert).
    loss = num_experts * Σ_i( mean_i² )
    Minimum when routing is perfectly uniform; increases with imbalance.

    routing_weights: (B, num_experts) — softmax output from gate.
    Returns scalar loss (already scaled by num_experts).
    """
    num_experts = routing_weights.size(1)
    mean_routing = routing_weights.mean(dim=0)  # (num_experts,)
    return num_experts * (mean_routing * mean_routing).sum()


# ── Soft Mixture of Experts ────────────────────────────────────────────────────


class SoftMixtureOfExperts(nn.Module):
    """End-to-end trainable Soft-MoE.

    Key differences from the legacy MixtureOfExperts:
      - Experts are full multiclass classifiers (not binary one-vs-rest).
      - Gating network is learned jointly with experts (no separate training stage).
      - Routing weights are continuous (soft), enabling gradients to flow to all experts.
      - Auxiliary load-balance loss (stored in self.last_aux_loss after each forward)
        prevents all tokens routing to a single expert.
      - Optional top-k hard routing with gradient straight-through for sparsity.

    Usage:
        model = SoftMixtureOfExperts(experts, in_features=1, num_classes=4)
        logits = model(x)          # also sets model.last_aux_loss
        loss = ce_loss + model.last_aux_loss   # Trainer does this automatically
    """

    def __init__(
        self,
        experts: list[nn.Module],
        in_features: int,
        num_classes: int,
        gate_hidden: int = 64,
        gate_dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = GatingNetwork(in_features, len(experts), gate_hidden, gate_dropout)
        self.num_experts = len(experts)
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.top_k = top_k
        # Populated after every forward pass; Trainer reads this for the total loss.
        self.last_aux_loss: Tensor = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, F). Returns (B, num_classes) logits."""
        # ── Routing weights ──────────────────────────────────────────────────
        routing_weights = self.gate(x)  # (B, num_experts)

        # Compute and store aux loss (read by Trainer after forward)
        self.last_aux_loss = self.aux_loss_weight * load_balance_loss(routing_weights)

        if self.top_k is not None and self.top_k < self.num_experts:
            # Top-k sparse routing: zero out non-selected, renormalise
            topk_vals, topk_idx = routing_weights.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(routing_weights)
            sparse.scatter_(1, topk_idx, topk_vals)
            routing_weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Expert forward passes ────────────────────────────────────────────
        # Stack: (B, num_experts, num_classes)
        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)

        # Weighted sum: routing_weights (B, num_experts, 1) × expert_logits → (B, num_classes)
        w = routing_weights.unsqueeze(-1)  # (B, num_experts, 1)
        return (w * expert_logits).sum(dim=1)  # (B, num_classes)

    def get_routing_weights(self, x: Tensor) -> Tensor:
        """Return routing weights for analysis without modifying last_aux_loss."""
        with torch.no_grad():
            return self.gate(x)


# ── Legacy binary MoE (deprecated) ───────────────────────────────────────────


class MixtureOfExperts(nn.Module):
    """[DEPRECATED] N binary one-vs-rest experts + MetaLearner.

    Kept for backward compatibility. Use SoftMixtureOfExperts for new code.
    """

    def __init__(
        self,
        experts: list[nn.Module],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(experts) != num_classes:
            raise ValueError(f"Need {num_classes} experts, got {len(experts)}")
        self.experts = nn.ModuleList(experts)
        self.meta = MetaLearner(num_classes * 2, num_classes, dropout)
        self.last_aux_loss: Tensor = torch.tensor(0.0)

    def get_expert_features(self, x: Tensor) -> Tensor:
        probs = [F.softmax(e(x), dim=-1) for e in self.experts]  # each (B, 2)
        return torch.cat(probs, dim=-1)  # (B, num_experts * 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.meta(self.get_expert_features(x))


# ── Expert builders ────────────────────────────────────────────────────────────


def build_moe_expert(
    in_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Binary GRU expert (bidirectional). Kept for backward compatibility."""
    return GRUClassifier(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        bidirectional=True,
    )


# Arch types supported as MoE experts
MOE_ARCH_TYPES: list[str] = ["gru", "lstm", "rnn", "cnn"]

# Named MoE combinations: maps combo name → list of arch type per expert slot.
# All lists must have length == num_experts (need not equal num_classes).
MOE_ARCH_TYPES: list[str] = ["gru", "lstm", "rnn", "cnn", "tcn"]

MOE_COMBOS: dict[str, list[str]] = {
    "MoE_4GRU": ["gru", "gru", "gru", "gru"],
    "MoE_4LSTM": ["lstm", "lstm", "lstm", "lstm"],
    "MoE_4TCN": ["tcn", "tcn", "tcn", "tcn"],
    "MoE_4CNN": ["cnn", "cnn", "cnn", "cnn"],
    "MoE_Mixed": ["gru", "lstm", "rnn", "cnn"],
    "MoE_Mixed2": ["gru", "lstm", "cnn", "tcn"],
    "MoE_2GRU2LSTM": ["gru", "gru", "lstm", "lstm"],
    "MoE_2TCN2GRU": ["tcn", "tcn", "gru", "gru"],
    "MoE_2CNN2TCN": ["cnn", "cnn", "tcn", "tcn"],
    "MoE_8Mixed": ["gru", "gru", "lstm", "lstm", "cnn", "cnn", "tcn", "tcn"],
}


def build_moe_expert_typed(
    arch_type: str,
    in_features: int,
    params: dict,
    num_classes: int = 4,
) -> nn.Module:
    """Build an expert of the given arch type.

    params keys (all optional, fall back to sensible defaults):
      gru/lstm/rnn: hidden_size, num_layers, bidirectional
      lstm:         use_attention
      cnn:          num_filters, num_blocks, kernel_set (0/1/2)
      all:          dropout

    num_classes: 4 for full multiclass (SoftMoE), 2 for legacy binary experts.
    """
    dropout: float = float(params.get("dropout", 0.3))

    match arch_type:
        case "gru":
            return GRUClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
            )
        case "lstm":
            return LSTMClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
                use_attention=bool(params.get("use_attention", False)),
            )
        case "rnn":
            return RNNClassifier(
                in_features=in_features,
                hidden_size=int(params.get("hidden_size", 128)),
                num_layers=int(params.get("num_layers", 2)),
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bool(params.get("bidirectional", True)),
            )
        case "cnn":
            ks_idx = int(params.get("kernel_set", 1))
            return CNNClassifier(
                in_features=in_features,
                num_filters=int(params.get("num_filters", 64)),
                num_blocks=int(params.get("num_blocks", 2)),
                num_classes=num_classes,
                dropout=dropout,
                kernel_sizes=_KERNEL_SETS[ks_idx],
            )
        case "tcn":
            return TCNClassifier(
                in_features=in_features,
                num_channels=int(params.get("num_channels", 128)),
                kernel_size=int(params.get("kernel_size", 3)),
                depth=int(params.get("depth", 4)),
                num_classes=num_classes,
                dropout=dropout,
            )
        case _:
            raise ValueError(f"Unknown MoE expert arch: {arch_type!r}")


def build_soft_moe(
    arch_list: list[str],
    in_features: int,
    num_classes: int,
    params_per_expert: list[dict] | None = None,
    gate_hidden: int = 64,
    gate_dropout: float = 0.1,
    aux_loss_weight: float = 0.01,
    top_k: int | None = None,
) -> SoftMixtureOfExperts:
    """Build a SoftMixtureOfExperts where expert i has architecture arch_list[i].

    All experts are full multiclass classifiers (num_classes outputs).
    params_per_expert[i] are passed to build_moe_expert_typed for expert i.
    """
    if params_per_expert is None:
        params_per_expert = [{} for _ in arch_list]
    experts = [
        build_moe_expert_typed(arch, in_features, params, num_classes)
        for arch, params in zip(arch_list, params_per_expert)
    ]
    return SoftMixtureOfExperts(
        experts=experts,
        in_features=in_features,
        num_classes=num_classes,
        gate_hidden=gate_hidden,
        gate_dropout=gate_dropout,
        aux_loss_weight=aux_loss_weight,
        top_k=top_k,
    )
