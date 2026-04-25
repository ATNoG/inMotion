"""Mamba-3 MIMO classifier — official mamba_ssm.modules.mamba3.Mamba3 when
available, pure-PyTorch MIMO selective-SSM fallback otherwise.

Install the library for full performance:
    pip install mamba-ssm  (requires CUDA + causal-conv1d)

MIMO (Multiple-Input Multiple-Output) mode runs R parallel SSM channels with
rank-R B/C projections and learned per-rank input/output mixing weights,
enabling richer multi-channel sequence dynamics.

  • Official path:  Mamba3(is_mimo=True)  — requires TileLang fused kernels
                   Mamba3(is_mimo=False) — SISO, only needs Triton (default)
  • Fallback path:  pure-PyTorch MIMO-SSM loop — runs everywhere.

References:
  Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective
      State Spaces"
  Dao & Gu (2024) "Transformers are SSMs" (Mamba-2)
  Gu et al. (2026) "Mamba-3"
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ── Official Mamba-3 availability ─────────────────────────────────────────────
try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3  # type: ignore[import-untyped]

    _MAMBA3_AVAILABLE = True
    # MIMO requires optional TileLang fused kernels (separate install)
    try:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import (  # type: ignore[import-untyped]
            mamba3_mimo as _mimo_fn,
        )

        _MIMO_KERNEL_AVAILABLE = _mimo_fn is not None
    except ImportError:
        _MIMO_KERNEL_AVAILABLE = False
except ImportError:
    _MAMBA3_AVAILABLE = False
    _MIMO_KERNEL_AVAILABLE = False


# ── Pure-PyTorch MIMO selective-SSM ──────────────────────────────────────────


class _SelectiveSSM(nn.Module):
    """MIMO-capable selective state-space layer (pure PyTorch, no CUDA ext).

    ``mimo_rank=1`` → standard SISO Mamba (single scan).
    ``mimo_rank=R`` → Mamba-3-style MIMO: R parallel selective scans with
    separate B/C projections and learned per-rank input/output mixing weights.

    Input/Output shape: ``(B, T, d_model)``.

    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        mimo_rank: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        R = mimo_rank
        dt_rank = dt_rank or max(1, d_model // 16)
        self.dt_rank = dt_rank

        # Input-dependent projections: Δ (dt) + B[R] + C[R]
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state * R, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Fixed log-diagonal A: eigenvalues ≈ -1 .. -(d_state+1)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(-A))  # (d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        # MIMO mixing weights — only allocated for R > 1
        if R > 1:
            self.mimo_x = nn.Parameter(torch.ones(R, d_model) / R)  # input mixing
            self.mimo_o = nn.Parameter(torch.ones(R, d_model) / R)  # output mixing

        nn.init.normal_(self.dt_proj.weight, std=0.01)
        nn.init.uniform_(
            self.dt_proj.bias,
            -math.log(d_state),
            math.log(d_state),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        B, T, d = x.shape
        R = self.mimo_rank
        d_s = self.d_state

        xz = self.x_proj(x)  # (B, T, dt_rank + 2*R*d_s)
        dt_raw = xz[..., : self.dt_rank]
        BC = xz[..., self.dt_rank :]  # (B, T, 2*R*d_s)
        B_all = BC[..., : R * d_s].reshape(B, T, R, d_s)  # (B, T, R, d_s)
        C_all = BC[..., R * d_s :].reshape(B, T, R, d_s)  # (B, T, R, d_s)

        dt = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_model)
        A = -torch.exp(self.A_log)  # (d_model, d_s) — negative
        # Ā = exp(Δ · A): same discretisation shared across all MIMO ranks
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, T, d_model, d_s)

        if R == 1:
            # ── SISO fast path ─────────────────────────────────────────────
            # B̄ = Δ ⊗ B_t  (outer product d_model × d_state)
            dB = dt.unsqueeze(-1) * B_all[:, :, 0].unsqueeze(2)  # (B, T, d_model, d_s)
            h = x.new_zeros(B, d, d_s)
            ys: list[Tensor] = []
            for t in range(T):
                # h_t = Ā_t h_{t-1} + B̄_t x_t
                h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
                # y_t = C_t · h_t  (dot over d_state)
                ys.append((h * C_all[:, t, 0].unsqueeze(1)).sum(-1))  # (B, d_model)
            out = torch.stack(ys, dim=1)  # (B, T, d_model)
        else:
            # ── MIMO path: R independent selective scans ───────────────────
            # Each rank r sees a scaled view of x via learned mimo_x[r].
            # Outputs are weighted by mimo_o[r] and summed (rank aggregation).
            out = x.new_zeros(B, T, d)
            for r in range(R):
                x_r = x * self.mimo_x[r]  # (B, T, d_model)
                dB_r = dt.unsqueeze(-1) * B_all[:, :, r].unsqueeze(2)
                h_r = x.new_zeros(B, d, d_s)
                ys_r: list[Tensor] = []
                for t in range(T):
                    h_r = dA[:, t] * h_r + dB_r[:, t] * x_r[:, t].unsqueeze(-1)
                    ys_r.append((h_r * C_all[:, t, r].unsqueeze(1)).sum(-1))
                # Weight rank-r output by mimo_o[r]
                out = out + torch.stack(ys_r, dim=1) * self.mimo_o[r]

        return out + self.D * x  # skip connection


class _MambaBlock(nn.Module):
    """Single Mamba-3-style residual block with MIMO SSM (pure PyTorch).

    Follows the Mamba paper: expand → (causal conv + MIMO-SSM) × gate → contract.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int | None = None,
        mimo_rank: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.ssm = _SelectiveSSM(d_inner, d_state=d_state, dt_rank=dt_rank, mimo_rank=mimo_rank)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)  # pre-norm
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # (B, T, d_inner) each
        # Channel-first conv; trim to original T (remove acausal padding)
        x_conv = self.conv(x_in.transpose(1, 2))[:, :, : x_in.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)
        y = self.ssm(x_conv)  # (B, T, d_inner)
        y = y * F.silu(z)  # multiplicative gate
        return self.drop(self.out_proj(y)) + residual


# ── Official Mamba-3 wrapper ──────────────────────────────────────────────────


class _OfficialMamba3Block(nn.Module):
    """Thin wrapper around ``mamba_ssm.modules.mamba3.Mamba3`` with residual.

    Selects MIMO mode (``is_mimo=True``) when TileLang fused kernels are
    present, otherwise falls back to SISO Mamba-3 (Triton only).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        mimo_rank: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        # headdim must divide d_inner exactly; halve until it does
        _hd = headdim
        while _hd > 1 and d_inner % _hd != 0:
            _hd //= 2

        self.norm = nn.LayerNorm(d_model)
        self._use_mimo = _MIMO_KERNEL_AVAILABLE
        self.mamba = _Mamba3(  # type: ignore[possibly-undefined]
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=_hd,
            is_mimo=self._use_mimo,
            mimo_rank=mimo_rank if self._use_mimo else 1,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.mamba(self.norm(x))) + x

    @property
    def is_mimo(self) -> bool:
        return self._use_mimo


# ── Classifier ────────────────────────────────────────────────────────────────


class MambaClassifier(nn.Module):
    """Mamba-3 MIMO sequence classifier.

    Uses the official ``mamba_ssm.modules.mamba3.Mamba3`` when available:
      - ``is_mimo=True`` (MIMO) when TileLang kernels are present
      - ``is_mimo=False`` (SISO) when only Triton kernels are installed

    Falls back to the pure-PyTorch MIMO-SSM when mamba-ssm is absent.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )

        if _MAMBA3_AVAILABLE:
            self.blocks: nn.ModuleList = nn.ModuleList(
                [
                    _OfficialMamba3Block(
                        d_model=d_model,
                        d_state=max(16, d_state),
                        expand=expand,
                        headdim=64,
                        mimo_rank=mimo_rank,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    _MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand,
                        mimo_rank=mimo_rank,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)  # (B, T, d_model)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        context = h.mean(dim=1)  # (B, d_model) — mean pooling over time
        return self.classifier(self.drop(context))

    @staticmethod
    def using_official_library() -> bool:
        """True when ``mamba_ssm.modules.mamba3.Mamba3`` is being used."""
        return _MAMBA3_AVAILABLE

    @staticmethod
    def using_mimo() -> bool:
        """True when full MIMO TileLang kernels are available."""
        return _MIMO_KERNEL_AVAILABLE
