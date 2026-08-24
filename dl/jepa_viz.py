"""JEPA pretraining monitor — visualize the "world model" being built.

During SSL pretraining, JEPA models learn a latent space that captures
temporal structure without labels. This module monitors:

  1. Pretrain loss curve — is the model learning?
  2. Uniformity — is the latent space collapsing? (higher = better spread)
  3. PCA projections — 2D snapshots of the latent space at checkpoints
  4. Latent statistics — mean/std per dimension, to detect dead neurons

After fine-tuning, you can compare:
  - Pre-finetune embeddings (world model)
  - Post-finetune embeddings (task-adapted)

Usage:
    viz = JEPAVisualizer(save_dir="models/exotic/viz")
    viz.log_pretrain(epoch, train_loss, val_loss, latents)
    viz.save_checkpoint_snapshot(epoch, latents, labels=None)
    viz.log_finetune(metrics)
    viz.finalize()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


# ── Optional imports (graceful fallback if matplotlib not available) ──────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from sklearn.decomposition import PCA

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class JEPAVisualizer:
    """Monitors JEPA pretraining progress with visualizations and metrics."""

    def __init__(
        self,
        save_dir: str | Path = "models/exotic/viz",
        model_name: str = "jepa",
        seed: int = 42,
    ) -> None:
        self.save_dir = Path(save_dir) / f"{model_name}_seed{seed}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Loss history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.ema_history: list[float] = []

        # Uniformity history (higher = better, detects collapse)
        self.uniformity_scores: list[float] = []

        # Latent statistics at checkpoints
        self.latent_stats: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # Logging
    # ═══════════════════════════════════════════════════════════════════════════

    def log_pretrain(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        latents: Tensor | None = None,
        ema_momentum: float = 0.0,
    ) -> None:
        """Log one pretraining epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.ema_history.append(ema_momentum)

        if latents is not None:
            self.uniformity_scores.append(self._compute_uniformity(latents))

    def log_finetune(self, metrics: dict) -> None:
        """Log fine-tuning metrics."""
        with open(self.save_dir / "finetune_metrics.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.floating))
                        else str(v) for k, v in metrics.items()}, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # Uniformity — collapse detection
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_uniformity(latents: Tensor, n_samples: int = 500) -> float:
        """Wang & Isola (2020) uniformity metric.

        Higher = better. Measures how uniformly the latent space is used.
        A collapsing model shows rapidly decreasing uniformity.

        Formula: -log(E[exp(-2 * ||z_i - z_j||^2)])
        """
        latents = latents.detach()
        if latents.size(0) > n_samples:
            idx = torch.randperm(latents.size(0), device=latents.device)[:n_samples]
            z = latents[idx]
        else:
            z = latents

        z = torch.nn.functional.normalize(z, dim=-1)
        if z.size(0) > 200:
            idx = torch.randperm(z.size(0), device=z.device)[:200]
            z = z[idx]

        sq_dist = torch.cdist(z, z, p=2).pow(2)
        uniformity = torch.log(torch.exp(-2.0 * sq_dist).mean() + 1e-10)
        return -uniformity.item()

    # ═══════════════════════════════════════════════════════════════════════════
    # PCA snapshot — "see the world"
    # ═══════════════════════════════════════════════════════════════════════════

    def save_checkpoint_snapshot(
        self,
        epoch: int,
        latents: Tensor,
        labels: Tensor | None = None,
        title: str = "Pretrain Latent Space",
    ) -> str | None:
        """Save PCA projection of latent vectors at a checkpoint.

        Args:
            epoch: current epoch
            latents: (N, D) latent vectors
            labels: (N,) optional class labels for coloring
            title: plot title

        Returns:
            Path to saved PNG, or None if matplotlib/sklearn unavailable
        """
        if not (_MPL_AVAILABLE and _SKLEARN_AVAILABLE):
            return None

        z = latents.detach().cpu().numpy()
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: PCA scatter
        ax = axes[0]
        if labels is not None:
            labs = labels.detach().cpu().numpy()
            for c in np.unique(labs):
                mask = labs == c
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=4, alpha=0.6, label=f"Class {c}")
            ax.legend(markerscale=3, fontsize=8)
        else:
            ax.scatter(z_2d[:, 0], z_2d[:, 1], s=4, alpha=0.6)
        ax.set_title(f"{title} — Epoch {epoch}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

        # Right: per-dimension mean/std (dead neuron detection)
        ax = axes[1]
        dim_means = z.mean(axis=0)
        dim_stds = z.std(axis=0)
        ax.bar(range(len(dim_means)), dim_stds, alpha=0.7, label="Std")
        ax.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="Dead threshold")
        ax.set_title(f"Latent Dimension Activity — Epoch {epoch}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Standard deviation")
        ax.legend(fontsize=8)
        n_dead = (dim_stds < 0.01).sum()
        ax.text(0.02, 0.95, f"Dead dims: {n_dead}/{len(dim_stds)}",
                transform=ax.transAxes, fontsize=9, va="top",
                color="red" if n_dead > len(dim_stds) * 0.1 else "green")

        path = self.save_dir / f"latents_epoch{epoch:04d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

        # Store stats
        self.latent_stats.append({
            "epoch": epoch,
            "pca_var_ratio": pca.explained_variance_ratio_.tolist(),
            "n_dead_dims": int(n_dead),
            "mean_std": float(dim_stds.mean()),
        })

        return str(path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Final plots
    # ═══════════════════════════════════════════════════════════════════════════

    def finalize(self) -> str | None:
        """Generate summary plots and save metrics JSON. Call once at end."""
        # Save metrics JSON
        summary = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "uniformity_scores": self.uniformity_scores,
            "ema_history": self.ema_history,
            "latent_stats": self.latent_stats,
        }
        with open(self.save_dir / "pretrain_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if not _MPL_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(self.train_losses, label="Train loss", alpha=0.8)
        if self.val_losses:
            ax.plot(self.val_losses, label="Val loss", alpha=0.8)
        ax.set_title("Pretrain Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 2. Uniformity (collapse detection)
        ax = axes[0, 1]
        if self.uniformity_scores:
            ax.plot(self.uniformity_scores, color="green")
            ax.axhline(y=-3.5, color="red", linestyle="--", alpha=0.5,
                       label="Collapse threshold")
        ax.set_title("Latent Space Uniformity\n(higher = better, red = collapse)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Uniformity")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3. EMA schedule
        ax = axes[1, 0]
        if self.ema_history:
            ax.plot(self.ema_history)
        ax.set_title("EMA Momentum Schedule")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Momentum")
        ax.grid(True, alpha=0.3)

        # 4. Latent dimension activity over time
        ax = axes[1, 1]
        if self.latent_stats:
            epochs = [s["epoch"] for s in self.latent_stats]
            dead = [s["n_dead_dims"] for s in self.latent_stats]
            ax.fill_between(epochs, dead, alpha=0.3, color="red")
            ax.plot(epochs, dead, "o-", markersize=3, color="red")
        ax.set_title("Dead Latent Dimensions Over Time\n(lower = better)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dead dims")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{self.model_name} — Pretraining Summary", fontsize=14)
        fig.tight_layout()
        path = self.save_dir / "pretrain_summary.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)
