"""T-JEPA pretraining loop — self-supervised feature-masking on RSSI sequences.

Trains the T-JEPA model to predict the latent representation of masked
timesteps from unmasked ones, without using any class labels.  Produces
a context encoder that can be fine-tuned for classification.

Usage:
    from dl.jepa_training import JEPATrainer
    trainer = JEPATrainer(config, model)
    trainer.pretrain(train_loader, val_loader, epochs=200)
    # Then fine-tune with model.context_encoder + classification head

Reference:
    Thimonier et al. (2024) "T-JEPA" — arXiv:2410.05016
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from dl.config import DLConfig
from dl.models.t_jepa import TJEPAModel


@dataclass
class JEPAPretrainResult:
    """Results from T-JEPA pretraining."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    uniformity_scores: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    collapsed: bool = False


class JEPATrainer:
    """Pretraining loop for T-JEPA on RSSI sequences.

    Does NOT use labels — purely self-supervised.
    """

    def __init__(
        self,
        config: DLConfig,
        model: TJEPAModel,
        wandb_project: str | None = None,
        wandb_name: str = "t-jepa-pretrain",
    ) -> None:
        self.config = config
        self.model = model
        self.device = config.resolve_device()
        self.model.to(self.device)

        self._wandb_run: object | None = None
        if _WANDB_AVAILABLE and config.use_wandb and wandb_project:
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "n_features": model.n_features,
                    "d_model": model.d_model,
                    "pretrain_epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                },
                reinit=True,
            )

    def _compute_uniformity(self, latents: Tensor, n_samples: int = 500) -> float:
        """Measure how uniformly the latent space is used.

        Higher = better (no collapse).  Based on Wang & Isola (2020).
        """
        with torch.no_grad():
            # Take a subset for efficiency
            if latents.size(0) > n_samples:
                idx = torch.randperm(latents.size(0), device=latents.device)[:n_samples]
                z = latents[idx]
            else:
                z = latents
            # Flatten across feature dimension: (B, feat, d) → (B*feat, d)
            z = z.reshape(-1, z.size(-1))
            # L2-normalize
            z = F.normalize(z, dim=-1)
            # Pairwise distances on a random subset
            if z.size(0) > 200:
                idx2 = torch.randperm(z.size(0), device=z.device)[:200]
                z = z[idx2]
            # Uniformity loss: log of mean exp(-2 * ||z_i - z_j||^2)
            sq_dist = torch.cdist(z, z, p=2).pow(2)
            uniformity = torch.log(torch.exp(-2.0 * sq_dist).mean() + 1e-10)
            return -uniformity.item()  # negate so higher = better

    def _build_pretrain_loader(
        self, X: np.ndarray, batch_size: int, shuffle: bool = True
    ) -> DataLoader:  # type: ignore[type-arg]
        """Build a DataLoader from raw RSSI features (no labels needed).

        X shape: (N, seq_len, in_features) from the standard pipeline.
        We extract the raw RSSI channel (index 0) and flatten to (N, seq_len).
        """
        # Extract raw RSSI: X is (N, 10, 4) → take channel 0 → (N, 10)
        X_flat = X[:, :, 0].astype(np.float32) if X.ndim == 3 else X.astype(np.float32)

        # Z-score normalize per feature (timestep)
        mean = X_flat.mean(axis=0, keepdims=True)
        std = X_flat.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X_flat - mean) / std

        dataset = TensorDataset(torch.from_numpy(X_norm))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    def pretrain(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        epochs: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 25,
        checkpoint_dir: str | Path = "models/dl",
    ) -> JEPAPretrainResult:
        """Run full T-JEPA pretraining.

        Args:
            train_loader: DataLoader yielding (X_batch,) tuples
            val_loader: Optional validation loader
            epochs: Number of pretraining epochs
            lr: Learning rate
            weight_decay: AdamW weight decay
            patience: Early stopping patience
            checkpoint_dir: Where to save the best checkpoint

        Returns:
            JEPAPretrainResult with loss history and collapse detection.
        """
        result = JEPAPretrainResult()
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: only context encoder + predictor (target encoder via EMA)
        params = list(self.model.context_encoder.parameters()) + list(
            self.model.predictor.parameters()
        )
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # ── Training ──────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x = batch[0].to(self.device)  # (B, n_features)

                loss, metrics = self.model.pretrain_step(x, epoch, epochs)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            result.train_losses.append(avg_train_loss)
            scheduler.step()

            # ── Validation ────────────────────────────────────────────────
            val_loss = float("inf")
            uniformity = 0.0
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_n = 0
                latents_for_uniformity: list[Tensor] = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(self.device)
                        tgt_latents, predictions, _ctx_m, _tgt_m = self.model.forward(x)
                        # Compute MSE on all positions for validation
                        for pred in predictions:
                            diff = (pred - tgt_latents).pow(2).mean()
                            val_loss_sum += diff.item()
                            val_n += 1
                        latents_for_uniformity.append(tgt_latents)

                val_loss = val_loss_sum / max(val_n, 1)
                result.val_losses.append(val_loss)

                # Uniformity check
                if latents_for_uniformity:
                    all_latents = torch.cat(latents_for_uniformity, dim=0)
                    uniformity = self._compute_uniformity(all_latents)

                # Collapse detection
                if uniformity < 0.5 and epoch > 20:
                    result.collapsed = True
                    print(
                        f"WARNING: Potential representation collapse detected "
                        f"(uniformity={uniformity:.3f}). [REG] tokens may help."
                    )
                result.uniformity_scores.append(uniformity)

            # ── Logging ───────────────────────────────────────────────────
            ema_val = self.model.ema_momentum
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"uniformity={uniformity:.3f} | "
                f"ema={ema_val:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if self._wandb_run is not None:
                wandb.log({
                    "pretrain/train_loss": avg_train_loss,
                    "pretrain/val_loss": val_loss,
                    "pretrain/uniformity": uniformity,
                    "pretrain/ema_momentum": ema_val,
                    "pretrain/lr": scheduler.get_last_lr()[0],
                    "pretrain/epoch": epoch,
                })

            # ── Checkpointing ─────────────────────────────────────────────
            if val_loss < best_loss:
                best_loss = val_loss
                result.best_val_loss = best_loss
                result.best_epoch = epoch
                patience_counter = 0

                self._save_checkpoint(checkpoint_dir / "t_jepa_best.pt", epoch, best_loss)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Final save
        self._save_checkpoint(
            checkpoint_dir / "t_jepa_last.pt",
            epoch if "epoch" in dir() else epochs - 1,
            best_loss,
        )

        if self._wandb_run is not None:
            self._wandb_run.finish()

        return result

    def _save_checkpoint(self, path: Path, epoch: int, loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "context_encoder": self.model.context_encoder.state_dict(),
                "target_encoder": self.model.target_encoder.state_dict(),
                "predictor": self.model.predictor.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        """Load pretrained checkpoint. Returns the epoch it was saved at."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.context_encoder.load_state_dict(ckpt["context_encoder"])
        self.model.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.model.predictor.load_state_dict(ckpt["predictor"])
        return ckpt["epoch"]


def prepare_jepa_data(config: DLConfig) -> tuple[DataLoader, DataLoader | None]:  # type: ignore[type-arg]
    """Prepare DataLoaders for T-JEPA pretraining from the standard DL pipeline.

    Returns:
        train_loader, val_loader (val may be None if not enough data)
    """
    from dl.data_loader import DLDataLoader

    loader = DLDataLoader(config)
    X_all, y_all = loader.load_and_preprocess()
    # Keep full (N, L, C) — models expect multi-channel input
    X_all = X_all.astype(np.float32)  # (N, L, C)

    # Normalize per feature across samples and timesteps
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X_all - mean) / std

    # Split: 80/20 (no stratification needed — no labels)
    n = len(X_norm)
    n_train = int(n * 0.8)
    indices = np.random.RandomState(config.seed).permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X_norm[train_idx]
    X_val = X_norm[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    ) if len(X_val) > 0 else None

    return train_loader, val_loader
