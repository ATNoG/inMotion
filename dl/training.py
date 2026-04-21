"""Generic Trainer with WandB logging, early stopping, and L1 regularization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from .config import DLConfig


class Loggable(Protocol):
    def log(self, data: dict[str, float | int | str]) -> None: ...


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_mccs: list[float] = field(default_factory=list)
    best_val_mcc: float = 0.0
    best_epoch: int = 0


class Trainer:
    def __init__(self, config: DLConfig, run_name: str = "") -> None:
        self.config = config
        self.device = config.resolve_device()
        self.run_name = run_name
        self._wandb_run: object | None = None

    def _l1_loss(self, model: nn.Module) -> Tensor:
        l1 = torch.tensor(0.0, device=self.device)
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        return self.config.l1_lambda * l1

    def _step(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train: bool,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train)
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                if train:
                    loss = loss + self._l1_loss(model)

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()

                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.cpu().numpy().tolist())

        n = len(all_targets)
        return (
            total_loss / n if n > 0 else 0.0,
            np.array(all_preds, dtype=np.int64),
            np.array(all_targets, dtype=np.int64),
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        val_loader: DataLoader[tuple[Tensor, Tensor]],
        save_path: Path | None = None,
    ) -> TrainResult:
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()
        result = TrainResult()
        patience_counter = 0
        best_state: dict[str, Tensor] = {}

        if self.config.use_wandb and _WANDB_AVAILABLE:
            try:
                self._wandb_run = wandb.init(  # type: ignore[union-attr]
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.run_name or None,
                    config={
                        "learning_rate": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay,
                        "l1_lambda": self.config.l1_lambda,
                        "batch_size": self.config.batch_size,
                        "num_epochs": self.config.num_epochs,
                        "dropout": self.config.dropout,
                        "model": type(model).__name__,
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[WandB] init failed ({e}), continuing without logging.")
                self._wandb_run = None

        for epoch in range(1, self.config.num_epochs + 1):
            tr_loss, _, _ = self._step(model, train_loader, criterion, optimizer, train=True)
            val_loss, val_preds, val_targets = self._step(
                model, val_loader, criterion, None, train=False
            )
            val_mcc = float(matthews_corrcoef(val_targets, val_preds))
            scheduler.step()

            result.train_losses.append(tr_loss)
            result.val_losses.append(val_loss)
            result.val_mccs.append(val_mcc)

            if self.config.use_wandb and _WANDB_AVAILABLE and self._wandb_run is not None:
                wandb.log(
                    {  # type: ignore[union-attr]
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
                        "val_mcc": val_mcc,
                    }
                )

            if val_mcc > result.best_val_mcc:
                result.best_val_mcc = val_mcc
                result.best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        if self.config.use_wandb and _WANDB_AVAILABLE and self._wandb_run is not None:
            wandb.log({"best_val_mcc": result.best_val_mcc, "best_epoch": result.best_epoch})  # type: ignore[union-attr]
            wandb.finish()  # type: ignore[union-attr]

        return result

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (loss, preds, targets)."""
        criterion = nn.CrossEntropyLoss()
        return self._step(model, loader, criterion, None, train=False)
