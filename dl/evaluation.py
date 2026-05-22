"""Metrics, cross-validation evaluation, and plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    matthews_corrcoef,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")

from .config import DLConfig
from .data_loader import DLDataLoader
from .training import Trainer

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, float | str]:
    mcc = float(matthews_corrcoef(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    acc = float((y_true == y_pred).mean())
    return {"mcc": mcc, "accuracy": acc, "report": report}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, ax=ax, colorbar=False
    )
    ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(result_rows: list[dict[str, object]], save_path: Path) -> None:
    """Plot val MCC per epoch for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in result_rows:
        name = str(row["name"])
        mccs = row["val_mccs"]
        if isinstance(mccs, list):
            ax.plot(mccs, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MCC")
    ax.set_title("Validation MCC per epoch")
    ax.legend(fontsize=7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def cross_validate(
    model_factory: type[nn.Module],
    model_kwargs: dict[str, object],
    data_loader: DLDataLoader,
    X: np.ndarray,
    y: np.ndarray,
    config: DLConfig,
    model_name: str = "model",
) -> dict[str, float]:
    """K-fold CV — returns mean / std MCC."""
    fold_mccs: list[float] = []
    device = config.resolve_device()

    for fold, (tr_idx, val_idx) in enumerate(data_loader.cv_splits(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
        val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)

        model: nn.Module = model_factory(**model_kwargs)  # type: ignore[call-arg]
        trainer = Trainer(
            config,
            run_name=f"{model_name}_fold{fold}",
        )
        # Disable wandb for individual CV folds to avoid clutter
        original_wandb = config.use_wandb
        config.use_wandb = False
        trainer.fit(model, tr_loader, val_loader)
        config.use_wandb = original_wandb

        _, preds, targets = trainer.evaluate(model, val_loader)
        fold_mcc = float(matthews_corrcoef(targets, preds))
        fold_mccs.append(fold_mcc)

    return {
        "cv_mcc_mean": float(np.mean(fold_mccs)),
        "cv_mcc_std": float(np.std(fold_mccs)),
        "cv_mccs": fold_mccs,  # type: ignore[dict-item]
    }


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    config: DLConfig,
    classes: list[str],
    model_name: str,
    log_wandb: bool = False,
) -> dict[str, float | str]:
    trainer = Trainer(config)
    device = config.resolve_device()
    model = model.to(device)
    _, preds, targets = trainer.evaluate(model, test_loader)
    metrics = compute_metrics(targets, preds, classes)

    cm_path = config.plots_dir / f"cm_{model_name}.png"
    plot_confusion_matrix(targets, preds, classes, f"{model_name} — Test CM", cm_path)

    if log_wandb and _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(
            {  # type: ignore[union-attr]
                f"{model_name}/test_mcc": metrics["mcc"],
                f"{model_name}/test_acc": metrics["accuracy"],
            }
        )

    return metrics


def save_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
