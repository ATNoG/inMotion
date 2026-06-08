"""Metrics, cross-validation evaluation, and plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")

from .config import DLConfig
from .data_loader import DLDataLoader
from .training import Trainer

CLASS_NAMES = ["AA", "AB", "BA", "BB"]

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict:
    mcc = float(matthews_corrcoef(y_true, y_pred))
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    )

    per_class = {
        classes[i]: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }
        for i in range(len(classes))
    }

    return {
        "mcc": mcc,
        "accuracy": acc,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


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
) -> dict:
    trainer = Trainer(config)
    device = config.resolve_device()
    model = model.to(device)
    _, preds, targets = trainer.evaluate(model, test_loader)
    metrics = compute_metrics(targets, preds, classes)

    if log_wandb and _WANDB_AVAILABLE and wandb.run is not None:
        wandb_log_dict = {
            f"{model_name}/test_mcc": metrics["mcc"],
            f"{model_name}/test_acc": metrics["accuracy"],
            f"{model_name}/test_f1_macro": metrics["f1_macro"],
            f"{model_name}/test_f1_weighted": metrics["f1_weighted"],
            f"{model_name}/test_precision_macro": metrics["precision_macro"],
            f"{model_name}/test_recall_macro": metrics["recall_macro"],
        }
        for cls_name, cls_metrics in metrics["per_class"].items():
            wandb_log_dict[f"{model_name}/test_per_class/{cls_name}/precision"] = cls_metrics["precision"]
            wandb_log_dict[f"{model_name}/test_per_class/{cls_name}/recall"] = cls_metrics["recall"]
            wandb_log_dict[f"{model_name}/test_per_class/{cls_name}/f1"] = cls_metrics["f1"]
        cm = metrics["confusion_matrix"]
        wandb_log_dict[f"{model_name}/test_confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=targets,
            preds=preds,
            class_names=classes,
        )
        wandb.log(wandb_log_dict)  # type: ignore[union-attr]

    return metrics


def save_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def evaluate_checkpoint_from_disk(
    checkpoint_path: Path,
    model_name: str,
    model_factory_fn,
    classes: list[str],
    config: DLConfig,
    data_loader: DLDataLoader,
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_wandb: bool = False,
) -> dict:
    """Load a checkpoint from disk, reconstruct the model, and evaluate.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_name: Name for logging/identification.
        model_factory_fn: Callable that returns a model given config (for single models)
                          or a dict of special build instructions.
        classes: List of class name strings.
        config: DLConfig instance.
        data_loader: DLDataLoader instance with pre-fitted scaler.
        X_test, y_test: Test data arrays.
        log_wandb: Whether to log metrics to WandB.

    Returns:
        Dict with all metrics plus per-class breakdown and confusion matrix.
    """
    test_loader = data_loader.make_loader(X_test, y_test, shuffle=False)
    model = model_factory_fn()
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    metrics = evaluate_model_on_test(
        model, test_loader, config, classes, model_name, log_wandb=log_wandb,
    )
    return metrics


def metrics_to_extended_row(
    model_name: str,
    model_type: str,
    seed: int,
    device: str,
    metrics: dict,
    best_val_mcc: float = 0.0,
    best_epoch: int = 0,
    train_time_s: float = 0.0,
    extra: dict | None = None,
) -> dict:
    """Convert metrics dict to a flat row dict suitable for CSV storage.

    Extra fields like members, arch_list, val_mccs can be passed via `extra`.
    """
    row = {
        "model": model_name,
        "type": model_type,
        "seed": seed,
        "device": device,
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "best_val_mcc": best_val_mcc,
        "best_epoch": best_epoch,
        "train_time_s": train_time_s,
    }
    # Per-class metrics
    for cls_name, cls_metrics in metrics["per_class"].items():
        row[f"{cls_name}_precision"] = cls_metrics["precision"]
        row[f"{cls_name}_recall"] = cls_metrics["recall"]
        row[f"{cls_name}_f1"] = cls_metrics["f1"]
        row[f"{cls_name}_support"] = cls_metrics["support"]

    # Confusion matrix components flattened
    cm = metrics["confusion_matrix"]
    classes_ordered = list(metrics["per_class"].keys())
    for i, true_cls in enumerate(classes_ordered):
        for j, pred_cls in enumerate(classes_ordered):
            row[f"CM_{true_cls}_pred_{pred_cls}"] = int(cm[i, j])

    # Store full CM as JSON for easy reconstruction
    row["Confusion_Matrix"] = json.dumps(cm.tolist())

    if extra:
        row.update(extra)
    return row
