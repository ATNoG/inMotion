"""Main DL pipeline — multi-GPU parallel training.

Each model trains on its own GPU (round-robin). No DataParallel.
HPO studies also distributed across GPUs in parallel.

Usage:
    uv run python run_dl.py [OPTIONS]

Options:
    --seed INT          Random seed (default 42)
    --epochs INT        Max epochs per model (default 150)
    --trials INT        Optuna trials per study (default 50)
    --batch-size INT    Batch size (default 64)
    --num-gpus INT      GPUs to use, 0 = auto-detect all (default 0)
    --no-wandb          Disable WandB logging
    --no-optuna         Skip Optuna HPO/NAS studies
    --data PATH         Path to dataset CSV (default dataset.csv)
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from dl.config import DLConfig
from dl.data_loader import DLDataLoader
from dl.evaluation import (
    evaluate_model_on_test,
    plot_training_curves,
    save_results_csv,
)
from dl.models.bilstm import BiLSTMClassifier
from dl.models.cnn import CNNClassifier
from dl.models.cnn2d_rnn import CNN2DRNNClassifier
from dl.models.ensemble import StackingEnsemble, VotingEnsemble
from dl.models.gru import GRUClassifier
from dl.models.lstm import LSTMClassifier
from dl.models.mamba import MambaClassifier
from dl.models.meta_fusion import MetaFusionClassifier
from dl.models.moe import (
    MOE_COMBOS,
    build_moe_expert,
    build_moe_expert_typed,
    build_soft_moe,
)
from dl.models.rnn import RNNClassifier
from dl.models.tcn import TCNClassifier
from dl.models.transformer import TransformerClassifier
from dl.optimization import (
    init_optuna_db,
    run_binary_moe_hpo_study,
    run_hpo_study,
    run_moe_multiobjective_study,
    run_nas_study,
    save_optuna_plots,
)
from dl.training import Trainer
from torch import nn
from torch.utils.data import DataLoader

SINGLE_MODEL_NAMES: list[str] = [
    "RNN",
    "GRU",
    "LSTM",
    "CNN",
    "TCN",
    "Transformer",
    "BiLSTM",
    "CNN2DLSTM",
    "CNN2DGRU",
    "Mamba",
]
HPO_TYPES: list[str] = [
    "rnn",
    "gru",
    "lstm",
    "cnn",
    "tcn",
    "transformer",
    "bilstm",
    "cnn2drnn",
    "mamba",
]

# Metadata model names — require (X, meta, y) dataloaders
META_MODEL_NAMES: list[str] = ["MetaFusion_LSTM", "MetaFusion_GRU"]

# 18 base variants for Deep Stacking (2 configs per arch)
DS_BASE_NAMES: list[str] = [
    "DS_RNN_A",
    "DS_RNN_B",
    "DS_GRU_A",
    "DS_GRU_B",
    "DS_LSTM_A",
    "DS_LSTM_B",
    "DS_CNN_A",
    "DS_CNN_B",
    "DS_TCN_A",
    "DS_TCN_B",
    "DS_Transformer_A",
    "DS_Transformer_B",
    "DS_BiLSTM_A",
    "DS_BiLSTM_B",
    "DS_CNN2DRNN_A",
    "DS_CNN2DRNN_B",
    "DS_Mamba_A",
    "DS_Mamba_B",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DL pipeline for inMotion WiFi classification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-gpus", type=int, default=0, help="0 = auto-detect all available")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-optuna", action="store_true")
    p.add_argument("--no-moe", action="store_true", help="Skip Mixture-of-Experts phase")
    p.add_argument("--no-deepstack", action="store_true", help="Skip Deep Stacking phase")
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--wandb-project", type=str, default=None, help="Override WandB project name")
    p.add_argument("--no-meta", action="store_true", help="Skip metadata-fusion model phases")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override output directory for saved models (default: models/dl)",
    )
    return p.parse_args()


def _available_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def _build_device_pool(n_gpus: int) -> list[str]:
    if torch.cuda.is_available():
        count = min(n_gpus, torch.cuda.device_count())
        return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


def _build_model_by_name(name: str, config: DLConfig) -> nn.Module:
    base_kw: dict[str, object] = dict(
        in_features=config.in_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    match name:
        case "RNN":
            return RNNClassifier(**base_kw)  # type: ignore[arg-type]
        case "GRU":
            return GRUClassifier(**base_kw)  # type: ignore[arg-type]
        case "LSTM":
            return LSTMClassifier(**base_kw, use_attention=True)  # type: ignore[arg-type]
        case "BiLSTM":
            return BiLSTMClassifier(**base_kw, use_attention=True)  # type: ignore[arg-type]
        case "CNN":
            return CNNClassifier(
                in_features=config.in_features,
                num_filters=128,
                num_blocks=3,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "TCN":
            return TCNClassifier(
                in_features=config.in_features,
                num_channels=256,
                kernel_size=3,
                depth=6,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "Transformer":
            return TransformerClassifier(
                in_features=config.in_features,
                d_model=128,
                n_heads=8,
                num_layers=4,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "CNN2DLSTM":
            return CNN2DRNNClassifier(
                in_features=config.in_features,
                num_filters=64,
                cnn_depth=3,
                hidden_size=config.hidden_size,
                num_rnn_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                rnn_type="lstm",
            )
        case "CNN2DGRU":
            return CNN2DRNNClassifier(
                in_features=config.in_features,
                num_filters=64,
                cnn_depth=3,
                hidden_size=config.hidden_size,
                num_rnn_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                rnn_type="gru",
            )
        case "Mamba":
            return MambaClassifier(
                in_features=config.in_features,
                d_model=config.d_model,
                d_state=16,
                num_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                mimo_rank=4,
            )
        case _:
            raise ValueError(f"Unknown model: {name}")


def _build_ds_variant(name: str, config: DLConfig) -> nn.Module:
    """Build one of the Deep-Stack base model variants."""
    match name:
        case "DS_RNN_A":
            return RNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_RNN_B":
            return RNNClassifier(
                config.in_features, 256, 4, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_GRU_A":
            return GRUClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_GRU_B":
            return GRUClassifier(
                config.in_features, 256, 3, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_LSTM_A":
            return LSTMClassifier(config.in_features, 64, 2, config.num_classes, 0.2, False, True)
        case "DS_LSTM_B":
            return LSTMClassifier(config.in_features, 256, 3, config.num_classes, 0.3, True, True)
        case "DS_CNN_A":
            return CNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2, [3, 5])
        case "DS_CNN_B":
            return CNNClassifier(config.in_features, 256, 4, config.num_classes, 0.3, [3, 5, 7, 9])
        case "DS_TCN_A":
            return TCNClassifier(config.in_features, 128, 3, 4, config.num_classes, 0.2)
        case "DS_TCN_B":
            return TCNClassifier(config.in_features, 512, 7, 8, config.num_classes, 0.3)
        case "DS_Transformer_A":
            return TransformerClassifier(config.in_features, 64, 4, 2, config.num_classes, 0.1)
        case "DS_Transformer_B":
            return TransformerClassifier(config.in_features, 256, 8, 4, config.num_classes, 0.15)
        case "DS_BiLSTM_A":
            return BiLSTMClassifier(config.in_features, 64, 2, config.num_classes, 0.2, True)
        case "DS_BiLSTM_B":
            return BiLSTMClassifier(config.in_features, 256, 3, config.num_classes, 0.3, True)
        case "DS_CNN2DRNN_A":
            return CNN2DRNNClassifier(
                config.in_features, 32, 2, 64, 2, config.num_classes, 0.2, "lstm"
            )
        case "DS_CNN2DRNN_B":
            return CNN2DRNNClassifier(
                config.in_features,
                128,
                3,
                256,
                3,
                config.num_classes,
                0.3,
                "lstm",
                bidirectional=True,
            )
        case "DS_Mamba_A":
            return MambaClassifier(
                config.in_features, 64, 16, 2, 2, config.num_classes, 0.2, mimo_rank=4
            )
        case "DS_Mamba_B":
            return MambaClassifier(
                config.in_features, 256, 32, 4, 2, config.num_classes, 0.3, mimo_rank=8
            )
        case _:
            raise ValueError(f"Unknown DS variant: {name}")


def _extract_logits(
    model: nn.Module,
    loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    device: torch.device,
) -> np.ndarray:
    """Run model on loader, return concatenated logits (N, num_classes)."""
    model.eval().to(device)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            parts.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Module-level picklable workers (spawn-safe, no DataParallel) ─────────────


def _single_model_worker(
    name: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Train one model on `device_str`, save checkpoint, return metrics."""
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()

    dl = DLDataLoader(config)
    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)

    model = _build_model_by_name(name, config)
    trainer = Trainer(config, run_name=name)
    save_path = config.models_dir / f"{name}_seed{config.seed}.pt"

    t0 = time.time()
    result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    metrics = evaluate_model_on_test(
        model, test_loader, config, classes, name, log_wandb=config.use_wandb
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": name,
        "type": "single",
        "seed": config.seed,
        "device": device_str,
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_val_mcc": result.best_val_mcc,
        "best_epoch": result.best_epoch,
        "train_time_s": round(elapsed, 1),
        "val_mccs": result.val_mccs,
    }


def _build_hpo_best_model(model_type: str, config: DLConfig, params: dict) -> nn.Module:
    """Rebuild the best Optuna model from stored HPO trial params."""
    dropout = float(params.get("dropout", config.dropout))
    kernel_sets: list[list[int]] = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9]]
    if model_type == "rnn":
        return RNNClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
        )
    elif model_type == "gru":
        return GRUClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
        )
    elif model_type == "lstm":
        return LSTMClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("bidirectional", False)),
            bool(params.get("use_attention", False)),
        )
    elif model_type == "bilstm":
        return BiLSTMClassifier(
            config.in_features,
            int(params.get("hidden_size", config.hidden_size)),
            int(params.get("num_layers", config.num_layers)),
            config.num_classes,
            dropout,
            bool(params.get("use_attention", True)),
        )
    elif model_type == "cnn":
        ks = kernel_sets[int(params.get("kernel_set", 0))]
        return CNNClassifier(
            config.in_features,
            int(params.get("num_filters", 64)),
            int(params.get("num_blocks", 2)),
            config.num_classes,
            dropout,
            ks,
        )
    elif model_type == "tcn":
        return TCNClassifier(
            config.in_features,
            int(params.get("num_channels", 128)),
            int(params.get("kernel_size", 3)),
            int(params.get("depth", 4)),
            config.num_classes,
            dropout,
        )
    elif model_type == "transformer":
        d_model = int(params.get("d_model", 64))
        n_heads = int(params.get("n_heads", 4))
        while d_model % n_heads != 0:
            n_heads = max(1, n_heads // 2)
        return TransformerClassifier(
            config.in_features,
            d_model,
            n_heads,
            int(params.get("num_layers", 2)),
            config.num_classes,
            dropout,
        )
    elif model_type == "cnn2drnn":
        return CNN2DRNNClassifier(
            in_features=config.in_features,
            num_filters=int(params.get("num_filters", 32)),
            cnn_depth=int(params.get("cnn_depth", 2)),
            hidden_size=int(params.get("hidden_size", config.hidden_size)),
            num_rnn_layers=int(params.get("num_rnn_layers", 2)),
            num_classes=config.num_classes,
            dropout=dropout,
            rnn_type=str(params.get("rnn_type", "lstm")),
            bidirectional=bool(params.get("bidirectional", False)),
        )
    elif model_type == "mamba":
        return MambaClassifier(
            in_features=config.in_features,
            d_model=int(params.get("d_model", config.d_model)),
            d_state=int(params.get("d_state", 16)),
            num_layers=int(params.get("num_layers", config.num_layers)),
            expand=int(params.get("expand", 2)),
            num_classes=config.num_classes,
            dropout=dropout,
            mimo_rank=int(params.get("mimo_rank", 4)),
        )
    else:
        raise ValueError(f"Unknown HPO model_type: {model_type!r}")


def _hpo_worker(
    model_type: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Run Optuna HPO, then rebuild best model, train it, and evaluate on test."""
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    study = run_hpo_study(model_type, config, dl, X_tr, y_tr, X_val, y_val)
    save_optuna_plots(study, model_type, config.plots_dir)

    # Rebuild best model and apply best training HPs
    best_params = study.best_trial.params
    best_cfg = copy.copy(config)
    best_cfg.learning_rate = float(best_params.get("lr", config.learning_rate))
    best_cfg.weight_decay = float(best_params.get("weight_decay", config.weight_decay))
    best_cfg.l1_lambda = float(best_params.get("l1_lambda", config.l1_lambda))
    best_cfg.gradient_clip = float(best_params.get("gradient_clip", config.gradient_clip))
    best_cfg.loss_type = str(best_params.get("loss_type", config.loss_type))
    best_cfg.focal_gamma = float(best_params.get("focal_gamma", config.focal_gamma))
    best_cfg.optimizer_type = str(best_params.get("optimizer_type", config.optimizer_type))
    best_cfg.momentum = float(best_params.get("momentum", config.momentum))
    best_cfg.scheduler_type = str(best_params.get("scheduler_type", config.scheduler_type))

    model = _build_hpo_best_model(model_type, best_cfg, best_params)
    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)
    model_name = f"HPO_{model_type.upper()}"
    save_path = best_cfg.models_dir / f"{model_name}_seed{best_cfg.seed}.pt"
    trainer = Trainer(best_cfg, run_name=model_name)
    t0 = time.time()
    result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    # ── Final-fit: retrain on X_tr + X_val to close the val→test gap ───────────
    # After HPO identifies the best params, we squeeze out more signal by
    # training on the full train+val set for exactly best_epoch iterations.
    X_full = np.concatenate([X_tr, X_val], axis=0)
    y_full = np.concatenate([y_tr, y_val], axis=0)
    full_loader = dl.make_loader(X_full, y_full, shuffle=True)
    final_cfg = copy.copy(best_cfg)
    final_cfg.num_epochs = max(result.best_epoch, 10)
    final_cfg.patience = final_cfg.num_epochs + 1  # no early stopping
    final_cfg.use_wandb = False
    final_model = _build_hpo_best_model(model_type, final_cfg, best_params)
    # NOTE: do NOT pass save_path here — the original HPO checkpoint (saved at best
    # val-MCC on the held-out split) must NOT be overwritten by final-fit which
    # evaluates on training data and would corrupt the saved checkpoint MCC.
    Trainer(final_cfg, run_name=f"{model_name}_finalfit").fit(final_model, full_loader, val_loader)
    # Evaluate both the HPO model (best val-MCC) and the final-fit model on val split,
    # then overwrite save_path only if final-fit is strictly better.
    from sklearn.metrics import matthews_corrcoef as _mcc_fn

    _eval_t = Trainer(final_cfg)
    _, ff_preds, ff_tgts = _eval_t.evaluate(final_model, val_loader)
    ff_val_mcc = float(_mcc_fn(ff_tgts, ff_preds))
    if ff_val_mcc > result.best_val_mcc:
        torch.save(final_model.state_dict(), save_path)
        final_model_to_eval = final_model
    else:
        final_model_to_eval = model
    metrics = evaluate_model_on_test(
        final_model_to_eval,
        test_loader,
        best_cfg,
        classes,
        model_name,
        log_wandb=best_cfg.use_wandb,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_type": model_type,
        "device": device_str,
        "best_val_mcc": study.best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "model": model_name,
        "type": "hpo",
        "seed": config.seed,
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_epoch": result.best_epoch,
        "train_time_s": round(elapsed, 1),
        "val_mccs": result.val_mccs,
    }


def _nas_worker(
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Optuna NAS: search arch + HPs, train best model, return metrics."""
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    study, nas_model = run_nas_study(config, dl, X_tr, y_tr, X_val, y_val)
    save_optuna_plots(study, "nas", config.plots_dir)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)
    metrics = evaluate_model_on_test(
        nas_model, test_loader, config, classes, "OptunaNet", log_wandb=config.use_wandb
    )
    torch.save(nas_model.state_dict(), config.models_dir / f"OptunaNet_seed{config.seed}.pt")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": "OptunaNet_NAS",
        "type": "optuna_nas",
        "seed": config.seed,
        "device": device_str,
        "best_arch": json.dumps(study.best_trial.params),
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_val_mcc": study.best_value,
        "val_mccs": [],
    }


def _moe_expert_worker(
    class_idx: int,
    class_name: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Train one binary MoE expert: class_idx vs rest. Returns save path."""
    config = copy.copy(config)
    config.device = device_str
    config.num_classes = 2
    config.make_dirs()
    dl = DLDataLoader(config)
    y_tr_bin = (y_tr == class_idx).astype(int)
    y_val_bin = (y_val == class_idx).astype(int)
    tr_loader = dl.make_loader(X_tr, y_tr_bin, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val_bin, shuffle=False)
    model = build_moe_expert(
        config.in_features, hidden_size=128, num_layers=2, dropout=config.dropout
    )
    trainer = Trainer(config)
    trainer.fit(model, tr_loader, val_loader)
    save_path = config.models_dir / f"MoE_expert_{class_name}_seed{config.seed}.pt"
    torch.save(model.state_dict(), save_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"class_idx": class_idx, "class_name": class_name, "save_path": str(save_path)}


def _moe_binary_hpo_worker(
    arch_type: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Run binary Optuna HPO for one MoE expert arch type (class 0 vs rest as proxy task).

    Deprecated: use _moe_per_class_hpo_worker for per-class tuning.
    """
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    y_tr_bin = (y_tr == 0).astype(int)
    y_val_bin = (y_val == 0).astype(int)
    study = run_binary_moe_hpo_study(arch_type, config, dl, X_tr, y_tr_bin, X_val, y_val_bin)
    save_optuna_plots(study, f"moe_binary_{arch_type}", config.plots_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "arch_type": arch_type,
        "device": device_str,
        "best_val_mcc": study.best_value,
        "best_params": study.best_trial.params,
    }


def _moe_per_class_hpo_worker(
    arch_type: str,
    class_idx: int,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Run binary Optuna HPO for one MoE expert arch type tuned for a specific class."""
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    y_tr_bin = (y_tr == class_idx).astype(int)
    y_val_bin = (y_val == class_idx).astype(int)
    study = run_binary_moe_hpo_study(
        arch_type,
        config,
        dl,
        X_tr,
        y_tr_bin,
        X_val,
        y_val_bin,
        study_suffix=f"_c{class_idx}",
    )
    save_optuna_plots(study, f"moe_binary_{arch_type}_c{class_idx}", config.plots_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "arch_type": arch_type,
        "class_idx": class_idx,
        "device": device_str,
        "best_val_mcc": study.best_value,
        "best_params": study.best_trial.params,
    }


def _moe_typed_expert_worker(
    class_idx: int,
    class_name: str,
    arch_type: str,
    combo_name: str,
    device_str: str,
    config: DLConfig,
    best_params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Train one binary MoE expert with a specific arch type and Optuna-tuned params."""
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.num_classes = 2
    # Apply training HPs found by Optuna
    config.learning_rate = float(best_params.get("lr", config.learning_rate))
    config.weight_decay = float(best_params.get("weight_decay", config.weight_decay))
    config.l1_lambda = float(best_params.get("l1_lambda", config.l1_lambda))
    config.gradient_clip = float(best_params.get("gradient_clip", config.gradient_clip))
    config.loss_type = str(best_params.get("loss_type", config.loss_type))
    config.focal_gamma = float(best_params.get("focal_gamma", config.focal_gamma))
    config.optimizer_type = str(best_params.get("optimizer_type", config.optimizer_type))
    config.momentum = float(best_params.get("momentum", config.momentum))
    config.scheduler_type = str(best_params.get("scheduler_type", config.scheduler_type))
    config.make_dirs()
    dl = DLDataLoader(config)
    y_tr_bin = (y_tr == class_idx).astype(int)
    y_val_bin = (y_val == class_idx).astype(int)
    tr_loader = dl.make_loader(X_tr, y_tr_bin, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val_bin, shuffle=False)
    model = build_moe_expert_typed(arch_type, config.in_features, best_params)
    # Balanced class weights to handle 1:3 imbalance in binary (one-vs-rest) training
    n = len(y_tr_bin)
    n_pos = int(y_tr_bin.sum())
    n_neg = n - n_pos
    class_weights: torch.Tensor | None = None
    if n_pos > 0 and n_neg > 0:
        class_weights = torch.tensor([n / (2.0 * n_neg), n / (2.0 * n_pos)], dtype=torch.float32)
    trainer = Trainer(config, class_weights=class_weights)
    result = trainer.fit(model, tr_loader, val_loader)
    save_path = (
        config.models_dir / f"MoE_{combo_name}_expert{class_idx}_{arch_type}_seed{config.seed}.pt"
    )
    torch.save(model.state_dict(), save_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "arch_type": arch_type,
        "save_path": str(save_path),
        "best_val_mcc": result.best_val_mcc,
    }


def _ds_base_worker(
    name: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Train one DS base model variant. Returns metrics + save path."""
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    model = _build_ds_variant(name, config)
    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)
    trainer = Trainer(config)
    result = trainer.fit(model, tr_loader, val_loader)
    metrics = evaluate_model_on_test(model, test_loader, config, classes, name, log_wandb=False)
    save_path = config.models_dir / f"{name}_seed{config.seed}.pt"
    torch.save(model, save_path)  # save full model to avoid arch mismatch on reload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": name,
        "type": "ds_base",
        "seed": config.seed,
        "device": device_str,
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_val_mcc": result.best_val_mcc,
        "val_mccs": result.val_mccs,
        "save_path": str(save_path),
    }


def _moe_mo_hpo_worker(
    combo_name: str,
    arch_list: list[str],
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Spawn-safe worker: run multi-objective (NSGA-II) HPO for one SoftMoE combo."""
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    study, best_params = run_moe_multiobjective_study(
        arch_list=arch_list,
        config=config,
        data_loader=dl,
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        combo_name=combo_name,
    )
    pareto = study.best_trials
    pareto_best_mcc = max((t.values[0] for t in pareto if t.values), default=0.0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "combo_name": combo_name,
        "device": device_str,
        "best_params": best_params,
        "pareto_best_mcc": pareto_best_mcc,
        "n_pareto": len(pareto),
    }


def _softmoe_combo_worker(
    combo_name: str,
    arch_list: list[str],
    device_str: str,
    config: DLConfig,
    best_params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Train one SoftMixtureOfExperts combo end-to-end with Optuna-tuned params.

    All experts are full multiclass classifiers. Gating network is jointly trained.
    Auxiliary load-balance loss (Switch-Transformer style) is handled automatically
    by the Trainer (reads model.last_aux_loss after each forward pass).
    """
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    # Apply training HPs from multi-objective HPO
    config.learning_rate = float(best_params.get("lr", config.learning_rate))
    config.weight_decay = float(best_params.get("weight_decay", config.weight_decay))
    config.l1_lambda = float(best_params.get("l1_lambda", config.l1_lambda))
    config.gradient_clip = float(best_params.get("gradient_clip", config.gradient_clip))
    config.optimizer_type = str(best_params.get("optimizer_type", "adamw"))
    config.scheduler_type = str(best_params.get("scheduler_type", config.scheduler_type))
    config.loss_type = "ce"
    config.make_dirs()
    dl = DLDataLoader(config)
    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)

    # Expert arch params extracted from HPO best params
    expert_params: dict = {
        k: best_params[k]
        for k in (
            "dropout",
            "hidden_size",
            "num_layers",
            "bidirectional",
            "use_attention",
            "num_filters",
            "num_blocks",
            "kernel_set",
        )
        if k in best_params
    }

    gate_hidden = int(best_params.get("gate_hidden", 64))
    gate_dropout = float(best_params.get("gate_dropout", 0.1))
    aux_loss_weight = float(best_params.get("aux_loss_weight", 0.01))
    top_k_raw = best_params.get("top_k", "none")
    top_k: int | None = None if top_k_raw == "none" else int(top_k_raw)
    if top_k is not None and top_k >= len(arch_list):
        top_k = None

    params_per_expert = [dict(expert_params) for _ in arch_list]
    model = build_soft_moe(
        arch_list=arch_list,
        in_features=config.in_features,
        num_classes=config.num_classes,
        params_per_expert=params_per_expert,
        gate_hidden=gate_hidden,
        gate_dropout=gate_dropout,
        aux_loss_weight=aux_loss_weight,
        top_k=top_k,
    )
    save_path = config.models_dir / f"{combo_name}_seed{config.seed}.pt"
    trainer = Trainer(config, run_name=combo_name)
    result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)
    metrics = evaluate_model_on_test(
        model, test_loader, config, classes, combo_name, log_wandb=config.use_wandb
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": combo_name,
        "type": "soft_moe",
        "seed": config.seed,
        "device": device_str,
        "arch_list": "+".join(arch_list),
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_val_mcc": result.best_val_mcc,
        "val_mccs": result.val_mccs,
    }


def _meta_model_worker(
    name: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    meta_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    meta_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    meta_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, object]:
    """Train one MetaFusion model that also ingests noise/path context."""
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()

    dl = DLDataLoader(config)
    tr_loader = dl.make_meta_loader(X_tr, meta_tr, y_tr, shuffle=True)
    val_loader = dl.make_meta_loader(X_val, meta_val, y_val, shuffle=False)
    test_loader = dl.make_meta_loader(X_test, meta_test, y_test, shuffle=False)

    rnn_type = "gru" if "GRU" in name else "lstm"
    model: nn.Module = MetaFusionClassifier(
        in_features=config.in_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        meta_embed_dim=config.meta_embed_dim,
        rnn_type=rnn_type,
    )
    save_path = config.models_dir / f"{name}_seed{config.seed}.pt"
    trainer = Trainer(config, run_name=name)
    t0 = time.time()
    result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    metrics = evaluate_model_on_test(
        model, test_loader, config, classes, name, log_wandb=config.use_wandb
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": name,
        "type": "meta_fusion",
        "seed": config.seed,
        "device": device_str,
        "test_mcc": metrics["mcc"],
        "test_acc": metrics["accuracy"],
        "best_val_mcc": result.best_val_mcc,
        "best_epoch": result.best_epoch,
        "train_time_s": round(elapsed, 1),
        "val_mccs": result.val_mccs,
    }


def main() -> None:
    args = parse_args()

    n_gpus = args.num_gpus if args.num_gpus > 0 else _available_gpus()
    devices = _build_device_pool(n_gpus)
    print(f"Device pool: {devices}  (parallel workers: {len(devices)})")

    config = DLConfig(
        data_path=args.data,
        seed=args.seed,
        num_epochs=args.epochs,
        n_trials=args.trials,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        use_metadata=not args.no_meta,
    )
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.models_dir is not None:
        config.models_dir = args.models_dir
    config.make_dirs()
    set_seed(config.seed)

    # ── Data (main process) ──────────────────────────────────────────────────
    data_loader = DLDataLoader(config)
    X, y = data_loader.load_and_preprocess()
    classes = data_loader.classes_
    print(f"Dataset: {X.shape}  classes: {classes}  seed: {config.seed}")

    X_train, X_test, y_train, y_test = data_loader.train_test_split(X, y)
    X_tr, X_val, y_tr, y_val = data_loader.train_test_split(X_train, y_train)

    all_results: list[dict[str, object]] = []
    curve_data: list[dict[str, object]] = []
    ctx = mp.get_context("spawn")

    # ── Phase 1: Parallel single-model training ──────────────────────────────
    # Each model lands on its own GPU (round-robin). 2 GPUs → 2 models run simultaneously.
    print(f"\n[Phase 1] {len(SINGLE_MODEL_NAMES)} models × {len(devices)} GPU(s)")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _single_model_worker,
                name,
                devices[i % len(devices)],
                config,
                X_tr,
                y_tr,
                X_val,
                y_val,
                X_test,
                y_test,
                classes,
            ): name
            for i, name in enumerate(SINGLE_MODEL_NAMES)
        }
        for future in as_completed(futures):
            row = future.result()
            val_mccs: list[float] = row.pop("val_mccs", [])
            all_results.append(row)
            curve_data.append({"name": str(row["model"]), "val_mccs": val_mccs})
            print(
                f"  [{row['model']}] {row['device']}  "
                f"test_mcc={float(row['test_mcc']):.4f}  "
                f"best_val_mcc={float(row['best_val_mcc']):.4f}  "
                f"epoch={row['best_epoch']}  {row['train_time_s']}s"
            )

    # ── Phase 1.5: Metadata-fusion models (noise + concurrent_noise_path) ────
    if config.use_metadata and not args.no_meta:
        print(
            f"\n[Phase 1.5] Metadata-fusion models ({len(META_MODEL_NAMES)}) — noise + path context"
        )
        meta_dl = DLDataLoader(config)
        X_m, meta_m, y_m = meta_dl.load_and_preprocess_with_meta()
        (
            X_m_train,
            X_m_test,
            meta_m_train,
            meta_m_test,
            y_m_train,
            y_m_test,
        ) = meta_dl.train_test_split_with_meta(X_m, meta_m, y_m)
        (
            X_m_tr,
            X_m_val,
            meta_m_tr,
            meta_m_val,
            y_m_tr,
            y_m_val,
        ) = meta_dl.train_test_split_with_meta(X_m_train, meta_m_train, y_m_train)

        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            meta_futures = {
                pool.submit(
                    _meta_model_worker,
                    name,
                    devices[i % len(devices)],
                    config,
                    X_m_tr,
                    meta_m_tr,
                    y_m_tr,
                    X_m_val,
                    meta_m_val,
                    y_m_val,
                    X_m_test,
                    meta_m_test,
                    y_m_test,
                    classes,
                ): name
                for i, name in enumerate(META_MODEL_NAMES)
            }
            for future in as_completed(meta_futures):
                row = future.result()
                mv: list[float] = row.pop("val_mccs", [])
                all_results.append(row)
                curve_data.append({"name": str(row["model"]), "val_mccs": mv})
                print(
                    f"  [{row['model']}] {row['device']}  "
                    f"test_mcc={float(row['test_mcc']):.4f}  "
                    f"best_val_mcc={float(row['best_val_mcc']):.4f}  "
                    f"{row['train_time_s']}s"
                )

    # ── Phase 2: Ensembles (main process — reload saved checkpoints) ─────────
    print("\n[Phase 2] Building ensembles…")
    trained_models: dict[str, nn.Module] = {}
    for name in SINGLE_MODEL_NAMES:
        m = _build_model_by_name(name, config)
        state = torch.load(
            config.models_dir / f"{name}_seed{config.seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        m.load_state_dict(state)
        trained_models[name] = m

    main_device = torch.device(devices[0])
    test_loader = data_loader.make_loader(X_test, y_test, shuffle=False)
    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)

    ensemble_combos: dict[str, list[str]] = {
        "Ensemble_RNN_GRU": ["RNN", "GRU"],
        "Ensemble_GRU_LSTM": ["GRU", "LSTM"],
        "Ensemble_LSTM_CNN": ["LSTM", "CNN"],
        "Ensemble_CNN_TCN": ["CNN", "TCN"],
        "Ensemble_TCN_Transformer": ["TCN", "Transformer"],
        "Ensemble_BiLSTM_LSTM": ["BiLSTM", "LSTM"],
        "Ensemble_Mamba_LSTM": ["Mamba", "LSTM"],
        "Ensemble_RNN_GRU_LSTM": ["RNN", "GRU", "LSTM"],
        "Ensemble_All": SINGLE_MODEL_NAMES,
    }
    for ens_name, member_names in ensemble_combos.items():
        members = [trained_models[n] for n in member_names]
        ensemble = VotingEnsemble(members).to(main_device)
        metrics = evaluate_model_on_test(
            ensemble, test_loader, config, classes, ens_name, log_wandb=config.use_wandb
        )
        torch.save(
            ensemble.state_dict(),
            config.models_dir / f"{ens_name}_seed{config.seed}.pt",
        )
        all_results.append(
            {
                "model": ens_name,
                "type": "voting_ensemble",
                "seed": config.seed,
                "device": str(main_device),
                "members": "+".join(member_names),
                "test_mcc": metrics["mcc"],
                "test_acc": metrics["accuracy"],
            }
        )
        print(f"  [{ens_name}] test_mcc={float(metrics['mcc']):.4f}")

    print("  [Stacking_All] training meta-learner…")
    stacking = StackingEnsemble(
        list(trained_models.values()), config.num_classes, config.dropout
    ).to(main_device)
    stacking.freeze_base_models()
    stack_cfg = copy.copy(config)
    stack_cfg.device = str(main_device)
    stack_trainer = Trainer(stack_cfg, run_name="Stacking_All")
    stack_save = config.models_dir / f"Stacking_All_seed{config.seed}.pt"
    stack_result = stack_trainer.fit(stacking, tr_loader, val_loader, save_path=stack_save)
    metrics = evaluate_model_on_test(
        stacking, test_loader, config, classes, "Stacking_All", log_wandb=config.use_wandb
    )
    all_results.append(
        {
            "model": "Stacking_All",
            "type": "stacking_ensemble",
            "seed": config.seed,
            "device": str(main_device),
            "test_mcc": metrics["mcc"],
            "test_acc": metrics["accuracy"],
            "best_val_mcc": stack_result.best_val_mcc,
        }
    )
    curve_data.append({"name": "Stacking_All", "val_mccs": stack_result.val_mccs})
    print(f"  [Stacking_All] test_mcc={float(metrics['mcc']):.4f}")

    # ── Phase 3: Parallel Optuna HPO + NAS ───────────────────────────────────
    if not args.no_optuna:
        print(f"\n[Phase 3] HPO ({len(HPO_TYPES)} studies) + NAS across {len(devices)} GPU(s)…")
        # Pre-init DB schema + studies in main process → no spawn race
        all_study_names = [f"{config.optuna_study_prefix}_{t}" for t in HPO_TYPES] + [
            f"{config.optuna_study_prefix}_nas"
        ]
        init_optuna_db(config.optuna_storage, all_study_names)
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            hpo_futures = {
                pool.submit(
                    _hpo_worker,
                    model_type,
                    devices[i % len(devices)],
                    config,
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): model_type
                for i, model_type in enumerate(HPO_TYPES)
            }
            for future in as_completed(hpo_futures):
                res = future.result()
                hpo_val_mccs: list[float] = list(res.pop("val_mccs", []))  # type: ignore[arg-type]
                all_results.append(
                    {
                        "model": res["model"],
                        "type": res["type"],
                        "seed": res["seed"],
                        "device": res["device"],
                        "test_mcc": res["test_mcc"],
                        "test_acc": res["test_acc"],
                        "best_val_mcc": res["best_val_mcc"],
                        "best_epoch": res.get("best_epoch", 0),
                        "train_time_s": res.get("train_time_s", 0.0),
                    }
                )
                curve_data.append({"name": str(res["model"]), "val_mccs": hpo_val_mccs})
                print(
                    f"  [HPO {res['model_type']}] {res['device']}  "
                    f"best_val_mcc={float(res['best_val_mcc']):.4f}  "
                    f"test_mcc={float(res['test_mcc']):.4f}  "
                    f"params={res['best_params']}"
                )

        print("  [NAS] architecture search…")
        nas_row = _nas_worker(
            devices[0],
            config,
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            y_test,
            classes,
        )
        nas_val_mccs: list[float] = nas_row.pop("val_mccs", [])  # type: ignore[assignment]
        all_results.append(nas_row)
        curve_data.append({"name": "OptunaNet_NAS", "val_mccs": nas_val_mccs})
        nas_mcc = float(nas_row["test_mcc"])  # type: ignore[arg-type]
        print(f"  [NAS] {nas_row['device']}  arch={nas_row['best_arch']}  test_mcc={nas_mcc:.4f}")

    # ── Phase 4: Mixture-of-Experts — end-to-end SoftMoE with multi-objective HPO ─
    if not args.no_moe:
        n_combos = len(MOE_COMBOS)
        print(
            f"\n[Phase 4a] SoftMoE multi-objective HPO (NSGA-II) — "
            f"{n_combos} combos across {len(devices)} GPU(s)…"
        )
        # Pre-create study entries in DB to avoid spawn-race
        mo_study_names = [f"{config.optuna_study_prefix}_moe_mo_{name}" for name in MOE_COMBOS]
        # Multi-objective studies use directions=[] so cannot use init_optuna_db directly;
        # they are created inside the worker / study call (load_if_exists=True).

        # Phase 4a: run multi-objective HPO for each combo in parallel across GPUs
        combo_best_params: dict[str, dict] = {}
        if not args.no_optuna:
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
                mo_futures = {
                    pool.submit(
                        _moe_mo_hpo_worker,
                        combo_name,
                        arch_list,
                        devices[i % len(devices)],
                        copy.copy(config),
                        X_tr,
                        y_tr,
                        X_val,
                        y_val,
                    ): combo_name
                    for i, (combo_name, arch_list) in enumerate(MOE_COMBOS.items())
                }
                for future in as_completed(mo_futures):
                    res = future.result()
                    combo_best_params[str(res["combo_name"])] = dict(res["best_params"])  # type: ignore[arg-type]
                    print(
                        f"  [MoE HPO {res['combo_name']}] {res['device']}  "
                        f"pareto_best_mcc={float(res['pareto_best_mcc']):.4f}  "
                        f"n_pareto={res['n_pareto']}"
                    )
        else:
            combo_best_params = {name: {} for name in MOE_COMBOS}

        # Phase 4b: train each combo end-to-end with best params
        print(
            f"\n[Phase 4b] SoftMoE e2e training — {n_combos} combos across {len(devices)} GPU(s)…"
        )
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            train_futures = {
                pool.submit(
                    _softmoe_combo_worker,
                    combo_name,
                    arch_list,
                    devices[i % len(devices)],
                    copy.copy(config),
                    combo_best_params.get(combo_name, {}),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): combo_name
                for i, (combo_name, arch_list) in enumerate(MOE_COMBOS.items())
            }
            for future in as_completed(train_futures):
                res = future.result()
                moe_val_mccs: list[float] = list(res.pop("val_mccs", []))  # type: ignore[arg-type]
                all_results.append(res)
                curve_data.append({"name": str(res["model"]), "val_mccs": moe_val_mccs})
                print(
                    f"  [{res['model']}] test_mcc={float(res['test_mcc']):.4f}"
                    f"  acc={float(res['test_acc']):.4f}"
                    f"  best_val_mcc={float(res['best_val_mcc']):.4f}"
                )

    # ── Phase 5: Deep Stacking ─────────────────────────────────────────────────
    if not args.no_deepstack:
        n_base = len(DS_BASE_NAMES)
        print(f"\n[Phase 5] DeepStack — {n_base} base models across {len(devices)} GPU(s)…")
        ds_config = copy.copy(config)
        dl_ds = DLDataLoader(ds_config)
        tr_loader_ds = dl_ds.make_loader(X_tr, y_tr, shuffle=True)
        val_loader_ds = dl_ds.make_loader(X_val, y_val, shuffle=False)
        test_loader_ds = dl_ds.make_loader(X_test, y_test, shuffle=False)

        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            ds_futures = {
                pool.submit(
                    _ds_base_worker,
                    name,
                    devices[i % len(devices)],
                    copy.copy(config),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): name
                for i, name in enumerate(DS_BASE_NAMES)
            }
            ds_base_results: list[dict[str, object]] = []
            for future in as_completed(ds_futures):
                res = future.result()
                ds_base_results.append(res)
                all_results.append({k: v for k, v in res.items() if k != "save_path"})
                print(
                    f"  [DS-base {res['model']}] test_mcc={float(res['test_mcc']):.4f}  "
                    f"best_val_mcc={float(res['best_val_mcc']):.4f}"
                )

        ds_base_results.sort(key=lambda d: str(d["model"]))

        # ── Reload base models and extract concat logits (N, n_base * num_classes) ─
        dev0 = torch.device(devices[0])
        base_models_loaded: list[nn.Module] = []
        for info in sorted(ds_base_results, key=lambda d: str(d["model"])):
            _raw = torch.load(str(info["save_path"]), map_location="cpu", weights_only=False)
            if isinstance(_raw, nn.Module):
                m = _raw  # new format: full model saved
            else:
                # legacy format: state dict only — rebuild from current arch definition
                m = _build_ds_variant(str(info["model"]), config)
                m.load_state_dict(_raw)
            base_models_loaded.append(m)

        def _concat_logits(loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]") -> np.ndarray:
            return np.concatenate(
                [_extract_logits(m, loader, dev0) for m in base_models_loaded], axis=-1
            )

        base_feat_tr = _concat_logits(tr_loader_ds)  # (N, n_base * C)
        base_feat_val = _concat_logits(val_loader_ds)
        base_feat_test = _concat_logits(test_loader_ds)

        # ── Stage 1: Freeze base models, train Level2 nets (full multiclass) ──
        # Level2Net now outputs C classes (not binary), preserving full inter-class signal.
        from dl.models.deep_stack import DeepMetaMLP as _DeepMetaMLP
        from dl.models.deep_stack import Level2Net as _Level2Net

        n_base_feats = base_feat_tr.shape[1]  # n_base * C
        l2_config = copy.copy(config)
        l2_dl = DLDataLoader(l2_config)
        level2_nets: list[nn.Module] = []

        print("  [DeepStack stage-1] training Level2 nets (multiclass)…")
        for cls_name in classes:
            l2_tr = l2_dl.make_loader(base_feat_tr, y_tr, shuffle=True)
            l2_val = l2_dl.make_loader(base_feat_val, y_val, shuffle=False)
            l2_net: nn.Module = _Level2Net(n_base_feats, config.num_classes, dropout=config.dropout)
            Trainer(l2_config).fit(l2_net, l2_tr, l2_val)
            save_path_l2 = config.models_dir / f"DS_L2_{cls_name}_seed{config.seed}.pt"
            torch.save(l2_net.state_dict(), save_path_l2)
            level2_nets.append(l2_net)

        # ── Extract Level2 features: concat softmax of each L2 net → (N, n_l2 * C) ──
        def _l2_feats(feats: np.ndarray) -> np.ndarray:
            feat_t = torch.tensor(feats, dtype=torch.float32).to(dev0)
            parts = []
            for l2 in level2_nets:
                l2.eval().to(dev0)
                with torch.no_grad():
                    parts.append(torch.softmax(l2(feat_t), dim=-1).cpu().numpy())
            return np.concatenate(parts, axis=-1)  # (N, n_l2 * C)

        l2_feat_tr = _l2_feats(base_feat_tr)
        l2_feat_val = _l2_feats(base_feat_val)
        l2_feat_test = _l2_feats(base_feat_test)

        # ── Train DeepMetaMLP on L2 features ──────────────────────────────────
        print("  [DeepStack stage-1] training DeepMetaMLP…")
        meta_ds_config = copy.copy(config)
        meta_ds_dl = DLDataLoader(meta_ds_config)
        ds_meta_tr = meta_ds_dl.make_loader(l2_feat_tr, y_tr, shuffle=True)
        ds_meta_val = meta_ds_dl.make_loader(l2_feat_val, y_val, shuffle=False)
        ds_meta_test = meta_ds_dl.make_loader(l2_feat_test, y_test, shuffle=False)

        n_l2_feats = l2_feat_tr.shape[1]  # n_l2 * C
        ds_meta: nn.Module = _DeepMetaMLP(n_l2_feats, config.num_classes, config.dropout)
        ds_meta_save = config.models_dir / f"DS_meta_seed{config.seed}.pt"
        stage1_trainer = Trainer(meta_ds_config, run_name="DeepStackMeta_S1")
        stage1_result = stage1_trainer.fit(ds_meta, ds_meta_tr, ds_meta_val, save_path=ds_meta_save)

        # ── Stage 2: joint end-to-end fine-tuning (DeepStackEnsemble) ─────────
        # Reassemble the full pipeline into a single nn.Module for joint fine-tune.
        # Use a low LR (10× smaller) to avoid overwriting learned representations.
        print("  [DeepStack stage-2] joint end-to-end fine-tuning…")
        from dl.models.deep_stack import DeepStackEnsemble as _DeepStackEnsemble

        # Rebuild Level2 nets with loaded state (on CPU initially)
        l2_nets_loaded: list[nn.Module] = []
        for cls_name in classes:
            l2_r = _Level2Net(n_base_feats, config.num_classes, dropout=config.dropout)
            l2_r.load_state_dict(
                torch.load(
                    config.models_dir / f"DS_L2_{cls_name}_seed{config.seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            l2_nets_loaded.append(l2_r)

        deep_stack = _DeepStackEnsemble(
            base_models=base_models_loaded,
            level2_nets=l2_nets_loaded,
            num_classes=config.num_classes,
            dropout=config.dropout,
        ).to(dev0)

        # Transfer meta weights from stage-1
        deep_stack.meta.load_state_dict(ds_meta.state_dict())

        # Freeze base models for stage 2 (they are already well-trained)
        deep_stack.freeze_base_models()

        ft_config = copy.copy(config)
        ft_config.device = str(dev0)
        ft_config.learning_rate = config.learning_rate * 0.1  # smaller LR for fine-tuning
        ft_config.num_epochs = min(50, config.num_epochs // 2)
        ft_config.patience = 12

        ft_tr = dl_ds.make_loader(X_tr, y_tr, shuffle=True)
        ft_val = dl_ds.make_loader(X_val, y_val, shuffle=False)
        ft_test = dl_ds.make_loader(X_test, y_test, shuffle=False)

        ft_trainer = Trainer(ft_config, run_name="DeepStackEnsemble")
        ft_save = config.models_dir / f"DeepStackEnsemble_seed{config.seed}.pt"
        ft_result = ft_trainer.fit(deep_stack, ft_tr, ft_val, save_path=ft_save)

        ds_metrics = evaluate_model_on_test(
            deep_stack,
            ft_test,
            ft_config,
            classes,
            "DeepStackEnsemble",
            log_wandb=config.use_wandb,
        )
        all_results.append(
            {
                "model": "DeepStackEnsemble",
                "type": "deep_stack",
                "seed": config.seed,
                "device": str(dev0),
                "test_mcc": ds_metrics["mcc"],
                "test_acc": ds_metrics["accuracy"],
                "best_val_mcc": ft_result.best_val_mcc,
                "val_mccs": ft_result.val_mccs,
            }
        )
        curve_data.append({"name": "DeepStackEnsemble", "val_mccs": ft_result.val_mccs})
        ds_mcc = float(ds_metrics["mcc"])
        ds_acc = float(ds_metrics["accuracy"])
        print(f"  [DeepStack] test_mcc={ds_mcc:.4f}  acc={ds_acc:.4f}")

    # ── Artefacts ─────────────────────────────────────────────────────────────
    results_csv = config.results_dir / f"dl_results_seed{config.seed}.csv"
    save_results_csv(all_results, results_csv)
    plot_training_curves(curve_data, config.plots_dir / "training_curves.png")

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Type':<22} {'MCC':>8} {'Acc':>8}")
    print("-" * 70)
    for row in sorted(all_results, key=lambda r: float(str(r.get("test_mcc", 0))), reverse=True):
        print(
            f"{str(row['model']):<30} {str(row['type']):<22} "
            f"{float(str(row.get('test_mcc', 0))):>8.4f} "
            f"{float(str(row.get('test_acc', 0))):>8.4f}"
        )
    print("=" * 70)
    print(f"\nResults → {results_csv}")
    _write_ai_result(all_results, config, devices)


def _write_ai_result(
    results: list[dict[str, object]], config: DLConfig, devices: list[str]
) -> None:
    path = Path("AI_RESULT.md")
    sorted_rows = sorted(results, key=lambda r: float(str(r.get("test_mcc", 0))), reverse=True)

    table_lines = [
        "| Model | Type | Device | Test MCC | Test Acc |",
        "|-------|------|--------|----------|----------|",
    ]
    for row in sorted_rows:
        mcc_str = str(row.get("test_mcc", 0))
        acc_str = str(row.get("test_acc", 0))
        table_lines.append(
            f"| {row['model']} | {row['type']} | {row.get('device', '-')} "
            f"| {float(mcc_str):.4f} | {float(acc_str):.4f} |"
        )

    section = f"""
## Deep Learning Results (seed={config.seed})

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: {devices}
- Dataset: {config.num_classes} classes, seq_len=10, in_features=1
- Regularisation: L1 λ={config.l1_lambda}, AdamW wd={config.weight_decay}, dropout={config.dropout}
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: {config.n_cv_folds}

### Results

{chr(10).join(table_lines)}

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `{config.wandb_project}`
"""
    mode = "a" if path.exists() else "w"
    with path.open(mode) as f:
        f.write(section)


if __name__ == "__main__":
    main()
