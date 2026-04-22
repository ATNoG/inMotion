"""Optuna HPO studies — per-model and NAS meta-model."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import optuna
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import DLConfig
from .data_loader import DLDataLoader
from .models.autoformer import AutoformerClassifier
from .models.cnn import CNNClassifier
from .models.gru import GRUClassifier
from .models.lstm import LSTMClassifier
from .models.optuna_net import build_optuna_model
from .models.rnn import RNNClassifier
from .training import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _train_and_eval(
    model: nn.Module,
    config: DLConfig,
    tr_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    trial: optuna.Trial,
) -> float:
    cfg = copy.copy(config)
    # LR + regularisation
    cfg.learning_rate = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    cfg.l1_lambda = trial.suggest_float("l1_lambda", 1e-8, 1e-2, log=True)
    cfg.gradient_clip = trial.suggest_float("gradient_clip", 0.5, 5.0)
    # Loss / optimiser / scheduler
    cfg.loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])  # noqa: E501
    cfg.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 5.0)  # only used when focal
    cfg.optimizer_type = trial.suggest_categorical("optimizer_type", ["adamw", "sgd", "rmsprop"])  # noqa: E501
    cfg.momentum = trial.suggest_float("momentum", 0.7, 0.99)  # only used when sgd
    cfg.scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "plateau"])  # noqa: E501
    cfg.use_wandb = False
    cfg.num_epochs = 80
    cfg.patience = 15

    trainer = Trainer(cfg)
    result = trainer.fit(model, tr_loader, val_loader)
    return result.best_val_mcc


def _make_rnn_hpo_objective(
    config: DLConfig,
    data_loader: DLDataLoader,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> optuna.Study:
    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 8)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        bidir = trial.suggest_categorical("bidirectional", [True, False])
        model = RNNClassifier(
            config.in_features,
            hidden_size,
            num_layers,
            config.num_classes,
            dropout,
            bidir,
        )
        return _train_and_eval(model, config, tr_loader, val_loader, trial)

    return objective  # type: ignore[return-value]


def init_optuna_db(storage_url: str, study_names: list[str]) -> None:
    """Pre-create SQLite schema + studies in the main process to avoid spawn race."""
    for name in study_names:
        optuna.create_study(
            direction="maximize",
            study_name=name,
            storage=storage_url,
            load_if_exists=True,
        )


def run_hpo_study(
    model_type: str,
    config: DLConfig,
    data_loader: DLDataLoader,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> optuna.Study:
    """Run HPO for a named model type. Returns the finished study."""
    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)
    study_name = f"{config.optuna_study_prefix}_{model_type}"

    def objective(trial: optuna.Trial) -> float:
        dropout: float = trial.suggest_float("dropout", 0.05, 0.7)
        model: nn.Module

        if model_type == "rnn":
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir: bool = bool(trial.suggest_categorical("bidirectional", [True, False]))
            model = RNNClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir
            )

        elif model_type == "gru":
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir = bool(trial.suggest_categorical("bidirectional", [True, False]))
            model = GRUClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir
            )

        elif model_type == "lstm":
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir = bool(trial.suggest_categorical("bidirectional", [True, False]))
            use_attn: bool = bool(trial.suggest_categorical("use_attention", [True, False]))
            model = LSTMClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir, use_attn
            )

        elif model_type == "cnn":
            num_filters = trial.suggest_int("num_filters", 32, 512, log=True)
            num_blocks = trial.suggest_int("num_blocks", 1, 12)
            ks_idx: int = int(trial.suggest_categorical("kernel_set", [0, 1, 2]))
            kernel_sets: list[list[int]] = [[3], [3, 5], [3, 5, 7]]
            model = CNNClassifier(
                config.in_features,
                num_filters,
                num_blocks,
                config.num_classes,
                dropout,
                kernel_sets[ks_idx],
            )

        else:  # autoformer
            d_model = trial.suggest_int("d_model_mult", 4, 32) * 8  # 32–256
            n_heads_raw: int = int(trial.suggest_categorical("n_heads", [1, 2, 4, 8]))
            n_heads = n_heads_raw
            while d_model % n_heads != 0:
                n_heads //= 2
            enc_layers = trial.suggest_int("num_enc_layers", 1, 6)
            factor = trial.suggest_int("factor", 1, 5)
            model = AutoformerClassifier(
                config.in_features,
                d_model,
                n_heads,
                enc_layers,
                config.num_classes,
                dropout,
                factor,
            )

        return _train_and_eval(model, config, tr_loader, val_loader, trial)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=config.optuna_storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)
    return study


def run_nas_study(
    config: DLConfig,
    data_loader: DLDataLoader,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[optuna.Study, nn.Module]:
    """NAS: Optuna searches over architectures AND hyperparams."""
    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)
    study_name = f"{config.optuna_study_prefix}_nas"

    def objective(trial: optuna.Trial) -> float:
        model = build_optuna_model(trial, config.in_features, config.num_classes)
        return _train_and_eval(model, config, tr_loader, val_loader, trial)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=config.optuna_storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_model = build_optuna_model(best_trial, config.in_features, config.num_classes)

    cfg = copy.copy(config)
    cfg.learning_rate = best_trial.params.get("lr", config.learning_rate)
    cfg.weight_decay = best_trial.params.get("weight_decay", config.weight_decay)
    cfg.l1_lambda = best_trial.params.get("l1_lambda", config.l1_lambda)

    trainer = Trainer(cfg, run_name="nas_best")
    trainer.fit(best_model, tr_loader, val_loader)

    return study, best_model


def run_binary_moe_hpo_study(
    arch_type: str,
    config: DLConfig,
    data_loader: DLDataLoader,
    X_tr: np.ndarray,
    y_tr_bin: np.ndarray,
    X_val: np.ndarray,
    y_val_bin: np.ndarray,
) -> optuna.Study:
    """HPO for a single binary MoE expert arch type.

    y_tr_bin / y_val_bin are 0/1 labels (class_k vs rest).
    Searches arch hyperparams + training hyperparams jointly.
    Returns the finished study; best_trial.params contains the full param set.
    """
    from .models.moe import build_moe_expert_typed

    tr_loader = data_loader.make_loader(X_tr, y_tr_bin, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val_bin, shuffle=False)
    study_name = f"{config.optuna_study_prefix}_moe_binary_{arch_type}"

    def objective(trial: optuna.Trial) -> float:
        dropout = trial.suggest_float("dropout", 0.05, 0.6)
        params: dict = {"dropout": dropout}

        if arch_type in ("gru", "lstm", "rnn"):
            params["hidden_size"] = trial.suggest_int("hidden_size", 32, 256, log=True)
            params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
            params["bidirectional"] = bool(
                trial.suggest_categorical("bidirectional", [True, False])
            )
            if arch_type == "lstm":
                params["use_attention"] = bool(
                    trial.suggest_categorical("use_attention", [True, False])
                )
        else:  # cnn
            params["num_filters"] = trial.suggest_int("num_filters", 32, 256, log=True)
            params["num_blocks"] = trial.suggest_int("num_blocks", 1, 6)
            params["kernel_set"] = int(trial.suggest_categorical("kernel_set", [0, 1, 2]))

        cfg = copy.copy(config)
        cfg.num_classes = 2
        model = build_moe_expert_typed(arch_type, cfg.in_features, params)
        return _train_and_eval(model, cfg, tr_loader, val_loader, trial)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=config.optuna_storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)
    return study


def save_optuna_plots(study: optuna.Study, model_name: str, plots_dir: Path) -> None:
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
        )

        plots_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_optimization_history(study)
        fig.write_image(str(plots_dir / f"optuna_history_{model_name}.png"))

        if len(study.trials) > 1:
            fig2 = plot_param_importances(study)
            fig2.write_image(str(plots_dir / f"optuna_importance_{model_name}.png"))
    except Exception:
        pass  # plotly not required
