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
from .models.cnn import CNNClassifier
from .models.gru import GRUClassifier
from .models.lstm import LSTMClassifier
from .models.optuna_net import build_optuna_model
from .models.rnn import RNNClassifier
from .models.tcn import TCNClassifier
from .models.transformer import TransformerClassifier
from .training import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial_run_name(prefix: str, trial: optuna.Trial) -> str:
    """Build a compact but fully descriptive WandB run name from all trial params.

    Format: ``{prefix}_t{number}[_arch_params][_training_params]``
    All params present in trial.params at call time are included.
    """
    p = trial.params
    parts: list[str] = [prefix, f"t{trial.number}"]

    # ── arch params ──────────────────────────────────────────────────────────
    if "arch" in p:
        parts.append(str(p["arch"]))
    if "hidden_size" in p:
        parts.append(f"h{p['hidden_size']}")
    if "num_layers" in p:
        parts.append(f"l{p['num_layers']}")
    if "bidirectional" in p:
        parts.append("bi" if p["bidirectional"] else "uni")
    if "use_attention" in p:
        parts.append("attn" if p["use_attention"] else "noattn")
    if "num_filters" in p:
        parts.append(f"f{p['num_filters']}")
    if "num_blocks" in p:
        parts.append(f"b{p['num_blocks']}")
    if "kernel_set" in p:
        parts.append(f"ks{p['kernel_set']}")
    if "d_model_mult" in p:
        parts.append(f"dm{p['d_model_mult']}")
    if "n_heads" in p:
        parts.append(f"nh{p['n_heads']}")
    if "num_enc_layers" in p:
        parts.append(f"el{p['num_enc_layers']}")
    if "factor" in p:
        parts.append(f"fa{p['factor']}")

    # ── training params ───────────────────────────────────────────────────────
    if "dropout" in p:
        parts.append(f"do{float(p['dropout']):.2f}")
    if "lr" in p:
        parts.append(f"lr{float(p['lr']):.1e}")
    if "weight_decay" in p:
        parts.append(f"wd{float(p['weight_decay']):.1e}")
    if "l1_lambda" in p:
        parts.append(f"l1{float(p['l1_lambda']):.1e}")
    if "gradient_clip" in p:
        parts.append(f"gc{float(p['gradient_clip']):.1f}")
    if "loss_type" in p:
        parts.append(str(p["loss_type"]))
    if "optimizer_type" in p:
        parts.append(str(p["optimizer_type"]))
    if "scheduler_type" in p:
        parts.append(str(p["scheduler_type"]))

    return "_".join(parts)


def _train_and_eval(
    model: nn.Module,
    config: DLConfig,
    tr_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    trial: optuna.Trial,
    study_prefix: str = "hpo",
    mode: str = "default",  # "default" | "tcn" | "transformer"
) -> float:
    """Train model for HPO trial and return best val MCC.

    mode="tcn"         : constrained LR (≤3e-3), no rmsprop
    mode="transformer" : constrained LR (≤3e-3), adamw only
    """
    cfg = copy.copy(config)
    # LR — TCN/Transformer need smaller LR for stability
    lr_max = 3e-3 if mode in ("tcn", "transformer") else 5e-2
    cfg.learning_rate = trial.suggest_float("lr", 5e-5, lr_max, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    cfg.l1_lambda = trial.suggest_float("l1_lambda", 1e-8, 1e-2, log=True)
    cfg.gradient_clip = trial.suggest_float("gradient_clip", 0.5, 5.0)
    # Loss / optimiser / scheduler
    cfg.loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
    cfg.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 5.0)
    if mode == "transformer":
        # Transformers need Adam-family; SGD/RMSprop are highly unstable here
        cfg.optimizer_type = trial.suggest_categorical("optimizer_type", ["adamw"])
    elif mode == "tcn":
        # TCN is stable with adamw or sgd but rmsprop causes divergence
        cfg.optimizer_type = trial.suggest_categorical("optimizer_type", ["adamw", "sgd"])
    else:
        cfg.optimizer_type = trial.suggest_categorical(
            "optimizer_type", ["adamw", "sgd", "rmsprop"]
        )
    cfg.momentum = trial.suggest_float("momentum", 0.7, 0.99)  # SGD only
    cfg.scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "plateau"])
    cfg.num_epochs = 100  # increased from 80 for better convergence
    cfg.patience = 20
    run_name = _trial_run_name(study_prefix, trial)
    trainer = Trainer(cfg, run_name=run_name, extra_wandb_config=dict(trial.params))
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
        model: nn.Module
        hpo_mode = "default"

        if model_type == "rnn":
            dropout = trial.suggest_float("dropout", 0.05, 0.7)
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir: bool = bool(trial.suggest_categorical("bidirectional", [True, False]))
            model = RNNClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir
            )

        elif model_type == "gru":
            dropout = trial.suggest_float("dropout", 0.05, 0.7)
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir = bool(trial.suggest_categorical("bidirectional", [True, False]))
            use_attn_gru: bool = bool(trial.suggest_categorical("use_attention", [True, False]))
            model = GRUClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir, use_attn_gru
            )

        elif model_type == "lstm":
            dropout = trial.suggest_float("dropout", 0.05, 0.7)
            hidden = trial.suggest_int("hidden_size", 32, 512, log=True)
            layers = trial.suggest_int("num_layers", 1, 8)
            bidir = bool(trial.suggest_categorical("bidirectional", [True, False]))
            use_attn: bool = bool(trial.suggest_categorical("use_attention", [True, False]))
            model = LSTMClassifier(
                config.in_features, hidden, layers, config.num_classes, dropout, bidir, use_attn
            )

        elif model_type == "cnn":
            dropout = trial.suggest_float("dropout", 0.05, 0.7)
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

        elif model_type == "tcn":
            # Constrained dropout — high dropout + SGD destroys TCN
            dropout = trial.suggest_float("dropout", 0.05, 0.4)
            num_channels = trial.suggest_int("num_channels", 32, 512, log=True)
            kernel_size = int(trial.suggest_categorical("kernel_size", [3, 5, 7]))
            depth = trial.suggest_int("depth", 2, 8)
            model = TCNClassifier(
                config.in_features, num_channels, kernel_size, depth, config.num_classes, dropout
            )
            hpo_mode = "tcn"

        elif model_type == "transformer":
            # Narrow dropout for attention stability
            dropout = trial.suggest_float("dropout", 0.05, 0.3)
            d_model = int(trial.suggest_categorical("d_model", [32, 64, 128, 256]))
            n_heads = int(trial.suggest_categorical("n_heads", [2, 4, 8]))
            num_layers = trial.suggest_int("num_layers", 1, 4)
            model = TransformerClassifier(
                config.in_features, d_model, n_heads, num_layers, config.num_classes, dropout
            )
            hpo_mode = "transformer"

        else:
            raise ValueError(f"Unknown model_type for HPO: {model_type!r}")

        return _train_and_eval(
            model, config, tr_loader, val_loader, trial, f"hpo_{model_type}", mode=hpo_mode
        )

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
        return _train_and_eval(model, config, tr_loader, val_loader, trial, "nas")

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
    study_suffix: str = "",
) -> optuna.Study:
    """HPO for a single binary MoE expert arch type.

    y_tr_bin / y_val_bin are 0/1 labels (class_k vs rest).
    Searches arch hyperparams + training hyperparams jointly.
    Returns the finished study; best_trial.params contains the full param set.
    study_suffix allows per-class studies, e.g. "_c0", "_c1".
    """
    from .models.moe import build_moe_expert_typed

    tr_loader = data_loader.make_loader(X_tr, y_tr_bin, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val_bin, shuffle=False)
    study_name = f"{config.optuna_study_prefix}_moe_binary_{arch_type}{study_suffix}"
    wandb_prefix = f"moe_bin_{arch_type}{study_suffix}"

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
        return _train_and_eval(model, cfg, tr_loader, val_loader, trial, wandb_prefix)

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


# ── Multi-objective MoE HPO (NSGA-II) ─────────────────────────────────────────


def _compute_load_imbalance(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> float:
    """Compute average load-balance loss for a SoftMixtureOfExperts on a dataset.

    Returns 0.0 for models without last_aux_loss.
    """
    import torch as _torch

    model.eval().to(device)
    total = 0.0
    n = 0
    with _torch.no_grad():
        for X_batch, _ in loader:
            _ = model(X_batch.to(device))
            aux = getattr(model, "last_aux_loss", None)
            if aux is not None and isinstance(aux, _torch.Tensor):
                total += aux.item() * len(X_batch)
                n += len(X_batch)
    return total / max(n, 1)


def run_moe_multiobjective_study(
    arch_list: list[str],
    config: DLConfig,
    data_loader: DLDataLoader,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    combo_name: str = "moe",
    n_trials: int | None = None,
) -> tuple[optuna.Study, dict]:
    """Multi-objective Optuna HPO for SoftMixtureOfExperts using NSGA-II.

    Objectives:
      1. val_mcc (maximize) — classification quality.
      2. load_imbalance (minimize) — routing balance (lower = more uniform).

    Searches jointly over:
      - Expert arch hyperparameters (hidden_size, num_layers, bidirectional).
      - Gating parameters (gate_hidden, gate_dropout, aux_loss_weight, top_k).
      - Training hyperparameters (lr, weight_decay, optimizer, scheduler).

    Returns:
      study: finished multi-objective study.
      best_params: params of the Pareto-optimal trial with highest val_mcc.
    """
    import torch as _torch

    from .models.moe import build_soft_moe

    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)
    study_name = f"{config.optuna_study_prefix}_moe_mo_{combo_name}"
    n_opt = n_trials if n_trials is not None else max(10, config.n_trials // 4)

    device = _torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if _torch.cuda.is_available() else "cpu")
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # ── Expert arch params (shared across all experts) ──────────────────
        dropout = trial.suggest_float("dropout", 0.05, 0.5)
        expert_params: dict = {"dropout": dropout}

        # Recurrent expert params (used for gru/lstm/rnn in arch_list)
        if any(a in ("gru", "lstm", "rnn") for a in arch_list):
            expert_params["hidden_size"] = trial.suggest_int("hidden_size", 32, 256, log=True)
            expert_params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
            expert_params["bidirectional"] = bool(
                trial.suggest_categorical("bidirectional", [True, False])
            )
        # LSTM attention (only used if lstm present)
        if "lstm" in arch_list:
            expert_params["use_attention"] = bool(
                trial.suggest_categorical("use_attention", [True, False])
            )
        # CNN params (only used if cnn present)
        if "cnn" in arch_list:
            expert_params["num_filters"] = trial.suggest_int("num_filters", 32, 256, log=True)
            expert_params["num_blocks"] = trial.suggest_int("num_blocks", 1, 6)
            expert_params["kernel_set"] = int(trial.suggest_categorical("kernel_set", [0, 1, 2]))

        # ── Gating params ───────────────────────────────────────────────────
        gate_hidden = trial.suggest_int("gate_hidden", 32, 128, log=True)
        gate_dropout = trial.suggest_float("gate_dropout", 0.0, 0.3)
        aux_loss_weight = trial.suggest_float("aux_loss_weight", 1e-4, 0.1, log=True)
        top_k_raw = trial.suggest_categorical("top_k", ["none", "1", "2"])
        top_k: int | None = None if top_k_raw == "none" else int(top_k_raw)
        if top_k is not None and top_k >= len(arch_list):
            top_k = None

        # ── Training params ─────────────────────────────────────────────────
        cfg = copy.copy(config)
        cfg.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        cfg.l1_lambda = trial.suggest_float("l1_lambda", 1e-8, 1e-3, log=True)
        cfg.gradient_clip = trial.suggest_float("gradient_clip", 0.5, 5.0)
        cfg.optimizer_type = trial.suggest_categorical("optimizer_type", ["adamw", "rmsprop"])
        cfg.scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "plateau"])
        cfg.loss_type = "ce"
        cfg.num_epochs = 60
        cfg.patience = 12
        cfg.use_wandb = False  # suppress per-trial WandB runs

        # ── Build + train SoftMoE ───────────────────────────────────────────
        params_per_expert = [dict(expert_params) for _ in arch_list]
        model = build_soft_moe(
            arch_list=arch_list,
            in_features=cfg.in_features,
            num_classes=cfg.num_classes,
            params_per_expert=params_per_expert,
            gate_hidden=gate_hidden,
            gate_dropout=gate_dropout,
            aux_loss_weight=aux_loss_weight,
            top_k=top_k,
        )
        trainer = Trainer(cfg, run_name="")
        result = trainer.fit(model, tr_loader, val_loader)

        val_mcc = result.best_val_mcc

        # Compute load imbalance on validation set with best-checkpoint model
        load_imbalance = _compute_load_imbalance(model, val_loader, device)

        return val_mcc, load_imbalance

    sampler = optuna.samplers.NSGAIISampler(seed=config.seed)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=config.optuna_storage,
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_opt, show_progress_bar=False)

    # Select best from Pareto front: highest MCC among Pareto-optimal trials
    pareto_trials = study.best_trials
    if not pareto_trials:
        pareto_trials = study.trials
    best_trial = max(pareto_trials, key=lambda t: t.values[0] if t.values else -1.0)
    best_params = best_trial.params

    return study, best_params
