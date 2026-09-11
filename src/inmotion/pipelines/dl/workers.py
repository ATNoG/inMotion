"""Spawn-safe worker functions for parallel training and HPO."""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from torch import nn

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import DLDataLoader
from inmotion.dl.evaluation import (
    evaluate_model_on_test,
    metrics_to_extended_row,
)
from inmotion.dl.models.meta_fusion import MetaFusionClassifier
from inmotion.dl.models.moe import (
    build_moe_expert,
    build_moe_expert_typed,
    build_soft_moe,
)
from inmotion.dl.optimization import (
    run_binary_moe_hpo_study,
    run_hpo_study,
    run_moe_multiobjective_study,
    run_nas_study,
    save_optuna_plots,
)
from inmotion.dl.training import Trainer
from inmotion.pipelines.dl.builders import (
    _build_ds_variant,
    _build_hpo_best_model,
    _build_model_by_name,
)
from inmotion.pipelines.dl.devices import set_seed


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
    """Train one model on `device_str`, save checkpoint, return metrics.

    Skips training if checkpoint already exists and --no-resume is not set.
    """
    set_seed(config.seed)
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()

    dl = DLDataLoader(config)
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)
    save_path = config.models_dir / f"{name}_seed{config.seed}.pt"

    if not config._force_retrain and save_path.exists():
        print(f"  [{name}] checkpoint exists, skipping training → {save_path}")
        model = _build_model_by_name(name, config)
        state = torch.load(str(save_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        metrics = evaluate_model_on_test(
            model,
            test_loader,
            config,
            classes,
            name,
            log_wandb=config.use_wandb,
        )
        return metrics_to_extended_row(
            name,
            "single",
            config.seed,
            device_str,
            metrics,
            extra={},
        )

    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)

    model = _build_model_by_name(name, config)
    trainer = Trainer(config, run_name=name)

    t0 = time.time()
    result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)
    elapsed = time.time() - t0

    metrics = evaluate_model_on_test(
        model, test_loader, config, classes, name, log_wandb=config.use_wandb
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    row = metrics_to_extended_row(
        name,
        "single",
        config.seed,
        device_str,
        metrics,
        best_val_mcc=result.best_val_mcc,
        best_epoch=result.best_epoch,
        train_time_s=round(elapsed, 1),
        extra={"val_mccs": result.val_mccs},
    )
    return row


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
    model_name = f"HPO_{model_type.upper()}"
    save_path = config.models_dir / f"{model_name}_seed{config.seed}.pt"
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)

    # ── Shortcut: if both Optuna study and checkpoint exist, skip training ──
    if not config._force_retrain and save_path.exists():
        try:
            import optuna

            from inmotion.dl.optimization import _get_optuna_storage

            study_name = f"{config.optuna_study_prefix}_{model_type}"
            study = optuna.load_study(
                study_name=study_name, storage=_get_optuna_storage(config.optuna_storage)
            )
            if study.trials:
                best_params = study.best_trial.params
                print(f"  [HPO {model_type}] study + checkpoint exist, skipping → {save_path}")
                model = _build_hpo_best_model(model_type, config, best_params)
                state = torch.load(str(save_path), map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                metrics = evaluate_model_on_test(
                    model,
                    test_loader,
                    config,
                    classes,
                    model_name,
                    log_wandb=config.use_wandb,
                )
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
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"],
                    "f1_weighted": metrics["f1_weighted"],
                    "per_class": metrics["per_class"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "best_epoch": 0,
                    "train_time_s": 0.0,
                    "val_mccs": [],
                }
        except Exception:
            pass  # fall through to full training

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
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
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
    return metrics_to_extended_row(
        "OptunaNet_NAS",
        "optuna_nas",
        config.seed,
        device_str,
        metrics,
        best_val_mcc=study.best_value,
        extra={"best_arch": json.dumps(study.best_trial.params), "val_mccs": []},
    )


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
    save_path = config.models_dir / f"{name}_seed{config.seed}.pt"
    test_loader = dl.make_loader(X_test, y_test, shuffle=False)

    if not config._force_retrain and save_path.exists():
        print(f"  [DS {name}] checkpoint exists, skipping → {save_path}")
        model = torch.load(str(save_path), map_location="cpu", weights_only=False)
        metrics = evaluate_model_on_test(model, test_loader, config, classes, name, log_wandb=False)
        return metrics_to_extended_row(
            name,
            "ds_base",
            config.seed,
            device_str,
            metrics,
            extra={"save_path": str(save_path)},
        )

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
    row = metrics_to_extended_row(
        name,
        "ds_base",
        config.seed,
        device_str,
        metrics,
        best_val_mcc=result.best_val_mcc,
        extra={"val_mccs": result.val_mccs, "save_path": str(save_path)},
    )
    return row


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
    row = metrics_to_extended_row(
        combo_name,
        "soft_moe",
        config.seed,
        device_str,
        metrics,
        best_val_mcc=result.best_val_mcc,
        extra={"arch_list": "+".join(arch_list), "val_mccs": result.val_mccs},
    )
    return row


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
    row = metrics_to_extended_row(
        name,
        "meta_fusion",
        config.seed,
        device_str,
        metrics,
        best_val_mcc=result.best_val_mcc,
        best_epoch=result.best_epoch,
        train_time_s=round(elapsed, 1),
        extra={"val_mccs": result.val_mccs},
    )
    return row
