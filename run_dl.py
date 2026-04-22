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
from dl.models.autoformer import AutoformerClassifier
from dl.models.cnn import CNNClassifier
from dl.models.deep_stack import Level2Net
from dl.models.ensemble import MetaLearner, StackingEnsemble, VotingEnsemble
from dl.models.gru import GRUClassifier
from dl.models.lstm import LSTMClassifier
from dl.models.moe import MOE_ARCH_TYPES, MOE_COMBOS, build_moe_expert, build_moe_expert_typed
from dl.models.rnn import RNNClassifier
from dl.optimization import (
    init_optuna_db,
    run_binary_moe_hpo_study,
    run_hpo_study,
    run_nas_study,
    save_optuna_plots,
)
from dl.training import Trainer
from torch import nn
from torch.utils.data import DataLoader

SINGLE_MODEL_NAMES: list[str] = ["RNN", "GRU", "LSTM", "CNN", "Autoformer"]
HPO_TYPES: list[str] = ["rnn", "gru", "lstm", "cnn", "autoformer"]

# 8 base variants for Deep Stacking (2 configs per arch)
DS_BASE_NAMES: list[str] = [
    "DS_RNN_A",
    "DS_RNN_B",
    "DS_GRU_A",
    "DS_GRU_B",
    "DS_LSTM_A",
    "DS_LSTM_B",
    "DS_CNN_A",
    "DS_CNN_B",
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
        case "CNN":
            return CNNClassifier(
                in_features=config.in_features,
                num_filters=64,
                num_blocks=2,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case "Autoformer":
            return AutoformerClassifier(
                in_features=config.in_features,
                d_model=config.d_model,
                n_heads=config.n_heads,
                num_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )
        case _:
            raise ValueError(f"Unknown model: {name}")


def _build_ds_variant(name: str, config: DLConfig) -> nn.Module:
    """Build one of the 8 Deep-Stack base model variants."""
    match name:
        case "DS_RNN_A":
            return RNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_RNN_B":
            return RNNClassifier(
                config.in_features, 128, 3, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_GRU_A":
            return GRUClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_GRU_B":
            return GRUClassifier(
                config.in_features, 128, 2, config.num_classes, 0.3, bidirectional=True
            )
        case "DS_LSTM_A":
            return LSTMClassifier(config.in_features, 64, 2, config.num_classes, 0.2, False, True)
        case "DS_LSTM_B":
            return LSTMClassifier(config.in_features, 128, 2, config.num_classes, 0.3, True, False)
        case "DS_CNN_A":
            return CNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2, [3, 5])
        case "DS_CNN_B":
            return CNNClassifier(config.in_features, 128, 3, config.num_classes, 0.3, [3, 5, 7])
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


def _hpo_worker(
    model_type: str,
    device_str: str,
    config: DLConfig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    """Run Optuna HPO for one model type on `device_str`."""
    config = copy.copy(config)
    config.device = device_str
    config.make_dirs()
    dl = DLDataLoader(config)
    study = run_hpo_study(model_type, config, dl, X_tr, y_tr, X_val, y_val)
    save_optuna_plots(study, model_type, config.plots_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_type": model_type,
        "device": device_str,
        "best_val_mcc": study.best_value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
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
    """Run binary Optuna HPO for one MoE expert arch type (class 0 vs rest as proxy task)."""
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
    trainer = Trainer(config)
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
    torch.save(model.state_dict(), save_path)
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
    )
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
        "Ensemble_CNN_Autoformer": ["CNN", "Autoformer"],
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
                ): model_type
                for i, model_type in enumerate(HPO_TYPES)
            }
            for future in as_completed(hpo_futures):
                res = future.result()
                print(
                    f"  [HPO {res['model_type']}] {res['device']}  "
                    f"best_val_mcc={float(res['best_val_mcc']):.4f}  "
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

    # ── Phase 4: Mixture-of-Experts (multi-arch, Optuna-tuned) ────────────────
    if not args.no_moe:
        n_combos = len(MOE_COMBOS)
        print(
            f"\n[Phase 4a] MoE HPO — binary tuning for {len(MOE_ARCH_TYPES)} arch types"
            f" across {len(devices)} GPU(s)…"
        )
        # Pre-init binary HPO study names in the main process to avoid SQLite spawn races
        binary_study_names = [
            f"{config.optuna_study_prefix}_moe_binary_{a}" for a in MOE_ARCH_TYPES
        ]
        init_optuna_db(config.optuna_storage, binary_study_names)

        best_params_per_arch: dict[str, dict] = {}
        if not args.no_optuna:
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
                bin_hpo_futures = {
                    pool.submit(
                        _moe_binary_hpo_worker,
                        arch_type,
                        devices[i % len(devices)],
                        copy.copy(config),
                        X_tr,
                        y_tr,
                        X_val,
                        y_val,
                    ): arch_type
                    for i, arch_type in enumerate(MOE_ARCH_TYPES)
                }
                for future in as_completed(bin_hpo_futures):
                    res = future.result()
                    best_params_per_arch[str(res["arch_type"])] = dict(res["best_params"])
                    print(
                        f"  [MoE HPO {res['arch_type']}] {res['device']}  "
                        f"best_val_mcc={float(res['best_val_mcc']):.4f}  "
                        f"params={res['best_params']}"
                    )
        else:
            # No HPO — use sensible defaults for each arch type
            best_params_per_arch = {a: {} for a in MOE_ARCH_TYPES}

        print(f"\n[Phase 4b] MoE training — {n_combos} combos across {len(devices)} GPU(s)…")
        dev0_moe = torch.device(devices[0])

        for combo_name, arch_list in MOE_COMBOS.items():
            print(f"  [{combo_name}] training {len(arch_list)} experts ({' '.join(arch_list)})…")

            # ── train experts in parallel ──────────────────────────────────────
            with ProcessPoolExecutor(
                max_workers=min(len(classes), len(devices)), mp_context=ctx
            ) as pool:
                exp_futures = [
                    pool.submit(
                        _moe_typed_expert_worker,
                        class_idx,
                        class_name,
                        arch_list[class_idx],
                        combo_name,
                        devices[class_idx % len(devices)],
                        copy.copy(config),
                        best_params_per_arch[arch_list[class_idx]],
                        X_tr,
                        y_tr,
                        X_val,
                        y_val,
                    )
                    for class_idx, class_name in enumerate(classes)
                ]
                combo_expert_infos = [f.result() for f in as_completed(exp_futures)]
            combo_expert_infos.sort(key=lambda d: int(d["class_idx"]))

            # ── reload experts, extract P(positive) probs ─────────────────────
            combo_experts: list[nn.Module] = []
            for info in combo_expert_infos:
                a = str(info["arch_type"])
                exp = build_moe_expert_typed(a, config.in_features, best_params_per_arch[a])
                exp.load_state_dict(
                    torch.load(str(info["save_path"]), map_location="cpu", weights_only=True)
                )
                combo_experts.append(exp)

            moe_combo_dl = DLDataLoader(copy.copy(config))
            tr_loader_moe = moe_combo_dl.make_loader(X_tr, y_tr, shuffle=True)
            val_loader_moe = moe_combo_dl.make_loader(X_val, y_val, shuffle=False)
            test_loader_moe = moe_combo_dl.make_loader(X_test, y_test, shuffle=False)

            def _combo_probs(
                loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
                experts: list[nn.Module] = combo_experts,
            ) -> np.ndarray:
                cols = []
                for exp in experts:
                    logits = _extract_logits(exp, loader, dev0_moe)
                    cols.append(torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy())
                return np.stack(cols, axis=-1)  # (N, num_classes)

            feat_tr_c = _combo_probs(tr_loader_moe)
            feat_val_c = _combo_probs(val_loader_moe)
            feat_test_c = _combo_probs(test_loader_moe)

            # ── train meta-learner on expert probs ────────────────────────────
            meta_cfg_c = copy.copy(config)
            meta_dl_c = DLDataLoader(meta_cfg_c)
            meta_tr_c = meta_dl_c.make_loader(feat_tr_c, y_tr, shuffle=True)
            meta_val_c = meta_dl_c.make_loader(feat_val_c, y_val, shuffle=False)
            meta_test_c = meta_dl_c.make_loader(feat_test_c, y_test, shuffle=False)

            meta_c: nn.Module = MetaLearner(config.num_classes, config.num_classes, config.dropout)
            Trainer(meta_cfg_c).fit(meta_c, meta_tr_c, meta_val_c)
            moe_c_metrics = evaluate_model_on_test(
                meta_c,
                meta_test_c,
                meta_cfg_c,
                classes,
                combo_name,
                log_wandb=config.use_wandb,
            )
            torch.save(
                meta_c.state_dict(),
                config.models_dir / f"{combo_name}_meta_seed{config.seed}.pt",
            )
            all_results.append(
                {
                    "model": combo_name,
                    "type": "moe_typed",
                    "seed": config.seed,
                    "device": devices[0],
                    "test_mcc": moe_c_metrics["mcc"],
                    "test_acc": moe_c_metrics["accuracy"],
                    "best_val_mcc": 0.0,
                    "members": "+".join(arch_list),
                    "val_mccs": [],
                }
            )
            print(
                f"  [{combo_name}] test_mcc={float(moe_c_metrics['mcc']):.4f}"
                f"  acc={float(moe_c_metrics['accuracy']):.4f}"
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

        # Reload base models and extract concat logits (N, 8*num_classes)
        dev0 = torch.device(devices[0])
        base_models_loaded: list[nn.Module] = []
        for info in sorted(ds_base_results, key=lambda d: str(d["model"])):
            m = _build_ds_variant(str(info["model"]), config)
            m.load_state_dict(
                torch.load(str(info["save_path"]), map_location="cpu", weights_only=True)
            )
            base_models_loaded.append(m)

        def _concat_logits(loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]") -> np.ndarray:
            return np.concatenate(
                [_extract_logits(m, loader, dev0) for m in base_models_loaded], axis=-1
            )

        base_feat_tr = _concat_logits(tr_loader_ds)  # (N, 8*4)
        base_feat_val = _concat_logits(val_loader_ds)
        base_feat_test = _concat_logits(test_loader_ds)

        n_base_feats = base_feat_tr.shape[1]
        l2_config = copy.copy(config)
        l2_config.num_classes = 2
        l2_dl = DLDataLoader(l2_config)
        level2_nets: list[nn.Module] = []

        for class_idx, cls_name in enumerate(classes):
            y_tr_bin = (y_tr == class_idx).astype(int)
            y_val_bin = (y_val == class_idx).astype(int)
            l2_tr = l2_dl.make_loader(base_feat_tr, y_tr_bin, shuffle=True)
            l2_val = l2_dl.make_loader(base_feat_val, y_val_bin, shuffle=False)
            l2_net: nn.Module = Level2Net(n_base_feats, dropout=config.dropout)
            Trainer(l2_config).fit(l2_net, l2_tr, l2_val)
            save_path = config.models_dir / f"DS_L2_{cls_name}_seed{config.seed}.pt"
            torch.save(l2_net.state_dict(), save_path)
            level2_nets.append(l2_net)

        # Extract L2 probs (N, 4)
        def _l2_probs(feats: np.ndarray) -> np.ndarray:
            feat_t = torch.tensor(feats, dtype=torch.float32).to(dev0)
            cols = []
            for l2 in level2_nets:
                l2.eval().to(dev0)
                with torch.no_grad():
                    logits_l2 = l2(feat_t)
                cols.append(torch.softmax(logits_l2, dim=-1)[:, 1].cpu().numpy())
            return np.stack(cols, axis=-1)

        l2_feat_tr = _l2_probs(base_feat_tr)
        l2_feat_val = _l2_probs(base_feat_val)
        l2_feat_test = _l2_probs(base_feat_test)

        meta_ds_config = copy.copy(config)
        meta_ds_dl = DLDataLoader(meta_ds_config)
        ds_meta_tr = meta_ds_dl.make_loader(l2_feat_tr, y_tr, shuffle=True)
        ds_meta_val = meta_ds_dl.make_loader(l2_feat_val, y_val, shuffle=False)
        ds_meta_test = meta_ds_dl.make_loader(l2_feat_test, y_test, shuffle=False)

        ds_meta: nn.Module = MetaLearner(config.num_classes, config.num_classes, config.dropout)
        Trainer(meta_ds_config).fit(ds_meta, ds_meta_tr, ds_meta_val)
        ds_metrics = evaluate_model_on_test(
            ds_meta,
            ds_meta_test,
            meta_ds_config,
            classes,
            "DeepStackEnsemble",
            log_wandb=config.use_wandb,
        )
        torch.save(ds_meta.state_dict(), config.models_dir / f"DS_meta_seed{config.seed}.pt")
        all_results.append(
            {
                "model": "DeepStackEnsemble",
                "type": "deep_stack",
                "seed": config.seed,
                "device": devices[0],
                "test_mcc": ds_metrics["mcc"],
                "test_acc": ds_metrics["accuracy"],
                "best_val_mcc": 0.0,
                "val_mccs": [],
            }
        )
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
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), Autoformer
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
