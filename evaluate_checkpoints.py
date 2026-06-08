#!/usr/bin/env python3
"""Retroactive DL checkpoint evaluation — produce extended results CSV with per-class metrics.

Usage:
    uv run python evaluate_checkpoints.py --seed 42 --data dataset.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from dl.config import DLConfig
from dl.data_loader import DLDataLoader
from dl.evaluation import (
    evaluate_model_on_test,
    metrics_to_extended_row,
)
from dl.training import Trainer
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
    build_moe_expert_typed,
    build_soft_moe,
)
from dl.models.rnn import RNNClassifier
from dl.models.tcn import TCNClassifier
from dl.models.transformer import TransformerClassifier
from dl.models.deep_stack import (
    DeepMetaMLP,
    DeepStackEnsemble,
    Level2Net,
)
from torch import nn


# ── Model-building helpers ────────────────────────────────────────────────────

def _build_single_model(name: str, config: DLConfig) -> nn.Module:
    """Build a single (non-HPO) model by name."""
    base_kw = dict(
        in_features=config.in_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    match name:
        case "RNN":  return RNNClassifier(**base_kw)
        case "GRU":  return GRUClassifier(**base_kw)
        case "LSTM": return LSTMClassifier(**base_kw, use_attention=True)
        case "BiLSTM": return BiLSTMClassifier(**base_kw, use_attention=True)
        case "CNN":
            return CNNClassifier(config.in_features, 128, 3, config.num_classes, config.dropout)
        case "TCN":
            return TCNClassifier(config.in_features, 256, 3, 6, config.num_classes, config.dropout)
        case "Transformer":
            return TransformerClassifier(config.in_features, 128, 8, 4, config.num_classes, config.dropout)
        case "CNN2DLSTM":
            return CNN2DRNNClassifier(config.in_features, 64, 3, config.hidden_size, config.num_layers,
                                       config.num_classes, config.dropout, "lstm")
        case "CNN2DGRU":
            return CNN2DRNNClassifier(config.in_features, 64, 3, config.hidden_size, config.num_layers,
                                       config.num_classes, config.dropout, "gru")
        case "Mamba":
            return MambaClassifier(config.in_features, config.d_model, 16, config.num_layers, 2,
                                    config.num_classes, config.dropout, mimo_rank=4)
        case "Autoformer":
            return MambaClassifier(config.in_features, config.d_model, 16, config.num_layers, 2,
                                    config.num_classes, config.dropout, mimo_rank=4)
        case _:
            raise ValueError(f"Unknown single model: {name}")


def _build_hpo_model(model_type: str, config: DLConfig, params: dict) -> nn.Module:
    """Rebuild a best HPO model from Optuna params."""
    dropout = float(params.get("dropout", config.dropout))
    kernel_sets = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9]]
    match model_type:
        case "rnn":
            return RNNClassifier(config.in_features, int(params.get("hidden_size", 64)),
                                  int(params.get("num_layers", 2)), config.num_classes, dropout,
                                  bool(params.get("bidirectional", False)))
        case "gru":
            return GRUClassifier(config.in_features, int(params.get("hidden_size", 64)),
                                  int(params.get("num_layers", 2)), config.num_classes, dropout,
                                  bool(params.get("bidirectional", False)),
                                  bool(params.get("use_attention", False)))
        case "lstm":
            return LSTMClassifier(config.in_features, int(params.get("hidden_size", 64)),
                                   int(params.get("num_layers", 2)), config.num_classes, dropout,
                                   bool(params.get("bidirectional", False)),
                                   bool(params.get("use_attention", False)))
        case "bilstm":
            return BiLSTMClassifier(config.in_features, int(params.get("hidden_size", 64)),
                                     int(params.get("num_layers", 2)), config.num_classes, dropout,
                                     bool(params.get("use_attention", True)))
        case "cnn":
            ks = kernel_sets[int(params.get("kernel_set", 0))]
            return CNNClassifier(config.in_features, int(params.get("num_filters", 64)),
                                  int(params.get("num_blocks", 2)), config.num_classes, dropout, ks)
        case "tcn":
            return TCNClassifier(config.in_features, int(params.get("num_channels", 128)),
                                  int(params.get("kernel_size", 3)), int(params.get("depth", 4)),
                                  config.num_classes, dropout)
        case "transformer":
            d_model = int(params.get("d_model", 64))
            n_heads = int(params.get("n_heads", 4))
            while d_model % n_heads != 0:
                n_heads = max(1, n_heads // 2)
            return TransformerClassifier(config.in_features, d_model, n_heads,
                                          int(params.get("num_layers", 2)), config.num_classes, dropout)
        case "cnn2drnn":
            return CNN2DRNNClassifier(in_features=config.in_features,
                                       num_filters=int(params.get("num_filters", 32)),
                                       cnn_depth=int(params.get("cnn_depth", 2)),
                                       hidden_size=int(params.get("hidden_size", 64)),
                                       num_rnn_layers=int(params.get("num_rnn_layers", 2)),
                                       num_classes=config.num_classes, dropout=dropout,
                                       rnn_type=str(params.get("rnn_type", "lstm")),
                                       bidirectional=bool(params.get("bidirectional", False)))
        case "mamba":
            return MambaClassifier(in_features=config.in_features,
                                    d_model=int(params.get("d_model", config.d_model)),
                                    d_state=int(params.get("d_state", 16)),
                                    num_layers=int(params.get("num_layers", config.num_layers)),
                                    expand=int(params.get("expand", 2)),
                                    num_classes=config.num_classes, dropout=dropout,
                                    mimo_rank=int(params.get("mimo_rank", 4)))
        case _:
            raise ValueError(f"Unknown HPO model type: {model_type}")


def _build_ds_variant(name: str, config: DLConfig) -> nn.Module:
    """Build one of the Deep-Stack base model variants."""
    match name:
        case "DS_RNN_A":
            return RNNClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_RNN_B":
            return RNNClassifier(config.in_features, 256, 4, config.num_classes, 0.3, bidirectional=True)
        case "DS_GRU_A":
            return GRUClassifier(config.in_features, 64, 2, config.num_classes, 0.2)
        case "DS_GRU_B":
            return GRUClassifier(config.in_features, 256, 3, config.num_classes, 0.3, bidirectional=True)
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
            return CNN2DRNNClassifier(config.in_features, 32, 2, 64, 2, config.num_classes, 0.2, "lstm")
        case "DS_CNN2DRNN_B":
            return CNN2DRNNClassifier(config.in_features, 128, 3, 256, 3, config.num_classes, 0.3, "lstm", bidirectional=True)
        case "DS_Mamba_A":
            return MambaClassifier(config.in_features, 64, 16, 2, 2, config.num_classes, 0.2, mimo_rank=4)
        case "DS_Mamba_B":
            return MambaClassifier(config.in_features, 256, 32, 4, 2, config.num_classes, 0.3, mimo_rank=8)
        case _:
            raise ValueError(f"Unknown DS variant: {name}")


# ── Checkpoint → model mapping ───────────────────────────────────────────────

def _load_hpo_best_params(model_type: str, config: DLConfig) -> dict | None:
    """Load best Optuna HPO params from the DB for a given model type."""
    try:
        import optuna
    except ImportError:
        print(f"[warn] optuna not installed, skipping HPO params for {model_type}")
        return None

    study_name = f"{config.optuna_study_prefix}_{model_type}"
    try:
        study = optuna.load_study(study_name=study_name, storage=config.optuna_storage)
        if study.trials:
            return study.best_trial.params
    except Exception as e:
        print(f"[warn] Could not load Optuna study '{study_name}': {e}")
    return None


def _load_moe_mo_best_params(combo_name: str, config: DLConfig) -> dict | None:
    """Load best multi-objective HPO params for a MoE combo."""
    try:
        import optuna
    except ImportError:
        return None

    study_name = f"{config.optuna_study_prefix}_moe_mo_{combo_name}"
    try:
        study = optuna.load_study(study_name=study_name, storage=config.optuna_storage)
        pareto = study.best_trials
        if pareto:
            best_mcc = max((t.values[0] for t in pareto if t.values), default=0.0)
            for t in pareto:
                if t.values and t.values[0] == best_mcc:
                    return dict(t.params)
    except Exception as e:
        print(f"[warn] Could not load MoE study '{study_name}': {e}")
    return None


def _load_nas_best_params(config: DLConfig) -> dict | None:
    """Load best NAS study params."""
    try:
        import optuna
    except ImportError:
        return None

    study_name = f"{config.optuna_study_prefix}_nas"
    try:
        study = optuna.load_study(study_name=study_name, storage=config.optuna_storage)
        if study.trials:
            return study.best_trial.params
    except Exception as e:
        print(f"[warn] Could not load NAS study: {e}")
    return None


# ── Main evaluation logic ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retroactive DL checkpoint evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", type=Path, default=Path("dataset.csv"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("models/dl"))
    p.add_argument("--output", type=Path, default=None, help="Output CSV path (default: results/dl/dl_detailed_seed{seed}.csv)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--optuna-db", type=str, default="sqlite:///optuna_dl_2.db")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_single_checkpoint(
    ckpt_path: Path,
    model_name: str,
    config: DLConfig,
    test_loader,
    classes: list[str],
    model_factory,
) -> dict | None:
    """Load a checkpoint, build model, evaluate. Returns metrics dict or None."""
    try:
        model = model_factory()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(state, nn.Module):
            # DeepStack base models saved as full model
            model = state
        else:
            model.load_state_dict(state, strict=False)
        metrics = evaluate_model_on_test(model, test_loader, config, classes, model_name)
        return metrics
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = DLConfig(
        data_path=args.data,
        seed=args.seed,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
    )
    if args.models_dir is not None:
        config.models_dir = args.models_dir
    config.optuna_storage = args.optuna_db
    config.make_dirs()

    checkpoints_dir = args.checkpoints_dir
    if not checkpoints_dir.exists():
        print(f"Checkpoints dir not found: {checkpoints_dir}")
        sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_loader = DLDataLoader(config)
    X, y = data_loader.load_and_preprocess()
    classes = data_loader.classes_
    print(f"Dataset: {X.shape}  classes: {classes}  seed: {config.seed}")

    X_train, X_test, y_train, y_test = data_loader.train_test_split(X, y)
    test_loader = data_loader.make_loader(X_test, y_test, shuffle=False)
    device = config.resolve_device()

    # ── Meta data for MetaFusion models ───────────────────────────────────────
    X_test_meta = None
    meta_test = None
    test_meta_loader = None
    try:
        meta_info = data_loader.load_and_preprocess_with_meta()
        if meta_info is not None:
            _, meta_data, _, y_meta = meta_info
            _, X_test_meta_full, _, meta_test_full = data_loader.train_test_split(
                data_loader._X_raw, meta_data, y_meta
            )
            X_test_meta = X_test_meta_full
            meta_test = meta_test_full
            test_meta_loader = data_loader.make_meta_loader(
                X_test_meta, meta_test, y_test, shuffle=False
            )
    except Exception as e:
        print(f"[warn] Meta data not available: {e}")

    # ── Collect all .pt files ─────────────────────────────────────────────────
    ckpt_files = sorted(checkpoints_dir.glob(f"*_seed{args.seed}.pt"))
    print(f"Found {len(ckpt_files)} checkpoint(s)")

    # ── Load Optuna best params ──────────────────────────────────────────────
    print("Loading Optuna best params for HPO models...")
    HPO_TYPES = ["rnn", "gru", "lstm", "cnn", "tcn", "transformer", "bilstm", "cnn2drnn", "mamba"]
    hpo_best_params: dict[str, dict] = {}
    for mt in HPO_TYPES:
        params = _load_hpo_best_params(mt, config)
        if params:
            hpo_best_params[mt] = params

    moe_mo_params: dict[str, dict] = {}
    for combo_name in MOE_COMBOS:
        params = _load_moe_mo_best_params(combo_name, config)
        if params:
            moe_mo_params[combo_name] = params

    nas_params = _load_nas_best_params(config)

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    rows: list[dict] = []

    SINGLE_MODELS = [
        "RNN", "GRU", "LSTM", "CNN", "TCN", "Transformer", "BiLSTM",
        "CNN2DLSTM", "CNN2DGRU", "Mamba",
    ]
    HPO_MODEL_MAP = {
        "HPO_RNN": "rnn", "HPO_GRU": "gru", "HPO_LSTM": "lstm", "HPO_CNN": "cnn",
        "HPO_TCN": "tcn", "HPO_TRANSFORMER": "transformer", "HPO_BILSTM": "bilstm",
        "HPO_CNN2DRNN": "cnn2drnn", "HPO_MAMBA": "mamba", "HPO_AUTOFORMER": "mamba",
    }
    DS_VARIANTS = [
        "DS_RNN_A", "DS_RNN_B", "DS_GRU_A", "DS_GRU_B", "DS_LSTM_A", "DS_LSTM_B",
        "DS_CNN_A", "DS_CNN_B", "DS_TCN_A", "DS_TCN_B", "DS_Transformer_A", "DS_Transformer_B",
        "DS_BiLSTM_A", "DS_BiLSTM_B", "DS_CNN2DRNN_A", "DS_CNN2DRNN_B", "DS_Mamba_A", "DS_Mamba_B",
    ]

    for ckpt_path in ckpt_files:
        stem = ckpt_path.stem  # e.g. "CNN_seed42"
        model_name = stem.replace(f"_seed{args.seed}", "")

        # Skip DeepStack + MoE internals (sub-expert checkpoints pre-saved by MoE process)
        if model_name.startswith("MoE_expert_") or model_name.startswith("MoE_MoE_"):
            continue

        try:
            metrics = None
            model = None

            # ── Case 1: Single models ───────────────────────────────────
            if model_name in SINGLE_MODELS:
                metrics = evaluate_single_checkpoint(
                    ckpt_path, model_name, config, test_loader, classes,
                    lambda n=model_name: _build_single_model(n, config),
                )
                if metrics:
                    rows.append(metrics_to_extended_row(model_name, "single", args.seed, "cpu",
                                                         metrics, extra={"member_count": None}))

            # ── Case 2: HPO models ──────────────────────────────────────
            elif model_name in HPO_MODEL_MAP:
                model_type = HPO_MODEL_MAP[model_name]
                params = hpo_best_params.get(model_type, {})
                if params:
                    metrics = evaluate_single_checkpoint(
                        ckpt_path, model_name, config, test_loader, classes,
                        lambda mt=model_type, p=params: _build_hpo_model(mt, config, p),
                    )
                    if metrics:
                        rows.append(metrics_to_extended_row(model_name, "hpo", args.seed, "cpu",
                                                             metrics, extra={"best_arch": json.dumps(params)}))
                else:
                    print(f"[skip] {model_name}: no HPO best params found in Optuna DB")

            # ── Case 3: DS base variants ────────────────────────────────
            elif model_name in DS_VARIANTS:
                # DS base models saved as full model via torch.save(model, ...)
                try:
                    model = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    model = model.to(device)
                    metrics = evaluate_model_on_test(model, test_loader, config, classes, model_name)
                    rows.append(metrics_to_extended_row(model_name, "ds_base", args.seed, "cpu", metrics))
                except Exception as e:
                    print(f"[warn] {model_name}: full-model load failed ({e}), trying state_dict...")
                    metrics = evaluate_single_checkpoint(
                        ckpt_path, model_name, config, test_loader, classes,
                        lambda n=model_name: _build_ds_variant(n, config),
                    )
                    if metrics:
                        rows.append(metrics_to_extended_row(model_name, "ds_base", args.seed, "cpu", metrics))

            # ── Case 4: MetaFusion models ───────────────────────────────
            elif model_name in ("MetaFusion_LSTM", "MetaFusion_GRU"):
                if test_meta_loader is not None and meta_test is not None:
                    rnn_type = "gru" if "GRU" in model_name else "lstm"
                    try:
                        model = MetaFusionClassifier(
                            in_features=config.in_features,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            num_classes=config.num_classes,
                            dropout=config.dropout,
                            meta_embed_dim=config.meta_embed_dim,
                            rnn_type=rnn_type,
                        )
                        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                        model.load_state_dict(state, strict=False)
                        metrics = evaluate_model_on_test(
                            model, test_meta_loader, config, classes, model_name,
                        )
                        rows.append(metrics_to_extended_row(model_name, "meta_fusion", args.seed, "cpu", metrics))
                    except Exception as e:
                        print(f"[ERROR] {model_name}: {e}")
                else:
                    print(f"[skip] {model_name}: meta data not available")

            # ── Case 5: SoftMoE variants ────────────────────────────────
            elif model_name in MOE_COMBOS:
                arch_list = MOE_COMBOS[model_name]
                best_params = moe_mo_params.get(model_name, {})
                if best_params:
                    try:
                        expert_params = {
                            k: best_params[k]
                            for k in ("dropout", "hidden_size", "num_layers", "bidirectional",
                                       "use_attention", "num_filters", "num_blocks", "kernel_set")
                            if k in best_params
                        }
                        gate_hidden = int(best_params.get("gate_hidden", 64))
                        gate_dropout = float(best_params.get("gate_dropout", 0.1))
                        aux_loss_weight = float(best_params.get("aux_loss_weight", 0.01))
                        top_k_raw = best_params.get("top_k", "none")
                        top_k = None if top_k_raw == "none" else int(top_k_raw)
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
                        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                        model.load_state_dict(state, strict=False)
                        metrics = evaluate_model_on_test(model, test_loader, config, classes, model_name)
                        rows.append(metrics_to_extended_row(
                            model_name, "soft_moe", args.seed, "cpu", metrics,
                            extra={"arch_list": "+".join(arch_list)},
                        ))
                    except Exception as e:
                        print(f"[ERROR] {model_name}: {e}")
                else:
                    print(f"[skip] {model_name}: no MoE HPO best params")

            # ── Case 6: OptunaNet NAS ───────────────────────────────────
            elif model_name == "OptunaNet_NAS" or model_name == "OptunaNet":
                params = nas_params
                if params:
                    arch = str(params.get("arch", "cnn"))
                    if arch == "cnn":
                        ks_idx = int(params.get("kernel_set", 0))
                        ks = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9]][ks_idx]
                        model = CNNClassifier(
                            config.in_features,
                            int(params.get("num_filters", 64)),
                            int(params.get("num_blocks", 2)),
                            config.num_classes,
                            float(params.get("dropout", config.dropout)),
                            ks,
                        )
                    else:
                        model = _build_single_model(arch.upper(), config)
                    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=False)
                    metrics = evaluate_model_on_test(model, test_loader, config, classes, "OptunaNet_NAS")
                    rows.append(metrics_to_extended_row(
                        "OptunaNet_NAS", "optuna_nas", args.seed, "cpu", metrics,
                        extra={"best_arch": json.dumps(params)},
                    ))
                else:
                    print(f"[skip] OptunaNet: no NAS best params")

            # ── Case 7: Voting/Stacking Ensembles ────────────────────────
            elif model_name.startswith("Ensemble_") or model_name.startswith("Stacking_"):
                # Extract member models from filename
                member_names_raw = model_name.replace("Ensemble_", "").replace("Stacking_", "").replace("All",
                    "RNN,GRU,LSTM,CNN,TCN,Transformer,BiLSTM,CNN2DLSTM,CNN2DGRU,Mamba",
                )
                member_names_src = member_names_raw.replace("_", ",")
                member_names = [m.strip() for m in member_names_src.split(",") if m.strip()]

                try:
                    members = []
                    for mn in member_names:
                        member_path = checkpoints_dir / f"{mn}_seed{args.seed}.pt"
                        if member_path.exists():
                            m = _build_single_model(mn, config)
                            s = torch.load(str(member_path), map_location="cpu", weights_only=True)
                            m.load_state_dict(s, strict=False)
                            m.eval().to(device)
                            members.append(m)
                        else:
                            print(f"[warn] {model_name}: member checkpoint not found: {member_path}")

                    if members:
                        if model_name.startswith("Ensemble_"):
                            model = VotingEnsemble(members).to(device)
                        else:
                            model = StackingEnsemble(
                                members, num_classes=config.num_classes, dropout=config.dropout,
                            ).to(device)
                            try:
                                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                                model.load_state_dict(state, strict=False)
                            except Exception as e:
                                print(f"[warn] {model_name}: ensemble state_dict load failed: {e}")

                        metrics = evaluate_model_on_test(model, test_loader, config, classes, model_name)
                        row_type = "voting_ensemble" if model_name.startswith("Ensemble_") else "stacking_ensemble"
                        rows.append(metrics_to_extended_row(
                            model_name, row_type, args.seed, "cpu", metrics,
                            extra={"members": "+".join(member_names)},
                        ))
                    else:
                        print(f"[skip] {model_name}: no member models could be loaded")
                except Exception as e:
                    print(f"[ERROR] {model_name}: {e}")

            # ── Case 8: DeepStackEnsemble ────────────────────────────────
            elif model_name == "DeepStackEnsemble":
                print(f"[skip] {model_name}: requires full DeepStack training pipeline — evaluate via run_dl.py")

            else:
                print(f"[skip] {model_name}: unrecognized model type")

        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()

        if model is not None and hasattr(model, "cpu"):
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────────
    if rows:
        output_path = args.output or (config.results_dir / f"dl_detailed_seed{args.seed}.csv")
        import pandas as pd
        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(rows)} evaluation rows → {output_path}")
    else:
        print("\nNo results produced.")


if __name__ == "__main__":
    main()
