"""Train the paper's HPO-best models using configs from docs/icaisf/hpo_supplementary.md.

The original Optuna DL DBs (optuna_dl_*.db) were deleted, so run_dl.py can no
longer skip HPO. This script rebuilds each HPO-best model from the paper's
documented config and trains it with the Trainer, saving checkpoints that the
mega-ensemble can load directly.

Configs sourced from docs/icaisf/hpo_supplementary.md (best val MCC in parens):
    HPO GRU      0.9030   HPO TCN      0.8526   HPO Mamba    0.8506
    HPO CNN      0.8488   HPO LSTM     0.8388   HPO BiLSTM   0.8410
    HPO RNN      0.6402

Usage:
    MAMBA_SSM_AVAILABLE=0 uv run python train_hpo_paper.py --data dataset.csv --seed 42
    # --models only gru tcn    (subset)
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from dl.config import DLConfig
from dl.data_loader import DLDataLoader
from dl.evaluation import evaluate_model_on_test
from dl.training import Trainer


# ═══════════════════════════════════════════════════════════════════════════════
# Paper HPO-best configs (docs/icaisf/hpo_supplementary.md)
# ═══════════════════════════════════════════════════════════════════════════════

PAPER_HPO_CONFIGS: dict[str, dict] = {
    "gru": {
        "hidden_size": 552, "num_layers": 6, "bidirectional": False,
        "use_attention": False, "dropout": 0.157,
        "loss_type": "focal", "focal_gamma": 4.18,
        "lr": 0.00119, "optimizer_type": "adamw", "scheduler_type": "cosine",
        "weight_decay": 9e-6, "l1_lambda": 0.0, "gradient_clip": 4.36,
    },
    "tcn": {
        "num_channels": 363, "depth": 4, "kernel_size": 5, "dropout": 0.064,
        "loss_type": "ce",
        "lr": 0.000132, "optimizer_type": "adamw", "scheduler_type": "cosine",
        "weight_decay": 2.7e-5, "l1_lambda": 4.7e-5, "gradient_clip": 1.94,
    },
    "mamba": {
        "d_model": 256, "num_layers": 6, "d_state": 16, "expand": 4,
        "mimo_rank": 1, "dropout": 0.247,
        "loss_type": "ce",
        "lr": 0.000102, "optimizer_type": "adamw", "scheduler_type": "cosine",
        "weight_decay": 0.0, "l1_lambda": 5e-5, "gradient_clip": 4.48,
    },
    "cnn": {
        "num_filters": 218, "num_blocks": 12, "kernel_set": 2, "dropout": 0.185,
        "loss_type": "focal", "focal_gamma": 2.96,
        "lr": 0.00317, "optimizer_type": "adamw", "scheduler_type": "cosine",
        "weight_decay": 3.4e-4, "l1_lambda": 1e-6, "gradient_clip": 2.85,
    },
    "lstm": {
        "hidden_size": 382, "num_layers": 9, "bidirectional": True,
        "use_attention": True, "dropout": 0.212,
        "loss_type": "focal", "focal_gamma": 2.59,
        "lr": 0.00124, "optimizer_type": "adamw", "scheduler_type": "plateau",
        "weight_decay": 0.00116, "l1_lambda": 0.0, "gradient_clip": 2.13,
    },
    "bilstm": {
        "hidden_size": 144, "num_layers": 5, "use_attention": True, "dropout": 0.502,
        "loss_type": "ce",
        "lr": 0.00308, "optimizer_type": "adamw", "scheduler_type": "plateau",
        "weight_decay": 1e-6, "l1_lambda": 0.0, "gradient_clip": 1.66,
    },
    "rnn": {
        "hidden_size": 305, "num_layers": 3, "bidirectional": True, "dropout": 0.153,
        "loss_type": "ce",
        "lr": 0.00343, "optimizer_type": "sgd", "scheduler_type": "cosine",
        "momentum": 0.903, "weight_decay": 8.0e-5, "l1_lambda": 6.2e-6,
        "gradient_clip": 2.998,
    },
}


def _build_model(model_type: str, config: DLConfig, p: dict) -> torch.nn.Module:
    """Rebuild an HPO-best model from params (mirrors run_dl._build_hpo_best_model)."""
    from dl.models.bilstm import BiLSTMClassifier
    from dl.models.cnn import CNNClassifier
    from dl.models.gru import GRUClassifier
    from dl.models.lstm import LSTMClassifier
    from dl.models.mamba import MambaClassifier
    from dl.models.rnn import RNNClassifier
    from dl.models.tcn import TCNClassifier

    dropout = float(p.get("dropout", config.dropout))
    kernel_sets: list[list[int]] = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 9]]
    match model_type:
        case "rnn":
            return RNNClassifier(config.in_features, int(p["hidden_size"]), int(p["num_layers"]),
                                 config.num_classes, dropout, bool(p.get("bidirectional", False)))
        case "gru":
            return GRUClassifier(config.in_features, int(p["hidden_size"]), int(p["num_layers"]),
                                 config.num_classes, dropout, bool(p.get("bidirectional", False)),
                                 bool(p.get("use_attention", False)))
        case "lstm":
            return LSTMClassifier(config.in_features, int(p["hidden_size"]), int(p["num_layers"]),
                                  config.num_classes, dropout, bool(p.get("bidirectional", False)),
                                  bool(p.get("use_attention", False)))
        case "bilstm":
            return BiLSTMClassifier(config.in_features, int(p["hidden_size"]), int(p["num_layers"]),
                                    config.num_classes, dropout, bool(p.get("use_attention", True)))
        case "cnn":
            ks = kernel_sets[int(p.get("kernel_set", 0))]
            return CNNClassifier(config.in_features, int(p["num_filters"]), int(p["num_blocks"]),
                                 config.num_classes, dropout, ks)
        case "tcn":
            return TCNClassifier(config.in_features, int(p["num_channels"]), int(p["kernel_size"]),
                                 int(p["depth"]), config.num_classes, dropout)
        case "mamba":
            return MambaClassifier(config.in_features, int(p["d_model"]), int(p["d_state"]),
                                   int(p["num_layers"]), int(p.get("expand", 2)),
                                   config.num_classes, dropout, mimo_rank=int(p.get("mimo_rank", 4)))
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def _apply_training_hps(cfg: DLConfig, p: dict) -> DLConfig:
    cfg.learning_rate = float(p.get("lr", cfg.learning_rate))
    cfg.weight_decay = float(p.get("weight_decay", cfg.weight_decay))
    cfg.l1_lambda = float(p.get("l1_lambda", cfg.l1_lambda))
    cfg.gradient_clip = float(p.get("gradient_clip", cfg.gradient_clip))
    cfg.loss_type = str(p.get("loss_type", cfg.loss_type))
    cfg.focal_gamma = float(p.get("focal_gamma", cfg.focal_gamma))
    cfg.optimizer_type = str(p.get("optimizer_type", cfg.optimizer_type))
    cfg.momentum = float(p.get("momentum", cfg.momentum))
    cfg.scheduler_type = str(p.get("scheduler_type", cfg.scheduler_type))
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience (run_dl HPO trials use 20)")
    ap.add_argument("--models", nargs="+", default=list(PAPER_HPO_CONFIGS),
                    choices=list(PAPER_HPO_CONFIGS))
    ap.add_argument("--models-dir", type=Path, default=Path("models/dl"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/dl"))
    args = ap.parse_args()

    for mt in args.models:
        p = PAPER_HPO_CONFIGS[mt]
        print(f"\n{'='*60}\n  Training HPO {mt.upper()} (paper config)\n{'='*60}")
        print(f"  params: {p}")

        config = DLConfig()
        config.data_path = args.data
        config.seed = args.seed
        config.num_epochs = args.epochs
        config.patience = args.patience
        config.use_wandb = False
        config.models_dir = args.models_dir
        config.results_dir = args.results_dir
        config = _apply_training_hps(config, p)

        dl = DLDataLoader(config)
        X, y = dl.load_and_preprocess()
        X_tr, X_te, y_tr, y_te = dl.train_test_split(X, y)
        X_tr, X_va, y_tr, y_va = dl.train_test_split(X_tr, y_tr)
        classes = dl.classes_
        print(f"  Train {len(X_tr)} / Val {len(X_va)} / Test {len(X_te)}")

        model = _build_model(mt, config, p)
        tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
        val_loader = dl.make_loader(X_va, y_va, shuffle=False)
        test_loader = dl.make_loader(X_te, y_te, shuffle=False)

        run_name = f"HPO_{mt.upper()}"
        save_path = config.models_dir / f"{run_name}_seed{config.seed}.pt"
        trainer = Trainer(config, run_name=run_name)
        result = trainer.fit(model, tr_loader, val_loader, save_path=save_path)

        # ── Final-fit (mirrors run_dl.py): retrain on train+val for best_epoch,
        #    keep the better of the two models ────────────────────────────────
        X_full = np.concatenate([X_tr, X_va], axis=0)
        y_full = np.concatenate([y_tr, y_va], axis=0)
        full_loader = dl.make_loader(X_full, y_full, shuffle=True)
        final_cfg = copy.copy(config)
        final_cfg.num_epochs = max(result.best_epoch, 10)
        final_cfg.patience = final_cfg.num_epochs + 1  # no early stopping
        final_model = _build_model(mt, final_cfg, p)
        Trainer(final_cfg, run_name=f"{run_name}_finalfit").fit(final_model, full_loader, val_loader)

        from sklearn.metrics import matthews_corrcoef as _mcc
        dev = next(final_model.parameters()).device
        final_model.eval()
        with torch.no_grad():
            ff_preds = final_model(torch.from_numpy(X_va.astype(np.float32)).to(dev)).argmax(1).cpu().numpy()
        ff_val_mcc = float(_mcc(y_va, ff_preds))
        if ff_val_mcc > result.best_val_mcc:
            torch.save(final_model.state_dict(), save_path)
            model = final_model
            print(f"  Final-fit better ({ff_val_mcc:.4f} > {result.best_val_mcc:.4f}) — using final-fit")

        metrics = evaluate_model_on_test(model, test_loader, config, classes, run_name)
        print(f"  Best val MCC: {result.best_val_mcc:.4f}")
        print(f"  Test MCC:     {metrics['mcc']:.4f}")
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
