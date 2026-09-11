#!/usr/bin/env python3
"""Permutation feature importance for DL movement classification models.

Computes MCC-drop when each of the 4 engineered channels × 10 timesteps
(= 40 features) is independently shuffled, for CNN, TCN, GRU, and Mamba.

DeepStackEnsemble is skipped unless all 8 DS base-variant checkpoints
already exist under models/dl/ (it requires 9 pre-trained sub-models).

Output plots (publication-quality PDF, 600 DPI, husl palette):
  docs/icaisf/paper/images/feature_importance_dl.pdf           heatmap
  docs/icaisf/paper/images/feature_importance_channels_dl.pdf   bar chart
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Pre-flight: torch must be available (run `uv sync` first if missing)
try:
    import torch  # noqa: F811
except ImportError:
    import sys as _sys
    _sys.stderr.write('ERROR: PyTorch not installed. Run:  uv sync\n')
    raise SystemExit(1)

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ── Project imports ─────────────────────────────────────────────────────────
# Repository root, for default artifact paths.
REPO_ROOT = Path(__file__).resolve().parents[3]

from dl.config import DLConfig
from dl.data_loader import DLDataLoader
from dl.models.cnn import CNNClassifier
from dl.models.gru import GRUClassifier
from dl.models.mamba import MambaClassifier
from dl.models.tcn import TCNClassifier
from dl.training import Trainer

matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_NAMES = [
    "Raw RSSI",
    r"Velocity ($\Delta$)",
    r"Acceleration ($\Delta^2$)",
    "Window Deviation",
]
CLASS_NAMES = ["AA", "AB", "BA", "BB"]
N_TIMESTEPS = 10
N_CHANNELS = 4
N_REPEATS = 10  # permutations per feature for robust estimates
SEED = 42

# Model names to evaluate (order matters for plot legends)
SINGLE_MODELS = ["CNN", "TCN", "GRU", "Mamba"]
DEEPSTACK_DS_VARIANTS = [
    "DS_RNN_A", "DS_RNN_B",
    "DS_GRU_A",
    "DS_CNN_A", "DS_CNN_B",
    "DS_TCN_A", "DS_TCN_B",
    "DS_CNN2DRNN_B",
    "DS_Mamba_A", "DS_Mamba_B",
]

# Output
OUTPUT_DIR = REPO_ROOT / "docs" / "icaisf" / "paper" / "images"
CHECKPOINT_DIR = REPO_ROOT / "models" / "dl"

# ── Plotting standards (match DLVisualizer precisely) ───────────────────────
PLOT_DPI = 600
FONT_SIZE = 18
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 16
FONT_SCALE = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_plot_style() -> None:
    """Apply DLVisualizer-compatible plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": TITLE_SIZE + 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_context("paper", font_scale=FONT_SCALE)


def build_model(name: str, config: DLConfig) -> nn.Module:
    """Construct model matching _build_model_by_name from run_dl.py."""
    kw: dict = dict(
        in_features=config.in_features,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    match name:
        case "CNN":
            return CNNClassifier(num_filters=128, num_blocks=3, **kw)  # type: ignore[arg-type]
        case "TCN":
            return TCNClassifier(num_channels=256, kernel_size=3, depth=6, **kw)  # type: ignore[arg-type]
        case "GRU":
            return GRUClassifier(
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                **kw,  # type: ignore[arg-type]
            )
        case "Mamba":
            return MambaClassifier(
                d_model=config.d_model,
                d_state=16,
                num_layers=config.num_layers,
                mimo_rank=4,
                **kw,  # type: ignore[arg-type]
            )
        case _:
            raise ValueError(f"Unknown model: {name}")


def train_or_load(
    name: str, config: DLConfig, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, device: torch.device,
) -> nn.Module:
    """Train model from scratch or load saved checkpoint."""
    checkpoint = CHECKPOINT_DIR / f"{name}_seed{SEED}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if checkpoint.exists():
        print(f"  ✓ Loading checkpoint: {checkpoint}")
        model = build_model(name, config)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval().to(device)
        return model

    print(f"  Training {name} from scratch ({config.num_epochs} epochs max)...")
    config = _clone_config(config)
    config.device = str(device)
    config.use_wandb = False

    dl = DLDataLoader(config)
    tr_loader = dl.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = dl.make_loader(X_val, y_val, shuffle=False)

    model = build_model(name, config)
    trainer = Trainer(config, run_name=f"fi_{name}")

    result = trainer.fit(model, tr_loader, val_loader, save_path=checkpoint)
    print(f"    Best val MCC: {result.best_val_mcc:.4f} @ epoch {result.best_epoch}")

    # Reload best checkpoint
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def _clone_config(config: DLConfig) -> DLConfig:
    """Shallow copy with needed overrides for standalone training."""
    import copy
    c = copy.copy(config)
    c.use_wandb = False
    c.num_epochs = 100
    c.patience = 20
    return c


def permutation_importance(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Return (importance[N_CHANNELS, N_TIMESTEPS], baseline_mcc).

    For each of the 40 feature positions, shuffles the column across samples
    N_REPEATS times and measures the mean MCC drop.
    """
    model.eval()

    # Baseline MCC
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        baseline_logits = model(X_t.to(device)).cpu().numpy()
    baseline_preds = baseline_logits.argmax(axis=1)
    baseline_mcc = float(matthews_corrcoef(y_test, baseline_preds))

    importance = np.zeros((N_CHANNELS, N_TIMESTEPS), dtype=np.float64)
    rng = np.random.RandomState(SEED)

    total = N_CHANNELS * N_TIMESTEPS
    for c in range(N_CHANNELS):
        for t in range(N_TIMESTEPS):
            mcc_drops: list[float] = []
            for _ in range(N_REPEATS):
                X_shuffled = X_test.copy()
                col = X_shuffled[:, t, c].copy()
                rng.shuffle(col)
                X_shuffled[:, t, c] = col

                X_s = torch.tensor(X_shuffled, dtype=torch.float32)
                with torch.no_grad():
                    logits_s = model(X_s.to(device)).cpu().numpy()
                preds_s = logits_s.argmax(axis=1)
                mcc_s = float(matthews_corrcoef(y_test, preds_s))
                mcc_drops.append(baseline_mcc - mcc_s)

            importance[c, t] = float(np.mean(mcc_drops))
            idx = c * N_TIMESTEPS + t + 1
            print(f"    [{idx}/{total}] {CHANNEL_NAMES[c]:25s} t={t+1:2d}  "
                  f"ΔMCC={importance[c,t]:.4f}")

    return importance, baseline_mcc


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_heatmap(
    all_importances: np.ndarray,       # (n_models, 4, 10)
    model_names: list[str],
    baseline_mccs: list[float],
) -> None:
    """Publication-quality heatmap: 4 channels × 10 timesteps, avg across models."""
    setup_plot_style()

    avg = all_importances.mean(axis=0)  # (4, 10)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        avg,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=[f"t={i + 1}" for i in range(N_TIMESTEPS)],
        yticklabels=CHANNEL_NAMES,
        ax=ax,
        cbar_kws={"label": "MCC Drop (importance)", "shrink": 0.85},
        vmin=0.0,
        linewidths=0.5,
        linecolor="white",
    )
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Channel")
    models_str = ", ".join(model_names)
    ax.set_title(
        f"Permutation Feature Importance — DL Models\n"
        f"({models_str} | avg baseline MCC={np.mean(baseline_mccs):.4f})"
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "feature_importance_dl.pdf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_model_heatmaps(
    all_importances: np.ndarray,       # (n_models, 4, 10)
    model_names: list[str],
    baseline_mccs: list[float],
) -> None:
    """Multi-panel figure: one 4x10 heatmap per model."""
    setup_plot_style()

    n_models = len(model_names)
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        squeeze=False,
    )

    vmax = all_importances.max()

    for i, (name, mcc) in enumerate(zip(model_names, baseline_mccs)):
        ax = axes[i // ncols][i % ncols]
        sns.heatmap(
            all_importances[i],
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[f"t={j + 1}" for j in range(N_TIMESTEPS)],
            yticklabels=CHANNEL_NAMES,
            ax=ax,
            cbar_kws={"label": "MCC Drop"},
            vmin=0.0,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
        )
        ax.set_title(f"{name}  (baseline MCC={mcc:.4f})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Channel")

    # Hide unused subplots
    for j in range(n_models, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "feature_importance_dl_per_model.pdf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_channel_bars(
    all_importances: np.ndarray,       # (n_models, 4, 10)
    model_names: list[str],
) -> None:
    """Channel-level bar chart: mean importance +/- std across timesteps & models."""
    setup_plot_style()
    colors = sns.color_palette("husl", 20)

    # Mean across timesteps per model -> (n_models, 4)
    per_model = all_importances.mean(axis=2)
    means = per_model.mean(axis=0)   # (4,)
    stds = per_model.std(axis=0)     # (4,)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(N_CHANNELS)
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=8,
        color=[colors[i % 20] for i in range(N_CHANNELS)],
        edgecolor="white",
        linewidth=1.5,
        width=0.55,
    )

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.3,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=LABEL_SIZE - 2,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Mean MCC Drop (importance)")
    ax.set_title("Channel-Level Feature Importance — DL Models")
    ymax = (means + stds).max() * 1.25 + 0.005
    ax.set_ylim(0, max(ymax, 0.02))

    plt.tight_layout()
    out = OUTPUT_DIR / "feature_importance_channels_dl.pdf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# DeepStack (optional — only when DS base checkpoints already exist)
# ═══════════════════════════════════════════════════════════════════════════════


def _try_load_deepstack(
    config: DLConfig,
    device: torch.device,
) -> nn.Module | None:
    """Build DeepStackEnsemble from saved DS base checkpoints, or return None."""
    from dl.models.deep_stack import DeepStackEnsemble, Level2Net

    base_models: list[nn.Module] = []
    for ds_name in DEEPSTACK_DS_VARIANTS:
        ckpt = CHECKPOINT_DIR / f"{ds_name}_seed{SEED}.pt"
        if not ckpt.exists():
            print(f"  DeepStack: missing {ds_name}_seed{SEED}.pt — skipping ensemble")
            return None
        model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        base_models.append(model)

    # 8 base models x 4 classes = 32 input features per Level2Net
    n_base = len(base_models)
    in_features = n_base * config.num_classes
    n_l2 = 5
    level2_nets = [
        Level2Net(in_features, config.num_classes, config.dropout)
        for _ in range(n_l2)
    ]

    ensemble = DeepStackEnsemble(
        base_models=base_models,
        level2_nets=level2_nets,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )

    # Try loading the ensemble checkpoint
    ensemble_ckpt = CHECKPOINT_DIR / f"DeepStackEnsemble_seed{SEED}.pt"
    if ensemble_ckpt.exists():
        print(f"  DeepStack: loading ensemble checkpoint: {ensemble_ckpt}")
        state = torch.load(str(ensemble_ckpt), map_location="cpu", weights_only=True)
        ensemble.load_state_dict(state, strict=False)
    else:
        print("  DeepStack: base models loaded but ensemble not trained — skipping")
        return None

    ensemble.eval().to(device)
    return ensemble


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Permutation feature importance for the DL baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=None,
                        help="Dataset CSV. Defaults to the processed dataset.csv.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per model.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for the generated figures.")
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINT_DIR,
                        help="Directory holding trained DL checkpoints.")
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Permutation repeats (default: module constant).")
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    # The plotting helpers read the module-level OUTPUT_DIR; point it at the
    # requested directory rather than threading a parameter through each one.
    global OUTPUT_DIR  # noqa: PLW0603 - module-level output location
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    config = DLConfig()
    config.seed = args.seed
    config.use_wandb = False
    config.num_epochs = args.epochs
    config.patience = args.patience
    config.models_dir = args.checkpoints_dir

    # ── Load data ─────────────────────────────────────────────────────────
    from inmotion.data.paths import resolve

    data_path = args.data or resolve("dataset.csv")
    if not data_path.exists():
        data_path = resolve("dataset_augmented.csv")
    print(f"\nLoading: {data_path}")

    loader = DLDataLoader(config)
    X, y = loader.load_and_preprocess(data_path)
    print(f"  X: {X.shape}  y: {y.shape}  classes: {loader.classes_}")

    # Stratified holdout: 64% train / 16% val / 20% test
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=0.2, random_state=SEED, stratify=y_tr_val,
    )
    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Evaluate models ───────────────────────────────────────────────────
    models_to_run = list(SINGLE_MODELS)

    # DeepStack: only attempt if all 8 DS base checkpoints exist
    ds_available = all(
        (CHECKPOINT_DIR / f"{ds_name}_seed{SEED}.pt").exists()
        for ds_name in DEEPSTACK_DS_VARIANTS
    )
    if ds_available:
        ds_ckpt = CHECKPOINT_DIR / f"DeepStackEnsemble_seed{SEED}.pt"
        if ds_ckpt.exists():
            models_to_run.append("DeepStackEnsemble")
            print("\n  DeepStackEnsemble checkpoint found — will include in analysis")
        else:
            print("\n  DS base checkpoints exist but ensemble not trained — skipping")
    else:
        print("\n  DS base checkpoints not found — skipping DeepStackEnsemble")

    all_importances: list[np.ndarray] = []
    all_baseline_mccs: list[float] = []
    model_labels: list[str] = []

    for name in models_to_run:
        print(f"\n{'─' * 60}")
        print(f"Model: {name}")
        print(f"{'─' * 60}")

        if name == "DeepStackEnsemble":
            model = _try_load_deepstack(config, device)
            if model is None:
                continue
        else:
            model = train_or_load(name, config, X_tr, y_tr, X_val, y_val, device)

        imp, baseline_mcc = permutation_importance(model, X_test, y_test, device)
        all_importances.append(imp)
        all_baseline_mccs.append(baseline_mcc)
        model_labels.append(name)

        print(f"\n  Baseline MCC: {baseline_mcc:.4f}")
        print(f"  Top 5 most important features:")
        flat = [
            (c, t, imp[c, t])
            for c in range(N_CHANNELS)
            for t in range(N_TIMESTEPS)
        ]
        flat.sort(key=lambda x: -x[2])
        for c, t, val in flat[:5]:
            print(f"    {CHANNEL_NAMES[c]:30s} t={t + 1:2d}  ΔMCC={val:.4f}")

        # Channel-level summary
        print(f"  Channel-level importance (mean ΔMCC across timesteps):")
        for c in range(N_CHANNELS):
            ch_mean = float(imp[c].mean())
            ch_std = float(imp[c].std())
            print(f"    {CHANNEL_NAMES[c]:30s}  {ch_mean:.4f} ± {ch_std:.4f}")

    if not all_importances:
        print("\nERROR: No models evaluated. Check that training succeeded.")
        sys.exit(1)

    # ── Plots ─────────────────────────────────────────────────────────────
    all_imp = np.array(all_importances)  # (n_models, 4, 10)

    print(f"\n{'─' * 60}")
    print("Generating plots...")

    plot_heatmap(all_imp, model_labels, all_baseline_mccs)
    plot_channel_bars(all_imp, model_labels)

    if len(model_labels) > 1:
        plot_per_model_heatmaps(all_imp, model_labels, all_baseline_mccs)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
