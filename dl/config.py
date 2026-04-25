"""DL pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DLConfig:
    # Data
    data_path: Path = field(default_factory=lambda: Path("dataset.csv"))
    feature_cols: list[str] = field(default_factory=lambda: [str(i) for i in range(1, 11)])
    label_col: str = "label"
    seq_len: int = 10
    in_features: int = 4  # 4 engineered channels: raw, Δ, Δ², window-deviation
    num_classes: int = 4

    # Reproducibility
    seed: int = 42

    # Training
    batch_size: int = 64
    num_epochs: int = 200
    patience: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2
    l1_lambda: float = 1e-5  # L1
    dropout: float = 0.3
    gradient_clip: float = 1.0

    # Label smoothing + Mixup regularisation
    label_smoothing: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2

    # Loss / optimiser / scheduler
    loss_type: str = "ce"  # "ce" | "focal"
    focal_gamma: float = 2.0
    optimizer_type: str = "adamw"  # "adamw" | "sgd" | "rmsprop" | "muon"
    scheduler_type: str = "cosine"  # "cosine" | "plateau"
    momentum: float = 0.9  # SGD only

    # Model defaults (overridden by Optuna)
    hidden_size: int = 64
    num_layers: int = 2
    d_model: int = 64
    n_heads: int = 4

    # Split / CV
    test_size: float = 0.2
    n_cv_folds: int = 5

    # Optuna
    n_trials: int = 100
    optuna_storage: str = "sqlite:///optuna_dl_2.db"
    optuna_study_prefix: str = "inMotion_dl_v3"

    # WandB
    wandb_project: str = "inMotion-dl-3"
    wandb_entity: str | None = None
    use_wandb: bool = True

    # Metadata columns (noise + concurrent path context)
    noise_col: str = "noise"
    noise_path_col: str = "concurrent_noise_path"
    meta_embed_dim: int = 8  # embedding size per metadata token
    use_metadata: bool = False  # set True to load noise/path cols

    # Paths
    models_dir: Path = field(default_factory=lambda: Path("models/dl"))
    results_dir: Path = field(default_factory=lambda: Path("results/dl"))
    plots_dir: Path = field(default_factory=lambda: Path("plots/dl"))

    # Device
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def make_dirs(self) -> None:
        for p in (self.models_dir, self.results_dir, self.plots_dir):
            p.mkdir(parents=True, exist_ok=True)
