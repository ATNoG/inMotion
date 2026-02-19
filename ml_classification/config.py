"""Configuration module for ML classification pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available for ML libraries.

    Can be overridden with FORCE_GPU=1 or FORCE_CPU=1 environment variables.
    """
    # Check for manual override
    if os.environ.get("FORCE_GPU", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False

    # Try PyTorch CUDA detection
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # Try XGBoost GPU detection
    try:
        import xgboost as xgb

        # Check if GPU support is compiled in
        build_info = str(xgb.build_info()).lower()
        if "gpu" in build_info or "cuda" in build_info:
            return True
    except (ImportError, AttributeError):
        pass

    # Fallback: check if nvidia-smi is accessible
    try:
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass
class Config:
    """Configuration settings for the ML classification pipeline."""

    # Paths
    data_path: Path = field(default_factory=lambda: Path("dataset.csv"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))

    # Random seed for reproducibility
    random_seed: int = 42

    # Data split
    test_size: float = 0.2
    validation_size: float = 0.1

    # Cross-validation
    n_cv_folds: int = 5

    # Optuna settings
    n_optuna_trials: int = 50
    optuna_timeout: int | None = None
    optuna_study_name: str = "inMotion_classification"
    optuna_storage: str = (
        "sqlite:///optuna_studies.db"  # Set to "sqlite:///optuna_studies.db" to persist
    )

    # Training settings
    n_jobs: int = -1
    verbose: int = 1

    # GPU settings
    use_gpu: bool = field(default_factory=check_gpu_availability)

    # Feature columns (RSSI readings)
    feature_columns: list[str] = field(
        default_factory=lambda: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    )
    target_column: str = "label"
    noise_column: str = "noise_label"
    mac_column: str = "mac"

    # Plot settings for publication-ready figures
    plot_format: str = "pdf"
    plot_dpi: int = 600
    plot_font_scale: float = 2.0
    plot_font_size: int = 16
    plot_title_size: int = 18
    plot_label_size: int = 16
    plot_tick_size: int = 18
    plot_legend_size: int = 18

    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if self.use_gpu:
            print(
                "🚀 GPU acceleration enabled for supported classifiers (XGBoost, LightGBM, CatBoost)"
            )
        else:
            print("⚠️ GPU not available, using CPU for all classifiers")
