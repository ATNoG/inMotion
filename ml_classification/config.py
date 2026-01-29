"""Configuration module for ML classification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


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
    optuna_study_name: str = "wifi_fingerprint_classification"
    optuna_storage: str | None = None

    # Training settings
    n_jobs: int = -1
    verbose: int = 1

    # Feature columns (RSSI readings)
    feature_columns: list[str] = field(
        default_factory=lambda: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    )
    target_column: str = "label"
    noise_column: str = "noise_label"
    mac_column: str = "mac"

    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
