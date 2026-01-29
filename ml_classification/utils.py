"""Utility functions for ML classification pipeline."""

import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(model: Any, path: Path, model_name: str) -> Path:
    """Save a trained model to disk using joblib."""
    filepath = path / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    return filepath


def load_model(path: Path) -> Any:
    """Load a trained model from disk."""
    return joblib.load(path)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"
