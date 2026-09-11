"""Configuration must not change between processes.

`DLConfig.optuna_storage` was built from `randint(1000, 9999)` at import time,
so every process opened a different Optuna database. Hyperparameter studies could
never be resumed, and `evaluate_checkpoints.py` had to hardcode a guessed name
(`optuna_dl_2.db`) that only matched if the suffix happened to come up.
"""

from __future__ import annotations

import subprocess
import sys

from dl.config import DLConfig


def test_optuna_storage_is_not_randomised() -> None:
    assert "randint" not in DLConfig().optuna_storage
    assert DLConfig().optuna_storage == DLConfig().optuna_storage


def test_optuna_storage_is_stable_across_processes() -> None:
    """Two interpreters must agree, or a study cannot be resumed."""
    code = "from dl.config import DLConfig; print(DLConfig().optuna_storage)"
    first = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert first == second, f"{first!r} != {second!r}"


def test_optuna_storage_can_be_overridden_by_env() -> None:
    """A one-off run should be able to point somewhere else."""
    code = (
        "import os; os.environ['INMOTION_OPTUNA_STORAGE']='sqlite:///custom.db';"
        "from dl.config import DLConfig; print(DLConfig().optuna_storage)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "sqlite:///custom.db"
