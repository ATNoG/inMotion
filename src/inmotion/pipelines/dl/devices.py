"""Device pool selection and seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _available_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def _build_device_pool(n_gpus: int) -> list[str]:
    if torch.cuda.is_available():
        count = min(n_gpus, torch.cuda.device_count())
        return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]
