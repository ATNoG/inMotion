"""Device selection and seeding."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu >= 0 else "cuda:0")
    return torch.device("cpu")
