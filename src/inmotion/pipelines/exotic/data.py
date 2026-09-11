"""Data loading for pretraining and supervised training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import DLDataLoader, RSSIDataset

if TYPE_CHECKING:
    pass


def load_pretrain_data(
    config: DLConfig,
) -> tuple[DataLoader, DataLoader | None, DataLoader, DataLoader]:
    """Load data for SSL pretraining.

    Returns (train_loader, val_loader, probe_train_loader, probe_val_loader).
    The probe loaders carry labels — used by the linear probe that monitors
    representation quality during pretraining (probe MCC drives early
    stopping and checkpoint selection, not the raw val loss).
    """
    loader = DLDataLoader(config)
    X_all, y_all = loader.load_and_preprocess()
    X_all = X_all.astype(np.float32)

    # Z-score normalize per channel
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X_all - mean) / std

    # 80/20 train/val split
    n = len(X_norm)
    n_train = int(n * 0.8)
    indices = np.random.RandomState(config.seed).permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train = torch.from_numpy(X_norm[train_idx])
    X_val = torch.from_numpy(X_norm[val_idx])

    from torch.utils.data import TensorDataset

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            TensorDataset(X_val),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        if len(X_val) > 0
        else None
    )

    # Labeled loaders for the probe (smaller batch, capped size)
    probe_batch = min(config.batch_size, 256)
    probe_train_loader = DataLoader(
        RSSIDataset(X_norm[train_idx], y_all[train_idx]),
        batch_size=probe_batch,
        shuffle=True,
        drop_last=False,
    )
    probe_val_loader = DataLoader(
        RSSIDataset(X_norm[val_idx], y_all[val_idx]),
        batch_size=probe_batch,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, probe_train_loader, probe_val_loader


def load_supervised_data(
    config: DLConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load data for supervised training.

    Returns (train_loader, val_loader, test_loader, classes).
    """
    dl = DLDataLoader(config)
    X, y = dl.load_and_preprocess()

    classes = dl.classes_
    label_map = {0: "AA", 1: "AB", 2: "BA", 3: "BB"}
    classes = [label_map.get(i, c) for i, c in enumerate(classes)]

    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=config.seed, stratify=y_tr
    )

    train_loader = DataLoader(RSSIDataset(X_tr, y_tr), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(RSSIDataset(X_va, y_va), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(RSSIDataset(X_te, y_te), batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, classes
