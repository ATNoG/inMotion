"""Data loading, preprocessing, and PyTorch Dataset/DataLoader utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import DLConfig


def _expand_features(X_seq: np.ndarray) -> np.ndarray:
    """Expand (N, T, 1) → (N, T, 4): raw RSSI, velocity (Δ), acceleration (Δ²), window-deviation.

    These four channels give the model positional, temporal, and distributional
    signal without any external data — a pure feature-engineering boost.
    """
    raw = X_seq[:, :, 0].astype(np.float32)  # (N, T)
    N, T = raw.shape
    zeros_col = np.zeros((N, 1), dtype=np.float32)
    diff1 = np.concatenate([zeros_col, np.diff(raw, axis=1)], axis=1)  # velocity
    diff2 = np.concatenate([zeros_col, np.diff(diff1, axis=1)], axis=1)  # acceleration
    dev = (raw - raw.mean(axis=1, keepdims=True)).astype(np.float32)  # window deviation
    return np.stack([raw, diff1, diff2, dev], axis=-1)  # (N, T, 4)


class RSSIDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for RSSI time-series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DLDataLoader:
    """Load, preprocess, and split the WiFi fingerprinting dataset."""

    def __init__(self, config: DLConfig) -> None:
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classes_: list[str] = []

    def load_and_preprocess(self, path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) shaped (N, seq_len, in_features) and (N,)."""
        df = pd.read_csv(path or self.config.data_path)
        X_raw: np.ndarray = df[self.config.feature_cols].values.astype(np.float32)
        y_raw: np.ndarray = df[self.config.label_col].values

        X_scaled: np.ndarray = self.scaler.fit_transform(X_raw).astype(np.float32)
        y: np.ndarray = self.label_encoder.fit_transform(y_raw).astype(np.int64)
        self.classes_ = list(self.label_encoder.classes_)

        # Always reshape to (N, seq_len, 1) raw, then expand to (N, seq_len, 4)
        X_seq_raw = X_scaled.reshape(-1, self.config.seq_len, 1)
        X_seq = _expand_features(X_seq_raw)  # → (N, seq_len, 4)
        return X_seq, y

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=y,
        )

    def cv_splits(self, X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_idx, val_idx) for stratified k-fold."""
        skf = StratifiedKFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.seed,
        )
        return list(skf.split(X, y))

    def make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        dataset = RSSIDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
