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


def _expand_features_rich(X_seq: np.ndarray) -> np.ndarray:
    """Expand (N, T, 1) → (N, T, C) with spectral, statistical, and shape features.

    The base 4 channels (raw, Δ, Δ², deviation) are deterministic transforms of
    one signal. These add genuinely new information: frequency content, window
    shape, and distributional statistics that raw values and their differences
    cannot express.
    """
    raw = X_seq[:, :, 0].astype(np.float32)  # (N, T)
    N, T = raw.shape

    # --- base temporal channels (same as _expand_features) ---
    zeros_col = np.zeros((N, 1), dtype=np.float32)
    diff1 = np.concatenate([zeros_col, np.diff(raw, axis=1)], axis=1)
    diff2 = np.concatenate([zeros_col, np.diff(diff1, axis=1)], axis=1)
    dev = raw - raw.mean(axis=1, keepdims=True)

    # --- spectral content: rolling FFT band energies ---
    # 3-sample window at each position; captures local frequency signature.
    pad = np.pad(raw, ((0, 0), (1, 1)), mode="edge")  # (N, T+2)
    windows = np.stack([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], axis=-1)  # (N, T, 3)
    spec = np.abs(np.fft.rfft(windows, axis=-1))  # (N, T, 2)
    band_dc = spec[:, :, 0]                       # mean level
    band_ac = spec[:, :, 1]                       # local oscillation energy

    # --- window statistics (rolling) ---
    roll_mean = windows.mean(axis=-1)
    roll_range = windows.max(axis=-1) - windows.min(axis=-1)
    roll_std = windows.std(axis=-1)
    # skew/kurtosis over a 3-sample window are near-degenerate; use a wider 5-sample
    # window for higher moments so the statistic actually varies.
    pad5 = np.pad(raw, ((0, 0), (2, 2)), mode="edge")
    win5 = np.stack([pad5[:, :-4], pad5[:, 1:-3], pad5[:, 2:-2], pad5[:, 3:-1], pad5[:, 4:]], axis=-1)
    mu = win5.mean(axis=-1, keepdims=True)
    s = win5.std(axis=-1, keepdims=True) + 1e-8
    roll_skew = ((win5 - mu) ** 3).mean(axis=-1) / (s[..., 0] ** 3)
    roll_kurt = ((win5 - mu) ** 4).mean(axis=-1) / (s[..., 0] ** 4)

    # --- cross-timestep shape ---
    peak_loc = np.argmax(np.abs(win5), axis=-1)      # 0..4 within window
    slope_sign = np.sign(diff1)                       # +1 / 0 / -1 per step
    sign_change = np.abs(np.diff(slope_sign, axis=1))  # 0/1/2 per boundary
    sign_change = np.concatenate([zeros_col, sign_change], axis=1)  # align to T

    # --- full-sequence STFT (4 frequency-band energies) ---
    # Sequence-level spectrum, broadcast to every timestep. Captures global
    # frequency structure the 3-sample rolling FFT cannot see.
    window = np.hanning(T).astype(np.float32)
    framed = raw * window[None, :]
    stft = np.abs(np.fft.rfft(framed, axis=1))  # (N, T//2 + 1)
    # 4 log-energy bands split across the positive frequencies
    n_bins = stft.shape[1]
    edges = np.linspace(0, n_bins, 5, dtype=np.int64)
    bands = []
    for i in range(4):
        b = stft[:, edges[i]:edges[i + 1]]
        bands.append(np.log1p(b.mean(axis=1)))  # (N,)
    stft_feats = np.stack(bands, axis=1)  # (N, 4)
    stft_feats = np.repeat(stft_feats[:, None, :], T, axis=1)  # (N, T, 4)

    feats = [
        raw, diff1, diff2, dev,      # base 4
        band_dc, band_ac,            # local spectral 2
        roll_mean, roll_range, roll_std, roll_skew, roll_kurt,  # stats 5
        peak_loc.astype(np.float32), slope_sign.astype(np.float32), sign_change.astype(np.float32),  # shape 3
    ]
    feats_t = np.stack(feats, axis=-1)  # (N, T, 14)
    return np.concatenate([feats_t, stft_feats], axis=-1)  # (N, T, 18)


# Canonical noise-path label → index mapping (unknown / not-applicable → 4)
_NOISE_PATH_LABELS: dict[str, int] = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}


class RSSIDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for RSSI time-series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MetaRSSIDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset that returns (X, meta, y) triples for metadata-fusion models.

    meta shape: (2,) int64
        meta[0] = noise_int       (0=False, 1=True)
        meta[1] = noise_path_idx  (0..3 = AA/AB/BA/BB, 4 = unknown/none)
    """

    def __init__(self, X: np.ndarray, meta: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.meta[idx], self.y[idx]


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

        # Always reshape to (N, seq_len, 1) raw, then expand to (N, seq_len, C)
        X_seq_raw = X_scaled.reshape(-1, self.config.seq_len, 1)
        X_seq = (
            _expand_features_rich(X_seq_raw)
            if getattr(self.config, "rich_features", False)
            else _expand_features(X_seq_raw)
        )
        return X_seq, y

    def load_and_preprocess_with_meta(
        self, path: Path | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, meta, y) where meta is (N, 2) int64 with noise + path indices."""
        df = pd.read_csv(path or self.config.data_path)
        X_raw: np.ndarray = df[self.config.feature_cols].values.astype(np.float32)
        y_raw: np.ndarray = df[self.config.label_col].values

        X_scaled: np.ndarray = self.scaler.fit_transform(X_raw).astype(np.float32)
        y: np.ndarray = self.label_encoder.fit_transform(y_raw).astype(np.int64)
        self.classes_ = list(self.label_encoder.classes_)

        X_seq_raw = X_scaled.reshape(-1, self.config.seq_len, 1)
        X_seq = (
            _expand_features_rich(X_seq_raw)
            if getattr(self.config, "rich_features", False)
            else _expand_features(X_seq_raw)
        )

        # ── Metadata encoding ─────────────────────────────────────────────────
        noise_col = self.config.noise_col
        path_col = self.config.noise_path_col

        if noise_col in df.columns:
            noise_int = (
                df[noise_col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "1": 1, "false": 0, "0": 0})
                .fillna(0)
                .values.astype(np.int64)
            )
        else:
            noise_int = np.zeros(len(df), dtype=np.int64)

        if path_col in df.columns:
            path_idx = (
                df[path_col]
                .astype(str)
                .map(lambda v: _NOISE_PATH_LABELS.get(v.upper(), 4))
                .values.astype(np.int64)
            )
        else:
            path_idx = np.full(len(df), 4, dtype=np.int64)

        meta = np.stack([noise_int, path_idx], axis=-1)  # (N, 2)
        return X_seq, meta, y

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

    def train_test_split_with_meta(
        self, X: np.ndarray, meta: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified split that keeps meta aligned with X and y."""
        idx = np.arange(len(y))
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=y,
        )
        return (
            X[idx_tr],
            X[idx_te],
            meta[idx_tr],
            meta[idx_te],
            y[idx_tr],
            y[idx_te],
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
        num_workers: int = 0,
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

    def make_meta_loader(
        self,
        X: np.ndarray,
        meta: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """DataLoader that yields (X, meta, y) triples for metadata-fusion models."""
        dataset = MetaRSSIDataset(X, meta, y)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
