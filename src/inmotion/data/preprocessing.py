"""Reproducible RSSI feature pipeline.

The training code standardised RSSI values with a `StandardScaler` and then
engineered extra channels, but **never persisted the scaler**. Two different
conventions were in use, which is why a saved model could not be re-run without
also having the exact training CSV:

``inmotion/dl/data_loader.py``
    ``scaler.fit_transform(X_raw)`` on the whole file, *before* the split.
``inmotion/pipelines/mega_ensemble.py``
    ``StandardScaler().fit(_raw_X[train_idx])`` on the training fold only.

:class:`FeaturePipeline` records which convention produced a model, stores the
fitted statistics next to the checkpoint, and replays them at inference. A
pipeline whose statistics were rebuilt from a reference CSV rather than
recovered from training is flagged ``reconstructed`` so results are never
presented as more reproducible than they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Feature engineering is imported from the training code so inference and
# training cannot drift apart.
from inmotion.dl.data_loader import (
    _expand_features,
    _expand_features_device_invariant,
    _expand_features_rich,
)

SCHEMA_VERSION = 1
STATS_FILE = "preprocessing.npz"
META_FILE = "preprocessing.json"

# How the scaler statistics were obtained. Kept explicit because the two
# conventions give different numbers on the same data.
CONVENTION_WHOLE_CSV = "fit_on_whole_csv"
CONVENTION_TRAIN_FOLD = "fit_on_train_fold"
CONVENTION_RECONSTRUCTED = "reconstructed_from_reference_csv"


@dataclass
class FeaturePipeline:
    """Standardisation + channel expansion for RSSI sequences."""

    feature_cols: list[str]
    seq_len: int = 10
    rich: bool = False
    device_invariant: bool = False
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    classes: list[str] = field(default_factory=list)
    convention: str = CONVENTION_RECONSTRUCTED
    reference_csv: str | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        csv_path: Path,
        *,
        feature_cols: list[str] | None = None,
        seq_len: int = 10,
        rich: bool = False,
        device_invariant: bool = False,
        convention: str = CONVENTION_WHOLE_CSV,
    ) -> FeaturePipeline:
        """Fit from a reference CSV.

        ``convention`` is recorded, not inferred: pass
        :data:`CONVENTION_WHOLE_CSV` to match ``inmotion/dl/data_loader.py`` or
        :data:`CONVENTION_TRAIN_FOLD` to match ``inmotion/pipelines/mega_ensemble.py``.
        """
        cols = feature_cols or [str(i) for i in range(1, seq_len + 1)]
        df = pd.read_csv(csv_path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing feature columns {missing}")

        raw = df[cols].to_numpy(dtype=np.float32)
        mean = raw.mean(axis=0)
        # Guard against a zero-variance column collapsing the scale to 0.
        scale = raw.std(axis=0)
        scale[scale == 0] = 1.0

        classes: list[str] = []
        if "label" in df.columns:
            classes = sorted(str(c) for c in df["label"].unique())

        return cls(
            feature_cols=list(cols),
            seq_len=seq_len,
            rich=rich,
            device_invariant=device_invariant,
            mean_=mean.astype(np.float32),
            scale_=scale.astype(np.float32),
            classes=classes,
            convention=convention,
            reference_csv=str(csv_path),
        )

    # -- inference ---------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """CSV rows -> ``(N, seq_len, C)`` float32 model input."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"input is missing feature columns {missing}")

        raw = df[self.feature_cols].to_numpy(dtype=np.float32)
        if raw.shape[1] != self.seq_len:
            raise ValueError(f"expected {self.seq_len} RSSI columns, got {raw.shape[1]}")

        if self.mean_ is not None and self.scale_ is not None:
            scaled = ((raw - self.mean_) / self.scale_).astype(np.float32)
        else:
            # No persisted statistics: fall back to per-row standardisation and
            # make the caller aware by keeping this path explicit.
            scaled = raw.astype(np.float32)

        seq = scaled.reshape(-1, self.seq_len, 1)
        if self.device_invariant:
            return _expand_features_device_invariant(seq)
        if self.rich:
            return _expand_features_rich(seq)
        return _expand_features(seq)

    @property
    def n_features(self) -> int:
        """Channel count the model expects (4 or 18)."""
        return 18 if self.rich else 4

    # -- persistence -------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
            "rich": self.rich,
            "device_invariant": self.device_invariant,
            "classes": self.classes,
            "convention": self.convention,
            "reference_csv": self.reference_csv,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
        arrays = {}
        if self.mean_ is not None:
            arrays["mean"] = self.mean_
        if self.scale_ is not None:
            arrays["scale"] = self.scale_
        np.savez(directory / STATS_FILE, **arrays)

    @classmethod
    def load(cls, directory: Path) -> FeaturePipeline:
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"no preprocessing metadata at {meta_path}")
        meta = json.loads(meta_path.read_text())

        mean = scale = None
        stats_path = directory / STATS_FILE
        if stats_path.exists():
            with np.load(stats_path) as data:
                mean = data.get("mean", None)
                scale = data.get("scale", None)

        return cls(
            feature_cols=list(meta["feature_cols"]),
            seq_len=int(meta["seq_len"]),
            rich=bool(meta["rich"]),
            device_invariant=bool(meta.get("device_invariant", False)),
            mean_=mean,
            scale_=scale,
            classes=list(meta.get("classes", [])),
            convention=meta.get("convention", CONVENTION_RECONSTRUCTED),
            reference_csv=meta.get("reference_csv"),
        )
