from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np

from .config import DemoConfig


class InferenceService:
    def __init__(self, config: DemoConfig) -> None:
        self._config = config
        self._model = None
        self._classes = list(config.classes)
        self._degraded = True

    def startup(self) -> None:
        if not self._config.model_path.exists():
            return
        model = joblib.load(self._config.model_path)
        if not hasattr(model, "predict_proba"):
            return
        self._model = model
        if hasattr(model, "classes_"):
            classes = [str(c) for c in model.classes_]
            filtered = [c for c in classes if c in self._config.classes]
            if filtered:
                self._classes = filtered
        self._degraded = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def predict(self, rssi_window: list[float]) -> dict:
        if len(rssi_window) != self._config.window_size:
            raise ValueError("RSSI window must contain 10 samples")

        if self._degraded:
            mean_rssi = float(np.mean(rssi_window))
            rng = np.random.default_rng(seed=int(abs(mean_rssi) * 1000) % (2**32 - 1))
            proba = rng.dirichlet(np.ones(len(self._classes)))
            source = "degraded"
        else:
            x = np.array(rssi_window, dtype=np.float32).reshape(1, -1)
            proba = self._model.predict_proba(x)[0]
            source = "model"

        idx = int(np.argmax(proba))
        predicted_class = self._classes[idx]
        probabilities = {
            cls: round(float(p), 6) for cls, p in zip(self._classes, proba, strict=False)
        }

        return {
            "timestamp": datetime.now(UTC),
            "predicted_class": predicted_class,
            "confidence": round(float(proba[idx]), 6),
            "probabilities": probabilities,
            "source": source,
        }
