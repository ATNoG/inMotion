from __future__ import annotations

import os
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
        self._debug = os.getenv("INMOTION_INFERENCE_DEBUG", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _debug_print(self, message: str) -> None:
        if self._debug:
            print(f"[INFERENCE DEBUG] {message}")

    def startup(self) -> None:
        self._degraded = True
        self._model = None
        if not self._config.model_path.exists():
            self._debug_print(
                f"Model file not found at {self._config.model_path}. Falling back to degraded mode."
            )
            return
        model = joblib.load(self._config.model_path)
        if not hasattr(model, "predict_proba"):
            self._debug_print(
                f"Model at {self._config.model_path} has no predict_proba(). Falling back to degraded mode."
            )
            return
        self._model = model
        if hasattr(model, "classes_"):
            classes = [str(c) for c in model.classes_]
            filtered = [c for c in classes if c in self._config.classes]
            if filtered:
                self._classes = filtered
        self._degraded = False
        self._debug_print(
            f"Model loaded from {self._config.model_path}. classes={self._classes} degraded={self._degraded}"
        )

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def predict(self, rssi_window: list[float]) -> dict:
        if len(rssi_window) != self._config.window_size:
            raise ValueError("RSSI window must contain 10 samples")

        self._debug_print(
            f"Input window={rssi_window} min={min(rssi_window):.2f} max={max(rssi_window):.2f} mean={float(np.mean(rssi_window)):.2f}"
        )

        if self._degraded or self._model is None:
            mean_rssi = float(np.mean(rssi_window))
            rng = np.random.default_rng(seed=int(abs(mean_rssi) * 1000) % (2**32 - 1))
            proba = rng.dirichlet(np.ones(len(self._classes)))
            source = "degraded"
        else:
            x = np.array(rssi_window, dtype=np.float32).reshape(1, -1)
            proba = self._model.predict_proba(x)[0]
            self._debug_print(f"Raw model output probabilities={proba} sum={sum(proba):.6f}")
            source = "model"

        idx = int(np.argmax(proba))
        predicted_class = self._classes[idx]
        probabilities = {
            cls: round(float(p), 6) for cls, p in zip(self._classes, proba, strict=False)
        }
        self._debug_print(
            f"source={source} predicted_class={predicted_class} confidence={float(proba[idx]):.6f} probabilities={probabilities}"
        )

        return {
            "timestamp": datetime.now(UTC),
            "predicted_class": predicted_class,
            "confidence": round(float(proba[idx]), 6),
            "probabilities": probabilities,
            "source": source,
        }
