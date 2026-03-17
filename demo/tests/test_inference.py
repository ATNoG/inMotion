from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from app.config import DemoConfig
from app.inference import InferenceService
from sklearn.ensemble import RandomForestClassifier


def make_config(model_path: Path) -> DemoConfig:
    return DemoConfig(
        root_dir=Path("."),
        model_path=model_path,
        fallback_csv_path=Path("dataset_only_pure.csv"),
        test_router_csv_path=Path("dataset.csv"),
        scan_interval_seconds=1.0,
        scan_mode="auto",
        router_ip="",
        router_user="root",
        router_pass="",
        router_key_file="",
        router_port=22,
        router_command_mode="auto",
    )


class TestInference(unittest.TestCase):
    def test_degraded_mode_when_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_model = Path(tmp) / "definitely-missing.joblib"
            config = make_config(missing_model)
            service = InferenceService(config)
            service.startup()
            out = service.predict([-50.0] * 10)

            self.assertTrue(service.degraded)
            self.assertEqual(out["source"], "degraded")

    def test_model_mode_when_model_is_valid(self) -> None:
        x = np.array(
            [
                [-40.0] * 10,
                [-60.0] * 10,
                [-45.0] * 10,
                [-55.0] * 10,
            ]
        )
        y = np.array(["AA", "BB", "AA", "BB"])
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(x, y)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "rf.joblib"
            joblib.dump(model, model_path)

            config = make_config(model_path)
            service = InferenceService(config)
            service.startup()
            out = service.predict([-41.0] * 10)

            self.assertFalse(service.degraded)
            self.assertEqual(out["source"], "model")
            self.assertIn(out["predicted_class"], service.classes)


if __name__ == "__main__":
    unittest.main()
