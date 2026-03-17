from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.config import _resolve_path


class TestConfig(unittest.TestCase):
    def test_resolve_path_uses_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "from_env.joblib"
            env_path.write_text("x")
            os.environ["INMOTION_TEST_PATH"] = str(env_path)
            try:
                result = _resolve_path("INMOTION_TEST_PATH", [Path("fallback")])
                self.assertEqual(result, env_path)
            finally:
                os.environ.pop("INMOTION_TEST_PATH", None)

    def test_resolve_path_uses_first_existing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.csv"
            existing = Path(tmp) / "existing.csv"
            existing.write_text("content")
            result = _resolve_path("INMOTION_TEST_PATH", [missing, existing])
            self.assertEqual(result, existing)


if __name__ == "__main__":
    unittest.main()
