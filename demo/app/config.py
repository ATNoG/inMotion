from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(env_key: str, candidates: list[Path]) -> Path:
    env_value = os.getenv(env_key)
    if env_value:
        return Path(env_value)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


@dataclass(frozen=True)
class DemoConfig:
    root_dir: Path
    model_path: Path
    fallback_csv_path: Path
    test_router_csv_path: Path
    scan_interval_seconds: float
    scan_mode: str
    router_ip: str
    router_user: str
    router_pass: str
    router_key_file: str
    router_port: int
    router_command_mode: str
    classes: tuple[str, ...] = ("AA", "AB", "BA", "BB")
    window_size: int = 10


def load_config() -> DemoConfig:
    app_dir = Path(__file__).resolve().parent
    demo_dir = app_dir.parent
    root_dir = demo_dir.parent

    model_path = _resolve_path(
        "INMOTION_MODEL_PATH",
        [
            Path("/app/models/RandomForest.joblib"),
            Path("/app/models/GaussianProcess.joblib"),
            demo_dir / "models" / "GaussianProcess.joblib",
            root_dir / "models" / "GaussianProcess.joblib",
        ],
    )
    fallback_csv_path = _resolve_path(
        "INMOTION_FALLBACK_CSV",
        [
            demo_dir / "dataset_only_pure.csv",
            root_dir / "dataset_only_pure.csv",
        ],
    )
    test_router_csv_path = _resolve_path(
        "INMOTION_TEST_ROUTER_CSV",
        [
            demo_dir / "dataset.csv",
            root_dir / "dataset.csv",
            fallback_csv_path,
        ],
    )

    return DemoConfig(
        root_dir=root_dir,
        model_path=model_path,
        fallback_csv_path=fallback_csv_path,
        test_router_csv_path=test_router_csv_path,
        scan_interval_seconds=float(os.getenv("INMOTION_SCAN_INTERVAL", "1.0")),
        scan_mode=os.getenv("INMOTION_SCAN_MODE", "auto"),
        router_ip=os.getenv("INMOTION_ROUTER_IP", ""),
        router_user=os.getenv("INMOTION_ROUTER_USER", "root"),
        router_pass=os.getenv("INMOTION_ROUTER_PASS", ""),
        router_key_file=os.getenv("INMOTION_ROUTER_KEY_FILE", ""),
        router_port=int(os.getenv("INMOTION_ROUTER_PORT", "22")),
        router_command_mode=os.getenv("INMOTION_ROUTER_COMMAND_MODE", "auto"),
    )
