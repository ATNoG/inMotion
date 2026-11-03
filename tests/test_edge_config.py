"""Profiles, run layout and the shared table writer (no device needed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from inmotion.edge import config, contracts


def test_profiles_have_the_contract_matrix() -> None:
    assert set(config.PROFILES) == {"smoke", "quick", "full"}
    smoke, quick, full = config.PROFILES["smoke"], config.PROFILES["quick"], config.PROFILES["full"]

    assert smoke.models == ("lejepa",)
    assert len(smoke.precisions) == 3
    assert smoke.reps == 1
    assert (quick.models, quick.reps) == (("lejepa",), 2)
    assert len(quick.precisions) == 3

    assert full.models == contracts.MODELS
    assert full.precisions == contracts.PRECISIONS_DEPLOYABLE
    assert full.reps == 3
    # 5 models x 9 precisions x 3 repetitions.
    assert full.cells == 135
    assert quick.cells == 6
    assert smoke.cells == 3


def test_full_profile_matches_the_declared_matrices() -> None:
    full = config.PROFILES["full"]
    # The contract's `--cpus` sets are prefix ranges: 1, 2, 3 and 4 cores.
    assert full.core_sets == ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    assert config.PROFILES["quick"].core_sets == full.core_sets
    assert full.concurrencies == (1, 2, 4)
    assert len(full.core_models) == 3
    assert len(full.core_precisions) == 3
    # core scaling: 3 models x 3 precisions x 4 core sets, plus concurrency.
    # 3 models x 3 precisions x (4 core sets + concurrency 2 and 4); the serial
    # concurrency=1 row is the 4-core core-matrix cell, never a duplicate.
    assert full.core_cells == 3 * 3 * (4 + 2)


def test_replay_counts_are_the_contract_ones() -> None:
    full = config.PROFILES["full"]
    assert full.replay_count("wave-router") == 703
    assert full.replay_count("wave-router-2") == 200
    assert config.PROFILES["smoke"].replay_count("local") == 16
    assert full.replay_count("nope") == 0


def test_stress_budgets() -> None:
    assert config.PROFILES["full"].stress_reps == 5
    assert config.PROFILES["full"].stress_seconds == 60.0
    assert config.PROFILES["quick"].stress_reps == 2
    assert config.PROFILES["quick"].stress_seconds == 15.0


def test_profile_lookup_names_the_choices() -> None:
    with pytest.raises(KeyError, match="full, quick, smoke"):
        config.profile("enormous")
    assert config.profile("quick").name == "quick"


def test_iteration_bounds_follow_the_contract() -> None:
    for name in ("quick", "full"):
        chosen = config.PROFILES[name]
        assert chosen.min_iterations == 30
        assert chosen.max_iterations == 500
        assert chosen.calib_iterations == 10


def test_expected_rows() -> None:
    rows = config.PROFILES["quick"].expected_rows()
    assert rows["bench"] == 6
    assert rows["load"] == 12
    assert rows["cores"] == 2 * (4 + 2)
    assert rows["replay"] == 2  # two replayed devices
    assert rows["stress_windows"] == 2 * 4 * 2


def test_utc_stamp_is_a_parseable_utc_id() -> None:
    stamp = config.utc_stamp(0)
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert stamp == "19700101T000000Z"
    assert config.stamp_epoch(stamp) == 0


def test_campaign_config_resolves_the_shared_layout(tmp_path: Path) -> None:
    cfg = config.CampaignConfig.build(
        "quick", ["wave-router"], results_root=tmp_path, run_dir=tmp_path / "runs" / "x"
    )
    assert cfg.tables_dir == tmp_path / "runs" / "x" / "tables"
    assert cfg.raw_dir == tmp_path / "runs" / "x" / "raw"
    assert cfg.figures_dir == tmp_path / "runs" / "x" / "figures"
    assert cfg.logs_dir == tmp_path / "runs" / "x" / "logs"
    assert cfg.artifacts_dir == tmp_path / "artifacts"
    assert cfg.windows_dir == tmp_path / "windows"
    cfg.ensure_dirs()
    assert cfg.tables_dir.is_dir()
    assert cfg.logs_dir.is_dir()


def test_campaign_config_rejects_unknown_devices_and_phases(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown device"):
        config.CampaignConfig.build("smoke", ["router-nine"], results_root=tmp_path)
    with pytest.raises(KeyError, match="unknown phase"):
        config.CampaignConfig.build(
            "smoke", ["local"], results_root=tmp_path, phases=["preflight", "polish"]
        )


def test_write_table_uses_the_contract_header_and_keeps_unavailable_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bench.csv"
    config.write_table(
        path,
        "bench",
        [
            {
                "device": "wave-router",
                "model": "lejepa",
                "precision": "fp32",
                "rep": 0,
                "p50_ms": 1.5,
                "peak_rss_kb": -1,
                "status": "ok",
                "not_a_column": "dropped",
                "empty": None,
            }
        ],
    )
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(contracts.COLUMNS["bench"])
    rows = config.read_table(path, "bench")
    assert rows[0]["p50_ms"] == "1.5"
    assert rows[0]["peak_rss_kb"] == "-1"
    assert "not_a_column" not in rows[0]


def test_read_table_refuses_a_drifted_header(tmp_path: Path) -> None:
    path = tmp_path / "bench.csv"
    path.write_text("device,model\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        config.read_table(path, "bench")


def test_cell_formats_booleans_and_nan() -> None:
    assert config.cell(True) == "True"
    assert config.cell(False) == "False"
    assert config.cell(float("nan")) == "-1"
    assert config.cell(0.0) == "0.0"
    assert config.cell(None) == ""
    assert config.cell(3) == "3"


def test_as_float_maps_garbage_to_the_default() -> None:
    assert config.as_float("1.25") == 1.25
    assert config.as_float("") == -1.0
    assert config.as_float(None) == -1.0
    assert config.as_float("nope", 7.0) == 7.0
