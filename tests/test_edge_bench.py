"""Bench matrices driven by a fake runtime on LocalTransport (no device needed)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from inmotion.edge import bench, config, contracts

_BENCH_JSON = (
    '{"schema":2,"model":"lejepa","artifact":"x","iterations":50,"warmup":2,'
    '"min_ms":1.8,"median_ms":2.0,"mean_ms":2.1,"p95_ms":2.6,"p99_ms":3.0,'
    '"max_ms":3.2,"throughput_per_s":500.0,"wall_ms":100.0,"total_ms":120.0,'
    '"load_ms":9.0,"cpu_ms_load":8.0,"cpu_ms_bench":90.0,"cpu_pct_bench":90.0,'
    '"cpu_ms_per_infer":1.8,"rss_after_load_kb":5500,"peak_rss_kb":5600,'
    '"artifact_bytes":4096,"cpus":null,"governor":"schedutil","freq_mhz":1500.0}'
)
_LOAD_JSON = (
    '{"schema":2,"model":"lejepa","artifact":"x","cold":false,'
    '"artifact_bytes":4096,"load_ms":8.0,"cpu_ms_load":7.0,'
    '"rss_after_load_kb":5500,"peak_rss_kb":5500}'
)

_FAKE_BINARY = f"""#!/bin/sh
case "$1" in
  --help)
    echo "usage: inmotion-infer bench --artifact DIR [--cpus L]"
    exit 2 ;;
  bench)
    echo '{_BENCH_JSON}'
    exit 0 ;;
  load)
    echo '{_LOAD_JSON}'
    exit 0 ;;
  stream|check)
    echo "usage: $1 --artifact DIR"
    exit 2 ;;
  *)
    echo "unknown command"
    exit 1 ;;
esac
"""


@pytest.fixture
def fake_runtime(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "inmotion-infer"
    binary.parent.mkdir(parents=True)
    binary.write_text(_FAKE_BINARY)
    binary.chmod(0o755)
    return binary


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    rows = [
        "model,precision,runtime_loadable,path,bytes,sha256,status,error",
        f"lejepa,fp32,true,{root / 'lejepa-fp32'},4096,deadbeef,ok,",
        f"lejepa,q4,true,{root / 'lejepa-q4'},2048,deadbeef,ok,",
    ]
    (root / "artifacts.csv").write_text("\n".join(rows) + "\n")
    for name in ("lejepa-fp32", "lejepa-q4"):
        directory = root / name
        directory.mkdir()
        (directory / contracts.ARTIFACT_HASH_FILE).write_bytes(name.encode())
        (directory / "meta.json").write_text("{}")
    return root


@pytest.fixture
def local(tmp_path: Path) -> object:
    from inmotion.edge import transport

    return transport.LocalTransport(root=str(tmp_path / "device"))


def _tiny_profile(**overrides: object) -> config.Profile:
    base = config.PROFILES["smoke"]
    values: dict[str, object] = {
        "models": ("lejepa",),
        "precisions": ("fp32",),
        "reps": 1,
        "core_models": (),
        "core_precisions": (),
        "core_sets": (),
        "concurrencies": (),
        "replay_samples": {"local": 4},
        "stress_reps": 0,
        "stress_seconds": 0.0,
        "target_seconds": 2.0,
        "min_iterations": 30,
        "max_iterations": 500,
    }
    values.update(overrides)
    return dataclasses.replace(base, **values)


def test_cpu_set_grammar_round_trips() -> None:
    assert bench.parse_cpu_set("0") == (0,)
    assert bench.parse_cpu_set("0-2") == (0, 1, 2)
    assert bench.parse_cpu_set("0,2-3") == (0, 2, 3)
    assert bench.format_cpu_set((0, 1, 2)) == "0-2"
    assert bench.format_cpu_set((0, 1, 2, 3)) == "0-3"
    assert bench.parse_cpu_set("0-3") == (0, 1, 2, 3)
    assert bench.format_cpu_set((0, 2, 3)) == "0,2-3"
    assert bench.format_cpu_set([3]) == "3"
    with pytest.raises(ValueError, match="invalid range"):
        bench.parse_cpu_set("2-0")
    with pytest.raises(ValueError, match="empty core set"):
        bench.format_cpu_set([])


def test_resolve_artifacts_prefers_the_manifest_then_the_two_known_spellings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    for name in ("lejepa-fp32", "lejepa_ft_seed42-q4", "lejepa_ft_seed42-bnb-int8"):
        (root / name).mkdir(parents=True)
    (root / "artifacts.csv").write_text(
        "model,precision,runtime_loadable,path,bytes,sha256,status,error\n"
        f"lejepa,fp32,true,{root / 'lejepa-fp32'},1,x,ok,\n"
    )
    mapping = bench.resolve_artifacts(root)
    assert mapping[("lejepa", "fp32")] == root / "lejepa-fp32"
    # Fallback spelling works without a manifest row, but a producer variant of
    # the same precision must not be mistaken for the in-house artifact.
    assert mapping[("lejepa", "q4")] == root / "lejepa_ft_seed42-q4"
    assert ("lejepa", "int8") not in mapping


def test_last_json_reads_the_last_object_line() -> None:
    class Result:
        stdout = 'note\n{"a": 1}\nnoise\n{"b": 2}\n'

    assert bench.last_json(Result()) == {"b": 2}

    class Broken:
        stdout = "not json\n"

    assert bench.last_json(Broken()) == {}


def test_capability_probe_distinguishes_old_from_new(fake_runtime: Path, local: object) -> None:
    capabilities = bench.probe_capabilities(local, str(fake_runtime))
    assert capabilities["bench"] is True
    assert capabilities["load"] is True
    assert capabilities["cpus"] is True
    # The fake has no `stress`/`loadgen` and says so explicitly.
    assert capabilities["stress"] is False
    assert capabilities["loadgen"] is False


def test_calibration_clamps_to_the_profile_bounds(fake_runtime: Path, local: object) -> None:
    profile = _tiny_profile(min_iterations=30, max_iterations=500, target_seconds=2.0)
    # The fake reports 2.0 ms, so 2 s would be 1000 iterations -> clamped to 500.
    assert bench.calibrate_iterations(local, str(fake_runtime), "/x/lejepa-fp32", profile) == 500
    slow = _tiny_profile(min_iterations=30, max_iterations=500, target_seconds=0.01)
    assert bench.calibrate_iterations(local, str(fake_runtime), "/x/lejepa-fp32", slow) == 30


def test_run_writes_three_tables_and_the_raw_cells(
    fake_runtime: Path, artifacts: Path, local: object, tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    raw = tmp_path / "raw"
    profile = _tiny_profile()
    path = bench.run(local, artifacts, tables, raw, profile, binary=fake_runtime)

    assert path == tables
    loads = config.read_table(tables / "load.csv", "load")
    assert len(loads) == 2
    assert loads[0]["cold"] == "True"
    assert loads[1]["cold"] == "False"
    assert loads[0]["load_ms"] == "8.0"
    assert loads[0]["artifact_bytes"] == "4096"

    rows = config.read_table(tables / "bench.csv", "bench")
    assert len(rows) == 1
    row = rows[0]
    assert row["device"] == "local"
    assert row["p50_ms"] == "2.0"
    assert row["p95_ms"] == "2.6"
    assert row["p99_ms"] == "3.0"
    assert row["throughput_s"] == "500.0"
    assert row["cpu_ms_per_infer"] == "1.8"
    assert row["status"] == "ok"
    expected_sha = hashlib.sha256(b"lejepa-fp32").hexdigest()
    assert row["artifact_sha256"] == expected_sha
    # No TDP for a local host: the model is unavailable, not zero.
    assert row["energy_mj_est"] == "-1.0"
    assert row["power_source"] == "none"

    assert (tables / "cores.csv").read_text().splitlines()[0] == ",".join(
        contracts.COLUMNS["cores"]
    )
    raw_cells = sorted((raw / "bench").glob("*.json"))
    assert len(raw_cells) == 1
    payload = json.loads(raw_cells[0].read_text())
    assert payload["iterations"] == 500
    assert payload["state_before"]["governor"]


def test_run_refuses_a_missing_artifact(
    fake_runtime: Path, artifacts: Path, local: object, tmp_path: Path
) -> None:
    profile = _tiny_profile(precisions=("q2",))
    with pytest.raises(FileNotFoundError, match="no artifact"):
        bench.run(local, artifacts, tmp_path / "t", tmp_path / "r", profile, binary=fake_runtime)


def test_unsupported_load_is_recorded_not_invented(
    fake_runtime: Path, artifacts: Path, local: object, tmp_path: Path, monkeypatch
) -> None:
    real_probe = bench.probe_capabilities

    def old_binary(device: object, binary: str) -> dict[str, bool]:
        caps = real_probe(device, binary)
        caps["load"] = False
        return caps

    monkeypatch.setattr(bench, "probe_capabilities", old_binary)
    tables = tmp_path / "tables"
    bench.run(local, artifacts, tables, tmp_path / "raw", _tiny_profile(), binary=fake_runtime)
    loads = config.read_table(tables / "load.csv", "load")
    assert {row["status"] for row in loads} == {"unsupported:load"}
    assert loads[0]["load_ms"] == "-1.0"


def test_sample_state_reads_this_host(local: object) -> None:
    state = bench.sample_state(local)
    assert isinstance(state.governor, str)
    assert state.governor
    assert state.mem_available_kb > 0
    assert state.loadavg >= 0
    assert -1 <= state.foreign_cpu_pct <= 100
