"""Stress: ABBA order, the busybox window script, parsers and the guards."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from inmotion.edge import config, contracts, stress


def _profile(**overrides: object) -> config.Profile:
    values: dict[str, object] = {
        "models": ("lejepa",),
        "precisions": ("fp32",),
        "stress_reps": 1,
        "stress_seconds": 0.5,
        "stress_workers": 1,
        "stress_model": "lejepa",
        "stress_precision": "fp32",
    }
    values.update(overrides)
    return dataclasses.replace(config.PROFILES["smoke"], **values)


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    directory = root / "lejepa-fp32"
    directory.mkdir(parents=True)
    (directory / contracts.ARTIFACT_HASH_FILE).write_bytes(b"weights")
    (directory / "meta.json").write_text("{}")
    return root


def _script(**overrides: object) -> str:
    values: dict[str, object] = {
        "window": "w1",
        "device": "local",
        "phase": "both",
        "repetition": 0,
        "seconds": 2.0,
        "binary": "/bin/true",
        "artifact": "/tmp/artifact",
        "cpus": "3",
        "out_dir": "/tmp/out",
        "probes": {"lan": "127.0.0.1", "fwd": ""},
    }
    values.update(overrides)
    return stress.build_window_script(**values)


def test_phase_order_balances_even_and_odd_repetitions() -> None:
    first = stress.phase_order(0)
    second = stress.phase_order(1)
    assert first == contracts.STRESS_PHASES
    assert second == tuple(reversed(contracts.STRESS_PHASES))
    assert stress.phase_order(2) == first
    # ABBA across two repetitions: every phase appears once early and once late.
    positions = {
        phase: [first.index(phase), second.index(phase)] for phase in contracts.STRESS_PHASES
    }
    assert {phase: sum(positions[phase]) for phase in positions} == {
        phase: len(contracts.STRESS_PHASES) - 1 for phase in contracts.STRESS_PHASES
    }


def test_window_id_is_filesystem_safe() -> None:
    assert stress.window_id("2026/10 20 x", "both", 2) == "2026-10-20-x-both-r2"
    assert stress.window_id("", "baseline", 0) == "run-baseline-r0"


def test_window_script_is_busybox_safe() -> None:
    script = _script()
    assert "trap cleanup EXIT INT TERM" in script
    assert "usleep 500000" in script
    assert "usleep 500000 2>/dev/null || sleep 1" in script
    # busybox `sleep` rejects fractions; a fractional sleep anywhere is a bug.
    assert "sleep 0." not in script
    assert "sleep 1" in script
    assert 'ping -c "$SECS" -i 1' in script
    assert "loadgen --artifact" in script
    assert "stress --mode spin" in script
    assert "MemAvailable" in script
    assert f"{stress.MEM_FLOOR_KB}" in script
    # An empty probe target must not launch a ping.
    assert "run_probe lan" in script
    assert "run_probe fwd" not in script


def test_window_script_passes_the_inference_interval() -> None:
    with_interval = _script(inference_interval_ms=2000)
    assert "--interval-ms 2000" in with_interval
    assert "--interval-ms" not in _script(inference_interval_ms=0)


def test_window_script_phases_are_exclusive() -> None:
    baseline = _script(phase="baseline")
    assert "loadgen --artifact" in baseline  # the command text exists...
    # ...but the baseline branch cannot execute it.
    assert 'if [ "$MODE" = "inference" ] || [ "$MODE" = "both" ]' in baseline
    assert "MODE=baseline" in baseline


def test_parse_sample_line_round_trips_every_field() -> None:
    line = (
        "w1|wave-router|stress|2|1.5|42.00|1.20|0.10|0.30|10.0;20.0|1900000|3.20|"
        "1000|2000|4|5|6|7|800|900|45000|1500000"
    )
    row = stress.parse_sample_line(line)
    assert row is not None
    assert row["window"] == "w1"
    assert row["phase"] == "stress"
    assert row["repetition"] == 2
    assert row["t_rel_s"] == 1.5
    assert row["cpu_busy_pct"] == 42.0
    assert row["cpu_iowait_pct"] == 1.2
    assert row["cpu_softirq_pct"] == 0.1
    assert row["cpu_sirq_pct"] == 0.3
    assert row["per_core_busy_pct"] == "10.0;20.0"
    assert row["mem_available_kb"] == 1_900_000
    assert row["loadavg_1m"] == 3.2
    assert (row["rx_bytes"], row["tx_bytes"]) == (1000, 2000)
    assert (row["rx_drops"], row["tx_drops"]) == (4, 5)
    assert (row["rx_errors"], row["tx_errors"]) == (6, 7)
    assert row["temp_c"] == 45.0
    assert row["freq_mhz"] == 1500.0
    assert stress.packets_from_line(line) == (800, 900)
    assert stress.packets_from_line("too|few|fields") is None


def test_parse_probe_line() -> None:
    row = stress.parse_probe_line("w1|wave-router|both|1|lan|3|7|0.456|0")
    assert row == {
        "window": "w1",
        "device": "wave-router",
        "phase": "both",
        "repetition": 1,
        "probe": "lan",
        "seq": 7,
        "rtt_ms": 0.456,
        "lost": 0,
    }
    assert stress.parse_probe_line("nope") is None


def test_parse_host_ping_anchors_on_the_first_reply_and_fills_trailing_losses() -> None:
    text = (
        "PING 192.168.93.13 (192.168.93.13) 56(84) bytes of data.\n"
        "64 bytes from 192.168.93.13: icmp_seq=1 ttl=64 time=0.412 ms\n"
        "64 bytes from 192.168.93.13: icmp_seq=2 ttl=64 time=0.398 ms\n"
        "64 bytes from 192.168.93.13: icmp_seq=4 ttl=64 time=0.501 ms\n"
    )
    rows = stress.parse_host_ping(
        text, window="w1", device="wave-router", phase="both", repetition=0, count=5
    )
    assert [row["seq"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["lost"] for row in rows] == [0, 0, 1, 0, 1]
    assert rows[0]["rtt_ms"] == pytest.approx(0.412)
    assert stress.probe_loss_pct(rows) == pytest.approx(40.0)
    assert stress.probe_loss_pct([], "lan") == -1.0


def test_run_writes_both_tables_and_the_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inmotion.edge import transport

    # The test host may be busy; make the guards unreachable so the window
    # completes and the tables are what is under test.
    monkeypatch.setattr(stress, "LOAD_CEILING", 10**9)
    monkeypatch.setattr(stress, "MEM_FLOOR_KB", 1)

    device = transport.LocalTransport(root=str(tmp_path / "device"))
    tables = tmp_path / "tables"
    raw = tmp_path / "raw"
    path = stress.run(device, _artifact(tmp_path), tables, raw, _profile(), token="test")
    assert path == tables
    samples = config.read_table(tables / "stress_raw.csv", "stress_raw")
    assert len(samples) == 4  # four phases, one window each
    assert {row["phase"] for row in samples} == set(contracts.STRESS_PHASES)
    assert samples[0]["window"].startswith("test-")

    probes = config.read_table(tables / "stress_probes.csv", "stress_probes")
    assert probes == []  # no probes are defined for the local transport

    windows = json.loads((raw / "stress" / "windows.json").read_text())
    assert windows["aborted"] is False
    assert len(windows["windows"]) == 4

    inference_rows = (raw / "stress" / "inference.csv").read_text().splitlines()
    assert inference_rows[0].startswith("window,device,phase,repetition")
    assert len(inference_rows) == 3  # inference + both, as unsupported:loadgen
    assert all("unsupported" in line for line in inference_rows[1:])

    interfaces = (raw / "stress" / "interfaces.csv").read_text().splitlines()
    assert len(interfaces) == 5


def test_a_guard_stops_the_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inmotion.edge import transport

    # A memory floor no machine can meet trips the device-side guard on the very
    # first sample; the run must stop and say why instead of continuing.
    monkeypatch.setattr(stress, "MEM_FLOOR_KB", 10**12)
    device = transport.LocalTransport(root=str(tmp_path / "device"))
    tables = tmp_path / "tables"
    raw = tmp_path / "raw"
    stress.run(
        device,
        _artifact(tmp_path),
        tables,
        raw,
        _profile(stress_reps=3),
        token="guard",
    )
    windows = json.loads((raw / "stress" / "windows.json").read_text())
    assert windows["aborted"] is True
    assert "device guard" in windows["reason"]
    assert len(windows["windows"]) == 1
    assert windows["windows"][0]["status"] == "aborted:device-guard"
