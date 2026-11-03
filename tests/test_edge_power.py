"""Energy: the model's monotonicity and range, meters, and the power table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inmotion.edge import contracts, power
from inmotion.edge.config import read_table, write_table


def _bench_rows(device: str = "wave-router", reps: int = 3) -> list[dict[str, object]]:
    return [
        {
            "device": device,
            "model": "lejepa",
            "precision": "fp32",
            "rep": rep,
            "cpus": "0-3",
            "concurrency": 1,
            "iterations": 500,
            "p50_ms": 2.0,
            "p95_ms": 2.6,
            "throughput_s": 500.0,
            "cpu_ms_bench": 900.0,
            "cpu_pct_bench": 90.0,
            "cpu_ms_per_infer": 1.8,
            "status": "ok",
        }
        for rep in range(reps)
    ]


def _write_bench(tmp_path: Path, device: str = "wave-router") -> Path:
    path = tmp_path / "bench.csv"
    write_table(path, "bench", _bench_rows(device))
    return tmp_path


def test_estimate_is_monotonic_in_utilisation() -> None:
    idle = power.estimate_energy(cpu_s=0.1, wall_s=1.0, cores=4, tdp_w=10.0, inferences=100)
    busy = power.estimate_energy(cpu_s=3.0, wall_s=1.0, cores=4, tdp_w=10.0, inferences=100)
    assert idle.utilization == pytest.approx(0.025)
    assert busy.utilization == pytest.approx(0.75)
    assert idle.mj_est < busy.mj_est
    assert idle.mj_lo <= idle.mj_est <= idle.mj_hi
    assert busy.mj_lo <= busy.mj_est <= busy.mj_hi
    # The band is the declared idle-fraction set, so it is never empty.
    assert busy.mj_hi > busy.mj_lo


def test_estimate_is_unavailable_not_zero_for_bad_inputs() -> None:
    for kwargs in (
        {"cpu_s": -1.0, "wall_s": 1.0, "cores": 4, "tdp_w": 10.0, "inferences": 100},
        {"cpu_s": 1.0, "wall_s": 0.0, "cores": 4, "tdp_w": 10.0, "inferences": 100},
        {"cpu_s": 1.0, "wall_s": 1.0, "cores": 0, "tdp_w": 10.0, "inferences": 100},
        {"cpu_s": 1.0, "wall_s": 1.0, "cores": 4, "tdp_w": -1.0, "inferences": 100},
        {"cpu_s": 1.0, "wall_s": 1.0, "cores": 4, "tdp_w": 10.0, "inferences": 0},
    ):
        estimate = power.estimate_energy(**kwargs)  # type: ignore[arg-type]
        assert estimate.mj_est == -1.0
        assert estimate.mj_lo == -1.0


def test_meter_json_shapes() -> None:
    assert power.parse_meter_text('{"power_w": 4.2}') == (4.2, "meter=json:power_w")
    watts, assumption = power.parse_meter_text('{"energy_j": 10.0, "seconds": 2.0}')
    assert watts == pytest.approx(5.0)
    assert "energy_j" in assumption


def test_meter_power_series_is_integrated() -> None:
    watts, assumption = power.parse_meter_text("t,power_w\n0,2\n1,4\n2,2\n")
    # Trapezoid: (2+4)/2*1 + (4+2)/2*1 = 6 J over 2 s -> 3 W.
    assert watts == pytest.approx(3.0)
    assert "power_series" in assumption


def test_meter_single_column_watts_and_joules() -> None:
    assert power.parse_meter_text("0.5\n1.5\n") == (1.0, "meter=file:power_watts(2 samples)")
    watts, assumption = power.parse_meter_text("energy_j\n100\n")
    assert watts == 100.0
    assert "joules" in assumption


def test_malformed_meters_raise_instead_of_returning_zero() -> None:
    for text in ("", "   \n", '{"nothing": 1}', "0\n0\n", "-1\n-2\n", "t,p\n1,0\n2,0\n"):
        with pytest.raises(power.MeterError):
            power.parse_meter_text(text)


def test_meter_file_missing(tmp_path: Path) -> None:
    with pytest.raises(power.MeterError, match="cannot read"):
        power.read_meter_file(tmp_path / "nope.csv")


def test_count_cores() -> None:
    assert power._count_cores("0-3") == 4
    assert power._count_cores("0,2") == 2
    assert power._count_cores("") == 0


def test_run_prefers_a_measured_meter(tmp_path: Path) -> None:
    from inmotion.edge import transport

    tables = _write_bench(tmp_path)
    meter = tmp_path / "meter.csv"
    meter.write_text('{"power_w": 5.0}')
    device = transport.LocalTransport(root=str(tmp_path / "device"))
    device.key = "wave-router"  # a router identity gives the model a TDP
    path = power.run(device, tables, meter_file=meter)
    rows = read_table(path, "power")
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "meter"
    assert float(row["mj_per_infer_meas"]) > 0
    assert row["cores"] == "4"
    assert row["assumptions"]


def test_run_falls_back_to_the_model_on_a_malformed_meter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inmotion.edge import transport

    tables = _write_bench(tmp_path)
    meter = tmp_path / "meter.csv"
    meter.write_text("this is not a meter\n")
    device = transport.LocalTransport(root=str(tmp_path / "device"))
    device.key = "wave-router"
    path = power.run(device, tables, meter_file=meter)
    rows = read_table(path, "power")
    assert rows[0]["source"] == "model"
    assert float(rows[0]["mj_per_infer_est"]) > 0
    assert float(rows[0]["mj_per_infer_meas"]) == -1.0
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "meter file ignored" in captured.err


def test_run_without_a_tdp_reports_none_not_zero(tmp_path: Path) -> None:
    from inmotion.edge import transport

    tables = _write_bench(tmp_path, device="local")
    device = transport.LocalTransport(root=str(tmp_path / "device"))
    path = power.run(device, tables)
    rows = read_table(path, "power")
    assert rows[0]["source"] == "none"
    assert float(rows[0]["mj_per_infer_est"]) == -1.0


def test_power_table_header(tmp_path: Path) -> None:
    from inmotion.edge import transport

    tables = _write_bench(tmp_path, device="local")
    device = transport.LocalTransport(root=str(tmp_path / "device"))
    path = power.run(device, tables)
    assert path.read_text().splitlines()[0] == ",".join(contracts.COLUMNS["power"])


def test_missing_bench_is_an_error(tmp_path: Path) -> None:
    from inmotion.edge import transport

    device = transport.LocalTransport(root=str(tmp_path / "device"))
    with pytest.raises(FileNotFoundError, match="run bench first"):
        power.run(device, tmp_path)


def test_unavailable_energy_columns_use_minus_one(tmp_path: Path) -> None:
    rows = _bench_rows(reps=1)
    rows[0]["iterations"] = -1
    write_table(tmp_path / "bench.csv", "bench", rows)
    from inmotion.edge import transport

    device = transport.LocalTransport(root=str(tmp_path / "device"))
    device.key = "wave-router"
    path = power.run(device, tmp_path)
    row = read_table(path, "power")[0]
    assert float(row["mj_per_infer_est"]) == -1.0
    assert row["source"] == "none"


def test_parse_meter_text_accepts_a_json_list_as_malformed() -> None:
    with pytest.raises(power.MeterError):
        power.parse_meter_text(json.dumps([1, 2, 3]))
