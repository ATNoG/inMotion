"""Energy per inference: measured sources first, a labelled model last.

Neither router exposes an energy counter (no powercap, no hwmon power/energy,
no battery), so every number this module writes is either a *measured* package
power from a source that happened to exist, or the CPU-package model from
:mod:`inmotion.edge.contracts` with its idle-fraction range. The model is
labelled as a model in the ``source`` column and printed with the range beside
it; it is never presented as a measurement.

Source order, per the contract: ``rapl`` → ``hwmon`` → ``meter`` → ``model`` →
``none``. ``none`` means "no number is available" and writes ``-1``, never 0 — a
zero here would read as "free" instead of "unknown".

A malformed meter file is a warning and a fallback, not a zero: the run continues
with the next source and the reason is recorded.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inmotion.edge import contracts, transport

#: Where a measured value came from, in preference order.
SOURCES = ("rapl", "hwmon", "meter", "model", "none")

#: The middle idle fraction is the reported point estimate; the range is the
#: contract's full set (0.15, 0.30, 0.50).
_CENTRAL_IDLE = 0.30

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


class MeterError(ValueError):
    """A meter file or command returned something this module cannot use."""


@dataclass(frozen=True)
class EnergyEstimate:
    """The model's answer for one measurement, in millijoules per inference."""

    utilization: float
    mj_est: float
    mj_lo: float
    mj_hi: float

    def as_row(self) -> dict[str, float]:
        return {
            "utilization": self.utilization,
            "mj_per_infer_est": self.mj_est,
            "mj_per_infer_lo": self.mj_lo,
            "mj_per_infer_hi": self.mj_hi,
        }


def _unavailable() -> EnergyEstimate:
    return EnergyEstimate(utilization=-1.0, mj_est=-1.0, mj_lo=-1.0, mj_hi=-1.0)


def estimate_energy(
    *,
    cpu_s: float,
    wall_s: float,
    cores: int,
    tdp_w: float,
    inferences: int,
) -> EnergyEstimate:
    """CPU-package energy per inference under the contract's linear model.

    ``u = cpu_seconds / (wall_seconds * cores)`` and
    ``P(u) = P_idle + (P_TDP - P_idle) * clamp(u, 0, 1)`` with
    ``P_idle = fraction * P_TDP``. The reported range is the idle-fraction set,
    so the reader always sees how much the unmeasurable idle power moves it.
    """
    if (
        not all(math.isfinite(x) for x in (cpu_s, wall_s, tdp_w))
        or cpu_s < 0
        or wall_s <= 0
        or cores <= 0
        or tdp_w <= 0
        or inferences <= 0
    ):
        return _unavailable()
    utilization = max(0.0, min(1.0, cpu_s / (wall_s * cores)))
    per_infer: list[float] = []
    for fraction in contracts.POWER_IDLE_FRACTIONS:
        idle = fraction * tdp_w
        power_w = idle + (tdp_w - idle) * utilization
        per_infer.append(power_w * wall_s * 1000.0 / inferences)
    central = contracts.POWER_IDLE_FRACTIONS[contracts.POWER_IDLE_FRACTIONS.index(_CENTRAL_IDLE)]
    idle = central * tdp_w
    power_w = idle + (tdp_w - idle) * utilization
    return EnergyEstimate(
        utilization=utilization,
        mj_est=power_w * wall_s * 1000.0 / inferences,
        mj_lo=min(per_infer),
        mj_hi=max(per_infer),
    )


# ── Meters ───────────────────────────────────────────────────────────────


def _floats(line: str) -> list[float]:
    return [float(match) for match in _NUMBER.findall(line)]


def parse_meter_text(text: str) -> tuple[float, str]:
    """A measured average power in watts, plus the assumption it rests on.

    Accepted shapes, in order:

    * JSON ``{"power_w": w}`` or ``{"energy_j": e, "seconds": s}``;
    * two numeric columns per row: instantaneous power in W, integrated by the
      trapezoid rule over the row timestamps/indices and divided by the window;
    * one numeric column: total joules when the header names energy/J, average
      watts otherwise.
    """
    stripped = text.strip()
    if not stripped:
        raise MeterError("meter returned no data")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        if isinstance(payload.get("power_w"), (int, float)):
            value = float(payload["power_w"])
            if math.isfinite(value) and value > 0:
                return value, "meter=json:power_w"
            raise MeterError(f"meter power_w is not positive: {value!r}")
        energy = payload.get("energy_j")
        seconds = payload.get("seconds")
        if isinstance(energy, (int, float)) and isinstance(seconds, (int, float)):
            if seconds <= 0 or not math.isfinite(float(energy)) or energy < 0:
                raise MeterError("meter energy_j/seconds is malformed")
            return float(energy) / float(seconds), "meter=json:energy_j/seconds"
        raise MeterError("meter JSON has neither power_w nor energy_j/seconds")

    rows = [
        line for line in stripped.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise MeterError("meter returned only comments")
    header = rows[0] if not _floats(rows[0]) else ""
    data_rows = rows[1:] if header else rows
    power_rows = [_floats(row) for row in data_rows]
    if not power_rows or any(len(row) < 1 for row in power_rows):
        raise MeterError("meter rows contain no numbers")
    if all(len(row) >= 2 for row in power_rows):
        watts = [row[-1] for row in power_rows]
        seconds = [row[0] for row in power_rows]
        if any(not math.isfinite(w) or w < 0 for w in watts):
            raise MeterError("meter contains a negative or non-finite power")
        duration = seconds[-1] - seconds[0]
        if duration <= 0:
            raise MeterError("meter timestamps do not advance")
        energy_j = sum(
            (watts[i] + watts[i + 1]) / 2.0 * (seconds[i + 1] - seconds[i])
            for i in range(len(watts) - 1)
        )
        average = energy_j / duration
        if average <= 0:
            raise MeterError("meter integrates to zero power")
        return average, f"meter=file:power_series({len(watts)} samples)"
    values = [row[0] for row in power_rows]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise MeterError("meter contains a negative or non-finite value")
    looks_energy = any(token in header.lower() for token in ("energy", "joule", "[j]", "_j"))
    if looks_energy:
        total = values[-1]
        return total, "meter=file:total_joules(assumes one window)"
    average = statistics.fmean(values)
    if average <= 0:
        raise MeterError("meter average power is zero")
    return average, f"meter=file:power_watts({len(values)} samples)"


def read_meter_file(path: Path) -> tuple[float, str]:
    """Parse a local meter file; raises :class:`MeterError` when it is unusable."""
    try:
        return parse_meter_text(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise MeterError(f"cannot read meter file {path}: {exc}") from None


def _read_number(path: str) -> float | None:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def measure_device_power(device: transport.DeviceTransport) -> tuple[float, str] | None:
    """Average package power from rapl or hwmon, when the device exposes one.

    Both routers are known to expose neither, so this normally returns ``None``
    and the caller falls back to the model. The code path exists because a
    different board *does* expose them and the campaign should then stop
    modelling what it can measure.
    """
    probe = device.run(
        "ls /sys/class/powercap/intel-rapl:0/energy_uj "
        "/sys/class/hwmon/hwmon*/power1_input 2>/dev/null",
        timeout=30.0,
    )
    candidates = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    rapl = [p for p in candidates if "powercap" in p]
    hwmon = [p for p in candidates if "hwmon" in p]
    if rapl:
        path = rapl[0]
        start = device.run(f"cat {path}", timeout=30.0)
        device.run("usleep 1000000 2>/dev/null || sleep 1", timeout=10.0)
        end = device.run(f"cat {path}", timeout=30.0)
        try:
            energy_j = (float(end.stdout.strip()) - float(start.stdout.strip())) / 1e6
        except ValueError:
            energy_j = -1.0
        if energy_j > 0:
            return energy_j / 1.0, f"rapl:{path}"
        return None
    if hwmon:
        path = hwmon[0]
        readings: list[float] = []
        for _ in range(5):
            result = device.run(f"cat {path}", timeout=30.0)
            try:
                microwatts = float(result.stdout.strip())
            except ValueError:
                readings = []
                break
            if microwatts > 0:
                readings.append(microwatts / 1e6)
        if readings:
            return statistics.fmean(readings), f"hwmon:{path}"
    return None


# ── Table ────────────────────────────────────────────────────────────────


def _energy_for_row(
    *, cpu_s: float, wall_s: float, cores: int, tdp_w: float, inferences: int
) -> EnergyEstimate:
    return estimate_energy(
        cpu_s=cpu_s, wall_s=wall_s, cores=cores, tdp_w=tdp_w, inferences=inferences
    )


def _device_spec(device_key: str) -> contracts.DeviceSpec | None:
    return contracts.DEVICES.get(device_key)


def run(
    device: transport.DeviceTransport,
    tables_dir: Path | str,
    meter_file: Path | str | None = None,
    meter_cmd: str | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Write ``power.csv`` from ``bench.csv``; returns the table path.

    One row per (device, model, precision, cpus): the reps are aggregated to a
    median wall time and a mean CPU time, because energy is a per-inference
    quantity and the repetition spread belongs in ``bench.csv``.
    """
    from inmotion.edge.config import as_float, read_table, write_table

    say = log or (lambda message: print(message, file=sys.stderr, flush=True))
    table_dir = Path(tables_dir)
    bench_path = table_dir / "bench.csv"
    if not bench_path.is_file():
        raise FileNotFoundError(f"power: {bench_path} is missing; run bench first")
    rows = read_table(bench_path, "bench")
    spec = _device_spec(device.key)

    warnings: list[str] = []
    measured: tuple[float, str] | None = None
    meter_assumption = ""
    if meter_file is not None:
        try:
            measured = read_meter_file(Path(meter_file))
            meter_assumption = measured[1]
        except MeterError as exc:
            warnings.append(f"meter file ignored: {exc}")
    if measured is None and meter_cmd:
        result = device.run(meter_cmd, timeout=120.0)
        if result.returncode == 0:
            try:
                measured = parse_meter_text(result.stdout)
                meter_assumption = measured[1]
            except MeterError as exc:
                warnings.append(f"meter command ignored: {exc}")
        else:
            warnings.append(f"meter command failed ({result.returncode}); using fallback")
    if measured is None:
        hardware = measure_device_power(device)
        if hardware is not None:
            measured = hardware
            meter_assumption = hardware[1]

    for warning in warnings:
        say(f"power: warning: {warning}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["model"], row["precision"], row["cpus"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for (model, precision, cpus), group in sorted(groups.items()):
        # `cpus` is empty for the serial matrix; the estimate needs the device's
        # own core count there, not zero.
        cores = _count_cores(cpus) or (spec.cores if spec is not None else 0)
        iterations = [as_float(row["iterations"]) for row in group]
        wall = []
        cpu = []
        utilizations = []
        for row in group:
            n = as_float(row["iterations"])
            cpu_ms = as_float(row["cpu_ms_bench"])
            throughput = as_float(row["throughput_s"])
            if n > 0 and throughput > 0:
                wall.append(n / throughput)
            elif as_float(row["p50_ms"]) > 0 and n > 0:
                wall.append(n * as_float(row["p50_ms"]) / 1000.0)
            if cpu_ms >= 0:
                cpu.append(cpu_ms / 1000.0)
            utilization = as_float(row["cpu_pct_bench"])
            if utilization >= 0:
                utilizations.append(utilization / 100.0)
        wall_s = statistics.median(wall) if wall else -1.0
        cpu_s = statistics.fmean(cpu) if cpu else -1.0
        inferences = statistics.median(iterations) if iterations else -1.0
        tdp = spec.tdp_w if spec is not None else -1.0

        estimate = _energy_for_row(
            cpu_s=cpu_s, wall_s=wall_s, cores=cores, tdp_w=tdp, inferences=int(inferences)
        )
        if measured is not None:
            power_w, source = measured
            mj_meas = power_w * wall_s * 1000.0 / inferences
            energy_source = source if source in ("rapl", "hwmon", "meter") else "meter"
            assumptions = (
                f"{meter_assumption}; measured package power applied per cell "
                f"({power_w:.3f} W average); wall from bench.csv"
            )
        elif estimate.mj_est >= 0:
            mj_meas = -1.0
            energy_source = "model"
            fractions = ",".join(f"{f:g}" for f in contracts.POWER_IDLE_FRACTIONS)
            assumptions = (
                f"tdp_w={tdp:g};idle_fractions={fractions};cores={cores};"
                "u=cpu_s/(wall_s*cores);range is the idle-fraction set"
            )
        else:
            mj_meas = -1.0
            energy_source = "none"
            assumptions = "no measured source and no TDP for this device"
        utilization = (
            estimate.utilization
            if estimate.utilization >= 0
            else (statistics.fmean(utilizations) if utilizations else -1.0)
        )
        output.append(
            {
                "device": device.key,
                "model": model,
                "precision": precision,
                "cores": cores,
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "utilization": utilization,
                "mj_per_infer_est": estimate.mj_est,
                "mj_per_infer_lo": estimate.mj_lo,
                "mj_per_infer_hi": estimate.mj_hi,
                "mj_per_infer_meas": mj_meas,
                "source": energy_source,
                "assumptions": assumptions,
            }
        )

    path = write_table(table_dir / "power.csv", "power", output)
    say(f"power: {path} ({len(output)} rows, source={output[0]['source'] if output else 'none'})")
    return path


def _count_cores(cpus: str) -> int:
    """Number of distinct cores in a ``--cpus`` set string."""
    total = 0
    for part in cpus.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                total += int(high) - int(low) + 1
            except ValueError:
                total += 1
        else:
            total += 1
    return total
