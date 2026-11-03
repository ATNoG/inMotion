"""Load, latency, CPU and memory matrices on a device.

One cell is one artifact measured once: a cold load, a warm load, then the
timed loop. Every cell records the device state before and after it (governor,
current frequency, temperature, load average, foreign CPU busy, available
memory) because a number measured on a router that was busy with something
else is not the same number, and the table has to say so.

Iterations are adaptive: calibrate on ten inferences, then aim at
:attr:`Profile.target_seconds` of timed work, bounded by the profile's
``[min_iterations, max_iterations]``. A fixed count is wrong in both
directions: too few on a fast artifact is noise, too many on a slow one is a
wasted afternoon.

The runtime is addressed by command line only. Schema-v2 commands (``load``,
``loadgen``, ``stress``) are probed once per device; an older binary is tolerated
and the missing values are recorded as ``-1`` with a status, never as zero.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from inmotion.edge import contracts, power, transport
from inmotion.edge.config import Profile, as_float, write_table

#: A bench command that has not returned by this many seconds past its budget is
#: treated as wedged; the transport kills the process group.
_BENCH_TIMEOUT_S = 180.0

_STATE_SCRIPT = "; ".join(
    (
        "CPUFREQ=/sys/devices/system/cpu/cpu0/cpufreq",
        'echo "governor=$(cat $CPUFREQ/scaling_governor 2>/dev/null || echo unknown)"',
        'echo "freq_khz=$(cat $CPUFREQ/scaling_cur_freq 2>/dev/null || echo -1)"',
        "echo \"loadavg=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo -1)\"",
        "echo \"mem_available_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)\"",
        'echo "temp_mc=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null '
        '| sort -n | tail -1)"',
        "prev=$(sed -n 's/^cpu //p' /proc/stat 2>/dev/null)",
        "usleep 200000 2>/dev/null || sleep 1",
        "cur=$(sed -n 's/^cpu //p' /proc/stat 2>/dev/null)",
        'foreign=$(printf \'%s\\n%s\\n\' "$prev" "$cur" | awk \''
        'NR==1 { split($0,x," "); next } '
        'NR==2 { split($0,y," "); t=0; id=0; '
        "for (i=1; i<=10; i++) { d=y[i]-x[i]; t+=d; if (i==4||i==5) id+=d } "
        'if (t>0) printf "%.2f", 100*(t-id)/t; else printf "-1" }\')',
        'echo "foreign_cpu_pct=${foreign:--1}"',
    )
)


@dataclass(frozen=True)
class DeviceState:
    """What the router was doing around a cell."""

    governor: str = "unknown"
    freq_mhz: float = -1.0
    temp_c: float = -1.0
    loadavg: float = -1.0
    foreign_cpu_pct: float = -1.0
    mem_available_kb: int = -1


# ── small helpers ────────────────────────────────────────────────────────


def parse_cpu_set(text: str) -> tuple[int, ...]:
    """Parse the runtime's ``--cpus`` grammar: ``0``, ``0-2``, ``0,2-3``."""
    cores: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty core selector in {text!r}")
        if "-" in part:
            low, _, high = part.partition("-")
            start, end = int(low), int(high)
            if start > end or start < 0:
                raise ValueError(f"invalid range {part!r}")
            cores.extend(range(start, end + 1))
        else:
            value = int(part)
            if value < 0:
                raise ValueError(f"invalid core {part!r}")
            cores.append(value)
    return tuple(dict.fromkeys(cores))


def format_cpu_set(cores: Iterable[int]) -> str:
    """Compress a core set back to the grammar (``0-2`` rather than ``0,1,2``)."""
    values = sorted({int(core) for core in cores})
    if not values:
        raise ValueError("an empty core set has no valid --cpus spelling")
    parts: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def _number(mapping: dict[str, object], *keys: str, default: float = -1.0) -> float:
    """First finite numeric field among ``keys``, else ``default``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                return number
    return default


def last_json(result: object) -> dict[str, object]:
    """The last JSON object a command printed, or ``{}`` when it printed none."""
    stdout = getattr(result, "stdout", "") or ""
    for line in reversed(str(stdout).splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _first_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


# ── device state and capabilities ────────────────────────────────────────


def sample_state(device: transport.DeviceTransport) -> DeviceState:
    """One round trip: governor, frequency, temperature, load, foreign CPU, memory."""
    result = device.run(_STATE_SCRIPT, timeout=60.0)
    if result.returncode != 0:
        return DeviceState()
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    freq_khz = as_float(fields.get("freq_khz"), -1.0)
    temp_mc = as_float(fields.get("temp_mc"), -1.0)
    return DeviceState(
        governor=fields.get("governor", "unknown") or "unknown",
        freq_mhz=freq_khz / 1000.0 if freq_khz > 0 else -1.0,
        temp_c=temp_mc / 1000.0 if temp_mc > 1_000 else temp_mc,
        loadavg=as_float(fields.get("loadavg"), -1.0),
        foreign_cpu_pct=as_float(fields.get("foreign_cpu_pct"), -1.0),
        mem_available_kb=int(as_float(fields.get("mem_available_kb"), -1.0)),
    )


def probe_capabilities(device: transport.DeviceTransport, binary: str) -> dict[str, bool]:
    """Ask the runtime which schema-v2 commands it has, once per device.

    The probe deliberately passes flags so an *old* binary answers "unknown
    command" instead of failing on a missing argument: the distinction is what
    keeps this module honest about which numbers are unavailable.
    """
    probes = {
        "load": f"{binary} load --artifact /nonexistent --json",
        "loadgen": f"{binary} loadgen --artifact /nonexistent --seconds 1 --json",
        "stress": (
            f"{binary} stress --artifact /nonexistent --mode spin --seconds 1 --workers 1 --json"
        ),
    }
    capabilities: dict[str, bool] = {"bench": False, "stream": False, "check": False, "cpus": False}
    for name in ("bench", "stream", "check"):
        result = device.run(f"{binary} {name}", timeout=60.0)
        text = f"{result.stdout}\n{result.stderr}".lower()
        capabilities[name] = "unknown command" not in text and bool(text.strip())
    for name, command in probes.items():
        result = device.run(command, timeout=60.0)
        text = f"{result.stdout}\n{result.stderr}".lower()
        capabilities[name] = "unknown command" not in text and bool(text.strip())
    # `--cpus` is silently ignored by a binary that predates it, which would turn
    # an unpinned run into a "core scaling" point. The usage text is the only
    # place the flag is declared, so that is what decides.
    usage = device.run(f"{binary} --help", timeout=60.0)
    capabilities["cpus"] = "--cpus" in f"{usage.stdout}\n{usage.stderr}"
    return capabilities


# ── artifacts on the device ──────────────────────────────────────────────


def resolve_artifacts(artifacts_dir: Path | str) -> dict[tuple[str, str], Path]:
    """Map (model, precision) to an artifact directory under ``artifacts_dir``.

    ``artifacts.csv`` is authoritative when present. Without it (the temporary
    artifact store, and unit fixtures) the contract's ``<model>-<precision>``
    layout is tried first, then the store's historical
    ``<model>_ft_seed42-<precision>`` names. Only those two spellings are
    accepted, so a producer variant such as ``-bnb-int8`` cannot be mistaken for
    the in-house ``int8`` artifact.
    """
    root = Path(artifacts_dir)
    mapping: dict[tuple[str, str], Path] = {}
    manifest = root / "artifacts.csv"
    if manifest.is_file():
        import csv

        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model = (row.get("model") or "").strip()
                precision = (row.get("precision") or "").strip()
                if not model or not precision:
                    continue
                if (row.get("status") or "ok").strip() not in ("", "ok"):
                    continue
                candidates = [
                    Path(row.get("path") or ""),
                    root / Path(row.get("path") or "").name,
                    root / f"{model}-{precision}",
                ]
                found = next((path for path in candidates if path.is_dir()), None)
                if found is not None:
                    mapping[(model, precision)] = found
    if not root.is_dir():
        return mapping
    for model in contracts.MODELS:
        for precision in (*contracts.PRECISIONS_DEPLOYABLE, *contracts.PRECISIONS_CAST):
            if (model, precision) in mapping:
                continue
            for name in (f"{model}-{precision}", f"{model}_ft_seed42-{precision}"):
                candidate = root / name
                if candidate.is_dir():
                    mapping[(model, precision)] = candidate
                    break
    return mapping


def ensure_remote_artifact(
    device: transport.DeviceTransport,
    local: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> str:
    """Upload an artifact once; return its remote directory.

    Verified by the contract's join key, the sha256 of ``model.safetensors``.
    When the router has no ``sha256sum`` the existing directory is trusted —
    re-uploading 45 artifacts every run is worse than the risk of a stale file
    that ``artifacts.csv`` would already have flagged.
    """
    say = log or (lambda message: print(message, file=sys.stderr, flush=True))
    remote = f"{device.root.rstrip('/')}/artifacts/{local.name}"
    marker = f"{remote}/{contracts.ARTIFACT_HASH_FILE}"
    expected = transport.collect_artifact_sha256(local)
    if device.exists(marker):
        remote_sha = device.sha256(marker)
        if not remote_sha or not expected or remote_sha == expected:
            return remote
        say(f"bench: {local.name} differs on {device.key}, re-uploading")
    device.mkdir(f"{device.root.rstrip('/')}/artifacts")
    device.upload(local, remote)
    uploaded = device.sha256(marker)
    if uploaded and expected and uploaded != expected:
        raise transport.TransportError(
            f"{device.key}: uploaded {local.name} hashes to {uploaded[:12]}, "
            f"expected {expected[:12]}"
        )
    return remote


def deploy_binary(
    device: transport.DeviceTransport,
    binary: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> str:
    """Upload the musl binary to ``<root>/bin/inmotion-infer`` when it changed."""
    say = log or (lambda message: print(message, file=sys.stderr, flush=True))
    local = Path(binary)
    if not local.is_file():
        raise FileNotFoundError(f"bench: runtime binary not found at {local}")
    remote = f"{device.root.rstrip('/')}/bin/inmotion-infer"
    expected = transport.binary_sha256(local)
    if device.exists(remote):
        actual = device.sha256(remote)
        if actual == expected:
            return remote
        say(f"bench: {device.key} binary differs, deploying {expected[:12]}")
    device.mkdir(f"{device.root.rstrip('/')}/bin")
    device.upload(local, remote)
    device.run(f"chmod +x {remote}", timeout=60.0, check=True)
    actual = device.sha256(remote)
    if actual and actual != expected:
        raise transport.TransportError(
            f"{device.key}: deployed binary hashes to {actual[:12]}, expected {expected[:12]}"
        )
    return remote


# ── runtime invocations ──────────────────────────────────────────────────


def _invoke(
    device: transport.DeviceTransport,
    binary: str,
    *,
    artifact: str,
    iterations: int | None = None,
    warmup: int | None = None,
    cpus: str | None = None,
    concurrency: int | None = None,
    seconds: float | None = None,
    timeout: float = _BENCH_TIMEOUT_S,
) -> object:
    command = f"{binary} bench --artifact {transport.shell_quote(artifact)}"
    if concurrency is not None:
        command += f" --concurrency {int(concurrency)} --seconds {float(seconds):g}"
        timeout = max(_BENCH_TIMEOUT_S, float(seconds) + 120.0)
    else:
        command += f" --iterations {int(iterations or 1)}"
        if warmup:
            command += f" --warmup {int(warmup)}"
    if cpus:
        command += f" --cpus {transport.shell_quote(cpus)}"
    return device.run(f"{command} --json", timeout=timeout)


def calibrate_iterations(
    device: transport.DeviceTransport,
    remote_binary: str,
    remote_artifact: str,
    profile: Profile,
) -> int:
    """Ten-inference calibration, then the iteration count that fills the budget."""
    result = _invoke(
        device,
        remote_binary,
        artifact=remote_artifact,
        iterations=profile.calib_iterations,
        warmup=max(1, profile.warmup // 2),
    )
    payload = last_json(result)
    median = _number(payload, "median_ms", "p50_ms")
    if median <= 0:
        return profile.min_iterations
    wanted = round(profile.target_seconds * 1000.0 / median)
    return int(min(profile.max_iterations, max(profile.min_iterations, wanted)))


def load_payload(
    device: transport.DeviceTransport, remote_binary: str, remote_artifact: str, *, cold: bool
) -> tuple[dict[str, object], object]:
    command = f"{remote_binary} load --artifact {transport.shell_quote(remote_artifact)}"
    if cold:
        command += " --cold"
    result = device.run(f"{command} --json", timeout=_BENCH_TIMEOUT_S)
    return last_json(result), result


# ── rows ─────────────────────────────────────────────────────────────────


def _energy(
    device_key: str, *, cpu_ms: float, wall_s: float, iterations: float
) -> power.EnergyEstimate:
    spec = contracts.DEVICES.get(device_key)
    cores = spec.cores if spec is not None else 0
    tdp = spec.tdp_w if spec is not None else -1.0
    return power.estimate_energy(
        cpu_s=cpu_ms / 1000.0 if cpu_ms >= 0 else -1.0,
        wall_s=wall_s,
        cores=cores,
        tdp_w=tdp,
        inferences=int(iterations) if iterations > 0 else 0,
    )


def _load_row(
    *,
    device_key: str,
    model: str,
    precision: str,
    rep: int,
    cold: bool,
    payload: dict[str, object],
    status: str,
    state: DeviceState,
    artifact_bytes: int,
) -> dict[str, object]:
    return {
        "device": device_key,
        "model": model,
        "precision": precision,
        "rep": rep,
        "cold": cold,
        "load_ms": _number(payload, "load_ms", "load_cold_ms", "total_ms"),
        "cpu_ms_load": _number(payload, "cpu_ms_load"),
        "rss_after_load_kb": int(_number(payload, "rss_after_load_kb")),
        "artifact_bytes": int(_number(payload, "artifact_bytes", default=float(artifact_bytes))),
        "governor": str(payload.get("governor") or state.governor),
        "freq_mhz": _number(payload, "freq_mhz", default=state.freq_mhz),
        "status": status,
    }


def _bench_row(
    *,
    device_key: str,
    model: str,
    precision: str,
    rep: int,
    cpus: str,
    concurrency: int,
    payload: dict[str, object],
    requested_iterations: int,
    before: DeviceState,
    after: DeviceState,
    artifact_bytes: int,
    binary_sha: str,
    artifact_sha: str,
    status: str,
) -> dict[str, object]:
    iterations = _number(payload, "iterations", default=float(requested_iterations))
    throughput = _number(payload, "throughput_per_s")
    wall_ms = _number(payload, "wall_ms")
    if wall_ms <= 0 and throughput > 0 and iterations > 0:
        wall_ms = iterations * 1000.0 / throughput
    cpu_ms = _number(payload, "cpu_ms_bench")
    estimate = _energy(device_key, cpu_ms=cpu_ms, wall_s=wall_ms / 1000.0, iterations=iterations)
    freq = _number(payload, "freq_mhz")
    if freq <= 0:
        freq = before.freq_mhz
    return {
        "device": device_key,
        "model": model,
        "precision": precision,
        "rep": rep,
        "cpus": cpus,
        "concurrency": concurrency,
        "iterations": int(iterations),
        "min_ms": _number(payload, "min_ms"),
        "p50_ms": _number(payload, "median_ms", "p50_ms"),
        "mean_ms": _number(payload, "mean_ms"),
        "p95_ms": _number(payload, "p95_ms"),
        "p99_ms": _number(payload, "p99_ms"),
        "max_ms": _number(payload, "max_ms"),
        "throughput_s": throughput,
        "cpu_ms_bench": cpu_ms,
        "cpu_pct_bench": _number(payload, "cpu_pct_bench"),
        "cpu_ms_per_infer": _number(payload, "cpu_ms_per_infer"),
        "peak_rss_kb": int(_number(payload, "peak_rss_kb")),
        "artifact_bytes": int(_number(payload, "artifact_bytes", default=float(artifact_bytes))),
        "energy_mj_est": estimate.mj_est,
        "energy_mj_lo": estimate.mj_lo,
        "energy_mj_hi": estimate.mj_hi,
        "energy_mj_meas": -1.0,
        "power_source": "model" if estimate.mj_est >= 0 else "none",
        "freq_mhz": freq,
        "temp_c_before": before.temp_c,
        "temp_c_after": after.temp_c,
        "foreign_cpu_pct": max(before.foreign_cpu_pct, after.foreign_cpu_pct),
        "loadavg_before": before.loadavg,
        "loadavg_after": after.loadavg,
        "binary_sha256": binary_sha,
        "artifact_sha256": artifact_sha,
        "status": status,
    }


def _cores_row(
    *,
    device_key: str,
    model: str,
    precision: str,
    rep: int,
    cpus: str,
    concurrency: int,
    payload: dict[str, object],
    state: DeviceState,
    status: str,
) -> dict[str, object]:
    return {
        "device": device_key,
        "model": model,
        "precision": precision,
        "rep": rep,
        "cpus": cpus,
        "cores": len(parse_cpu_set(cpus)) if cpus else -1,
        "concurrency": concurrency,
        "p50_ms": _number(payload, "median_ms", "p50_ms", "latency_ms"),
        "p95_ms": _number(payload, "p95_ms"),
        "throughput_s": _number(payload, "throughput_per_s"),
        "cpu_ms_per_infer": _number(payload, "cpu_ms_per_infer"),
        "foreign_cpu_pct": state.foreign_cpu_pct,
        "status": status,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


# ── the run ──────────────────────────────────────────────────────────────


def run(
    device: transport.DeviceTransport,
    artifacts_dir: Path | str,
    tables_dir: Path | str,
    raw_dir: Path | str,
    profile: Profile,
    *,
    binary: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Measure ``profile`` on ``device``; write load.csv, bench.csv, cores.csv."""
    say = log or (lambda message: print(message, file=sys.stderr, flush=True))
    table_dir = Path(tables_dir)
    raw = Path(raw_dir) / "bench"
    raw.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifacts(artifacts_dir)
    wanted = [(model, precision) for model in profile.models for precision in profile.precisions]
    missing = [pair for pair in wanted if pair not in artifacts]
    if missing:
        listing = ", ".join(f"{model}-{precision}" for model, precision in missing[:8])
        raise FileNotFoundError(
            f"bench: no artifact for {listing} under {artifacts_dir}; "
            "run `inmotion edge artifacts` or pass --artifacts-dir"
        )

    remote_binary = f"{device.root.rstrip('/')}/bin/inmotion-infer"
    if binary is not None:
        remote_binary = deploy_binary(device, Path(binary), log=say)
    binary_sha = transport.binary_sha256(Path(binary)) if binary is not None else ""
    capabilities = probe_capabilities(device, remote_binary)
    say(f"bench: {device.key} runtime capabilities {capabilities}")

    loads: list[dict[str, object]] = []
    benches: list[dict[str, object]] = []
    cores_rows: list[dict[str, object]] = []
    calibration: dict[tuple[str, str], int] = {}
    remote_paths: dict[tuple[str, str], str] = {}

    def remote_for(model: str, precision: str) -> str:
        pair = (model, precision)
        if pair not in remote_paths:
            local = artifacts[pair]
            remote_paths[pair] = ensure_remote_artifact(device, local, log=say)
        return remote_paths[pair]

    for model in profile.models:
        for precision in profile.precisions:
            local_artifact = artifacts[(model, precision)]
            artifact_sha = transport.collect_artifact_sha256(local_artifact)
            size = _artifact_bytes(local_artifact)
            remote_artifact = remote_for(model, precision)
            iterations = calibrate_iterations(device, remote_binary, remote_artifact, profile)
            calibration[(model, precision)] = iterations
            say(f"bench: {device.key} {model}-{precision} calibrates to {iterations} iterations")
            for rep in range(profile.reps):
                before = sample_state(device)
                cell: dict[str, object] = {
                    "device": device.key,
                    "model": model,
                    "precision": precision,
                    "rep": rep,
                    "iterations": iterations,
                    "state_before": before.__dict__,
                }
                for cold in (True, False):
                    if capabilities["load"]:
                        payload, result = load_payload(
                            device, remote_binary, remote_artifact, cold=cold
                        )
                        status = "ok" if payload else f"error:{_first_line(result.stderr)}"
                    else:
                        payload, status = {}, "unsupported:load"
                    loads.append(
                        _load_row(
                            device_key=device.key,
                            model=model,
                            precision=precision,
                            rep=rep,
                            cold=cold,
                            payload=payload,
                            status=status,
                            state=before,
                            artifact_bytes=size,
                        )
                    )
                    cell["cold" if cold else "warm"] = payload
                result = _invoke(
                    device,
                    remote_binary,
                    artifact=remote_artifact,
                    iterations=iterations,
                    warmup=profile.warmup,
                )
                payload = last_json(result)
                status = "ok" if payload else f"error:{_first_line(result.stderr)}"
                after = sample_state(device)
                benches.append(
                    _bench_row(
                        device_key=device.key,
                        model=model,
                        precision=precision,
                        rep=rep,
                        cpus="",
                        concurrency=1,
                        payload=payload,
                        requested_iterations=iterations,
                        before=before,
                        after=after,
                        artifact_bytes=size,
                        binary_sha=binary_sha,
                        artifact_sha=artifact_sha,
                        status=status,
                    )
                )
                cell["bench"] = payload
                cell["state_after"] = after.__dict__
                cell["status"] = status
                _write_json(raw / f"{device.key}-{model}-{precision}-r{rep}.json", cell)

    for model in profile.core_models:
        for precision in profile.core_precisions:
            if (model, precision) not in artifacts:
                say(f"bench: skipping core matrix for missing {model}-{precision}")
                continue
            remote_artifact = remote_for(model, precision)
            iterations = calibration.get((model, precision)) or calibrate_iterations(
                device, remote_binary, remote_artifact, profile
            )
            for cores in profile.core_sets:
                cpus = format_cpu_set(cores)
                if not capabilities["cpus"]:
                    cores_rows.append(
                        _cores_row(
                            device_key=device.key,
                            model=model,
                            precision=precision,
                            rep=0,
                            cpus=cpus,
                            concurrency=1,
                            payload={},
                            state=DeviceState(),
                            status="unsupported:cpus",
                        )
                    )
                    continue
                state = sample_state(device)
                result = _invoke(
                    device,
                    remote_binary,
                    artifact=remote_artifact,
                    iterations=iterations,
                    warmup=profile.warmup,
                    cpus=cpus,
                )
                payload = last_json(result)
                cores_rows.append(
                    _cores_row(
                        device_key=device.key,
                        model=model,
                        precision=precision,
                        rep=0,
                        cpus=cpus,
                        concurrency=1,
                        payload=payload,
                        state=state,
                        status="ok" if payload else f"error:{_first_line(result.stderr)}",
                    )
                )
            for concurrency in profile.concurrencies:
                if concurrency <= 1:
                    # The serial 4-core row from the core matrix already is the
                    # concurrency=1 baseline, and it carries a real p95.
                    continue
                cpus = format_cpu_set(range(_core_count(device)))
                state = sample_state(device)
                result = _invoke(
                    device,
                    remote_binary,
                    artifact=remote_artifact,
                    cpus=cpus,
                    concurrency=concurrency,
                    seconds=profile.concurrency_seconds,
                )
                payload = last_json(result)
                cores_rows.append(
                    _cores_row(
                        device_key=device.key,
                        model=model,
                        precision=precision,
                        rep=0,
                        cpus=cpus,
                        concurrency=concurrency,
                        payload=payload,
                        state=state,
                        status="ok" if payload else f"error:{_first_line(result.stderr)}",
                    )
                )

    write_table(table_dir / "load.csv", "load", loads)
    write_table(table_dir / "bench.csv", "bench", benches)
    write_table(table_dir / "cores.csv", "cores", cores_rows)
    say(
        f"bench: {device.key} wrote load={len(loads)} "
        f"bench={len(benches)} cores={len(cores_rows)} rows"
    )
    return table_dir


def _core_count(device: transport.DeviceTransport) -> int:
    spec = contracts.DEVICES.get(device.key)
    return spec.cores if spec is not None else (os.cpu_count() or 1)
