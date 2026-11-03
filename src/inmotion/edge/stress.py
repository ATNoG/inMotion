"""Controlled interference: what inference and synthetic load do to a router.

Four phases, each a fixed-length *window*:

``baseline``   nothing but the probes and the sampler
``inference``  ``loadgen --cpus 3`` (the runtime itself, on one core)
``stress``     ``stress --mode spin --workers 3`` on cores 0-2
``both``       both at once

The order of the phases is reversed on every other repetition (ABBA-style), so
a drift in the lab over the campaign cannot be attributed to one phase. The
repetitions are the bootstrap blocks in :mod:`inmotion.edge.analysis`.

Windows run *on the device*: one busybox-safe shell script starts the load,
samples ``/proc`` every 0.5 s, runs the probes, and cleans up on every exit
path. The host runs the control ping and a watchdog on the SSH channel, and
stops the campaign when a guard trips:

* ``MemAvailable`` below 400 MB, or a 1-minute load average above 10, inside a
  window (the device script writes an ``abort`` marker and stops sampling);
* the control SSH channel unresponsive for more than 10 s (the watchdog kills
  the window);
* more than 5 % control-ping loss over a window (the window is marked invalid
  and no further windows are started).

The raw window directory is kept under ``raw/stress/``: it holds the exact
samples, the two probe directories, and the runtime's own JSON, so every table
row can be traced back to the file it came from.
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from inmotion.edge import contracts, transport
from inmotion.edge.config import Profile, as_float, write_table

#: Guards from the approved plan.
MEM_FLOOR_KB = 400 * 1024
LOAD_CEILING = 10.0
SSH_UNRESPONSIVE_S = 10.0
CONTROL_LOSS_PCT = 5.0

#: Sampler at 2 Hz; five 1 Hz probe workers per path; 20 Hz control ping.
SAMPLER_HZ = 2.0
PROBE_WORKERS = 5
CONTROL_HZ = 20.0

#: Samples per window line: window|device|phase|rep|t|cpu(4)|percore|mem|load|
#: net(8)|temp|freq.
SAMPLE_FIELDS = 22

#: Where each device sends its two probes. ``lan`` is the device-to-device link;
#: ``fwd`` is the gateway route (on device 2 that path is forwarded by device 1).
PROBE_TARGETS: dict[str, dict[str, str]] = {
    "wave-router": {"lan": "10.33.0.113", "fwd": "192.168.93.1"},
    "wave-router-2": {"lan": "192.168.93.13", "fwd": "192.168.93.1"},
}

_PROBE_AWK = r"""
BEGIN { last = 0; base = 0; started = 0 }
/bytes from/ {
  seq = -1; rtt = -1
  for (k = 1; k <= NF; k++) {
    if ($k ~ /seq=/) { s = $k; sub(/.*seq=/, "", s); gsub(/[^0-9]/, "", s); seq = s + 0 }
    if ($k ~ /^time=/) { v = $k; sub(/.*time=/, "", v); gsub(/[^0-9.]/, "", v); rtt = v + 0 }
    if ($k == "time<1") { rtt = 0.5 }
  }
  if (seq < 0) next
  # busybox numbers the first packet 0, iputils 1; anchor on the first reply
  # so a healthy stream never reports a phantom lost packet.
  if (started == 0) { started = 1; base = seq; last = seq }
  while (last < seq) {
    printf "%s|%s|%s|%s|%s|%d|%d|%s|%d\n", win, dev, ph, rep, probe, worker, last, "-1", 1
    last++
  }
  printf "%s|%s|%s|%s|%s|%d|%d|%s|%d\n", win, dev, ph, rep, probe, worker, seq, rtt, 0
  last = seq + 1
}
END {
  while (last < base + count) {
    printf "%s|%s|%s|%s|%s|%d|%d|%s|%d\n", win, dev, ph, rep, probe, worker, last, "-1", 1
    last++
  }
}
"""


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ── window script ────────────────────────────────────────────────────────


def window_id(token: str, phase: str, repetition: int) -> str:
    """A filesystem-safe, unique id for one window."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", token)[:40].strip("-") or "run"
    return f"{safe}-{phase}-r{repetition}"


def _indent(text: str, width: int) -> str:
    pad = " " * width
    return "\n".join(pad + line if line else line for line in text.strip().splitlines())


def build_window_script(
    *,
    window: str,
    device: str,
    phase: str,
    repetition: int,
    seconds: float,
    binary: str,
    artifact: str,
    cpus: str,
    out_dir: str,
    probes: dict[str, str],
    stress_workers: int = 3,
    sampler_hz: float = SAMPLER_HZ,
    inference_interval_ms: int = 0,
) -> str:
    """The device-side window: load, sampler, probes, guards and cleanup.

    Busybox-safe on purpose: no arrays, no ``local``, fractional waits via
    ``usleep`` (busybox ``sleep`` rejects fractions), integer counters, and one
    ``awk`` per sample instead of a shell loop per field. This file is executed
    by ``sh`` on the device, so unlike a remote command it keeps its newlines.
    """
    interval_us = int(1_000_000 / sampler_hz)
    steps = int(round(seconds * sampler_hz))
    interval_flag = (
        f" --interval-ms {int(inference_interval_ms)}" if inference_interval_ms > 0 else ""
    )
    probe_calls = "\n".join(
        f"run_probe {shlex.quote(name)} {shlex.quote(target)}"
        for name, target in probes.items()
        if target
    )
    return f"""#!/bin/sh
# Generated by inmotion.edge.stress -- one measurement window.
# Parameters are baked in so the window cannot be re-parameterised by accident.
set -u
WINDOW={shlex.quote(window)}
DEVICE={shlex.quote(device)}
PHASE={shlex.quote(phase)}
REP={shlex.quote(str(repetition))}
SECS={int(round(seconds))}
MODE={shlex.quote(phase)}
BIN={shlex.quote(binary)}
ART={shlex.quote(artifact)}
CPUS={shlex.quote(cpus)}
OUT={shlex.quote(out_dir)}
mkdir -p "$OUT" || exit 1
: > "$OUT/pids"
: > "$OUT/abort"

cleanup() {{
  if [ -f "$OUT/pids" ]; then
    for pid in $(cat "$OUT/pids" 2>/dev/null); do
      kill "$pid" 2>/dev/null
    done
    rm -f "$OUT/pids"
  fi
}}
trap cleanup EXIT INT TERM

if [ "$MODE" = "stress" ] || [ "$MODE" = "both" ]; then
  if [ -x "$BIN" ]; then
    "$BIN" stress --mode spin --seconds $((SECS + 3)) --workers {stress_workers} --json \\
      > "$OUT/stress.json" 2> "$OUT/stress.err" &
    echo $! >> "$OUT/pids"
  fi
fi

if [ "$MODE" = "inference" ] || [ "$MODE" = "both" ]; then
  if [ -x "$BIN" ] && [ -d "$ART" ]; then
    "$BIN" loadgen --artifact "$ART" --seconds $((SECS + 3)) --cpus "$CPUS"{interval_flag} \\
      --json > "$OUT/loadgen.json" 2> "$OUT/loadgen.err" &
    echo $! >> "$OUT/pids"
  fi
fi

run_probe() {{
  probe=$1
  target=$2
  worker=0
  while [ "$worker" -lt {PROBE_WORKERS} ]; do
    ping -c "$SECS" -i 1 "$target" 2>/dev/null | awk \\
      -v win="$WINDOW" -v dev="$DEVICE" -v ph="$PHASE" -v rep="$REP" \\
      -v probe="$probe" -v worker="$worker" -v count="$SECS" '
{_indent(_PROBE_AWK, 0)}
      ' >> "$OUT/probes.$probe.$worker.csv" &
    worker=$((worker + 1))
  done
}}

{probe_calls}

prev=$(sed -n 's/^cpu //p' /proc/stat 2>/dev/null)
sed -n 's/^cpu\\([0-9][0-9]*\\) /\\1 /p' /proc/stat 2>/dev/null > "$OUT/percore.prev"
i=0
while [ "$i" -lt {steps} ]; do
  cur=$(sed -n 's/^cpu //p' /proc/stat 2>/dev/null)
  sed -n 's/^cpu\\([0-9][0-9]*\\) /\\1 /p' /proc/stat 2>/dev/null > "$OUT/percore.cur"
  cpu=$(printf '%s\\n%s\\n' "$prev" "$cur" | awk '
    NR == 1 {{ split($0, x, " "); next }}
    NR == 2 {{
      split($0, y, " "); t = 0; idle = 0; iow = 0; hi = 0; si = 0
      for (k = 1; k <= 10; k++) {{
        d = y[k] - x[k]; t += d
        if (k == 4 || k == 5) idle += d
        if (k == 5) iow += d
        if (k == 6) hi += d
        if (k == 7) si += d
      }}
      if (t <= 0) {{ printf "-1|-1|-1|-1"; next }}
      b1 = 100 * (t - idle) / t; b2 = 100 * iow / t
      b3 = 100 * si / t; b4 = 100 * hi / t
      printf "%.2f|%.2f|%.2f|%.2f", b1, b2, b3, b4
    }}')
  percore=$(awk 'NR == FNR {{ p[$1] = $0; next }}
    {{ split(p[$1], a, " "); t = 0; idle = 0
       for (k = 2; k <= 11; k++) t += ($k - a[k])
       idle = ($5 - a[5]) + ($6 - a[6])
       if (t > 0) printf "%.1f;", 100 * (t - idle) / t
       else printf "-1;" }}' "$OUT/percore.prev" "$OUT/percore.cur")
  percore=${{percore%";"}}
  mem=$(awk '/^MemAvailable:/{{print $2}}' /proc/meminfo 2>/dev/null)
  load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)
  net=$(awk -F'[ :]+' 'NR > 2 {{
      sub(/^[ \t]+/, "")
      if ($1 == "lo") next
      rx += $2; rxp += $3; rxe += $4; rxd += $5
      tx += $10; txp += $11; txe += $12; txd += $13
    }}
    END {{ printf "%.0f|%.0f|%.0f|%.0f|", rx, tx, rxd, txd
           printf "%.0f|%.0f|%.0f|%.0f", rxe, txe, rxp, txp }}' \\
    /proc/net/dev 2>/dev/null)
  temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  frequency=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
  t=$(awk -v i="$i" 'BEGIN {{ printf "%.1f", i / {sampler_hz:g} }}')
  echo "$WINDOW|$DEVICE|$PHASE|$REP|$t|$cpu|$percore|$mem|$load|$net|$temp|$frequency" \\
    >> "$OUT/samples.csv"
  guard=0
  if [ -n "$mem" ] && [ "$mem" -lt {MEM_FLOOR_KB} ]; then
    guard=1
    echo "mem_available_kb=$mem" >> "$OUT/abort"
  fi
  if awk -v l="$load" 'BEGIN {{ exit !(l > {LOAD_CEILING:g}) }}'; then
    guard=1
    echo "loadavg_1m=$load" >> "$OUT/abort"
  fi
  if [ "$guard" -eq 1 ]; then
    break
  fi
  usleep {interval_us} 2>/dev/null || sleep 1
  prev=$cur
  cp "$OUT/percore.cur" "$OUT/percore.prev" 2>/dev/null
  i=$((i + 1))
done
wait
echo done > "$OUT/done"
exit 0
"""


# ── parsing the device output ────────────────────────────────────────────


def parse_sample_line(line: str) -> dict[str, object] | None:
    """One ``samples.csv`` row as a canonical ``stress_raw`` row."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != SAMPLE_FIELDS or not parts[0]:
        return None
    window, device, phase, rep, t = parts[0:5]
    busy, iowait, softirq, hardirq = parts[5:9]
    per_core, mem, load = parts[9], parts[10], parts[11]
    rx, tx, rxd, txd, rxe, txe = parts[12:18]
    temp, freq = parts[20], parts[21]
    temp_value = as_float(temp)
    freq_value = as_float(freq)
    return {
        "window": window,
        "device": device,
        "phase": phase,
        "repetition": int(as_float(rep, 0)),
        "t_rel_s": as_float(t),
        "cpu_busy_pct": as_float(busy),
        "cpu_iowait_pct": as_float(iowait),
        "cpu_softirq_pct": as_float(softirq),
        # The contract's `sirq` column carries the hard-IRQ share (the soft-IRQ
        # share is its own column above); both are pinned out here, not guessed.
        "cpu_sirq_pct": as_float(hardirq),
        "per_core_busy_pct": per_core,
        "mem_available_kb": int(as_float(mem)),
        "loadavg_1m": as_float(load),
        "rx_bytes": int(as_float(rx)),
        "tx_bytes": int(as_float(tx)),
        "rx_drops": int(as_float(rxd)),
        "tx_drops": int(as_float(txd)),
        "rx_errors": int(as_float(rxe)),
        "tx_errors": int(as_float(txe)),
        "temp_c": temp_value / 1000.0 if temp_value > 1000 else temp_value,
        "freq_mhz": freq_value / 1000.0 if freq_value > 0 else -1.0,
    }


def packets_from_line(line: str) -> tuple[int, int] | None:
    """(rx_packets, tx_packets) from a sample line, for the drop-rate denominator."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != SAMPLE_FIELDS:
        return None
    return int(as_float(parts[18], -1)), int(as_float(parts[19], -1))


def parse_probe_line(line: str) -> dict[str, object] | None:
    """One ``probes.*.csv`` row as a canonical ``stress_probes`` row."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != 9 or not parts[0]:
        return None
    window, device, phase, rep, probe, _worker, seq, rtt, lost = parts
    return {
        "window": window,
        "device": device,
        "phase": phase,
        "repetition": int(as_float(rep, 0)),
        "probe": probe,
        "seq": int(as_float(seq)),
        "rtt_ms": as_float(rtt),
        "lost": int(as_float(lost)),
    }


_HOST_PING = re.compile(r"icmp_seq=(\d+).*time=([\d.]+)\s*ms")


def parse_host_ping(
    text: str,
    *,
    window: str,
    device: str,
    phase: str,
    repetition: int,
    count: int | None = None,
) -> list[dict[str, object]]:
    """Parse iputils ``ping`` output into per-packet rows with gap-filled losses.

    The control probe is the host's own 20 Hz ping, so it is parsed here rather
    than in the device script; the sequence numbers give both the RTT and the
    loss count (a sent packet has a sequence number whether or not it replies).
    """
    rows: list[dict[str, object]] = []
    last = 0
    base = 0
    started = False
    for line in text.splitlines():
        match = _HOST_PING.search(line)
        if match is None:
            continue
        seq = int(match.group(1))
        # iputils numbers the first packet 1; anchor on the first reply so a
        # healthy stream never reports a phantom lost packet.
        if not started:
            started, base, last = True, seq, seq
        while last < seq:
            rows.append(
                {
                    "window": window,
                    "device": device,
                    "phase": phase,
                    "repetition": repetition,
                    "probe": "control",
                    "seq": last,
                    "rtt_ms": -1.0,
                    "lost": 1,
                }
            )
            last += 1
        rows.append(
            {
                "window": window,
                "device": device,
                "phase": phase,
                "repetition": repetition,
                "probe": "control",
                "seq": seq,
                "rtt_ms": float(match.group(2)),
                "lost": 0,
            }
        )
        last = seq + 1
    if count is not None and started:
        while last < base + count:
            rows.append(
                {
                    "window": window,
                    "device": device,
                    "phase": phase,
                    "repetition": repetition,
                    "probe": "control",
                    "seq": last,
                    "rtt_ms": -1.0,
                    "lost": 1,
                }
            )
            last += 1
    return rows


def probe_loss_pct(rows: Iterable[dict[str, object]], probe: str | None = None) -> float:
    """Loss from the packet counters: lost packets / packets sent, in percent."""
    selected = [row for row in rows if probe is None or row.get("probe") == probe]
    if not selected:
        return -1.0
    lost = sum(1 for row in selected if int(row.get("lost") or 0) == 1)
    return 100.0 * lost / len(selected)


def phase_order(
    repetition: int, phases: Sequence[str] = contracts.STRESS_PHASES
) -> tuple[str, ...]:
    """ABBA-style balance: the phase order reverses on every other repetition."""
    ordered = tuple(phases)
    return ordered if repetition % 2 == 0 else tuple(reversed(ordered))


# ── host-side watchdog and control ping ──────────────────────────────────


@dataclass
class _Watchdog:
    """SSH liveness watchdog: one cheap command every few seconds."""

    device: transport.DeviceTransport
    script_path: str
    interval_s: float = 5.0
    unresponsive: bool = False
    last_error: str = ""
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> _Watchdog:
        self._thread = threading.Thread(target=self._loop, name="edge-watchdog", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                result = self.device.run("echo watchdog", timeout=SSH_UNRESPONSIVE_S)
                if result.returncode != 0:
                    raise transport.TransportError(result.stderr.strip()[:120] or "no reply")
            except transport.TransportError as exc:
                self.unresponsive = True
                self.last_error = str(exc)
                self._kill_remote()
                return

    def _kill_remote(self) -> None:
        # The window is a foreground SSH command, so killing the remote shell is
        # the only way to stop it. A wedged channel may not accept this either;
        # that is swallowed because the window's own timeout still ends it.
        with contextlib.suppress(transport.TransportError):
            self.device.run(f"pkill -9 -f {shlex.quote(self.script_path)}", timeout=15.0)

    def stop(self) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=SSH_UNRESPONSIVE_S + 5.0)
        return self.unresponsive


def _start_control_ping(host: str, seconds: float) -> subprocess.Popen[str] | None:
    """20 Hz host->device-1 ping; the lab host has iputils, the devices do not."""
    count = max(1, int(seconds * CONTROL_HZ))
    try:
        return subprocess.Popen(
            ["ping", "-c", str(count), "-i", "0.05", "-W", "1", host],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except (OSError, ValueError):
        return None


# ── the run ──────────────────────────────────────────────────────────────


def run(
    device: transport.DeviceTransport,
    artifacts_dir: Path | str,
    tables_dir: Path | str,
    raw_dir: Path | str,
    profile: Profile,
    *,
    binary: Path | str | None = None,
    token: str = "",
    log: Callable[[str], None] | None = None,
) -> Path:
    """Run the interference protocol on ``device``; write the two stress tables."""
    say = log or _say
    table_dir = Path(tables_dir)
    if profile.stress_reps <= 0 or profile.stress_seconds <= 0:
        say(f"stress: profile {profile.name} skips the interference study")
        write_table(table_dir / "stress_raw.csv", "stress_raw", [])
        write_table(table_dir / "stress_probes.csv", "stress_probes", [])
        return table_dir

    from inmotion.edge.bench import deploy_binary, ensure_remote_artifact, resolve_artifacts

    raw = Path(raw_dir) / "stress"
    raw.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifacts(artifacts_dir)
    pair = (profile.stress_model, profile.stress_precision)
    if pair not in artifacts:
        raise FileNotFoundError(
            f"stress: no artifact for {pair[0]}-{pair[1]} under {artifacts_dir}"
        )
    remote_binary = f"{device.root.rstrip('/')}/bin/inmotion-infer"
    if binary is not None:
        remote_binary = deploy_binary(device, Path(binary), log=say)
    remote_artifact = ensure_remote_artifact(device, artifacts[pair], log=say)
    remote_scripts = f"{device.root.rstrip('/')}/stress/scripts"
    device.mkdir(remote_scripts)
    probes = PROBE_TARGETS.get(device.key, {})

    samples: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    interfaces: list[dict[str, object]] = []
    inference: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    aborted = False
    abort_reason = ""

    for repetition in range(profile.stress_reps):
        for phase in phase_order(repetition):
            if aborted:
                break
            window = window_id(token or "stress", phase, repetition)
            script = build_window_script(
                window=window,
                device=device.key,
                phase=phase,
                repetition=repetition,
                seconds=profile.stress_seconds,
                binary=remote_binary,
                artifact=remote_artifact,
                cpus="3",
                out_dir=f"{device.root.rstrip('/')}/stress/windows/{window}",
                probes=probes,
                stress_workers=profile.stress_workers,
                inference_interval_ms=profile.inference_interval_ms,
            )
            script_remote = f"{remote_scripts}/{window}.sh"
            device.write_text(script_remote, script)
            # The window lasts SECS + the loadgen/stress tail, so the control
            # ping covers the tail too; its expected packet count is derived from
            # the same number, so the two can never disagree.
            control_seconds = profile.stress_seconds + 3.0
            control = (
                _start_control_ping(contracts.DEVICES["wave-router"].host, control_seconds)
                if device.key != "local"
                else None
            )
            watchdog = _Watchdog(device, script_remote).start()
            started = time.monotonic()
            status = "ok"
            detail = ""
            control_rows: list[dict[str, object]] = []
            try:
                result = device.run(
                    f"sh {shlex.quote(script_remote)}", timeout=profile.stress_seconds + 120.0
                )
                if result.returncode != 0:
                    status = "error"
                    detail = (result.stderr.strip() or result.stdout.strip())[-300:]
            except transport.TransportError as exc:
                status = "error"
                detail = str(exc)
            finally:
                elapsed = time.monotonic() - started
                unresponsive = watchdog.stop()

            if control is not None:
                try:
                    control_text = control.communicate(timeout=60.0)[0] or ""
                except subprocess.TimeoutExpired:
                    control.kill()
                    control_text = ""
                control_rows = parse_host_ping(
                    control_text,
                    window=window,
                    device=device.key,
                    phase=phase,
                    repetition=repetition,
                    count=max(1, int(profile.stress_seconds * CONTROL_HZ)),
                )
                probe_rows.extend(control_rows)
                loss = probe_loss_pct(control_rows)
                if loss > CONTROL_LOSS_PCT:
                    aborted = True
                    abort_reason = f"control loss {loss:.2f}% > {CONTROL_LOSS_PCT:g}%"
                    status = "aborted:control-loss"
                expected = max(1, int(control_seconds * CONTROL_HZ))
                if control_rows and len(control_rows) != expected:
                    say(
                        f"stress: {window}: control ping returned {len(control_rows)} rows, "
                        f"expected {expected}"
                    )

            remote_out = f"{device.root.rstrip('/')}/stress/windows/{window}"
            local_window = _download_window(device, remote_out, raw, window)
            abort_marker = ""
            if local_window is not None:
                marker = Path(local_window) / "abort"
                if marker.is_file():
                    abort_marker = marker.read_text(encoding="utf-8", errors="replace").strip()
            if abort_marker:
                status = "aborted:device-guard"
                aborted = True
                abort_reason = f"device guard tripped: {abort_marker}"
            if unresponsive:
                status = "aborted:ssh-unresponsive"
                aborted = True
                abort_reason = watchdog.last_error or "SSH channel unresponsive"

            if local_window is None:
                windows.append(
                    {
                        "window": window,
                        "phase": phase,
                        "repetition": repetition,
                        "status": status,
                        "reason": detail or "raw window not downloaded",
                        "elapsed_s": elapsed,
                        "samples": 0,
                        "probe_rows": len(control_rows),
                    }
                )
                if aborted:
                    break
                continue

            window_samples, window_probes, window_interfaces = _absorb_window(
                Path(local_window), window, device.key, phase, repetition
            )
            samples.extend(window_samples)
            probe_rows.extend(window_probes)
            interfaces.extend(window_interfaces)
            loadgen = _read_window_json(Path(local_window) / "loadgen.json")
            stress = _read_window_json(Path(local_window) / "stress.json")
            if loadgen:
                iterations = as_float(loadgen.get("iterations"))
                cpu_ms = as_float(loadgen.get("cpu_ms"))
                inference.append(
                    {
                        "window": window,
                        "device": device.key,
                        "phase": phase,
                        "repetition": repetition,
                        "infer_p50_ms": as_float(loadgen.get("p50_ms")),
                        "infer_p95_ms": as_float(loadgen.get("p95_ms")),
                        "infer_cpu_ms_per_infer": (
                            cpu_ms / iterations if iterations > 0 and cpu_ms >= 0 else -1.0
                        ),
                        "errors": int(as_float(loadgen.get("errors"))),
                        "status": "ok" if not loadgen.get("errors") else "errors",
                    }
                )
            elif phase in ("inference", "both"):
                inference.append(
                    {
                        "window": window,
                        "device": device.key,
                        "phase": phase,
                        "repetition": repetition,
                        "infer_p50_ms": -1.0,
                        "infer_p95_ms": -1.0,
                        "infer_cpu_ms_per_infer": -1.0,
                        "errors": -1,
                        "status": "unsupported:loadgen",
                    }
                )
            windows.append(
                {
                    "window": window,
                    "phase": phase,
                    "repetition": repetition,
                    "status": status,
                    "reason": detail or abort_marker,
                    "elapsed_s": elapsed,
                    "samples": len(window_samples),
                    "probe_rows": len(window_probes) + len(control_rows),
                    "inference_interval_ms": profile.inference_interval_ms,
                    "inference_seconds": profile.stress_seconds,
                    # The runtime does not echo the interval, so the achieved
                    # iterations over the window are the evidence that the duty
                    # cycle applied.
                    "loadgen_interval_ms": loadgen.get("interval_ms"),
                    "loadgen_iterations": loadgen.get("iterations"),
                    "loadgen_seconds": loadgen.get("seconds"),
                    "loadgen": loadgen,
                    "stress": stress,
                }
            )
            say(
                f"stress: {device.key} {phase} r{repetition} {status} "
                f"samples={len(window_samples)} probes={len(window_probes) + len(control_rows)}"
            )
        if aborted:
            break

    write_table(table_dir / "stress_raw.csv", "stress_raw", samples)
    write_table(table_dir / "stress_probes.csv", "stress_probes", probe_rows)
    _write_csv(
        raw / "inference.csv",
        (
            "window",
            "device",
            "phase",
            "repetition",
            "infer_p50_ms",
            "infer_p95_ms",
            "infer_cpu_ms_per_infer",
            "errors",
            "status",
        ),
        inference,
    )
    _write_csv(
        raw / "interfaces.csv",
        (
            "window",
            "device",
            "phase",
            "repetition",
            "rx_packets",
            "tx_packets",
            "rx_drops",
            "tx_drops",
            "rx_errors",
            "tx_errors",
            "duration_s",
        ),
        interfaces,
    )
    (raw / "windows.json").write_text(
        json.dumps(
            {"aborted": aborted, "reason": abort_reason, "windows": windows},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    say(
        f"stress: {device.key} wrote samples={len(samples)} probes={len(probe_rows)} "
        f"windows={len(windows)} aborted={aborted}"
    )
    return table_dir


def _read_window_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _download_window(
    device: transport.DeviceTransport, remote_out: str, local_root: Path, window: str
) -> Path | None:
    try:
        return device.download(remote_out, local_root / window)
    except transport.TransportError as exc:
        _say(f"stress: could not download {window}: {exc}")
        return None


def _absorb_window(
    local_window: Path, window: str, device_key: str, phase: str, repetition: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read one downloaded window directory into table rows."""
    samples: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    interfaces: list[dict[str, object]] = []
    samples_path = local_window / "samples.csv"
    if samples_path.is_file():
        first: dict[str, object] | None = None
        last: dict[str, object] | None = None
        first_packets: tuple[int, int] | None = None
        last_packets: tuple[int, int] | None = None
        for line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_sample_line(line)
            if row is not None:
                samples.append(row)
                first = first or row
                last = row
            packets = packets_from_line(line)
            if packets is not None:
                first_packets = first_packets or packets
                last_packets = packets
        if first is not None and last is not None and first_packets and last_packets:
            duration = as_float(last.get("t_rel_s")) - as_float(first.get("t_rel_s"))
            interfaces.append(
                {
                    "window": window,
                    "device": device_key,
                    "phase": phase,
                    "repetition": repetition,
                    "rx_packets": last_packets[0] - first_packets[0],
                    "tx_packets": last_packets[1] - first_packets[1],
                    "rx_drops": int(as_float(last.get("rx_drops")))
                    - int(as_float(first.get("rx_drops"))),
                    "tx_drops": int(as_float(last.get("tx_drops")))
                    - int(as_float(first.get("tx_drops"))),
                    "rx_errors": int(as_float(last.get("rx_errors")))
                    - int(as_float(first.get("rx_errors"))),
                    "tx_errors": int(as_float(last.get("tx_errors")))
                    - int(as_float(first.get("tx_errors"))),
                    "duration_s": duration if duration > 0 else -1.0,
                }
            )
    for probe_file in sorted(local_window.glob("probes.*.csv")):
        for line in probe_file.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_probe_line(line)
            if row is not None:
                probes.append(row)
    return samples, probes, interfaces


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, -1) for key in header})
    tmp.replace(path)
