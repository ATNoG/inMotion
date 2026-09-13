"""Drive the ESP32 firmware over serial and record measured latency and RAM.

The device protocol is one line in, one RESULT line out. Each input line is a
real test window from the golden vectors (dataset.csv's test split), so the
device is being asked the same question the parity harness asks on the host.

The CPU clock is *measured*, not assumed: the boot log prints `cpu_start: cpu
freq: <Hz>` from the image the device actually booted, and that value is what
the artifact records. A run that cannot read it fails instead of writing a
measurement whose clock is unknown — the figures label bars from this field, so
a guessed clock is not a labelling nicety, it is a wrong number.

Usage:
    sg dialout -c 'uv run --extra esp python firmware/tools/run_on_device.py \
        --port /dev/ttyUSB0 --cases /tmp/lejepa_int8/cases.tsv \
        --out results/quant/esp32_lejepa_240mhz.json --repeats 3'

To measure a different clock, boot the device with an image built for it: the
clock is a property of the flashed image, not of this script, which is why this
script never takes it as a parameter.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import serial

#: ESP-IDF's own cpu_start log: `cpu_start: cpu freq: 240000000 Hz`. The value
#: comes from the running image, so it is the clock the device actually booted
#: with, not the clock anyone asked for.
CPU_FREQ_RE = re.compile(r"cpu freq:\s*(\d+)\s*Hz")

#: ESP-IDF colourises its log lines; those bytes have no business in a recorded
#: field. Strip them so the artifact and the tracked evidence read the same.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """The log line without its terminal colour escapes."""
    return ANSI_RE.sub("", text)


def parse_clock_hz(banner: list[str]) -> int | None:
    """The CPU frequency the device reported at boot, in Hz."""
    for line in banner:
        match = CPU_FREQ_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def selftest() -> int:
    """Check the parser against the real boot lines, with no device attached.

    Run as ``python firmware/tools/run_on_device.py --selftest``. It is the
    smallest thing that fails if the clock stops being read out of the log.
    """
    cases = [
        ("\x1b[0;32mI (426) cpu_start: cpu freq: 240000000 Hz\x1b[0m\n", 240_000_000),
        ("I (426) cpu_start: cpu freq: 160000000 Hz\n", 160_000_000),
        ("I (31) boot: ESP-IDF v5.3.2 2nd stage bootloader\n", None),
        ("", None),
    ]
    for line, expected in cases:
        got = parse_clock_hz([line])
        if got != expected:
            print(f"FAIL parse_clock_hz({line!r}) = {got}, expected {expected}")
            return 1
    clock_line = strip_ansi(
        next(line for line in cases[0][0].splitlines() if CPU_FREQ_RE.search(line))
    )
    if not clock_line.strip():
        print("FAIL clock_source line is empty")
        return 1
    if ANSI_RE.search(clock_line):
        print(f"FAIL clock_source still carries escapes: {clock_line!r}")
        return 1
    print(f"selftest ok: 4 boot-log cases, clock_source={clock_line.strip()!r}")
    return 0


def read_result(port: serial.Serial, timeout_s: float) -> dict[str, object]:
    """Read until a RESULT or ERR line, returning the parsed fields."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = port.readline().decode("utf-8", "replace").strip()
        if not raw:
            continue
        if raw.startswith("ERR"):
            raise RuntimeError(f"device error: {raw}")
        if not raw.startswith("RESULT"):
            continue
        fields: dict[str, object] = {}
        for token in raw.split()[1:]:
            key, _, value = token.partition("=")
            try:
                fields[key] = float(value) if "." in value else int(value)
            except ValueError:
                fields[key] = value
        return fields
    raise TimeoutError(f"no RESULT within {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="passes over the case set; more passes make the timing median meaningful",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the clock parser against real boot lines and exit",
    )
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    if args.cases is None or args.out is None:
        ap.error("--cases and --out are required unless --selftest is given")

    cases: list[list[float]] = []
    for line in args.cases.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cases.append([float(v) for v in line.split("\t")[0].split(",")])

    with serial.Serial(args.port, args.baud, timeout=1) as port:
        # The DTR/RTS lines are wired to EN and GPIO0 on the usual dev boards.
        # Leaving DTR asserted holds the chip in reset (or in download mode) and
        # the port yields line noise rather than boot output, so release both and
        # then pulse EN to get a clean boot.
        port.setDTR(False)
        port.setRTS(False)
        time.sleep(0.3)
        port.reset_input_buffer()
        port.setRTS(True)
        time.sleep(0.15)
        port.setRTS(False)

        # Wait for READY, which the firmware prints after the model is loaded.
        boot_deadline = time.time() + 20
        banner: list[str] = []
        while time.time() < boot_deadline:
            line = port.readline().decode("utf-8", "replace").strip()
            if line:
                banner.append(line)
            if line == "READY":
                break
        else:
            raise TimeoutError(
                "firmware never printed READY; last lines:\n" + "\n".join(banner[-20:])
            )

        # The boot log carries the clock the image booted with. Refuse to write
        # an artifact whose clock is unknown: a figure label guessed from a
        # filename is exactly the defect this runner has to make impossible.
        clock_hz = parse_clock_hz(banner)
        if clock_hz is None:
            raise TimeoutError(
                "boot log never reported `cpu_start: cpu freq: <Hz>`; refusing to "
                "write a measurement whose clock is unknown. Last lines:\n"
                + "\n".join(banner[-20:])
            )
        clock_line = strip_ansi(next(line for line in banner if "cpu freq:" in line))

        runs: list[dict[str, object]] = []
        for _ in range(args.repeats):
            for window in cases:
                port.write((" ".join(repr(v) for v in window) + "\n").encode())
                port.flush()
                runs.append(read_result(port, 10.0))

    latencies = [float(r["latency_ms"]) for r in runs]
    heap_free = [int(r["heap_free"]) for r in runs]
    heap_delta = [int(r["heap_delta"]) for r in runs]
    flash_bytes = [int(r["flash_bytes"]) for r in runs]

    summary = {
        "device": "ESP32-D0WD-V3",
        "model": "lejepa",
        "precision": "int8",
        # The clock this run booted with, reported by the device itself.
        "clock_mhz": clock_hz // 1_000_000,
        "clock_hz": clock_hz,
        "clock_source": clock_line,
        "weight_blob_bytes": 617604,
        "n_runs": len(runs),
        "cases_per_pass": len(cases),
        "repeats": args.repeats,
        # Top-level aliases so the latency is readable without knowing the nest.
        "median_ms": statistics.median(latencies),
        "p95_ms": max(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "latency_ms": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "mean": statistics.fmean(latencies),
            "max": max(latencies),
        },
        # Measured by the firmware with esp_get_free_heap_size around the call,
        # not estimated: the forward pass uses a static arena and never allocates.
        "heap_free_bytes_after_run": {
            "min": min(heap_free),
            "max": max(heap_free),
        },
        "heap_delta_bytes_during_run": {"min": min(heap_delta), "max": max(heap_delta)},
        "flash_weight_bytes_per_forward": {
            "min": min(flash_bytes),
            "max": max(flash_bytes),
        },
        "predictions": [
            {"class": int(r["class"]), "confidence": float(r["confidence"])}
            for r in runs[: len(cases)]
        ],
        "raw": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"runs={len(runs)} clock={summary['clock_mhz']} MHz "
        f"median={summary['latency_ms']['median']:.3f} ms "
        f"heap_free_min={summary['heap_free_bytes_after_run']['min']} "
        f"flash_per_fwd={summary['flash_weight_bytes_per_forward']['min']} B"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
