"""Drive the ESP32 firmware over serial and record measured latency and RAM.

The device protocol is one line in, one RESULT line out. Each input line is a
real test window from the golden vectors (dataset.csv's test split), so the
device is being asked the same question the parity harness asks on the host.

Usage:
    sg dialout -c 'uv run --extra esp python firmware/tools/run_on_device.py \
        --port /dev/ttyUSB0 --cases /tmp/lejepa_int8/cases.tsv --out results/quant/esp32_lejepa.json'
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import serial


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
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=1,
                    help="passes over the case set; more passes make the timing median meaningful")
    args = ap.parse_args()

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
        "weight_blob_bytes": 617604,
        "n_runs": len(runs),
        "cases_per_pass": len(cases),
        "repeats": args.repeats,
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
    print(f"runs={len(runs)} median={summary['latency_ms']['median']:.3f} ms "
          f"heap_free_min={summary['heap_free_bytes_after_run']['min']} "
          f"flash_per_fwd={summary['flash_weight_bytes_per_forward']['min']} B")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
