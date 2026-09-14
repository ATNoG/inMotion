#!/usr/bin/env python3
"""Measure the Rust runtime on a device: RAM, CPU, energy and latency.

Latency alone does not answer "will this fit and what does it cost to run", so
this runs `inmotion-infer bench` and samples the process while it works:

    uv run python results/campaign/measure_device.py \\
        --artifacts /tmp/hpo-art --device this-pc \\
        --out results/campaign/perf_device.csv

Every column is either measured or empty, never guessed:

  peak_rss_mb    highest VmHWM seen while sampling /proc/<pid>/status
  cpu_percent    (utime + stime) / wall clock, the top(1) convention, so a
                 multi-threaded run may exceed 100
  energy_j       difference of two RAPL reads around the measured window
  flops_per_s    joined from flops_per_model.csv, measured with torch's own
                 FlopCounterMode; FLOPs are a property of the model, not of the
                 runtime, so one measurement per model serves both devices

ENERGY NEEDS PRIVILEGES. `/sys/class/powercap/*/energy_uj` is mode 0400 root, so
energy_j is empty unless this runs as root. That is deliberate — the column says
"not measured" rather than 0. To fill it:

    sudo -E uv run python results/campaign/measure_device.py --artifacts ... --device ...

The wave-router has no RAPL at all, so its energy column stays empty there
regardless; that is a property of the hardware, not of this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: RAPL package domain on Intel. Overridable for a different machine.
DEFAULT_RAPL = Path("/sys/class/powercap/intel-rapl:0")

TICKS_PER_SECOND = 100.0  # USER_HZ; the value /proc/<pid>/stat counts in.


def read_energy_uj(rapl: Path) -> int | None:
    """Cumulative microjoules, or None when the counter is not readable."""
    try:
        return int((rapl / "energy_uj").read_text().strip())
    except (OSError, ValueError):
        return None


def read_max_energy_uj(rapl: Path) -> int | None:
    try:
        return int((rapl / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        return None


def energy_delta(start: int, end: int, max_range: int | None) -> int:
    """Microjoules between two reads, handling a counter wrap."""
    if end >= start:
        return end - start
    if max_range is None:
        return end - start  # cannot be corrected; reported as-is
    return (max_range - start) + end


class Sampler(threading.Thread):
    """Samples a process's peak RSS and CPU time until stopped."""

    def __init__(self, pid: int, interval: float = 0.005) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_rss_kb = 0
        self.cpu_ticks = 0
        self._stop = threading.Event()

    def _sample_once(self) -> None:
        try:
            with open(f"/proc/{self.pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        self.peak_rss_kb = max(self.peak_rss_kb, int(line.split()[1]))
                        break
            with open(f"/proc/{self.pid}/stat") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            # After the comm field: state, ppid, ... utime=11, stime=12 (0-based 11, 12).
            self.cpu_ticks = int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            pass

    def run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1.0)


def measure(
    binary: Path,
    artifact: Path,
    *,
    iterations: int,
    warmup: int,
    rapl: Path,
) -> dict[str, object]:
    """Run one bench and return the sampled measurements."""
    max_range = read_max_energy_uj(rapl)
    energy_before = read_energy_uj(rapl)

    started = time.monotonic()
    proc = subprocess.Popen(
        [
            str(binary),
            "bench",
            "--artifact",
            str(artifact),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler = Sampler(proc.pid)
    sampler.start()
    stdout, stderr = proc.communicate()
    wall = time.monotonic() - started
    sampler.stop()

    energy_after = read_energy_uj(rapl)
    energy_uj = (
        energy_delta(energy_before, energy_after, max_range)
        if energy_before is not None and energy_after is not None
        else None
    )

    if proc.returncode != 0:
        raise RuntimeError(f"bench failed for {artifact}: {stderr.strip()[:200]}")

    return {
        "bench": json.loads(stdout),
        "wall_s": wall,
        "peak_rss_mb": sampler.peak_rss_kb / 1024.0,
        "cpu_percent": sampler.cpu_ticks / TICKS_PER_SECOND / wall * 100.0,
        "energy_j": energy_uj / 1e6 if energy_uj is not None else None,
    }


def load_flops() -> dict[str, float]:
    """FLOPs per inference, keyed by model name, if measured."""
    path = HERE / "flops_per_model.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["model"]: float(row["flops"]) for row in csv.DictReader(handle)}


#: Producer-specific artifact suffixes that are not `<model>-<precision>`. The
#: store holds several quantizers, and their directories have to stay
#: distinguishable in the output without corrupting the model name, which is
#: what the FLOPs table is joined on.
PRODUCER_SUFFIXES = ("-torchao-int8",)


def parse_artifact(name: str) -> tuple[str, str]:
    """`lejepa_ft_seed42-q4` -> ("lejepa_ft_seed42", "q4").

    A producer suffix becomes part of the PRECISION, not of the model:
    `lejepa_ft_seed42-torchao-int8` -> ("lejepa_ft_seed42", "torchao-int8"),
    so the row stays distinct from the in-house int8 and still joins to FLOPs.
    """
    for suffix in PRODUCER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    model, _, precision = name.rpartition("-")
    return (model or name), (precision or "fp32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory of artifacts.")
    parser.add_argument("--device", required=True, help="Label for the CSV, e.g. this-pc.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rapl", type=Path, default=DEFAULT_RAPL)
    args = parser.parse_args()

    # Only runtime artifacts: `model.safetensors` is what the loader reads.
    # The store also holds bitsandbytes and torch.ao directories, and the bnb
    # ones carry no runtime metadata at all — globbing on meta.json alone tried
    # to benchmark them and aborted the whole run.
    artifacts = sorted(
        p
        for p in args.artifacts.iterdir()
        if (p / "meta.json").is_file() and (p / "model.safetensors").is_file()
    )
    if not artifacts:
        print(f"no artifacts under {args.artifacts}", file=sys.stderr)
        return 1

    flops = load_flops()
    if not flops:
        print("note: flops_per_model.csv absent, flops columns will be empty")

    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        model, precision = parse_artifact(artifact.name)
        result = measure(
            args.binary,
            artifact,
            iterations=args.iterations,
            warmup=args.warmup,
            rapl=args.rapl,
        )
        bench = result["bench"]
        per_infer_flops = flops.get(model)
        throughput = float(bench["throughput_per_s"])
        rows.append(
            {
                "device": args.device,
                "model": model,
                "precision": precision,
                "iterations": bench["iterations"],
                "warmup": bench["warmup"],
                "min_ms": bench["min_ms"],
                "median_ms": bench["median_ms"],
                "p95_ms": bench["p95_ms"],
                "max_ms": bench["max_ms"],
                "throughput_per_s": throughput,
                "flops_per_infer": per_infer_flops if per_infer_flops is not None else "",
                "flops_per_s": (
                    per_infer_flops * throughput if per_infer_flops is not None else ""
                ),
                "peak_rss_mb": round(float(result["peak_rss_mb"]), 2),
                "cpu_percent": round(float(result["cpu_percent"]), 1),
                # -1 is the sentinel for "this device cannot supply it", so the
                # column stays numeric. An empty cell makes a numeric column
                # silently non-numeric and forces every reader to special-case
                # it. This counter is root-only here; the router has none at all.
                "energy_j": (
                    round(float(result["energy_j"]), 4) if result["energy_j"] is not None else -1
                ),
                "energy_per_infer_mj": (
                    round(float(result["energy_j"]) / bench["iterations"] * 1000.0, 4)
                    if result["energy_j"] is not None
                    else -1
                ),
            }
        )
        energy = f"{result['energy_j']:.2f} J" if result["energy_j"] is not None else "n/a"
        print(
            f"  {model:24} {precision:5} {bench['median_ms']:>8.4f} ms  "
            f"rss={result['peak_rss_mb']:>7.2f} MB  cpu={result['cpu_percent']:>5.1f}%  "
            f"energy={energy}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    if all(row["energy_j"] == "" for row in rows):
        print("energy is -1: the counter is root-only here. Re-run under sudo to fill it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
