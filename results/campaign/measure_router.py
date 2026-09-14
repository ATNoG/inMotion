#!/usr/bin/env python3
"""Measure the Rust runtime on the wave-router, from here.

The router is busybox: no Python, no scp, no base64. The binary and artifacts
are transferred by `tar` over `ssh`, sampled there by a POSIX shell loop, and the
resulting CSV is parsed locally. This script is that whole path, so the router
number is reproducible rather than a one-off someone typed.

    uv run python results/campaign/measure_router.py --host wave-router \\
        --remote /mnt/extra/cargo/inmotion/campaign --out results/campaign

CPU PERCENTAGE. /proc counts CPU over the whole process, so the divisor must be
the whole process too — loading the artifact, the warmup and the timed loop. The
binary reports that as `total_ms`; an earlier version estimated it from the
warmup count, which produced 103% on a single-threaded run.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The router's kernel clock tick. getconf is absent there, and 100 is the x86
#: Linux default; the value only scales the CPU column.
USER_HZ = 100.0


def ssh(host: str, command: str, timeout: int = 1800) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh {host}: {result.stderr.strip()[:300]}")
    # The login banner goes to stdout on this host.
    return "\n".join(
        line
        for line in result.stdout.splitlines()
        if line and "SECURED" not in line and "authorized" not in line.lower()
    )


def parse_rows(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line and not line.startswith("Device")]
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","), strict=False)) for line in lines[1:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="wave-router")
    parser.add_argument("--remote", default="/mnt/extra/cargo/inmotion/campaign")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", default="wave-router-j3455")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    print(f"measuring on {args.host} ...")
    raw = ssh(
        args.host,
        f"cd {args.remote} && sh router-measure.sh hpo-art {args.iterations} {args.warmup}",
    )
    rows = parse_rows(raw)

    out_rows = []
    for row in rows:
        cpu_seconds = float(row["cpu_ticks"]) / USER_HZ
        process_window_s = float(row["total_ms"]) / 1000.0
        out_rows.append(
            {
                "device": args.device,
                "model": row["model"],
                "precision": row["precision"],
                "iterations": args.iterations,
                "warmup": args.warmup,
                "min_ms": row["min_ms"],
                "median_ms": row["median_ms"],
                "p95_ms": row["p95_ms"],
                "max_ms": row["max_ms"],
                "throughput_per_s": row["throughput_per_s"],
                "flops_per_infer": "",
                "flops_per_s": "",
                "peak_rss_mb": round(float(row["peak_rss_kb"]) / 1024.0, 2),
                "cpu_percent": round(cpu_seconds / process_window_s * 100.0, 1),
                "energy_j": "",
                "energy_per_infer_mj": "",
            }
        )
        print(
            f"  {row['model']:22} {row['precision']:5} {float(row['median_ms']):>8.3f} ms  "
            f"rss={float(row['peak_rss_kb']) / 1024:>7.2f} MB  "
            f"cpu={out_rows[-1]['cpu_percent']:>5.1f}%"
        )

    # FLOPs are a property of the model, so the same table serves both devices.
    flops_path = HERE / "flops_per_model.csv"
    if flops_path.exists():
        with flops_path.open(newline="") as handle:
            flops = {r["model"]: float(r["flops"]) for r in csv.DictReader(handle)}
        for row in out_rows:
            per_infer = flops.get(row["model"])
            if per_infer is not None:
                row["flops_per_infer"] = per_infer
                row["flops_per_s"] = round(per_infer * float(row["throughput_per_s"]), 1)

    path = args.out / "perf_device_router.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {path} ({len(out_rows)} rows)")

    print("sweeping concurrency ...")
    scaling = []
    for artifact in sorted({f"{r['model']}-{r['precision']}" for r in rows}):
        baseline = None
        for concurrency in (1, 2, 4):
            raw = ssh(
                args.host,
                f"cd {args.remote} && ./inmotion-infer bench --artifact hpo-art/{artifact} "
                f"--concurrency {concurrency} --seconds {args.seconds} --json",
                timeout=300,
            )
            line = next((x for x in raw.splitlines() if x.startswith("{")), None)
            if line is None:
                continue
            import json

            data = json.loads(line)
            throughput = float(data["throughput_per_s"])
            if baseline is None:
                baseline = throughput
            scaling.append(
                {
                    "device": args.device,
                    "model": artifact.rsplit("-", 1)[0],
                    "precision": artifact.rsplit("-", 1)[1],
                    "concurrency": concurrency,
                    "seconds": data["seconds"],
                    "iterations": data["iterations"],
                    "throughput_per_s": round(throughput, 2),
                    "latency_ms": round(float(data["latency_ms"]), 4),
                    "speedup": round(throughput / baseline, 3),
                    "efficiency": round(throughput / baseline / concurrency, 3),
                }
            )
            print(
                f"  {artifact:22} c={concurrency} {throughput:>7.1f}/s x{throughput / baseline:.2f}"
            )

    path = args.out / "scaling_router.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaling[0]))
        writer.writeheader()
        writer.writerows(scaling)
    print(f"wrote {path} ({len(scaling)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
