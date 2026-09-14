#!/usr/bin/env python3
"""How throughput scales with concurrent inferences, and where it saturates.

Batch-1 inference here is 4-11 MFLOP, small enough that splitting one inference
across threads costs more than it saves. What can scale is the number of
inferences in flight, which is the deployment question: one device serving many
requests. This sweeps concurrency and reports the loaded latency beside the
throughput, because throughput saturating while latency keeps falling is the
saturation point.

    uv run python results/campaign/measure_scaling.py \\
        --binary rust/target/release/inmotion-infer \\
        --artifacts /tmp/hpo-art --device this-pc --seconds 2 \\
        --out results/campaign/scaling_this-pc.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: Concurrency levels to sweep. Includes 1 so every row has a baseline, and goes
#: past the physical core count to show what hyperthreading does or does not buy.
LEVELS = (1, 2, 4, 8)


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
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--levels", default=",".join(str(n) for n in LEVELS))
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    artifacts = sorted(
        p
        for p in args.artifacts.iterdir()
        if (p / "meta.json").is_file() and (p / "model.safetensors").is_file()
    )
    if not artifacts:
        print(f"no artifacts under {args.artifacts}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        model, precision = parse_artifact(artifact.name)
        baseline = None
        for concurrency in levels:
            result = subprocess.run(
                [
                    str(args.binary),
                    "bench",
                    "--artifact",
                    str(artifact),
                    "--concurrency",
                    str(concurrency),
                    "--seconds",
                    str(args.seconds),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(f"  {model} {precision} c={concurrency}: {result.stderr.strip()[:80]}")
                continue
            data = json.loads(result.stdout)
            throughput = float(data["throughput_per_s"])
            if baseline is None:
                baseline = throughput
            rows.append(
                {
                    "device": args.device,
                    "model": model,
                    "precision": precision,
                    "concurrency": concurrency,
                    "seconds": data["seconds"],
                    "iterations": data["iterations"],
                    "throughput_per_s": round(throughput, 2),
                    "latency_ms": round(float(data["latency_ms"]), 4),
                    "speedup": round(throughput / baseline, 3) if baseline else "",
                    "efficiency": round(throughput / baseline / concurrency, 3) if baseline else "",
                }
            )
            print(
                f"  {model:22} {precision:5} c={concurrency:<2} "
                f"{throughput:>9.1f}/s  x{throughput / baseline:>5.2f}  "
                f"lat={data['latency_ms']:>7.3f} ms"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
