#!/usr/bin/env python3
"""Export every model at every precision into the model store, well named.

The quantized artifacts used to live in /tmp, which is ephemeral: the device
measurements read them, and a reboot would have taken them with it. They now sit
beside the checkpoints they came from, in the store docs/reports/ARTIFACTS.md
already defines for models, one directory per (model, precision).

Naming is `<model>-<precision>`, which is what measure_device.py and
measure_scaling.py parse, so the measurement tooling reads this layout directly.

Every artifact is parity-checked against its own golden vectors as it is written,
so the manifest says whether each one is usable rather than only that it exists.

    uv run python results/campaign/export_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The checkpoints this campaign covers.
MODELS = (
    "lejepa_ft_seed42",
    "t_jepa_ft_seed42",
    "ts_jepa_ft_seed42",
    "cf_jepa_ft_seed42",
    "sigreg_seed42",
)

#: Every precision the runtime can LOAD, cheapest last so the output reads in
#: cost order. fp16, bf16 and fp8 are absent deliberately: they are cast
#: precisions the accuracy sweep measures, but the runtime reads F32, I32 and U8
#: only, so no artifact can carry them. Exporting them raises a clear error.
PRECISIONS = (
    "fp32",
    "int8",
    "q8",
    "q6",
    "q5",
    "q4",
    "q3",
    "q2",
    "q1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO / "20-aug" / "models" / "exotic" / "normal-new-ds" / "full-after-hpo",
    )
    parser.add_argument("--out", type=Path, default=None, help="Default: <models-dir>/artifacts")
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "rust" / "target" / "release" / "inmotion-infer",
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts_manifest.csv")
    parser.add_argument("--force", action="store_true", help="Re-export what exists.")
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    out: Path = args.out or models_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for model in MODELS:
        checkpoint = models_dir / f"{model}.pt"
        if not checkpoint.exists():
            print(f"skip {model}: {checkpoint.name} not present")
            continue
        for precision in PRECISIONS:
            target = out / f"{model}-{precision}"
            if target.exists() and not args.force:
                pass  # already exported; still measured and checked below
            else:
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "inmotion",
                        "quantize",
                        "export",
                        str(checkpoint),
                        "--out",
                        str(target),
                        "--format",
                        precision,
                    ],
                    cwd=REPO,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    failures.append(f"{model}-{precision}: export failed")
                    # The exporter creates the directory before it does any
                    # work, so a failure leaves an empty one behind. An empty
                    # artifact directory is worse than none: it looks present.
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"  FAIL {model}-{precision}: {result.stderr.strip()[:90]}")
                    continue

            check = subprocess.run(
                [str(args.binary), "check", "--artifact", str(target), "--golden", str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            parity = "PARITY OK" in check.stdout
            if not parity:
                failures.append(f"{model}-{precision}: parity")

            files = sorted(p.name for p in target.iterdir() if p.is_file())
            bytes_total = sum(p.stat().st_size for p in target.iterdir() if p.is_file())
            rows.append(
                {
                    "model": model,
                    "precision": precision,
                    "directory": f"{model}-{precision}",
                    "files": " ".join(files),
                    "bytes": bytes_total,
                    "parity": "ok" if parity else "FAIL",
                }
            )
            print(
                f"  {model:22} {precision:5} {bytes_total / 1e6:>7.3f} MB  "
                f"{'parity ok' if parity else 'PARITY FAILED'}"
            )

    if not rows:
        print("nothing exported", file=sys.stderr)
        return 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(int(r["bytes"]) for r in rows)
    print(f"\nwrote {args.manifest} ({len(rows)} artifacts, {total / 1e6:.1f} MB)")
    print(f"artifacts live in {out}")
    if failures:
        print(f"\n{len(failures)} PROBLEMS:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("every artifact replayed its golden vectors: parity ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
