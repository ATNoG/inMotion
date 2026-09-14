#!/usr/bin/env python3
"""Export every model with torch.ao's quantizer, into the model store.

The quantization lives in the package now — `inmotion.quant.producers` — and is
reachable from the CLI:

    uv run inmotion quantize export <checkpoint> --out DIR --producer torch-ao

This is only the loop over the whole store, so the artifacts and the report stay
in step with `export_artifacts.py` and the device harnesses read them the same
way. Parity is checked here because an artifact that loads and computes the wrong
thing is worse than one that does not exist.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

PRODUCER = "torch-ao"
SCHEME = "int8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "20-aug/models/exotic/normal-new-ds/full-after-hpo/artifacts",
    )
    parser.add_argument("--binary", type=Path, default=REPO / "rust/target/release/inmotion-infer")
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts_torch_ao.csv")
    args = parser.parse_args()

    from inmotion.quant.export import export, write_golden

    models_dir = args.artifacts.parent
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for checkpoint in sorted(models_dir.glob("*_ft_seed42.pt")) + sorted(
        models_dir.glob("sigreg_seed42.pt")
    ):
        model = checkpoint.stem
        out = args.artifacts / f"{model}-torchao-{SCHEME}"
        report = export(checkpoint, out, producer=PRODUCER, quant=SCHEME)
        write_golden(checkpoint, out, producer=PRODUCER, quant=SCHEME)

        check = subprocess.run(
            [str(args.binary), "check", "--artifact", str(out), "--golden", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        parity = "PARITY OK" in check.stdout
        if not parity:
            failures.append(f"{model}: {check.stdout.strip()[:120]}")

        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        rows.append(
            {
                "model": model,
                "producer": PRODUCER,
                "scheme": SCHEME,
                "directory": out.name,
                "quantized_tensors": report.quantized_tensors,
                "bytes": size,
                "parity": "ok" if parity else "FAIL",
            }
        )
        print(
            f"  {model:22} {report.quantized_tensors:>3} weights by torch.ao  "
            f"{size / 1e6:>6.3f} MB  {'parity ok' if parity else 'PARITY FAILED'}"
        )

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.manifest} ({len(rows)} artifacts)")
    if failures:
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
