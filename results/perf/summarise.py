"""Turn the raw per-run CSVs into the before/after table in summary.csv.

Each (device, model, precision, build) cell is measured REPS times. On a shared
machine the tail of the distribution is machine load, not the kernel, so the
reported figure is the *best* run per cell: the least-contaminated estimate of
what the kernel costs. The full spread stays in the raw CSVs.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("lejepa", "t_jepa", "ts_jepa", "cf_jepa", "sigreg")
PRECS = ("fp32", "int8", "q4")


def best_by_cell(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    cells: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in csv.DictReader(path.open()):
        if row.get("device") is None or row.get("build") is None or row.get("median_ms") is None:
            # A trailing status line (`DONE`, appended by a runner) lands here as
            # a bare word in the device column. A row that has some columns but
            # not all of them is runner schema drift, and must not be dropped.
            if sum(v is not None for v in row.values()) > 1:
                raise SystemExit(f"{path}: partial row, runner columns have drifted: {row!r}")
            continue
        cells[(row["device"], row["model"], row["precision"], row["build"])].append(row)
    return {k: min(v, key=lambda r: float(r["median_ms"])) for k, v in cells.items()}


def main() -> int:
    cells: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for name in ("bench_host.csv", "bench_router.csv"):
        path = HERE / name
        if path.exists():
            cells.update(best_by_cell(path))
        else:
            print(f"missing {path}; skipping", file=sys.stderr)

    devices = sorted({key[0] for key in cells})
    out = []
    for device in devices:
        for model in MODELS:
            for prec in PRECS:
                before = cells.get((device, model, prec, "before"))
                after = cells.get((device, model, prec, "after"))
                if before is None or after is None:
                    continue
                bm, am = float(before["median_ms"]), float(after["median_ms"])
                bp, ap = float(before["p95_ms"]), float(after["p95_ms"])
                out.append(
                    {
                        "device": device,
                        "model": model,
                        "precision": prec,
                        "before_median_ms": f"{bm:.4f}",
                        "after_median_ms": f"{am:.4f}",
                        "speedup_median": f"{bm / am:.2f}",
                        "before_p95_ms": f"{bp:.4f}",
                        "after_p95_ms": f"{ap:.4f}",
                        "speedup_p95": f"{bp / ap:.2f}",
                        "before_min_ms": f"{float(before['min_ms']):.4f}",
                        "after_min_ms": f"{float(after['min_ms']):.4f}",
                    }
                )
    if not out:
        print("no cells to summarise", file=sys.stderr)
        return 1
    with (HERE / "summary.csv").open("w", newline="") as fh:
        # LF, matching the committed CSVs and the runners; the csv module's
        # default dialect would write CRLF.
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(
            f"{row['device']:7s} {row['model']:8s} {row['precision']:5s} "
            f"median {row['before_median_ms']:>10s} -> {row['after_median_ms']:>9s} ms "
            f"{row['speedup_median']:>5s}x  |  p95 {row['before_p95_ms']:>10s} -> "
            f"{row['after_p95_ms']:>9s} ms {row['speedup_p95']:>5s}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
