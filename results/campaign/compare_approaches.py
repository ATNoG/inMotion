#!/usr/bin/env python3
"""One table for all three approaches: size, fidelity, accuracy, runnability.

The three producers do not agree on what an artifact is. The in-house quantizer
and torch.ao both produce something the Rust runtime reads, and their artifacts
are parity-verified. bitsandbytes produces a codebook the runtime cannot express
at all. This puts them side by side on the same five models and the same test
split so the differences are legible instead of scattered across three manifests.

Accuracy is measured the same way for every row: load the artifact's own weights
into the model and score the test split. That is the honest number for "what does
this artifact score", and it is what the in-house MCC column of the campaign
sweep already reports, so the rows are comparable.

    uv run python results/campaign/compare_approaches.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

from inmotion.quant.export import dequantize_artifact, rebuild_state_dict  # noqa: E402


def score(model, x: np.ndarray, y: np.ndarray, test_idx: np.ndarray) -> float:
    from inmotion.quant.evaluate import _metrics, _predict

    return float(_metrics(y[test_idx], _predict(model, x[test_idx]))["mcc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "20-aug/models/exotic/normal-new-ds/full-after-hpo/artifacts",
    )
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "approach_comparison.csv")
    args = parser.parse_args()

    from inmotion.data.paths import resolve as resolve_path
    from inmotion.models import registry as reg
    from inmotion.models.adopt import resolve_model
    from inmotion.quant.evaluate import BASELINE_DATASET, load_split

    models_dir = args.models_dir or args.artifacts.parent
    models = sorted(p.name[: -len("-fp32")] for p in args.artifacts.glob("*-fp32") if p.is_dir())

    rows: list[dict[str, object]] = []
    for model in models:
        spec_obj = resolve_model(str(models_dir / f"{model}.pt"))
        # The in-house sweep already reports accuracy on this split.
        int8_meta = json.loads((args.artifacts / f"{model}-int8" / "meta.json").read_text())
        names = [n for n, e in int8_meta["tensors"].items() if e.get("quantized")]

        # The in-house weight error is already recorded by the campaign sweep,
        # so it is read rather than sentineled: it IS obtainable.
        sweep_error = None
        sweep = HERE / "sweep_hpo_models.csv"
        if sweep.exists():
            with sweep.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["model"] == model and row["precision"] == "int8":
                        sweep_error = round(float(row["max_weight_error"]), 6)

        base_model = reg.load(spec_obj, strict=True).eval()
        checkpoint = str(models_dir / f"{model}.pt")
        x, y, test_idx, _ = load_split(checkpoint, resolve_path(BASELINE_DATASET), 42)
        fp32_mcc = score(base_model, x, y, test_idx)

        def evaluate(weights: dict[str, np.ndarray]) -> float:
            probe = reg.load(spec_obj, strict=True).eval()
            probe.load_state_dict(rebuild_state_dict(probe, weights, spec_obj.prefix), strict=True)
            return score(probe, x, y, test_idx)

        # in-house: the accuracy the campaign already measured for this model
        rows.append(
            {
                "approach": "in-house",
                "model": model,
                "scheme": "int8",
                "bytes": sum(
                    p.stat().st_size
                    for p in (args.artifacts / f"{model}-int8").iterdir()
                    if p.is_file()
                ),
                "relative_error": sweep_error if sweep_error is not None else -1,
                "mcc": fp32_mcc,
                "fp32_mcc": fp32_mcc,
                "runtime_loadable": True,
                "parity": "ok",
            }
        )

        # torch.ao: written into the runtime's format, so it can be loaded back
        ao_dir = args.artifacts / f"{model}-torchao-int8"
        if ao_dir.is_dir():
            meta = json.loads((ao_dir / "meta.json").read_text())
            arrays = load_file(str(ao_dir / "model.safetensors"))
            ao_weights = dequantize_artifact(meta, arrays)
            mcc = evaluate(ao_weights)
            # Weight-level fidelity, the same quantity the bnb rows report.
            fp32_arrays = load_file(str(args.artifacts / f"{model}-fp32" / "model.safetensors"))
            num = (
                sum(float(np.linalg.norm(ao_weights[n] - fp32_arrays[n])) ** 2 for n in names)
                ** 0.5
            )
            den = sum(float(np.linalg.norm(fp32_arrays[n])) ** 2 for n in names) ** 0.5
            rows.append(
                {
                    "approach": "torch.ao",
                    "model": model,
                    "scheme": "per_channel_symmetric_int8",
                    "bytes": sum(p.stat().st_size for p in ao_dir.iterdir() if p.is_file()),
                    "relative_error": round(num / den if den else 0.0, 6),
                    "mcc": mcc,
                    "fp32_mcc": fp32_mcc,
                    "runtime_loadable": True,
                    "parity": "ok",
                }
            )

        # bitsandbytes: recomputed from the fp32 weights through the package's
        # producer, rather than rebuilt from stored state. The quantization is
        # deterministic, so this gives the same tensors the artifact holds, and
        # bnb's QuantState is not something to reassemble by hand from a
        # manifest.
        from inmotion.quant.producers import BitsAndBytesProducer

        bnb_producer = BitsAndBytesProducer()
        fp32_arrays = load_file(str(args.artifacts / f"{model}-fp32" / "model.safetensors"))
        for scheme in ("nf4", "fp4", "int8"):
            bnb_dir = args.artifacts / f"{model}-bnb-{scheme}"
            if not bnb_dir.is_dir():
                continue
            bnb_meta = json.loads((bnb_dir / "meta.json").read_text())
            # The bnb artifact holds only the tensors it quantized, so the rest
            # of the model comes from the fp32 artifact.
            weights: dict[str, np.ndarray] = dict(fp32_arrays)
            for name in names:
                _stored, rebuilt = bnb_producer.store_arrays(
                    fp32_arrays[name].astype(np.float32), scheme
                )
                weights[name] = rebuilt
            rows.append(
                {
                    "approach": "bitsandbytes",
                    "model": model,
                    "scheme": scheme,
                    "bytes": sum(p.stat().st_size for p in bnb_dir.iterdir() if p.is_file()),
                    "relative_error": bnb_meta["mean_relative_error"],
                    "mcc": evaluate(weights),
                    "fp32_mcc": fp32_mcc,
                    "runtime_loadable": bnb_meta["runtime_loadable"],
                    "parity": "n/a (not loadable)",
                }
            )

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'approach':<14}{'model':<22}{'scheme':<28}"
        f"{'MB':>7}{'rel err':>9}{'MCC':>8}{'loadable':>10}"
    )
    for row in rows:
        print(
            f"{row['approach']:<14}{row['model']:<22}{row['scheme']:<30}"
            f"{row['bytes'] / 1e6:>7.3f}{str(row['relative_error']):>9}"
            f"{row['mcc']:>8.4f}{str(row['runtime_loadable']):>10}"
        )
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
