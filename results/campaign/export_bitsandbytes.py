#!/usr/bin/env python3
"""Store each model as quantized by bitsandbytes, and say which are runnable.

bitsandbytes offers three weight formats, and they are not alike:

  nf4   a 4-bit NON-UNIFORM codebook at the quantiles of a normal distribution.
        Not expressible in the runtime's format, which dequantizes every scheme
        as `(u - zero_point) * scale` — a single affine map.
  fp4   the same machinery with a uniform 4-bit float codebook. Also non-affine.
  int8  blockwise, one absmax per block, which IS affine and therefore could be
        written as a runtime artifact at a matching block size.

This script stores all three as `bnb.safetensors` plus a `meta.json` recording
the scheme, the per-tensor pairing and the reconstruction error, and marks
whether the runtime could load it. nf4 and fp4 are marked not loadable, and that
is a property of the format rather than a shortfall of this script.

    uv run python results/campaign/export_bitsandbytes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

SCHEMES = ("nf4", "fp4", "int8")

#: The residual above which a scheme is not a single affine map.
AFFINE_TOLERANCE = 0.01


def affine_residual(code: torch.Tensor) -> float:
    """How far a codebook is from a straight line through its own indices.

    The runtime dequantizes EVERY scheme as one affine map, `(u - zp) * scale`,
    so a scheme is readable exactly when its code->value map is affine. Fitting
    the codebook itself is the right test: it is what the runtime has to
    reproduce, and it works the same for a 16-entry 4-bit book and a 256-entry
    8-bit one.

    This is measured rather than assumed. bitsandbytes' 8-bit path LOOKS like
    plain int8 and is not: its book has 256 non-uniformly spaced entries and the
    best affine fit through them leaves a residual of most of the value range.
    """
    values = code.detach().cpu().numpy().ravel().astype(np.float64)
    if values.size < 2:
        return 1.0
    index = np.arange(values.size, dtype=np.float64)
    design = np.vstack([index, np.ones_like(index)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, values, rcond=None)
    span = float(np.abs(values).max()) or 1.0
    return float(np.abs(values - (slope * index + intercept)).max() / span)


def quantize_scheme(weight: torch.Tensor, scheme: str) -> tuple[dict[str, np.ndarray], float]:
    """Quantize one weight and return its stored arrays plus relative error."""
    import bitsandbytes.functional as BF

    if scheme in ("nf4", "fp4"):
        packed, state = BF.quantize_4bit(weight, quant_type=scheme)
        rebuilt = BF.dequantize_4bit(packed, state).float()
        arrays = {
            "packed": packed.cpu().numpy(),
            "absmax": state[0].cpu().numpy().astype(np.float32),
        }
    else:
        # One block per row, so the absmax pairs with the row it scales.
        blocksize = int(weight.shape[-1])
        packed, state = BF.quantize_blockwise(weight, blocksize=blocksize)
        rebuilt = BF.dequantize_blockwise(packed, state).float()
        arrays = {
            "packed": packed.cpu().numpy(),
            "absmax": state[0].cpu().numpy().astype(np.float32),
        }

    # Both paths carry an explicit codebook; that is what decides readability.
    residual = affine_residual(state.code) if getattr(state, "code", None) is not None else 1.0
    error = (rebuilt - weight).norm().item()
    base = weight.norm().item()
    return arrays, (error / base if base else 0.0), residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "20-aug/models/exotic/normal-new-ds/full-after-hpo/artifacts",
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts_bitsandbytes.csv")
    args = parser.parse_args()

    models = sorted(p.name[: -len("-fp32")] for p in args.artifacts.glob("*-fp32") if p.is_dir())
    if not models:
        print(f"no in-house fp32 artifacts under {args.artifacts}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for model in models:
        fp32 = load_file(str(args.artifacts / f"{model}-fp32" / "model.safetensors"))
        int8_meta = json.loads((args.artifacts / f"{model}-int8" / "meta.json").read_text())
        # The same tensor set the in-house int8 artifact quantizes, so the
        # approaches differ in the quantizer and not in the selection.
        names = [n for n, e in int8_meta["tensors"].items() if e.get("quantized")]

        for scheme in SCHEMES:
            out = args.artifacts / f"{model}-bnb-{scheme}"
            out.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, np.ndarray] = {}
            per_tensor: dict[str, dict[str, object]] = {}
            errors: list[float] = []
            residuals: list[float] = []
            for name in names:
                weight = torch.from_numpy(fp32[name].astype(np.float32))
                stored, relative, residual = quantize_scheme(weight, scheme)
                residuals.append(residual)
                for key, value in stored.items():
                    arrays[f"{name}.{key}"] = value
                per_tensor[name] = {
                    "shape": list(weight.shape),
                    "relative_error": round(relative, 6),
                }
                errors.append(relative)

            loadable = float(np.max(residuals)) <= AFFINE_TOLERANCE
            save_file(arrays, str(out / "bnb.safetensors"))
            (out / "meta.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "producer": f"bitsandbytes.functional.{scheme}",
                        "scheme": scheme,
                        "complete_model": False,
                        "complete_note": (
                            "holds only the tensors bitsandbytes quantized; the "
                            "remaining tensors are unchanged and come from the "
                            "model's fp32 artifact"
                        ),
                        "runtime_loadable": loadable,
                        "runtime_note": (
                            "the runtime dequantizes every scheme as one affine "
                            "map (u - zp) * scale; this scheme's code->value map is "
                            "not affine, by the residual recorded here"
                            if not loadable
                            else "affine within tolerance"
                        ),
                        "affine_residual": round(float(np.max(residuals)), 6),
                        "affine_tolerance": AFFINE_TOLERANCE,
                        "tensors": per_tensor,
                        "mean_relative_error": round(float(np.mean(errors)), 6),
                        "max_relative_error": round(float(np.max(errors)), 6),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
            rows.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "directory": out.name,
                    "tensors": len(names),
                    "bytes": size,
                    "mean_relative_error": round(float(np.mean(errors)), 6),
                    "runtime_loadable": loadable,
                    "affine_residual": round(float(np.max(residuals)), 6),
                }
            )
            print(
                f"  {model:22} bnb-{scheme:5} {len(names):>3} tensors  "
                f"{size / 1e6:>6.3f} MB  rel err {np.mean(errors):.4f}  "
                f"loadable={loadable}"
            )

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.manifest} ({len(rows)} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
