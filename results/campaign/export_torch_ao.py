#!/usr/bin/env python3
"""Store each model as quantized by torch.ao, in the runtime's artifact format.

Why per tensor and not the whole model: `torch.ao.quantization.quantize_dynamic`
raises inside `nn.TransformerEncoderLayer.forward` on every architecture here, so
the module-level pipeline cannot produce an artifact at all. torch.ao's own
per-channel quantizer can, applied weight by weight, and the result is still
torch's numerics: its observer picks the scale, its `quantize_per_channel`
produces the codes.

The set of tensors quantized is taken from the in-house `int8` artifact rather
than recomputed, so the two artifacts differ in WHO quantized the weights and not
in WHICH weights were quantized. Anything else would make the comparison
meaningless.

The artifact is written in the runtime's format — codes stored biased by
`2**(bits-1)`, zero point folded, arrays raveled — so the Rust runtime executes
torch.ao's quantization and its own parity check says whether it did so
correctly.

    uv run python results/campaign/export_torch_ao.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: Who quantized the weights. Recorded in meta.json so an artifact can be traced
#: to the tool that produced it rather than only to this repository.
PRODUCER = "torch.ao.quantization.per_channel_symmetric_int8"
QUANT_NAME = "torchao_per_channel_symmetric_int8"


def torch_ao_scale(weight: torch.Tensor) -> torch.Tensor:
    """torch.ao's own per-channel symmetric scale for one weight."""
    from torch.ao.quantization.observer import PerChannelMinMaxObserver

    observer = PerChannelMinMaxObserver(
        ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    observer(weight)
    scale, _zero_point = observer.calculate_qparams()
    return scale.to(torch.float32)


def quantize_like_torch_ao(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(biased codes, raveled scale, folded zero point) for one weight."""
    from inmotion.quant.packing import fold_bias_into_zero_point, pack

    tensor = torch.from_numpy(weight.astype(np.float32))
    scale = torch_ao_scale(tensor).clamp(min=1e-12)
    zero_point = torch.zeros(tensor.shape[0], dtype=torch.int32)

    quantized = torch.quantize_per_channel(tensor, scale, zero_point, 0, torch.qint8)
    codes = quantized.int_repr().numpy()  # signed, as torch produced them

    # Store the way the runtime reads: biased codes, folded zero point.
    packed = pack(codes, 8, symmetric=True).ravel()
    folded = fold_bias_into_zero_point(zero_point.numpy(), 8, symmetric=True).ravel()
    return packed, scale.numpy().astype(np.float32).ravel(), folded.astype(np.int32)


def dequantize_artifact(meta: dict, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The float weights an artifact stands for: (u - zp) * scale, exactly."""
    out: dict[str, np.ndarray] = {}
    for name, entry in meta["tensors"].items():
        if not entry.get("quantized"):
            out[name] = arrays[name]
            continue
        shape = tuple(entry["shape"])
        scale_shape = tuple(entry["scale_shape"])
        codes = arrays[f"{name}.q"].reshape(shape).astype(np.float32)
        zero_point = arrays[f"{name}.zp"].reshape(scale_shape).astype(np.float32)
        scale = arrays[f"{name}.scale"].reshape(scale_shape).astype(np.float32)
        out[name] = ((codes - zero_point) * scale).reshape(shape)
    return out


def rebuild_state_dict(
    model: torch.nn.Module, weights: dict[str, np.ndarray], prefix: str | None
) -> dict[str, torch.Tensor]:
    """The model's OWN state dict, rebuilt from artifact-shaped tensors.

    Three of the exporter's transforms have to be undone, and each is exact:

      - the name prefix (`TupleWrap` nests SIGReg under `m.`),
      - `_split_qkv`, which sliced `in_proj_weight` into q/k/v rows,
      - weight-norm folding, whose inverse is `g = ||w||`, `v = w / ||w||`,
        because `g * v / ||v||` then returns `w` exactly.
    """
    from inmotion.quant.export import _SKIP_BUFFERS

    lookup = {f"{prefix}{k}" if prefix else k: v for k, v in weights.items()}
    state: dict[str, torch.Tensor] = {}
    # Buffers are inference state too: BatchNorm's running statistics are not
    # parameters, and an artifact without them loads into nothing usable.
    every = [(n, p) for n, p in model.named_parameters()] + [
        (n, b) for n, b in model.named_buffers()
    ]
    # The exporter drops training bookkeeping rather than shipping it, so the
    # rebuild must drop it too or it will look like a missing tensor.
    every = [(n, t) for n, t in every if not any(s in n for s in _SKIP_BUFFERS)]
    names = {name for name, _ in every}

    for name, param in every:
        if name.endswith("self_attn.in_proj_weight"):
            head = name[: -len("in_proj_weight")]
            parts = [lookup[f"{head}{tag}.weight"] for tag in ("q", "k", "v")]
            state[name] = torch.from_numpy(np.concatenate(parts, axis=0))
        elif name.endswith("self_attn.in_proj_bias"):
            head = name[: -len("in_proj_bias")]
            parts = [lookup[f"{head}{tag}.bias"] for tag in ("q", "k", "v")]
            state[name] = torch.from_numpy(np.concatenate(parts, axis=0))
        elif name.endswith(".weight_g") or name.endswith(".weight_v"):
            continue  # rebuilt below, once the folded weight is known
        elif name in lookup:
            state[name] = torch.from_numpy(lookup[name])

    # weight-norm: a folded `weight` becomes its `weight_g` and `weight_v`.
    for name, param in model.named_parameters():
        if not name.endswith(".weight_g"):
            continue
        folded = lookup.get(name[: -len(".weight_g") + len("weight")])
        if folded is None:
            continue
        tensor = torch.from_numpy(folded)
        dim = 0  # `WeightNorm.apply(..., dim=0)` is what these checkpoints used
        norm = tensor.norm(dim=dim, keepdim=True).clamp(min=1e-12)
        state[name] = norm.reshape(param.shape)
        state[name[: -len("weight_g")] + "weight_v"] = tensor / norm

    missing = names - set(state)
    if missing:
        raise SystemExit(f"could not rebuild {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    return state


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

    model_dir = args.artifacts.parent
    source_models = sorted(
        p.name[: -len("-fp32")] for p in args.artifacts.glob("*-fp32") if p.is_dir()
    )
    if not source_models:
        print(f"no in-house fp32 artifacts under {args.artifacts}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for model in source_models:
        fp32_meta = json.loads((args.artifacts / f"{model}-fp32" / "meta.json").read_text())
        int8_meta = json.loads((args.artifacts / f"{model}-int8" / "meta.json").read_text())
        fp32_tensors = load_file(str(args.artifacts / f"{model}-fp32" / "model.safetensors"))

        # The tensor SET comes from the in-house int8 artifact; the values come
        # from the fp32 one and are quantized here by torch.ao.
        quantized_names = [n for n, e in int8_meta["tensors"].items() if e.get("quantized")]

        out = args.artifacts / f"{model}-torchao-int8"
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        index: dict[str, dict[str, object]] = {}

        for name, entry in fp32_meta["tensors"].items():
            if name not in quantized_names:
                arrays[name] = fp32_tensors[name]
                index[name] = entry
                continue
            reference = int8_meta["tensors"][name]
            codes, scale, zero_point = quantize_like_torch_ao(fp32_tensors[name])
            arrays[f"{name}.q"] = codes
            arrays[f"{name}.scale"] = scale
            arrays[f"{name}.zp"] = zero_point
            index[name] = {
                "axis": 0,
                "bits": 8,
                "granularity": "per_channel",
                "group_size": reference.get("group_size", 32),
                "n_values": int(np.prod(reference["shape"])),
                "quantized": True,
                "scale_shape": reference["scale_shape"],
                "shape": reference["shape"],
            }

        meta = dict(fp32_meta)
        meta["tensors"] = index
        meta["quant"] = {
            "axis": 0,
            "bits": 8,
            "granularity": "per_channel",
            "group_size": 32,
            "name": QUANT_NAME,
            "rounding": "half_to_even",
            "symmetric": True,
        }
        meta["export"] = dict(fp32_meta.get("export", {}))
        meta["export"]["quantized_by"] = PRODUCER
        meta["export"]["source_artifact"] = f"{model}-fp32"

        save_file(arrays, str(out / "model.safetensors"))
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # The golden vectors must come from THIS artifact's weights. Copying the
        # fp32 ones would compare a quantized artifact against an unquantized
        # model and fail by exactly the quantization error, which says nothing.
        golden_source = args.artifacts / f"{model}-fp32" / "golden.json"
        if golden_source.exists():
            from inmotion.models import registry as reg
            from inmotion.models.adopt import resolve_model

            golden = json.loads(golden_source.read_text())
            spec_obj = resolve_model(str(Path(model_dir) / f"{model}.pt"))
            torch_model = reg.load(spec_obj, strict=True).eval()
            weights = dequantize_artifact(meta, arrays)
            state = rebuild_state_dict(torch_model, weights, spec_obj.prefix)
            torch_model.load_state_dict(state, strict=True)

            cases = []
            with torch.no_grad():
                for case in golden["cases"]:
                    x = torch.from_numpy(np.asarray(case["input"], dtype=np.float32))
                    logits = torch_model(x.unsqueeze(0))[0].numpy().tolist()
                    cases.append({"input": case["input"], "logits": logits})
            golden["cases"] = cases
            golden["quant"] = QUANT_NAME
            (out / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
            print(f"    golden rebuilt from the torch.ao weights ({len(cases)} cases)")

        check = subprocess.run(
            [str(args.binary), "check", "--artifact", str(out), "--golden", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        parity = "PARITY OK" in check.stdout
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        rows.append(
            {
                "model": model,
                "producer": PRODUCER,
                "directory": out.name,
                "quantized_tensors": len(quantized_names),
                "bytes": size,
                "parity": "ok" if parity else "FAIL",
            }
        )
        print(
            f"  {model:22} {len(quantized_names):>3} weights quantized by torch.ao  "
            f"{size / 1e6:>6.3f} MB  {'parity ok' if parity else 'PARITY FAILED'}"
        )
        if not parity:
            print(f"    {check.stdout.strip()[:200]}")

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.manifest} ({len(rows)} artifacts)")
    return 0 if all(r["parity"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
