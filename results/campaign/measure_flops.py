#!/usr/bin/env python3
"""FLOPs per inference, per model, counted from the modules that actually ran.

FLOPs are a property of the architecture and the input shape, not of the runtime,
so this is measured once in Python and joined into the device measurements by
model name.

WHY NOT torch.utils.flop_counter. It is tried first and its answer is reported
alongside, but for these graphs it undercounts by about an order of magnitude:
cf_jepa's transformer alone holds 1.19M parameters, so one forward through them
costs at least ~2.4 MFLOP, and the counter reports 0.74 MFLOP for the WHOLE
model. Rather than publish a number that fails its own sanity check, MACs are
counted here from the shapes each leaf module is actually called with, captured
by forward hooks. The invariant that makes this checkable: a dense forward must
cost at least as many FLOPs as the model has parameters, and every row is
printed against that ratio.

    uv run python results/campaign/measure_flops.py \\
        --models 20-aug/models/exotic/normal-new-ds/full-after-hpo/*_ft_seed42.pt \\
        --out results/campaign/flops_per_model.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The capture window every model in this project consumes.
SEQ_LEN = 10
PLAIN_CHANNELS = 4
RICH_CHANNELS = 18

#: FLOPs per multiply-accumulate: one multiply, one add.
FLOPS_PER_MAC = 2


def count_macs(model: torch.nn.Module, x: torch.Tensor) -> tuple[int, list[str]]:
    """MACs for one forward, from the shape each leaf module is called with."""
    macs = 0
    notes: list[str] = []
    handles = []

    def linear_hook(module: torch.nn.Linear, inputs, output) -> None:
        nonlocal macs
        rows = inputs[0].numel() // module.in_features
        macs += module.in_features * module.out_features * max(rows, 1)

    def conv_hook(module: torch.nn.Conv1d, inputs, output) -> None:
        nonlocal macs
        macs += (
            module.in_channels
            * module.out_channels
            * module.kernel_size[0]
            * output.shape[-1]
            * output.shape[0]
        )

    def norm_hook(module: torch.nn.LayerNorm, inputs, output) -> None:
        nonlocal macs
        macs += inputs[0].numel()  # one scale and shift per element

    def embed_hook(module: torch.nn.Embedding, inputs, output) -> None:
        nonlocal macs
        macs += output.numel()  # a gather, not a multiply

    for module in model.modules():
        hook = None
        if isinstance(module, torch.nn.Linear):
            hook = linear_hook
        elif isinstance(module, torch.nn.Conv1d):
            hook = conv_hook
        elif isinstance(module, torch.nn.LayerNorm):
            hook = norm_hook
        elif isinstance(module, torch.nn.Embedding):
            hook = embed_hook
        if hook is not None:
            handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    if macs == 0:
        notes.append("no countable leaf modules were called")
    return macs, notes


def torch_counter_flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    """torch's own count, kept for comparison rather than for the result."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x)
        # get_total_flops() disagrees with its own per-op breakdown on these
        # graphs, so the breakdown is summed instead. Both are wrong here; the
        # comparison is the point.
        return sum(int(v) for per in counter.get_flop_counts().values() for v in per.values())
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from inmotion.models import registry as reg
    from inmotion.models.adopt import resolve_model

    rows = []
    print(f"{'model':<24}{'MFLOP/infer':>13}{'params':>12}{'flop/param':>12}{'torch':>12}")
    for arg in args.models:
        spec = resolve_model(arg)
        model = reg.load(spec, strict=True).eval()
        channels = RICH_CHANNELS if spec.rich else PLAIN_CHANNELS
        x = torch.zeros(1, SEQ_LEN, channels)

        macs, notes = count_macs(model, x)
        flops = macs * FLOPS_PER_MAC
        params = sum(p.numel() for p in model.parameters())
        reference = torch_counter_flops(model, x)

        rows.append({"model": spec.name, "flops": flops, "params": params, "macs": macs})
        print(
            f"{spec.name:<24}{flops / 1e6:>12.3f}M{params:>12,}"
            f"{flops / params:>12.2f}{reference / 1e6:>11.3f}M"
        )
        for note in notes:
            print(f"    note: {note}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "flops", "params", "macs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} models)")
    print("flop/param must be > 1 for a dense forward; a row below that is a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
