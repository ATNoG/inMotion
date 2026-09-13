# Rust port

Inference outside Python, verified against PyTorch rather than assumed correct.

```
rust/
  inmotion-core/    kernels, artifact reader, the model
  inmotion-cli/     `inmotion-infer` — describe / predict / check
```

## Why not ONNX Runtime

The graphs are small and fixed-shape, and the deployable targets are two baseline
x86-64 builds. A flat tensor bag plus a small JSON description is lighter and
easier to audit than an ONNX runtime, and the safetensors reader is short enough
to write here rather than depend on. (The ESP32 demo is separate C firmware under
`firmware/`; it does not share this crate.)

## Verifying it

A port that is not checked against the original is worthless, so the Python
exporter writes golden input/output pairs **from the same quantized weights the
artifact contains**, and this crate must reproduce them:

```bash
# Python side
uv run inmotion quantize export lejepa --out /tmp/art --format q4

# Rust side
cd rust
cargo run --release -p inmotion-cli -- check --artifact /tmp/art --golden /tmp/art
```

`tests/test_quant_export.py::test_rust_port_reproduces_pytorch` runs exactly that
for fp32, int8 and q4.

## Measured parity

Max absolute deviation from PyTorch over 16 real test windows, tolerance 2e-4,
measured with `inmotion-infer check` on freshly exported artifacts:

| model | fp32 | int8 | q4 |
|---|---:|---:|---:|
| lejepa | 4.1e-6 | 4.2e-6 | 3.3e-6 |
| t_jepa | 2.6e-6 | 4.1e-6 | 6.2e-6 |
| ts_jepa | 2.7e-6 | 2.9e-6 | 5.0e-6 |
| cf_jepa | 2.2e-6 | 3.0e-6 | 1.9e-6 |
| sigreg | 8.3e-7 | 8.3e-7 | 9.5e-7 |

Every case keeps more than 30x of headroom. `sigreg` was re-measured after the
dataset.csv retrain (`results/perf/README.md` records the pre-retrain run used for
the kernel before/after A/B, and `results/perf/parity_before_after.csv` has both).

## Scope

All five reported families are served by one shared stack:

* `lejepa`, `t_jepa`, `ts_jepa` and `cf_jepa` are the same pre-norm transformer
  plus attention pool. They differ only in the tokenizer — a per-timestep linear,
  a `Conv1d` patch projection, or T-JEPA's per-slot scale with a feature-index
  embedding — and in the stage between the transformer and the pool (a final
  `LayerNorm`, or LeJEPA's projection plus `BatchNorm1d`). T-JEPA additionally
  prepends `[REG]` slots and projects before the transformer.
* `sigreg` is a multi-scale CNN with a Gaussian latent bottleneck. It shares
  `ops.rs` and nothing else, and is deliberately not routed through the
  transformer path.

`arch.family` in the artifact selects the graph, followed by the fields that
graph validates against the tensor index (`tokenizer`, `tail`, `n_reg`,
`input_proj`, `num_filters`, `num_blocks`, ...). The runtime never infers
architecture, and an artifact whose declared architecture disagrees with its
weights is rejected at load rather than at inference.

## Bugs the parity check caught

Both were silent in Rust and only visible against the reference:

1. `proj` was transposed into `BatchNorm1d` by **reinterpreting memory** rather
   than transposing it, and the result was reinterpreted again on the way out.
   A reshape is not a transpose, and it reordered every element.
2. Attention pooling applies `softmax(dim=1)` over the **timestep** axis, not the
   trailing singleton. Using the usual last-axis softmax returned all ones and
   silently produced a plain mean.

## Platform notes

Two targets are supported and measured: baseline `x86_64-unknown-linux-gnu` and
baseline `x86_64-unknown-linux-musl` (no `target-cpu`, no AVX/AVX2/FMA), the
latter being what the wave-router J3455 runs. The crate is `std`-linked; only the
transcendentals come from elsewhere.

`libm` supplies `erf`/`exp`/`sqrt` so the arithmetic is identical on every target
and every libc instead of drifting with the platform's libm — that is what makes
the parity numbers above reproducible rather than approximate. BatchNorm is folded
into two fused multiply-adds per element at load time.
