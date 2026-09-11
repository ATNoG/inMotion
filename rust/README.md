# Rust port

Inference outside Python, verified against PyTorch rather than assumed correct.

```
rust/
  inmotion-core/    kernels, artifact reader, the model
  inmotion-cli/     `inmotion-infer` — describe / predict / check
```

## Why not ONNX Runtime

The graph is small and fixed-shape, and the target includes an ESP32-S3. A flat
tensor bag plus a small JSON description is lighter and easier to audit than an
ONNX runtime, and the safetensors reader is short enough to write here rather
than depend on.

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

| precision | max abs deviation on 16 real test windows | tolerance |
|---|---:|---:|
| fp32 | 2.9e-6 | 2e-4 |
| int8 | 3.5e-6 | 2e-4 |
| q4 | 3.2e-6 | 2e-4 |

## Scope

Only `lejepa` is ported. The other three reported JEPA models have structurally
different encoders — T-JEPA has a feature-index tokenizer, TS-JEPA and CF-JEPA are
patch-based with a `context_encoder` — so each needs its own `forward`, sharing
the ops in `ops.rs`. The artifact format and the parity harness already support
them; only the model module is missing.

## Bugs the parity check caught

Both were silent in Rust and only visible against the reference:

1. `proj` was transposed into `BatchNorm1d` by **reinterpreting memory** rather
   than transposing it, and the result was reinterpreted again on the way out.
   A reshape is not a transpose, and it reordered every element.
2. Attention pooling applies `softmax(dim=1)` over the **timestep** axis, not the
   trailing singleton. Using the usual last-axis softmax returned all ones and
   silently produced a plain mean.

## Platform notes

`libm` supplies `erf`/`exp`/`sqrt` so the crate can build without `std` math for a
microcontroller target. BatchNorm is folded into two fused multiply-adds per
element at load time.
