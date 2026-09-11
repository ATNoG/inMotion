"""Export a model to a portable artifact a non-Python runtime can execute.

The artifact is a directory:

    model.safetensors   tensors, quantized ones packed
    meta.json           architecture, quantization spec, tensor index

Why not ONNX
------------
ONNX Runtime is a poor fit for an MCU and heavy even for an x86 router, and the
graphs here are small and fixed-shape. A flat tensor bag plus a small JSON
description is easier to audit and is what embedded runtimes actually do.

Export-time transformations
---------------------------
Two source conventions are resolved here rather than in every Rust reader:

``in_proj_weight`` is split into separate Q, K and V tensors. PyTorch packs them
into one ``(3*d_model, d_model)`` matrix; splitting means the Rust side does not
have to know that packing convention.

``weight_norm`` is folded into an ordinary weight. It is what broke FX tracing
and ONNX export for ``tcn``/``deep_stack``, and folding removes the problem
instead of working around it.

Two export modes
----------------
``dtype="fp32"`` writes plain float weights. Use it to verify that the *port* is
correct before introducing quantization, so a mismatch can only be one of the two
things.

``quant=QuantSpec`` writes packed integer codes plus scales and zero points. The
Rust reader dequantizes exactly as :mod:`inmotion.quant.reference` does, so
parity holds by construction and is asserted by the golden-vector tests.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from inmotion.data.paths import resolve as resolve_path
from inmotion.models import registry as reg
from inmotion.quant.evaluate import load_split
from inmotion.quant.model import _is_quantizable, apply, apply_cast, quantize_tensor
from inmotion.quant.packing import fold_bias_into_zero_point, pack
from inmotion.quant.spec import FLOAT_PRECISIONS, QuantSpec
from inmotion.quant.spec import resolve as resolve_spec

ARTIFACT_VERSION = 1
WEIGHTS_NAME = "model.safetensors"
META_NAME = "meta.json"


@dataclass
class ExportReport:
    """What was written and how big it is."""

    path: Path
    tensors: int
    quantized_tensors: int
    total_bytes: int
    weights_bytes: int
    mode: str

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "mode": self.mode,
            "tensors": self.tensors,
            "quantized_tensors": self.quantized_tensors,
            "total_MB": round(self.total_bytes / 1e6, 4),
            "weights_MB": round(self.weights_bytes / 1e6, 4),
        }


def _split_qkv(state: dict[str, np.ndarray], model: torch.nn.Module) -> None:
    """Replace packed ``in_proj_weight``/``in_proj_bias`` with separate Q, K, V."""
    for name, param in list(model.named_parameters()):
        if not name.endswith("self_attn.in_proj_weight"):
            continue
        prefix = name[: -len("in_proj_weight")]
        d_model = param.shape[1]
        w = param.detach().cpu().numpy().astype(np.float32)
        for i, tag in enumerate(("q", "k", "v")):
            state[f"{prefix}{tag}.weight"] = w[i * d_model : (i + 1) * d_model].copy()
        state.pop(name, None)

        bias_name = f"{prefix}in_proj_bias"
        bias_param = dict(model.named_parameters()).get(bias_name)
        if bias_param is not None:
            b = bias_param.detach().cpu().numpy().astype(np.float32)
            for i, tag in enumerate(("q", "k", "v")):
                state[f"{prefix}{tag}.bias"] = b[i * d_model : (i + 1) * d_model].copy()
            state.pop(bias_name, None)


#: Buffers that are training bookkeeping rather than inference state.
_SKIP_BUFFERS = ("num_batches_tracked",)


def _fold_weight_norm(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Collect every tensor inference needs: parameters *and* buffers.

    Buffers matter here. BatchNorm's `running_mean`/`running_var` are buffers, not
    parameters, so walking only `named_parameters` silently omits the statistics
    the model normalises with at inference — the exporter produced an artifact the
    Rust reader could not load.
    """
    out: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if "parametrizations.weight" in name:
            continue
        value = param.detach().cpu().numpy().astype(np.float32)
        out[name] = value

    for name, buf in model.named_buffers():
        if any(skip in name for skip in _SKIP_BUFFERS):
            continue
        out[name] = buf.detach().cpu().numpy().astype(np.float32)

    # weight_norm stores weight_g/weight_v; the effective weight is g * v/||v||.
    for name in list(out):
        if not name.endswith("weight_g"):
            continue
        base = name[: -len("weight_g")]
        v = out.pop(f"{base}weight_v", None)
        g = out.pop(name, None)
        if v is None or g is None:
            continue
        dims = tuple(range(1, v.ndim))
        norm = np.sqrt((v**2).sum(axis=dims, keepdims=True)) + 1e-12
        out[f"{base}weight"] = (g * v / norm).astype(np.float32)
    return out


def export(
    model_name: str,
    out_dir: Path,
    *,
    quant: str | None = None,
    seed: int = 42,
) -> ExportReport:
    """Write ``model_name`` as a portable artifact directory."""
    spec_obj = reg.by_name(model_name)
    model = reg.load(spec_obj, strict=True).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _fold_weight_norm(model)
    _split_qkv(raw, model)

    quant_spec: QuantSpec | None = resolve_spec(quant) if quant else None
    tensors: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, object]] = {}

    quantized_count = 0
    for name in sorted(raw):
        array = raw[name]
        is_weight = _is_quantizable(name, torch.from_numpy(array))
        if quant_spec is not None and is_weight:
            qt, _ = quantize_tensor(torch.from_numpy(array), quant_spec)
            tensors[f"{name}.q"] = pack(qt.q.ravel(), quant_spec.bits)
            tensors[f"{name}.scale"] = qt.scale.astype(np.float32).ravel()
            tensors[f"{name}.zp"] = fold_bias_into_zero_point(
                qt.zero_point.ravel(), quant_spec.bits
            ).astype(np.int32)
            index[name] = {
                "quantized": True,
                "shape": list(qt.shape),
                "bits": quant_spec.bits,
                "granularity": quant_spec.granularity.value,
                "group_size": quant_spec.group_size,
                "axis": quant_spec.axis,
                "n_values": int(qt.q.size),
                "scale_shape": list(qt.scale.shape),
            }
            quantized_count += 1
        else:
            tensors[name] = array.astype(np.float32)
            index[name] = {
                "quantized": False,
                "shape": list(array.shape),
                "dtype": "fp32",
            }

    weights_path = out_dir / WEIGHTS_NAME
    save_file(tensors, str(weights_path))

    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model_name,
        "factory": spec_obj.factory,
        "arch": spec_obj.arch,
        "rich_features": spec_obj.rich,
        "seq_len": 10,
        "in_channels": 18 if spec_obj.rich else 4,
        "num_classes": 4,
        "quant": quant_spec.to_dict() if quant_spec else None,
        "eps": {"layernorm": 1e-5, "batchnorm": 1e-5},
        "tensors": index,
        "export": {"seed": seed, "source_git": _git_sha()},
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    total = weights_path.stat().st_size + (out_dir / META_NAME).stat().st_size
    return ExportReport(
        path=out_dir,
        tensors=len(index),
        quantized_tensors=quantized_count,
        total_bytes=total,
        weights_bytes=weights_path.stat().st_size,
        mode=quant or "fp32",
    )


def _git_sha() -> str | None:
    """Best-effort current commit, so an artifact can be traced to its code."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


GOLDEN_NAME = "golden.json"


def write_golden(
    model_name: str,
    out_dir: Path,
    *,
    quant: str | None = None,
    n_samples: int = 16,
    dataset: Path | None = None,
    seed: int = 42,
) -> Path:
    """Write input/output pairs for a non-Python runtime to reproduce.

    Inputs come from the real test split rather than a normal distribution, so
    the parity test exercises the value ranges the model actually sees.

    ``quant`` must match the artifact's scheme. The golden outputs are produced by
    the *same* quantized weights the artifact contains, because the runtime
    dequantizes before the matmul: comparing a quantized artifact against
    unquantized outputs would measure quantization error, not port correctness,
    and would be reported as a parity failure.
    """
    spec_obj = reg.by_name(model_name)
    path = dataset or resolve_path("dataset.icaisf.csv")
    X, _y, test_idx, classes = load_split(model_name, path, seed)

    model = reg.load(spec_obj, strict=True).eval()
    if quant is not None:
        if quant in FLOAT_PRECISIONS:
            model, _ = apply_cast(model, quant)
        else:
            model, _ = apply(model, resolve_spec(quant))
    chosen = test_idx[:n_samples]
    batch = torch.from_numpy(X[chosen])
    with torch.no_grad():
        logits = model(batch).numpy()

    payload = {
        "model": model_name,
        "seq_len": int(X.shape[1]),
        "in_channels": int(X.shape[2]),
        "num_classes": int(logits.shape[1]),
        "classes": classes,
        "dataset": path.name,
        "tol": 2e-4,
        "quant": quant,
        "cases": [
            {"input": batch[i].numpy().tolist(), "logits": logits[i].tolist()}
            for i in range(len(chosen))
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / GOLDEN_NAME
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path
