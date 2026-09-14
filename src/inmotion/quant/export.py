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
from inmotion.models.adopt import resolve_model
from inmotion.models.wrappers import strip_prefix
from inmotion.quant.evaluate import BASELINE_DATASET, load_split
from inmotion.quant.model import _is_quantizable, apply, apply_cast
from inmotion.quant.packing import fold_bias_into_zero_point, pack
from inmotion.quant.spec import FLOAT_PRECISIONS
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


def _transformer_arch(model: torch.nn.Module) -> dict[str, object]:
    """Describe a JEPA transformer for a non-Python runtime.

    The registry's ``arch`` holds the *factory arguments*, and it is empty for
    models whose factory takes its defaults (``ts_jepa``, ``cf_jepa``). The
    runtime cannot load those without ``num_layers``/``nhead``, and it selects
    its tokenizer and tail from these fields, so they are read from the module
    that is actually being exported rather than from a name.

    Returns ``{}`` for models that are not one of the transformer families, so
    this stays callable from a generic exporter.
    """
    if not _is_transformer(model):
        return {}
    encoder = model.encoder
    layers = encoder.transformer.layers
    first = layers[0]
    arch: dict[str, object] = {
        "family": "transformer",
        "d_model": int(first.linear1.in_features),
        "nhead": int(first.self_attn.num_heads),
        "num_layers": len(layers),
        "tail": "proj_bn" if hasattr(encoder, "bn") else "layernorm",
        # T-JEPA prepends [REG] slots (dropped again before pooling) and projects
        # the tokens before the transformer; nothing else does. Both are recorded
        # so the runtime selects a code path instead of inferring one.
        "n_reg": int(getattr(encoder, "n_reg_tokens", 0)),
        "input_proj": hasattr(encoder, "input_proj"),
    }
    tokenizer = encoder.tokenizer
    if hasattr(tokenizer, "feat_idx_embed"):
        # T-JEPA: one token per input slot, scaled elementwise, plus a
        # feature-index embedding instead of a positional one.
        arch["tokenizer"] = "feature"
        arch["total_features"] = int(tokenizer.total_features)
        arch["n_tokens"] = arch["n_reg"] + arch["total_features"]
    elif hasattr(tokenizer, "proj"):
        arch["tokenizer"] = "patch"
        arch["patch_size"] = int(tokenizer.proj.kernel_size[0])
        arch["n_tokens"] = int(encoder.pos_embed.shape[1])
    else:
        arch["tokenizer"] = "linear"
        arch["n_tokens"] = int(encoder.pos_embed.shape[1])
    return arch


def _sigreg_arch(model: torch.nn.Module) -> dict[str, object]:
    """Describe the SIGReg CNN for a non-Python runtime.

    Read from the module, for the same reason `_transformer_arch` is: the runtime
    selects its code path from these fields and must not have to assume them.

    Returns ``{}`` for models that are not the SIGReg CNN.
    """
    inner = getattr(model, "m", model)  # `TupleWrap` nests the model under `m.`
    encoder = getattr(inner, "encoder", None)
    if encoder is None or not hasattr(encoder, "blocks"):
        return {}
    return {
        "family": "sigreg_cnn",
        "in_features": int(encoder.conv3.in_channels),
        "num_filters": int(encoder.bn.num_features),
        "num_blocks": len(encoder.blocks),
        "front_kernels": [
            int(encoder.conv3.kernel_size[0]),
            int(encoder.conv5.kernel_size[0]),
            int(encoder.conv7.kernel_size[0]),
        ],
        "block_kernels": [
            int(encoder.blocks[0].block[0].kernel_size[0]),
            int(encoder.blocks[0].block[3].kernel_size[0]),
        ],
        "latent_dim": int(inner.latent_proj.out_features),
        "num_classes": int(inner.classifier.out_features),
    }


def _is_transformer(model: torch.nn.Module) -> bool:
    """True for the JEPA classifiers, whose block stack `_transformer_arch` reads."""
    encoder = getattr(model, "encoder", None)
    return encoder is not None and hasattr(encoder, "transformer")


def dequantize_artifact(meta: dict, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The float weights an artifact stands for: `(u - zero_point) * scale`.

    The artifact's inverse, and the only place the stored bias is undone. The
    codes are stored biased by `2**(bits-1)` and the zero point folded to match,
    so the subtract cancels both.
    """
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
    model: torch.nn.Module,
    weights: dict[str, np.ndarray],
    prefix: str | None,
) -> dict[str, torch.Tensor]:
    """The model's OWN state dict, from artifact-shaped tensors.

    Three of the exporter's transforms have to be undone, and each is exact:

      - the name prefix (`TupleWrap` nests SIGReg under `m.`),
      - `_split_qkv`, which sliced `in_proj_weight` into q/k/v rows,
      - weight-norm folding, whose inverse is `g = ||w||`, `v = w / ||w||`,
        because `g * v / ||v||` returns `w` exactly.

    Buffers are included: BatchNorm's running statistics are not parameters, and
    an artifact without them loads into nothing usable.
    """
    lookup = {f"{prefix}{k}" if prefix else k: v for k, v in weights.items()}
    every = [(n, p) for n, p in model.named_parameters()] + [
        (n, b) for n, b in model.named_buffers()
    ]
    # The exporter drops training bookkeeping rather than shipping it, so the
    # rebuild drops it too or it looks like a missing tensor.
    every = [(n, t) for n, t in every if not any(s in n for s in _SKIP_BUFFERS)]
    known = {name for name, _ in every}

    state: dict[str, torch.Tensor] = {}
    for name, param in every:
        if name.endswith("self_attn.in_proj_weight"):
            head = name[: -len("in_proj_weight")]
            state[name] = torch.from_numpy(
                np.concatenate([lookup[f"{head}{t}.weight"] for t in ("q", "k", "v")], axis=0)
            )
        elif name.endswith("self_attn.in_proj_bias"):
            head = name[: -len("in_proj_bias")]
            state[name] = torch.from_numpy(
                np.concatenate([lookup[f"{head}{t}.bias"] for t in ("q", "k", "v")], axis=0)
            )
        elif name.endswith((".weight_g", ".weight_v")):
            continue  # rebuilt below, once the folded weight is known
        elif name in lookup:
            state[name] = torch.from_numpy(lookup[name])

    for name, param in model.named_parameters():
        if not name.endswith(".weight_g"):
            continue
        folded = lookup.get(name[: -len("weight_g")] + "weight")
        if folded is None:
            continue
        tensor = torch.from_numpy(folded)
        norm = tensor.norm(dim=0, keepdim=True).clamp(min=1e-12)
        state[name] = norm.reshape(param.shape)
        state[name[: -len("weight_g")] + "weight_v"] = tensor / norm

    missing = known - set(state)
    if missing:
        raise ValueError(f"could not rebuild {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    return state


def model_with_producer(model_name: str, producer: str, quant: str | None):
    """The model carrying a producer's quantization in its weights.

    The golden vectors must come from the artifact's own weights, or the parity
    check reports quantization error as a port bug. This lays the model out
    exactly as the artifact does — folded weight-norm, split QKV, stripped
    prefix — quantizes each weight with the producer, and puts the result back
    through `rebuild_state_dict`.

    `rebuild_state_dict` is needed despite per-channel quantization commuting
    with the QKV split (verified: identical codes and scales): weight-norm
    folding does NOT commute, and its inverse is what turns a folded weight back
    into a `weight_g`/`weight_v` pair.
    """
    from inmotion.quant.producers import PRODUCERS

    spec_obj = resolve_model(model_name)
    model = reg.load(spec_obj, strict=True).eval()
    implementation = PRODUCERS[producer]()
    scheme = quant or implementation.schemes[0]

    raw = _fold_weight_norm(model)
    _split_qkv(raw, model)
    if spec_obj.prefix:
        raw = strip_prefix(raw, spec_obj.prefix)

    for name, weight in list(raw.items()):
        if not _is_quantizable(name, torch.from_numpy(weight)):
            continue
        # `round_trip` rather than `quantize`: every producer can say what its
        # scheme reconstructs, including bitsandbytes, whose formats have no
        # artifact representation at all.
        raw[name] = implementation.round_trip(weight, scheme)

    model.load_state_dict(rebuild_state_dict(model, raw, spec_obj.prefix), strict=True)
    return model.eval()


def resolve_producer(producer: str, quant: str | None):
    """The per-weight quantizer and the `quant` metadata block for a producer.

    One code path serves every producer: they differ in how a weight is
    quantized, not in how an artifact is laid out. `in-house` goes through the
    model-level spec, the others are applied weight by weight because their
    module-level pipelines cannot run on these architectures.

    A scheme the runtime cannot read is refused here rather than written, with
    the measured reason, because an artifact that loads and computes the wrong
    thing is worse than one that does not exist.
    """
    from inmotion.quant.producers import (
        PRODUCERS,
        InHouseProducer,
        parse_in_house_scheme,
    )

    if producer == "in-house":
        spec = parse_in_house_scheme(quant) if quant else None
        if spec is None:
            return None, None, None
        implementation = InHouseProducer(spec)
        return (
            (lambda weight: implementation.quantize(weight, quant)),
            spec.to_dict(),
            quant,
        )

    if producer not in PRODUCERS:
        raise ValueError(f"unknown producer {producer!r}; known: {', '.join(sorted(PRODUCERS))}")
    implementation = PRODUCERS[producer]()
    scheme = quant or implementation.schemes[0]
    if scheme not in implementation.schemes:
        raise ValueError(
            f"{producer} has no scheme {scheme!r}; known: {', '.join(implementation.schemes)}"
        )
    if not implementation.expressible(scheme):
        raise ValueError(
            f"{producer} {scheme} cannot be written as a runtime artifact: its "
            f"codebook is not affine (residual "
            f"{implementation.residual(scheme):.4f} against tolerance "
            f"{implementation.scheme_meta(scheme)['affine_tolerance']}), and the "
            f"runtime dequantizes every scheme as (u - zero_point) * scale."
        )
    return (
        (lambda weight: implementation.quantize(weight, scheme)),
        implementation.scheme_meta(scheme),
        scheme,
    )


def export(
    model_name: str,
    out_dir: Path,
    *,
    quant: str | None = None,
    producer: str = "in-house",
    seed: int = 42,
) -> ExportReport:
    """Write ``model_name`` as a portable artifact directory."""
    spec_obj = resolve_model(model_name)
    model = reg.load(spec_obj, strict=True).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _fold_weight_norm(model)
    _split_qkv(raw, model)
    if spec_obj.prefix:
        # `TupleWrap` (SIGReg) nests the model under `m.`, so a raw state dict
        # would read `m.encoder...`. The artifact is the runtime's interface, so
        # it carries the model's own names; the registry already declares the
        # prefix, so nothing is renamed by hand.
        raw = strip_prefix(raw, spec_obj.prefix)

    quantize_weight, quant_meta, resolved_scheme = resolve_producer(producer, quant)
    tensors: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, object]] = {}

    quantized_count = 0
    for name in sorted(raw):
        array = raw[name]
        is_weight = _is_quantizable(name, torch.from_numpy(array))
        if quantize_weight is not None and is_weight:
            produced = quantize_weight(array)
            tensors[f"{name}.q"] = pack(
                produced.codes.ravel(), produced.bits, symmetric=produced.signed_codes
            )
            tensors[f"{name}.scale"] = produced.scale.astype(np.float32).ravel()
            tensors[f"{name}.zp"] = fold_bias_into_zero_point(
                produced.zero_point.ravel(),
                produced.bits,
                symmetric=produced.signed_codes,
            ).astype(np.int32)
            index[name] = {
                "quantized": True,
                # The *original* shape, as `TensorInfo.shape` promises. The codes
                # are the flattened quantization view (a Conv1d (out, in, k)
                # becomes (out, in*k) under per-group), and recording that made a
                # 3-D weight load as a matrix.
                "shape": list(array.shape),
                "bits": produced.bits,
                "granularity": produced.granularity.value,
                "group_size": (quant_meta or {}).get("group_size", 32),
                "axis": produced.axis,
                "n_values": int(produced.codes.size),
                "scale_shape": list(produced.scale.shape),
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

    # Class names come from the dataset, and the runtime needs them to say "BA"
    # rather than "class 2". Recorded here rather than left in golden.json,
    # which is a parity artifact and not part of the runtime's interface.
    classes: list[str] = []
    try:
        from inmotion.quant.evaluate import BASELINE_DATASET, load_split

        _x, _y, _idx, classes = load_split(model_name, resolve_path(BASELINE_DATASET), seed)
    except (FileNotFoundError, OSError):
        # An artifact is still useful without labels; the runtime falls back to
        # the index, so a missing dataset is not a reason to fail the export.
        classes = []

    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "model": spec_obj.name,
        "factory": spec_obj.factory,
        # Registry args first, then what the module actually is: the two
        # describe different things and the module wins on conflict. Only one of
        # the two descriptors ever returns anything.
        "arch": {**spec_obj.arch, **_transformer_arch(model), **_sigreg_arch(model)},
        "rich_features": spec_obj.rich,
        "seq_len": 10,
        "in_channels": 18 if spec_obj.rich else 4,
        "num_classes": 4,
        "classes": classes,
        "quant": quant_meta,
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
        mode=(f"{producer}/{resolved_scheme}" if producer != "in-house" else (quant or "fp32")),
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
    producer: str = "in-house",
    n_samples: int = 16,
    dataset: Path | None = None,
    seed: int = 42,
) -> Path:
    """Write input/output pairs for a non-Python runtime to reproduce.

    Inputs come from the real test split of the same dataset the accuracy
    harness uses (``evaluate.BASELINE_DATASET``) rather than a normal
    distribution, so the parity test exercises the value ranges the model
    actually sees and no second dataset creeps back into the quant path.

    ``quant`` must match the artifact's scheme. The golden outputs are produced by
    the *same* quantized weights the artifact contains, because the runtime
    dequantizes before the matmul: comparing a quantized artifact against
    unquantized outputs would measure quantization error, not port correctness,
    and would be reported as a parity failure.
    """
    spec_obj = resolve_model(model_name)
    path = dataset or resolve_path(BASELINE_DATASET)
    X, _y, test_idx, classes = load_split(model_name, path, seed)

    model = reg.load(spec_obj, strict=True).eval()
    if producer != "in-house":
        model = model_with_producer(model_name, producer, quant)
    elif quant is not None:
        if quant in FLOAT_PRECISIONS:
            model, _ = apply_cast(model, quant)
        else:
            model, _ = apply(model, resolve_spec(quant))
    chosen = test_idx[:n_samples]
    batch = torch.from_numpy(X[chosen])
    with torch.no_grad():
        logits = model(batch).numpy()

    payload = {
        "model": spec_obj.name,
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
