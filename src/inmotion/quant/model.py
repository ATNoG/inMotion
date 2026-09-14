"""Weight-only quantization of a model, and honest size accounting.

What gets quantized
-------------------
Only the *weights* of ``Linear`` and ``Conv1d``/``Conv2d`` layers — the tensors
with two or more dimensions. Everything else stays fp32:

- biases, LayerNorm/BatchNorm weights and running statistics are 1-D and a tiny
  share of the parameters; quantizing them costs accuracy and saves nothing;
- learned tokens and position embeddings (``mask_token``, ``query_tokens``,
  ``pos_embed``) are initialization state, not weights to be approximated.

Activations stay fp32. That is deliberate: activations peak at ~10 KB against
0.57-2.35 MB of weights, so keeping them fp32 costs nothing measurable and
removes the need for calibration data.

Accuracy measurement
--------------------
:func:`apply` replaces each weight with ``dequantize(quantize(w))``. Running the
model afterwards measures exactly what a device does, because the device also
reconstructs fp32 weights from codes before the matmul. No forward pass needs to
be rewritten, so the measurement cannot drift from the model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from inmotion.quant.packing import n_bytes_for
from inmotion.quant.reference import QuantizedTensor, dequantize, quantize
from inmotion.quant.spec import Granularity, QuantSpec


@dataclass
class TensorRecord:
    """One quantized weight tensor."""

    name: str
    shape: tuple[int, ...]
    n_values: int
    payload_bytes: int
    """Packed code bytes."""
    metadata_bytes: int
    """float32 scales plus int8 zero points."""
    reconstruction_error: float
    """max |w - dequantize(quantize(w))|, for reporting."""

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes


@dataclass
class QuantReport:
    """Size and error summary for one (model, spec) pair."""

    spec_label: str
    bits: int
    records: list[TensorRecord] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    fp32_weight_bytes: int = 0
    skipped_param_bytes: int = 0
    """fp32 bytes of parameters left unquantized (norms, biases, tokens)."""
    buffer_bytes: int = 0
    """fp32 bytes of buffers (BatchNorm/LayerNorm running statistics)."""

    @property
    def quantized_bytes(self) -> int:
        return sum(r.total_bytes for r in self.records)

    @property
    def packed_bytes(self) -> int:
        return sum(r.payload_bytes for r in self.records)

    @property
    def unquantized_weight_bytes(self) -> int:
        return sum(r.n_values * 4 for r in self.records)

    @property
    def compression(self) -> float:
        """fp32 weight bytes divided by quantized weight bytes."""
        if not self.quantized_bytes:
            return 1.0
        return self.unquantized_weight_bytes / self.quantized_bytes

    @property
    def max_error(self) -> float:
        return max((r.reconstruction_error for r in self.records), default=0.0)

    @property
    def total_bytes(self) -> int:
        """What the device must hold: quantized weights plus everything else.

        This is the number that decides whether a model fits in PSRAM, so it
        counts the unquantized parameters and buffers too rather than reporting
        only the part quantization touched.
        """
        return self.quantized_bytes + self.skipped_param_bytes + self.buffer_bytes

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.spec_label,
            "bits": self.bits,
            "tensors": len(self.records),
            "skipped_tensors": len(self.skipped),
            "fp32_weight_MB": round(self.unquantized_weight_bytes / 1e6, 4),
            "quantized_MB": round(self.quantized_bytes / 1e6, 4),
            "compression": round(self.compression, 3),
            "max_weight_error": round(self.max_error, 6),
            "skipped_MB": round(self.skipped_param_bytes / 1e6, 4),
            "buffers_MB": round(self.buffer_bytes / 1e6, 4),
            "total_MB": round(self.total_bytes / 1e6, 4),
        }


#: Parameters that are initialization state rather than weight matrices, so
#: quantizing them saves space but changes what the model *is*.
_NOT_A_WEIGHT_MATRIX = (
    "pos_embed",
    "position",
    "mask_token",
    "query_token",
    "future_pos",
    "ctx_pos",
)


def _is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """True for a weight *matrix*, whatever PyTorch happens to call it.

    The rule is shape- and role-based rather than name-suffix-based. Matching
    only ``.weight`` silently skipped ``MultiheadAttention.in_proj_weight``,
    which is the packed QKV projection: 196,608 parameters, 35% of `lejepa`, and
    a dense matrix that is exactly what weight-only quantization is for.
    """
    if tensor.ndim < 2:
        return False
    lowered = name.lower()
    return not any(token in lowered for token in _NOT_A_WEIGHT_MATRIX)


def _flatten_for_grouping(w: np.ndarray, spec: QuantSpec) -> np.ndarray:
    """Give per-group a 2-D (rows, cols) view so grouping spans the whole row.

    A ``Conv1d`` weight is ``(out, in, k)``; grouping along ``k`` alone would
    produce groups of three, so the trailing axes are flattened into one.
    """
    if spec.granularity is Granularity.PER_GROUP and w.ndim > 2:
        return w.reshape(w.shape[0], -1)
    return w


def quantize_tensor(w: torch.Tensor, spec: QuantSpec) -> tuple[QuantizedTensor, np.ndarray]:
    """Quantize one weight tensor.

    Returns the :class:`QuantizedTensor` (which refers to the possibly-flattened
    view used for grouping) and the reconstructed tensor in the original shape.
    The two are returned together because reshaping the metadata independently of
    the codes is a bug: per-group codes are flattened but ``scale`` still
    describes the flattened row layout, so the original shape must be restored on
    the *output*, never by rewriting ``qt.shape`` or ``qt.q``.
    """
    array = w.detach().cpu().numpy().astype(np.float32)
    flat = _flatten_for_grouping(array, spec)
    qt = quantize(flat, spec)
    rebuilt = dequantize(qt).reshape(array.shape).astype(np.float32)
    return qt, rebuilt


def apply(
    model: nn.Module, spec: QuantSpec, *, in_place: bool = False
) -> tuple[nn.Module, QuantReport]:
    """Return a model whose quantizable weights have been quantized and rebuilt.

    The returned model is numerically what a device computes, so evaluating it
    after this call is the accuracy measurement. Set ``in_place=False`` (the
    default) to keep the original intact.
    """
    target = model if in_place else copy.deepcopy(model)
    report = QuantReport(spec_label=spec.label, bits=spec.bits)
    report.buffer_bytes = sum(int(b.numel()) * 4 for b in target.buffers())

    for name, param in list(target.named_parameters()):
        if not _is_quantizable(name, param):
            reason = (
                "1-D (norm/bias)" if param.ndim < 2 else "initialization state, not a weight matrix"
            )
            report.skipped[name] = reason
            report.skipped_param_bytes += int(param.numel()) * 4
            continue

        qt, rebuilt_np = quantize_tensor(param, spec)
        rebuilt = torch.from_numpy(rebuilt_np).to(param.dtype)
        error = float((rebuilt - param.detach().cpu()).abs().max())

        record = TensorRecord(
            name=name,
            shape=tuple(int(d) for d in param.shape),
            n_values=int(param.numel()),
            payload_bytes=n_bytes_for(int(param.numel()), spec.bits),
            metadata_bytes=4 * int(qt.scale.size) + int(qt.zero_point.size),
            reconstruction_error=error,
        )
        report.records.append(record)
        report.fp32_weight_bytes += record.n_values * 4

        with torch.no_grad():
            param.copy_(rebuilt.to(param.device))

    return target, report


#: Cast-only precisions. These are dtype round-trips, not affine quantization:
#: the weights are rounded to the narrower float format and read back.
CAST_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

#: Storage width of each cast, in bytes. Kept beside the dtypes so a new cast
#: cannot be added without stating what it costs on disk.
CAST_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def apply_cast(
    model: nn.Module, precision: str, *, in_place: bool = False
) -> tuple[nn.Module, QuantReport]:
    """Round every parameter and buffer to a narrower float format.

    Included so the accuracy and size curves can show bf16/fp16 alongside the
    integer schemes on the same axes. The storage cost is 4 bytes per value for
    fp32, 2 for fp16/bf16.
    """
    if precision not in CAST_DTYPES:
        raise KeyError(f"unknown precision {precision!r}; known: {sorted(CAST_DTYPES)}")
    dtype = CAST_DTYPES[precision]
    width = CAST_BYTES[precision]

    target = model if in_place else copy.deepcopy(model)
    report = QuantReport(spec_label=precision, bits=width * 8)

    with torch.no_grad():
        for name, param in target.named_parameters():
            rounded = param.detach().to(dtype).to(param.dtype)
            error = float((rounded.cpu() - param.detach().cpu()).abs().max())
            if _is_quantizable(name, param):
                report.records.append(
                    TensorRecord(
                        name=name,
                        shape=tuple(int(d) for d in param.shape),
                        n_values=int(param.numel()),
                        payload_bytes=int(param.numel()) * width,
                        metadata_bytes=0,
                        reconstruction_error=error,
                    )
                )
                report.fp32_weight_bytes += int(param.numel()) * 4
            else:
                report.skipped_param_bytes += int(param.numel()) * width
            param.copy_(rounded)
        report.buffer_bytes = sum(int(b.numel()) for b in target.buffers()) * width
        for buf in target.buffers():
            buf.copy_(buf.detach().to(dtype).to(buf.dtype))

    return target, report
