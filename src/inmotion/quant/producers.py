"""The quantizers, behind one interface.

Three producers quantize these models and they are not interchangeable:

  in-house      this repository's affine scheme, per-tensor/channel/group,
                symmetric or asymmetric. Model-level: one spec governs every
                weight, and `reference.quantize` IS the specification.
  torch-ao      torch.ao's per-channel symmetric int8. Per-tensor, because
                `quantize_dynamic` on a whole model raises inside
                `nn.TransformerEncoderLayer.forward` on every architecture here.
  bitsandbytes  nf4, fp4 and blockwise int8. Per-tensor and NOT affine, so the
                runtime cannot read it — see `expressible`.

A producer answers three questions per weight: what are the codes, what scale and
zero point do they carry, and can the Rust runtime read the result at all. The
artifact writer in `export.py` handles the packing and the bias; a producer deals
only in signed codes.

Why `expressible` is measured, not declared: the runtime dequantizes every scheme
as one affine map `(u - zp) * scale`. bitsandbytes' 8-bit path looks like plain
int8 and is not — its 256-entry codebook is non-uniformly spaced and leaves a 62%
residual against the best straight line through it. A producer that claimed
otherwise would produce artifacts the runtime silently misreads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from inmotion.quant.spec import Granularity, QuantSpec

#: How many bits the sentinel-observing helpers below need, and the tolerance at
#: which a codebook stops being a straight line.
AFFINE_TOLERANCE = 0.01


@dataclass(frozen=True)
class ProducedWeight:
    """One weight as a producer left it, ready for the artifact writer.

    ``codes`` are SIGNED, in the weight's original shape. The writer applies
    whatever bias the scheme needs, so a producer never has to know that a
    symmetric 8-bit code is stored as `q + 128`.
    """

    codes: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    bits: int
    granularity: Granularity
    axis: int
    #: Whether the codes are STORED biased by `2**(bits-1)`. This is not the
    #: same question as whether the scheme is symmetric: torch's qint8-affine
    #: emits signed codes with a non-zero zero point, so it stores biased while
    #: its scheme is asymmetric. The writer needs the storage answer, and the
    #: metadata carries the scheme answer separately.
    signed_codes: bool


class Producer(Protocol):
    """What the export path needs from a quantizer."""

    name: str
    schemes: tuple[str, ...]

    def expressible(self, scheme: str) -> bool:
        """Whether the Rust runtime can load an artifact of this scheme."""

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        """The `quant` block for meta.json."""

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        """Quantize one weight."""


# --------------------------------------------------------------------------
# in-house
# --------------------------------------------------------------------------


def parse_in_house_scheme(label: str) -> QuantSpec:
    """Parse `<bits>[_<granularity>][_g<group>][_asym]` into a spec.

    The presets (`int8`, `q1`..`q8`) are the short forms of this grammar, so
    `q4` and `q4_pg32` are the same scheme and the whole space is reachable
    without enumerating names by hand:

        q4            per-group 32, symmetric (the preset)
        q4_pc         per-channel symmetric
        q4_pt         per-tensor symmetric
        q4_pg128      per-group 128
        q4_asym       per-group 32, asymmetric
        int8_pt       per-tensor 8-bit
    """
    from inmotion.quant.spec import resolve as resolve_spec

    if label in PRESET_LIKE:
        return resolve_spec(label)

    parts = label.split("_")
    head = parts[0]
    if head.startswith("q") and head[1:].isdigit():
        bits = int(head[1:])
    elif head == "int8":
        bits = 8
    else:
        raise ValueError(
            f"unknown in-house scheme {label!r}; use a preset (int8, q1..q8) or "
            f"the grammar q<bits>[_pt|_pc|_pg<N>][_asym]"
        )

    granularity = Granularity.PER_GROUP
    group_size = 32
    symmetric = True
    for part in parts[1:]:
        if part == "pt":
            granularity = Granularity.PER_TENSOR
        elif part == "pc":
            granularity = Granularity.PER_CHANNEL
        elif part.startswith("pg"):
            granularity = Granularity.PER_GROUP
            group_size = int(part[2:])
        elif part == "asym":
            symmetric = False
        elif part.isdigit() and head == "int8":
            continue
        else:
            raise ValueError(f"unknown scheme modifier {part!r} in {label!r}")

    return QuantSpec(
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        symmetric=symmetric,
    )


#: Names `parse_in_house_scheme` defers to `spec.resolve` for.
PRESET_LIKE = ("int8", "q1", "q2", "q3", "q4", "q5", "q6", "q8")


class InHouseProducer:
    """This repository's own scheme, through `reference.quantize`."""

    name = "in-house"

    #: The presets, plus the grammar's own shapes. `schemes()` enumerates the
    #: cross product the CLI can list; any grammar string is accepted.
    schemes = ("int8", "q1", "q2", "q3", "q4", "q5", "q6", "q8")

    def __init__(self, spec: QuantSpec) -> None:
        self.spec = spec

    def expressible(self, scheme: str) -> bool:
        return True

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        return self.spec.to_dict()

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        # `quantize_tensor`, not `reference.quantize`: for per-group schemes on a
        # 3-D weight the tensor is flattened first, and calling the reference
        # directly would silently group a Conv1d differently.
        import torch

        from inmotion.quant.model import quantize_tensor

        qt, _rebuilt = quantize_tensor(torch.from_numpy(weight.astype(np.float32)), self.spec)
        return ProducedWeight(
            codes=qt.q,
            scale=qt.scale.astype(np.float32),
            zero_point=qt.zero_point,
            bits=self.spec.bits,
            granularity=self.spec.granularity,
            axis=self.spec.axis,
            signed_codes=self.spec.symmetric,
        )

    def round_trip(self, weight: np.ndarray, scheme: str) -> np.ndarray:
        """The weight as the artifact would reconstruct it.

        Taken from `quantize_tensor`, which returns the reconstruction: for a
        per-group scheme the codes are the flattened view while the scale is in
        the reduced shape, so re-deriving it here would need the group broadcast
        and that is one more place to get it wrong.
        """
        import torch

        from inmotion.quant.model import quantize_tensor

        _qt, rebuilt = quantize_tensor(torch.from_numpy(weight.astype(np.float32)), self.spec)
        return rebuilt.reshape(weight.shape)


# --------------------------------------------------------------------------
# torch.ao
# --------------------------------------------------------------------------


def affine_residual(code: np.ndarray) -> float:
    """How far a codebook is from a straight line through its own indices.

    This is the test of whether the runtime can express a scheme, and it is run
    rather than assumed. Fitting the codebook is the right object: it is exactly
    what the runtime has to reproduce, and it works the same for a 16-entry 4-bit
    book and a 256-entry 8-bit one.
    """
    values = np.asarray(code, dtype=np.float64).ravel()
    if values.size < 2:
        return 1.0
    index = np.arange(values.size, dtype=np.float64)
    design = np.vstack([index, np.ones_like(index)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, values, rcond=None)
    span = float(np.abs(values).max()) or 1.0
    return float(np.abs(values - (slope * index + intercept)).max() / span)


class TorchAOProducer:
    """torch.ao's quantizers, weight by weight.

    Per tensor because the module-level pipeline cannot run: `quantize_dynamic`
    raises inside `nn.TransformerEncoderLayer.forward` on every architecture
    here. The numerics are still torch's — its observers pick the scale and zero
    point, its `quantize_per_tensor`/`quantize_per_channel` produce the codes.

    The scheme names are the cross product torch actually supports, read from
    `torch.quantize_per_tensor`'s accepted dtypes and torch's own qscheme
    constants rather than assumed:

        qint8_pt_sym      qint8_pt_affine
        qint8_pc_sym      qint8_pc_affine
        quint8_pt_affine  quint8_pc_affine

    `qint32` is accepted by both calls and is not a quantization — it is int32
    storage — so it is deliberately absent.
    """

    name = "torch-ao"

    schemes = (
        "qint8_pt_sym",
        "qint8_pt_affine",
        "qint8_pc_sym",
        "qint8_pc_affine",
        "quint8_pt_affine",
        "quint8_pc_affine",
    )

    def expressible(self, scheme: str) -> bool:
        return True

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        dtype, granularity, qscheme = self._parse(scheme)
        span = "tensor" if granularity is Granularity.PER_TENSOR else "channel"
        symmetry = "symmetric" if qscheme == "sym" else "affine"
        return {
            "name": f"torchao_{scheme}",
            "bits": 8,
            "granularity": granularity.value,
            "group_size": 32,
            "axis": 0,
            "symmetric": qscheme == "sym",
            "rounding": "half_to_even",
            "torch_dtype": dtype,
            "torch_qscheme": f"per_{span}_{symmetry}",
        }

    @staticmethod
    def _parse(scheme: str) -> tuple[str, Granularity, str]:
        parts = scheme.split("_")
        if len(parts) != 3 or parts[0] not in ("qint8", "quint8"):
            raise ValueError(
                f"unknown torch-ao scheme {scheme!r}; use {{qint8,quint8}}_{{pt,pc}}_{{sym,affine}}"
            )
        dtype, where, qscheme = parts
        if where not in ("pt", "pc") or qscheme not in ("sym", "affine"):
            raise ValueError(f"unknown torch-ao scheme {scheme!r}")
        granularity = Granularity.PER_TENSOR if where == "pt" else Granularity.PER_CHANNEL
        return dtype, granularity, qscheme

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        import torch
        from torch.ao.quantization.observer import (
            MinMaxObserver,
            PerChannelMinMaxObserver,
        )

        dtype_name, granularity, qscheme = self._parse(scheme)
        torch_dtype = getattr(torch, dtype_name)
        tensor = torch.from_numpy(weight.astype(np.float32))
        per_channel = granularity is Granularity.PER_CHANNEL
        torch_qscheme = (
            torch.per_channel_symmetric
            if per_channel and qscheme == "sym"
            else torch.per_channel_affine
            if per_channel
            else torch.per_tensor_symmetric
            if qscheme == "sym"
            else torch.per_tensor_affine
        )

        if per_channel:
            observer = PerChannelMinMaxObserver(ch_axis=0, dtype=torch_dtype, qscheme=torch_qscheme)
        else:
            observer = MinMaxObserver(dtype=torch_dtype, qscheme=torch_qscheme)
        observer(tensor)
        scale, zero_point = observer.calculate_qparams()

        if per_channel:
            quantized = torch.quantize_per_channel(
                tensor, scale, zero_point.to(torch.int32), 0, torch_dtype
            )
            reduced = [1] * weight.ndim
            reduced[0] = weight.shape[0]
            scale_array = scale.to(torch.float32).numpy().astype(np.float32).reshape(reduced)
            zp_array = zero_point.to(torch.int32).numpy().astype(np.int32).reshape(reduced)
        else:
            quantized = torch.quantize_per_tensor(
                tensor, scale, zero_point.to(torch.int32), torch_dtype
            )
            scale_array = np.asarray(float(scale), dtype=np.float32)
            zp_array = np.asarray(int(zero_point), dtype=np.int32)

        return ProducedWeight(
            codes=quantized.int_repr().numpy(),
            scale=scale_array,
            zero_point=zp_array,
            bits=8,
            granularity=granularity,
            axis=0,
            # qint8 emits signed codes, quint8 unsigned. The writer biases the
            # signed ones; the scheme's own symmetry is a separate question and
            # lives in the metadata.
            signed_codes=dtype_name == "qint8",
        )

    def round_trip(self, weight: np.ndarray, scheme: str) -> np.ndarray:
        """The weight as the artifact would reconstruct it."""
        produced = self.quantize(weight, scheme)
        scale = produced.scale.astype(np.float32)
        zero_point = produced.zero_point.astype(np.float32)
        return ((produced.codes.astype(np.float32) - zero_point) * scale).reshape(weight.shape)


# --------------------------------------------------------------------------
# bitsandbytes
# --------------------------------------------------------------------------


class BitsAndBytesProducer:
    """bitsandbytes' weight formats.

    The scheme names come from its own functional signatures, not from memory:
    `quantize_4bit(..., blocksize, compress_statistics, quant_type)` and
    `quantize_blockwise(..., blocksize, nested)`.

        nf4, fp4              4-bit, one absmax per block of 64
        nf4_dq, fp4_dq        the same with double quantization of the absmax
        nf4_b128, fp4_b256    a different block size
        int8                  blockwise 8-bit, one absmax per row
        int8_dq               the same with nested (double) quantization
        int8_b128             a different block size

    An optional dependency: the wheel is CUDA-sized and this is a CPU inference
    project whose point is artifact size, so it is installed on demand rather
    than declared. Everything here imports it lazily and says so if absent.
    """

    name = "bitsandbytes"
    schemes = (
        "nf4",
        "nf4_dq",
        "nf4_b128",
        "nf4_b256",
        "fp4",
        "fp4_dq",
        "fp4_b128",
        "int8",
        "int8_dq",
        "int8_b128",
    )

    def __init__(self) -> None:
        self._residuals: dict[str, float] = {}

    def _functional(self):
        try:
            from bitsandbytes import functional
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "bitsandbytes is not installed. It is deliberately not a project "
                "dependency: install it on demand with `uv pip install bitsandbytes`."
            ) from exc
        return functional

    @staticmethod
    def _parse(scheme: str) -> tuple[str, int | None, bool]:
        """(kind, blocksize, double_quantization) from a scheme name."""
        parts = scheme.split("_")
        kind = parts[0]
        if kind not in ("nf4", "fp4", "int8"):
            raise ValueError(
                f"unknown bitsandbytes scheme {scheme!r}; use "
                f"{{nf4,fp4}}[_dq][_b<N>] or int8[_dq][_b<N>]"
            )
        blocksize: int | None = None
        double = False
        for part in parts[1:]:
            if part == "dq":
                double = True
            elif part.startswith("b") and part[1:].isdigit():
                blocksize = int(part[1:])
            else:
                raise ValueError(f"unknown scheme modifier {part!r} in {scheme!r}")
        return kind, blocksize, double

    def expressible(self, scheme: str) -> bool:
        """Measured, never assumed: every bnb format fails the affine test."""
        kind = self._parse(scheme)[0]
        self._residuals.setdefault(kind, self._measure_residual(kind))
        return self._residuals[kind] <= AFFINE_TOLERANCE

    def residual(self, scheme: str) -> float:
        """The codebook's distance from affine, for the record."""
        kind = self._parse(scheme)[0]
        self._residuals.setdefault(kind, self._measure_residual(kind))
        return self._residuals[kind]

    def _measure_residual(self, kind: str) -> float:
        functional = self._functional()
        probe = np.linspace(-1.0, 1.0, 1024, dtype=np.float32).reshape(32, 32)
        import torch

        tensor = torch.from_numpy(probe)
        if kind in ("nf4", "fp4"):
            _packed, state = functional.quantize_4bit(tensor, quant_type=kind)
        else:
            _packed, state = functional.quantize_blockwise(tensor, blocksize=32)
        code = state.code
        return affine_residual(code.detach().cpu().numpy() if code is not None else [])

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        kind, blocksize, double = self._parse(scheme)
        return {
            "name": f"bitsandbytes_{scheme}",
            "bits": 4 if kind in ("nf4", "fp4") else 8,
            "granularity": Granularity.PER_GROUP.value,
            # For 8-bit without an explicit blocksize the block is one row,
            # whose length depends on the tensor, so it is reported as 0 with a
            # note rather than as a number that would be wrong for most weights.
            "group_size": blocksize or (64 if kind in ("nf4", "fp4") else 0),
            "group_size_note": (
                "one row per block" if blocksize is None and kind == "int8" else ""
            ),
            "axis": 0,
            "symmetric": False,
            "rounding": "nearest",
            "double_quantization": double,
            "affine_residual": round(self.residual(scheme), 6),
            "affine_tolerance": AFFINE_TOLERANCE,
        }

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        raise NotImplementedError(
            "bitsandbytes formats are not affine, so they cannot be written as a "
            "runtime artifact. Use `store_arrays`, which keeps the packed codes "
            "and block maxima in bitsandbytes' own layout."
        )

    def store_arrays(
        self, weight: np.ndarray, scheme: str
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """bnb's own storage for one weight.

        Packed codes, block maxima, and the dequantized values so a caller can
        measure what it lost.

        A separate method because these are NOT artifact arrays: no
        `(u - zero_point) * scale` reconstructs them, which is exactly why they
        are not expressible.
        """
        import torch  # noqa: PLC0415 - optional dependency, imported on use

        functional = self._functional()
        kind, blocksize, double = self._parse(scheme)
        tensor = torch.from_numpy(weight.astype(np.float32))
        if kind in ("nf4", "fp4"):
            kwargs = {"quant_type": kind, "compress_statistics": double}
            if blocksize is not None:
                kwargs["blocksize"] = blocksize
            packed, state = functional.quantize_4bit(tensor, **kwargs)
            rebuilt = functional.dequantize_4bit(packed, state).float()
        else:
            packed, state = functional.quantize_blockwise(
                tensor,
                blocksize=blocksize or int(tensor.shape[-1]),
                nested=double,
            )
            rebuilt = functional.dequantize_blockwise(packed, state).float()
        arrays = {
            "packed": packed.detach().cpu().numpy(),
            "absmax": state[0].detach().cpu().numpy().astype(np.float32),
        }
        return arrays, rebuilt.reshape(tensor.shape).detach().cpu().numpy()

    def round_trip(self, weight: np.ndarray, scheme: str) -> np.ndarray:
        """bitsandbytes' own reconstruction, from its own storage."""
        _arrays, rebuilt = self.store_arrays(weight, scheme)
        return rebuilt

    def stored_bytes(self, weight: np.ndarray, scheme: str) -> int:
        """What one weight costs in bitsandbytes' layout, metadata included."""
        arrays, _rebuilt = self.store_arrays(weight, scheme)
        return sum(array.nbytes for array in arrays.values())


#: Every producer, by name. Exposed so the CLI can list them and the export path
#: can dispatch without importing the optional backends.
PRODUCERS: dict[str, type] = {
    "in-house": InHouseProducer,
    "torch-ao": TorchAOProducer,
    "bitsandbytes": BitsAndBytesProducer,
}
