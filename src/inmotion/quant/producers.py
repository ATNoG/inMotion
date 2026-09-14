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
    symmetric: bool


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


class InHouseProducer:
    """This repository's own scheme, through `reference.quantize`."""

    name = "in-house"

    schemes = ("fp32", "int8", "q1", "q2", "q3", "q4", "q5", "q6", "q8")

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
            symmetric=self.spec.symmetric,
        )


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
    """torch.ao's per-channel symmetric int8, applied weight by weight.

    Per tensor because the module-level pipeline cannot run: `quantize_dynamic`
    raises inside `nn.TransformerEncoderLayer.forward` on every architecture
    here. The numerics are still torch's — its observer picks the scale, its
    `quantize_per_channel` produces the codes.
    """

    name = "torch-ao"
    schemes = ("int8",)

    def expressible(self, scheme: str) -> bool:
        return True

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        return {
            "name": "torchao_per_channel_symmetric_int8",
            "bits": 8,
            "granularity": Granularity.PER_CHANNEL.value,
            "group_size": 32,
            "axis": 0,
            "symmetric": True,
            "rounding": "half_to_even",
        }

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        import torch
        from torch.ao.quantization.observer import PerChannelMinMaxObserver

        tensor = torch.from_numpy(weight.astype(np.float32))
        observer = PerChannelMinMaxObserver(
            ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
        observer(tensor)
        scale, zero_point = observer.calculate_qparams()
        scale = scale.to(torch.float32).clamp(min=1e-12)

        quantized = torch.quantize_per_channel(
            tensor, scale, zero_point.to(torch.int32), 0, torch.qint8
        )
        reduced = [1] * weight.ndim
        reduced[0] = weight.shape[0]
        return ProducedWeight(
            codes=quantized.int_repr().numpy(),
            scale=scale.numpy().astype(np.float32).reshape(reduced),
            zero_point=np.zeros(weight.shape[0], dtype=np.int32).reshape(reduced),
            bits=8,
            granularity=Granularity.PER_CHANNEL,
            axis=0,
            symmetric=True,
        )


# --------------------------------------------------------------------------
# bitsandbytes
# --------------------------------------------------------------------------


class BitsAndBytesProducer:
    """bitsandbytes' nf4, fp4 and blockwise int8.

    An optional dependency: the wheel is CUDA-sized and this is a CPU inference
    project whose point is artifact size, so it is installed on demand rather
    than declared. Everything here imports it lazily and says so if it is absent.
    """

    name = "bitsandbytes"
    schemes = ("nf4", "fp4", "int8")

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

    def expressible(self, scheme: str) -> bool:
        """Measured, never assumed: all three bnb schemes fail the affine test."""
        self._residuals.setdefault(scheme, self._measure_residual(scheme))
        return self._residuals[scheme] <= AFFINE_TOLERANCE

    def residual(self, scheme: str) -> float:
        """The codebook's distance from affine, for the record."""
        self._residuals.setdefault(scheme, self._measure_residual(scheme))
        return self._residuals[scheme]

    def _measure_residual(self, scheme: str) -> float:
        functional = self._functional()
        probe = np.linspace(-1.0, 1.0, 1024, dtype=np.float32).reshape(32, 32)
        import torch

        tensor = torch.from_numpy(probe)
        if scheme in ("nf4", "fp4"):
            _packed, state = functional.quantize_4bit(tensor, quant_type=scheme)
        else:
            _packed, state = functional.quantize_blockwise(tensor, blocksize=32)
        code = state.code
        return affine_residual(code.detach().cpu().numpy() if code is not None else [])

    def scheme_meta(self, scheme: str) -> dict[str, object]:
        return {
            "name": f"bitsandbytes_{scheme}",
            "bits": 4 if scheme in ("nf4", "fp4") else 8,
            "granularity": Granularity.PER_GROUP.value,
            "group_size": 64 if scheme in ("nf4", "fp4") else 32,
            "axis": 0,
            "symmetric": False,
            "rounding": "nearest",
            "affine_residual": round(self.residual(scheme), 6),
            "affine_tolerance": AFFINE_TOLERANCE,
        }

    def quantize(self, weight: np.ndarray, scheme: str) -> ProducedWeight:
        raise NotImplementedError(
            "bitsandbytes schemes are not affine, so they cannot be written as a "
            "runtime artifact. Use `store_arrays`, which keeps the packed codes "
            "and block maxima in bitsandbytes' own layout."
        )

    def store_arrays(
        self, weight: np.ndarray, scheme: str
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """bnb's own storage for one weight.

        Packed codes, block maxima, and the dequantized values so a caller can
        measure what it lost.

        A separate method because these are NOT artifact arrays: no `(u - zp) *
        scale` reconstructs them, which is exactly why they are not expressible.
        """
        import torch  # noqa: PLC0415 - optional dependency, imported on use

        functional = self._functional()
        tensor = torch.from_numpy(weight.astype(np.float32))
        if scheme in ("nf4", "fp4"):
            packed, state = functional.quantize_4bit(tensor, quant_type=scheme)
            rebuilt = functional.dequantize_4bit(packed, state).float()
        else:
            packed, state = functional.quantize_blockwise(tensor, blocksize=int(tensor.shape[-1]))
            rebuilt = functional.dequantize_blockwise(packed, state).float()
        arrays = {
            "packed": packed.detach().cpu().numpy(),
            "absmax": state[0].detach().cpu().numpy().astype(np.float32),
        }
        return arrays, rebuilt.reshape(tensor.shape).detach().cpu().numpy()


#: Every producer, by name. Exposed so the CLI can list them and the export path
#: can dispatch without importing the optional backends.
PRODUCERS: dict[str, type] = {
    "in-house": InHouseProducer,
    "torch-ao": TorchAOProducer,
    "bitsandbytes": BitsAndBytesProducer,
}
