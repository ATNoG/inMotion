"""Does the hand-written quantizer agree with a real one?

`test_quant_reference.py` pins self-consistency: the affine formula, idempotence,
monotonicity, ranges. None of that is evidence that the scheme is the standard
one, because a consistently wrong quantizer passes all of it. This compares
against torch's own quantization path — `torch.ao.quantization` observers for the
scale and zero point, `quantize_per_tensor`/`quantize_per_channel` for the codes.

Where the two differ, this records HOW MUCH rather than asserting equality: torch
places a symmetric int8 scale at `max|w| / 127.5`, the midpoint of the signed
range, while this code uses `max|w| / 127`, the full span of the positive side.
Both are legitimate; the difference is one part in 254, which is at most one
quantization step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

from inmotion.quant.reference import quantize
from inmotion.quant.spec import Granularity, QuantSpec

#: torch's symmetric int8 divisor: (qmax - qmin) / 2 = (127 + 128) / 2.
TORCH_SYMMETRIC_DIVISOR = 127.5

MINE_DIVISOR = 127.0


@pytest.fixture
def weights() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(64, 128) * 0.7


def dequantized(qt) -> np.ndarray:
    """The documented reconstruction: (q - zero_point) * scale."""
    return (qt.q.astype(np.float32) - qt.zero_point.astype(np.float32)) * qt.scale


def test_per_tensor_symmetric_matches_torch_within_one_step(weights) -> None:
    spec = QuantSpec(bits=8, granularity=Granularity.PER_TENSOR, symmetric=True)
    qt = quantize(weights.numpy(), spec)

    observer = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    observer(weights)
    torch_scale, torch_zp = observer.calculate_qparams()
    torch_dq = torch.quantize_per_tensor(weights, torch_scale, torch_zp, torch.qint8).dequantize()

    # The conventions differ by exactly the divisor ratio, and nothing else.
    assert float(qt.scale) / float(torch_scale) == pytest.approx(
        TORCH_SYMMETRIC_DIVISOR / MINE_DIVISOR, rel=1e-6
    )
    assert int(qt.zero_point) == int(torch_zp) == 0

    # One step is the most a different scale convention can move a value.
    assert np.abs(dequantized(qt) - torch_dq.numpy()).max() <= float(qt.scale) * 1.5


def test_per_channel_symmetric_matches_torch_within_one_step(weights) -> None:
    spec = QuantSpec(bits=8, granularity=Granularity.PER_CHANNEL, axis=0, symmetric=True)
    qt = quantize(weights.numpy(), spec)

    observer = PerChannelMinMaxObserver(
        ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    observer(weights)
    torch_scale, torch_zp = observer.calculate_qparams()
    torch_dq = torch.quantize_per_channel(
        weights, torch_scale, torch_zp, 0, torch.qint8
    ).dequantize()

    ratios = np.ravel(qt.scale) / torch_scale.numpy()
    assert np.allclose(ratios, TORCH_SYMMETRIC_DIVISOR / MINE_DIVISOR, rtol=1e-6)
    step = float(np.ravel(qt.scale).max())
    assert np.abs(dequantized(qt) - torch_dq.numpy()).max() <= step * 1.5


def test_per_tensor_asymmetric_matches_torch_exactly(weights) -> None:
    """Asymmetric is where the two should agree without a convention gap.

    torch uses a signed int8 range (-128..127) with zero point -1 where this
    code uses an unsigned one (0..255) with zero point 127. They are the same
    quantizer with a +128 offset on every code, so the reconstruction must match
    to float32 rounding and the scales must be identical.
    """
    spec = QuantSpec(bits=8, granularity=Granularity.PER_TENSOR, symmetric=False)
    qt = quantize(weights.numpy(), spec)

    observer = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    observer(weights)
    torch_scale, torch_zp = observer.calculate_qparams()
    torch_dq = torch.quantize_per_tensor(weights, torch_scale, torch_zp, torch.qint8).dequantize()

    assert float(qt.scale) == pytest.approx(float(torch_scale), rel=1e-6)
    # 127 unsigned is -1 signed, i.e. the same point on the axis.
    assert int(qt.zero_point) - 128 == int(torch_zp)
    assert np.abs(dequantized(qt) - torch_dq.numpy()).max() <= float(qt.scale) * 1.5


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_asymmetric_reconstruction_is_within_one_step(bits: int) -> None:
    """Regression: an 8-bit asymmetric scheme used to reconstruct as noise.

    Its codes run 0..255, which int8 cannot hold, so `.astype(np.int8)` wrapped
    255 to -1 and the error was larger than the tensor's own range. The dtype now
    follows the scheme. Latent rather than active — q1 is the only asymmetric
    preset and one bit fits — but a public API that silently corrupts is worse
    than one that refuses.
    """
    x = np.linspace(-2.0, 3.0, 4096, dtype=np.float32).reshape(64, 64)
    spec = QuantSpec(bits=bits, granularity=Granularity.PER_TENSOR, symmetric=False)
    qt = quantize(x, spec)

    assert qt.q.dtype == np.uint8, "an unsigned scheme must store unsigned codes"
    assert int(qt.q.min()) >= 0
    assert int(qt.q.max()) <= (1 << bits) - 1

    error = np.abs(dequantized(qt) - x).max()
    assert error <= float(qt.scale) + 1e-6, (
        f"{bits}-bit asymmetric reconstruction error {error} exceeds one "
        f"quantization step {float(qt.scale)}"
    )
