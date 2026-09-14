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


def test_torch_ao_dynamic_linear_agrees_to_within_one_code_level() -> None:
    """torch.ao's own dynamic int8 Linear, not just its observers.

    Only the LAYER path is compared: `torch.ao.quantization.quantize_dynamic` on
    a whole model raises inside `nn.TransformerEncoderLayer.forward` on these
    architectures, so the module-level pipeline cannot be used as a reference
    here at all. This is the same quantizer one layer down, and it must agree.
    """
    import torch.ao.nn.quantized.dynamic as nqd
    from torch.ao.quantization import QConfig
    from torch.ao.quantization.observer import PerChannelMinMaxObserver, PlaceholderObserver

    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 256, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(256, 512) * 0.6)
    layer.qconfig = QConfig(
        activation=PlaceholderObserver.with_args(dtype=torch.float),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
        ),
    )

    quantized = nqd.Linear.from_float(layer).eval()
    reference = quantized.weight()
    torch_codes = reference.int_repr().numpy().astype(int)

    spec = QuantSpec(bits=8, granularity=Granularity.PER_CHANNEL, axis=0, symmetric=True)
    qt = quantize(layer.weight.detach().numpy(), spec)

    # Same scheme, same level count: the codes may differ by one step and no more.
    assert torch_codes.shape == qt.q.shape
    assert np.abs(torch_codes - qt.q.astype(int)).max() <= 1

    scales = reference.q_per_channel_scales().numpy()
    ratios = np.ravel(qt.scale) / scales
    assert np.allclose(ratios, TORCH_SYMMETRIC_DIVISOR / MINE_DIVISOR, rtol=1e-5)


def test_symmetric_schemes_pay_for_a_zero_point_that_is_always_zero() -> None:
    """Records a known cost rather than asserting it is fine.

    A symmetric scheme has zero_point == 0 for every group, yet the artifact
    stores an int32 per group. At group 32 that is one bit per weight, so q4
    costs 6.00 effective bits where 5.00 would do. bitsandbytes NF4 gets better
    reconstruction at 4.50 effective bits because it stores no zero point at all.
    This test fails if the overhead silently grows, and should be rewritten when
    the format stops paying it.
    """
    spec = QuantSpec(bits=4, granularity=Granularity.PER_GROUP, group_size=32, symmetric=True)
    assert spec.symmetric
    weight = np.random.default_rng(0).normal(0, 0.6, (64, 128)).astype(np.float32)
    qt = quantize(weight, spec)
    assert np.all(qt.zero_point == 0), "a symmetric zero point is always zero"

    code_bits = spec.bits
    scale_bits = 32 / spec.group_size
    zero_point_bits = 32 / spec.group_size
    assert code_bits + scale_bits + zero_point_bits == 6.0
