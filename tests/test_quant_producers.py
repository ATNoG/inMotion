"""The three quantizers behind one interface.

Each producer must answer the same three questions — what are the codes, what
scale and zero point do they carry, and can the runtime read the result — and
the answers must be measured, not declared. The bitsandbytes case is why: its
8-bit path looks like plain int8 and is not affine, and a producer that claimed
otherwise would write artifacts the runtime silently misreads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inmotion.quant.producers import (
    AFFINE_TOLERANCE,
    PRODUCERS,
    BitsAndBytesProducer,
    InHouseProducer,
    TorchAOProducer,
    affine_residual,
)
from inmotion.quant.spec import resolve

WEIGHT = np.random.default_rng(0).normal(0, 0.6, (64, 128)).astype(np.float32)


def test_the_registry_covers_the_three_producers() -> None:
    assert set(PRODUCERS) == {"in-house", "torch-ao", "bitsandbytes"}


def test_in_house_producer_is_the_model_level_quantizer() -> None:
    """It must go through `quantize_tensor`, or per-group convs change meaning.

    `reference.quantize` on a 3-D weight would group it differently from the
    model-level entry point, which flattens first.
    """
    spec = resolve("q4")
    weight3d = np.random.default_rng(1).normal(0, 0.6, (128, 128, 3)).astype(np.float32)
    produced = InHouseProducer(spec).quantize(weight3d, "q4")

    from inmotion.quant.model import quantize_tensor

    qt, _ = quantize_tensor(torch.from_numpy(weight3d), spec)
    assert np.array_equal(produced.codes, qt.q)
    assert np.array_equal(produced.scale, qt.scale)


def test_torch_ao_producer_is_per_channel_symmetric_int8() -> None:
    produced = TorchAOProducer().quantize(WEIGHT, "int8")
    assert produced.bits == 8
    assert produced.symmetric
    assert produced.axis == 0
    assert produced.codes.shape == WEIGHT.shape
    assert produced.scale.shape == (WEIGHT.shape[0], 1)
    assert np.all(produced.zero_point == 0)


def test_torch_ao_and_in_house_differ_only_by_the_scale_convention() -> None:
    """The one real difference between the two int8 producers.

    torch.ao places the scale at `max|w| / 127.5`, the midpoint of the signed
    range; this repository uses `max|w| / 127`, the full positive span. Both
    standard, and the codes differ by at most one level of 255.
    """
    mine = InHouseProducer(resolve("int8")).quantize(WEIGHT, "int8")
    theirs = TorchAOProducer().quantize(WEIGHT, "int8")

    ratio = np.ravel(mine.scale) / np.ravel(theirs.scale)
    assert np.allclose(ratio, 127.5 / 127.0, rtol=1e-5)
    assert np.abs(mine.codes.astype(int) - theirs.codes.astype(int)).max() <= 1


@pytest.mark.parametrize("scheme", ["nf4", "fp4", "int8"])
def test_bitsandbytes_is_not_affine_and_says_so(scheme: str) -> None:
    """Measured, not assumed. All three fail the affine test.

    Its 8-bit path is the surprising one: it looks like plain int8, and its
    256-entry codebook is non-uniformly spaced.
    """
    pytest.importorskip("bitsandbytes")
    producer = BitsAndBytesProducer()
    residual = producer.residual(scheme)

    assert residual > AFFINE_TOLERANCE, (
        f"{scheme} residual {residual:.4f} would be readable by an affine "
        f"runtime; if that is real, the store should carry it"
    )
    assert producer.expressible(scheme) is False


def test_affine_residual_is_zero_for_a_straight_line() -> None:
    """The test itself, so a broken metric cannot pass everything."""
    line = np.arange(256, dtype=np.float64)
    assert affine_residual(line) == pytest.approx(0.0, abs=1e-12)
    bent = line**2
    assert affine_residual(bent) > AFFINE_TOLERANCE


def test_per_channel_quantization_commutes_with_the_qkv_split() -> None:
    """Why a producer golden can be built from the model's own weights.

    The artifact splits a fused `in_proj_weight` into q/k/v and quantizes each;
    the model holds it fused. Per-channel means one scale per row, and the rows
    are the same either way, so the two agree exactly. Weight-norm folding does
    NOT commute, which is why `rebuild_state_dict` still exists.
    """
    fused = np.random.default_rng(2).normal(0, 0.3, (3 * 128, 128)).astype(np.float32)
    parts = [fused[i * 128 : (i + 1) * 128] for i in range(3)]

    producer = TorchAOProducer()
    whole = producer.quantize(fused, "int8")
    split = [producer.quantize(part, "int8") for part in parts]

    assert np.array_equal(whole.codes, np.concatenate([s.codes for s in split], axis=0))
    assert np.allclose(whole.scale.ravel(), np.concatenate([s.scale.ravel() for s in split]))


def test_export_refuses_a_scheme_the_runtime_cannot_read() -> None:
    """Refused with the measured reason, not written and hoped for."""
    from inmotion.quant.export import resolve_producer

    with pytest.raises(ValueError, match="not affine") as excinfo:
        resolve_producer("bitsandbytes", "nf4")
    assert "residual" in str(excinfo.value)


def test_export_rejects_an_unknown_producer() -> None:
    from inmotion.quant.export import resolve_producer

    with pytest.raises(ValueError, match="unknown producer"):
        resolve_producer("not-a-producer", None)
