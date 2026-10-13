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
    parse_in_house_scheme,
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
    produced = TorchAOProducer().quantize(WEIGHT, "qint8_pc_sym")
    assert produced.bits == 8
    assert produced.signed_codes
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
    theirs = TorchAOProducer().quantize(WEIGHT, "qint8_pc_sym")

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
    whole = producer.quantize(fused, "qint8_pc_sym")
    split = [producer.quantize(part, "qint8_pc_sym") for part in parts]

    assert np.array_equal(whole.codes, np.concatenate([s.codes for s in split], axis=0))
    assert np.allclose(whole.scale.ravel(), np.concatenate([s.scale.ravel() for s in split]))


def test_export_refuses_a_scheme_the_runtime_cannot_read() -> None:
    """Refused with the measured reason, not written and hoped for."""
    # bitsandbytes is installed on demand, so a machine without it can only be
    # told that the backend is missing — the refusal needs it to measure.
    pytest.importorskip("bitsandbytes")
    from inmotion.quant.export import resolve_producer

    with pytest.raises(ValueError, match="not affine") as excinfo:
        resolve_producer("bitsandbytes", "nf4")
    assert "residual" in str(excinfo.value)


def test_export_rejects_an_unknown_producer() -> None:
    from inmotion.quant.export import resolve_producer

    with pytest.raises(ValueError, match="unknown producer"):
        resolve_producer("not-a-producer", None)


@pytest.mark.parametrize(
    ("label", "bits", "granularity", "group_size", "symmetric"),
    [
        ("int8", 8, "per_channel", 32, True),
        ("q4", 4, "per_group", 32, True),
        ("q4_pc", 4, "per_channel", 32, True),
        ("q4_pt", 4, "per_tensor", 32, True),
        ("q4_pg128", 4, "per_group", 128, True),
        ("q4_asym", 4, "per_group", 32, False),
        ("int8_pt", 8, "per_tensor", 32, True),
        ("q2_pg16_asym", 2, "per_group", 16, False),
    ],
)
def test_in_house_scheme_grammar(
    label: str, bits: int, granularity: str, group_size: int, symmetric: bool
) -> None:
    """The presets are short forms of a grammar, so the space is not a list.

    `q4` and `q4_pg32` are the same scheme; the whole cross product of bit
    width, granularity, group size and symmetry is reachable without naming
    every combination by hand.
    """
    spec = parse_in_house_scheme(label)
    assert spec.bits == bits
    assert spec.granularity.value == granularity
    assert spec.group_size == group_size
    assert spec.symmetric is symmetric


def test_in_house_scheme_grammar_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="sideways"):
        parse_in_house_scheme("q4_sideways")
    with pytest.raises(ValueError, match="unknown in-house scheme"):
        parse_in_house_scheme("nonsense")


def test_torch_ao_covers_the_grid_the_library_supports() -> None:
    """Read from torch, not from memory: qint8/quint8 x pt/pc x sym/affine.

    `qint32` is accepted by both quantize calls and is not a quantization — it
    is int32 storage — so it is deliberately absent.
    """
    producer = TorchAOProducer()
    assert set(producer.schemes) == {
        "qint8_pt_sym",
        "qint8_pt_affine",
        "qint8_pc_sym",
        "qint8_pc_affine",
        "quint8_pt_affine",
        "quint8_pc_affine",
    }
    for scheme in producer.schemes:
        produced = producer.quantize(WEIGHT, scheme)
        dtype, granularity, _qscheme = producer._parse(scheme)
        assert produced.codes.shape == WEIGHT.shape
        assert produced.signed_codes is (dtype == "qint8")
        if granularity.value == "per_tensor":
            assert produced.scale.shape == ()
        else:
            assert produced.scale.shape == (WEIGHT.shape[0], 1)


@pytest.mark.parametrize(
    ("scheme", "kind", "blocksize", "double"),
    [
        ("nf4", "nf4", None, False),
        ("nf4_dq", "nf4", None, True),
        ("nf4_b128", "nf4", 128, False),
        ("fp4_b256", "fp4", 256, False),
        ("int8", "int8", None, False),
        ("int8_dq", "int8", None, True),
        ("int8_b128", "int8", 128, False),
    ],
)
def test_bitsandbytes_scheme_names_parse(
    scheme: str, kind: str, blocksize: int | None, double: bool
) -> None:
    """`<kind>[_dq][_b<N>]`, from its own functional signatures.

    An absent block size stays absent here and is resolved where it is used:
    64 for 4-bit, one row for 8-bit, which only the tensor knows.
    """
    parsed_kind, parsed_block, parsed_double = BitsAndBytesProducer._parse(scheme)
    assert parsed_kind == kind
    assert parsed_block == blocksize
    assert parsed_double is double


def test_every_producer_can_say_what_it_reconstructs() -> None:
    """`round_trip` is what the accuracy harness uses, for all three.

    bitsandbytes has no artifact representation, so without this the accuracy
    path could only measure the producers that can be exported.
    """
    cases = [
        (InHouseProducer(resolve("q4")), "q4"),
        (InHouseProducer(resolve("int8")), "int8"),
        (TorchAOProducer(), "qint8_pc_sym"),
        (TorchAOProducer(), "quint8_pt_affine"),
    ]
    pytest.importorskip("bitsandbytes")
    cases += [(BitsAndBytesProducer(), "nf4"), (BitsAndBytesProducer(), "int8")]

    for producer, scheme in cases:
        rebuilt = producer.round_trip(WEIGHT, scheme)
        assert rebuilt.shape == WEIGHT.shape, f"{producer.name} {scheme}"
        relative = float(np.linalg.norm(rebuilt - WEIGHT) / np.linalg.norm(WEIGHT))
        assert 0.0 < relative < 0.5, f"{producer.name} {scheme} relative error {relative}"
