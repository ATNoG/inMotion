"""Contract for the quantization reference implementation.

This is the normative definition the Rust kernels are verified against, so the
tests pin exact arithmetic rather than approximate behaviour:

- dequantize is exactly ``(q - zero_point) * scale`` evaluated in float32;
- re-quantizing a dequantized tensor is idempotent;
- rounding ties follow the declared mode, because numpy rounds half to even and
  Rust rounds half away from zero and the two must not be conflated;
- packing round-trips for every bit width and every tail length;
- the byte accounting reflects the packed size, not an int8-per-value payload.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from inmotion.quant import evaluate as EV
from inmotion.quant.packing import fold_bias_into_zero_point, n_bytes_for, pack, unpack
from inmotion.quant.reference import dequantize, quantize
from inmotion.quant.spec import Granularity, QuantSpec, Rounding, resolve


def _tensor(shape=(8, 16), seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, scale, shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_integer_range_matches_bit_width() -> None:
    for bits, (lo, hi) in {1: (-1, 0), 2: (-2, 1), 4: (-8, 7), 8: (-128, 127)}.items():
        spec = QuantSpec(bits=bits, symmetric=True)
        assert (spec.qmin, spec.qmax) == (lo, hi)


def test_asymmetric_range_is_unsigned() -> None:
    spec = QuantSpec(bits=8, symmetric=False)
    assert (spec.qmin, spec.qmax) == (0, 255)


def test_bits_are_bounded() -> None:
    with pytest.raises(ValueError, match="bits must be in 1..8"):
        QuantSpec(bits=9)
    with pytest.raises(ValueError, match="bits must be in 1..8"):
        QuantSpec(bits=0)


def test_presets_are_distinguishable() -> None:
    labels = {name: resolve(name).label for name in ("int8", "q8", "q4", "q2")}
    assert len(set(labels.values())) == len(labels), labels


# ---------------------------------------------------------------------------
# exact dequantization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["int8", "q8", "q4", "q2"])
def test_dequantize_is_exactly_the_affine_formula(label: str) -> None:
    """Every element must equal ``(code - zero_point[group]) * scale[group]``.

    Checked element by element against the documented formula, so this does not
    simply re-run the implementation it is testing.
    """
    spec = resolve(label)
    x = _tensor(shape=(6, 40))
    qt = quantize(x, spec)
    got = dequantize(qt)
    assert got.shape == x.shape

    codes = qt.q.astype(np.float32).ravel()
    n = codes.size

    if spec.granularity is Granularity.PER_TENSOR:
        scale = np.full(n, float(qt.scale), dtype=np.float32)
        zp = np.full(n, float(qt.zero_point), dtype=np.float32)
    elif spec.granularity is Granularity.PER_CHANNEL:
        axis = spec.axis % len(qt.shape)
        per_channel = np.repeat(np.arange(qt.shape[axis]), np.prod(qt.shape[axis + 1 :], dtype=int))
        per_channel = np.tile(per_channel, int(np.prod(qt.shape[:axis], dtype=int)))
        scale = qt.scale.reshape(-1)[per_channel].astype(np.float32)
        zp = qt.zero_point.reshape(-1)[per_channel].astype(np.float32)
    else:
        per_channel = np.arange(n) % spec.group_size + spec.group_size * (
            np.arange(n) // spec.group_size
        )
        groups = np.repeat(np.arange(np.prod(qt.shape[:-1], dtype=int)), qt.shape[-1]) * (
            -(-qt.shape[-1] // spec.group_size)
        )
        groups = groups + (np.arange(n) % qt.shape[-1]) // spec.group_size
        scale = qt.scale.reshape(-1)[groups].astype(np.float32)
        zp = qt.zero_point.reshape(-1)[groups].astype(np.float32)

    expected = ((codes - zp) * scale).reshape(qt.shape).astype(np.float32)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("label", ["int8", "q8", "q4", "q2"])
def test_codes_stay_inside_the_declared_range(label: str) -> None:
    spec = resolve(label)
    qt = quantize(_tensor(seed=3), spec)
    assert qt.q.min() >= spec.qmin
    assert qt.q.max() <= spec.qmax


@pytest.mark.parametrize("label", ["int8", "q8", "q4", "q2"])
def test_requantizing_is_idempotent(label: str) -> None:
    """Dequantized values must map back to the same codes, or error compounds."""
    spec = resolve(label)
    x = _tensor(seed=7)
    first = quantize(x, spec)
    second = quantize(dequantize(first), spec)
    assert np.array_equal(first.q, second.q)


def test_higher_precision_is_never_worse() -> None:
    """Monotonicity: more bits cannot increase the maximum error."""
    x = _tensor(seed=11)
    errors = [
        np.abs(x - dequantize(quantize(x, resolve(name)))).max()
        for name in ("q2", "q4", "q8", "int8")
    ]
    assert errors == sorted(errors, reverse=True), errors


def test_per_group_is_at_least_as_good_as_per_tensor() -> None:
    """Finer granularity gives more scales, so it cannot be worse."""
    x = _tensor(shape=(16, 64), seed=13)
    coarse = QuantSpec(bits=4, granularity=Granularity.PER_TENSOR, symmetric=True)
    fine = QuantSpec(bits=4, granularity=Granularity.PER_GROUP, group_size=16, symmetric=True)
    err_coarse = np.abs(x - dequantize(quantize(x, coarse))).max()
    err_fine = np.abs(x - dequantize(quantize(x, fine))).max()
    assert err_fine <= err_coarse


# ---------------------------------------------------------------------------
# rounding
# ---------------------------------------------------------------------------


def test_rounding_modes_differ_only_on_ties() -> None:
    """0.5 and 1.5 are the tie cases where the two modes diverge."""
    from inmotion.quant.reference import _round

    vals = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.4, 0.6], dtype=np.float32)
    even = _round(vals, Rounding.HALF_TO_EVEN).astype(int).tolist()
    away = _round(vals, Rounding.HALF_AWAY).astype(int).tolist()

    # half to even rounds ties to the nearest even integer
    assert even == [0, 2, 2, 4, 0, -2, -2, 0, 1]
    # half away from zero always rounds ties outward
    assert away == [1, 2, 3, 4, -1, -2, -3, 0, 1]


def test_half_to_even_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    x = (rng.integers(-1000, 1000, 500) / 64.0).astype(np.float32).reshape(1, -1)
    spec = QuantSpec(bits=8, granularity=Granularity.PER_TENSOR, rounding=Rounding.HALF_TO_EVEN)
    qt = quantize(x, spec)
    scale = np.max(np.abs(x)) / np.float32(127)
    assert np.array_equal(
        qt.q.astype(np.int64).ravel(), np.round(x / scale).astype(np.int64).ravel()
    )


# ---------------------------------------------------------------------------
# all-zero and degenerate tensors
# ---------------------------------------------------------------------------


def test_all_zero_tensor_does_not_divide_by_zero() -> None:
    spec = resolve("q4")
    x = np.zeros((4, 8), dtype=np.float32)
    qt = quantize(x, spec)
    assert np.all(qt.q == 0)
    assert np.array_equal(dequantize(qt), x)
    assert np.all(np.isfinite(qt.scale))
    assert np.all(qt.scale > 0)


def test_single_element_tensor() -> None:
    spec = resolve("int8")
    x = np.array([[3.5]], dtype=np.float32)
    assert np.allclose(dequantize(quantize(x, spec)), x, atol=1e-6)


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", list(range(1, 9)))
@pytest.mark.parametrize("count", [1, 3, 7, 8, 9, 63, 64, 65, 1000, 2049])
def test_pack_round_trips(bits: int, count: int) -> None:
    rng = np.random.default_rng(bits * 100 + count)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    codes = rng.integers(lo, hi + 1, count).astype(np.int8)
    assert np.array_equal(codes, unpack(pack(codes, bits), bits, count))


@pytest.mark.parametrize("bits", list(range(1, 9)))
def test_packed_size_is_bits_per_value(bits: int) -> None:
    assert n_bytes_for(1024, bits) == 1024 * bits // 8
    assert pack(np.zeros(1024, dtype=np.int8), bits).size == 1024 * bits // 8


def test_pack_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match="out of range"):
        pack(np.array([8], dtype=np.int8), 4)  # q4 max is 7
    with pytest.raises(ValueError, match="out of range"):
        pack(np.array([-9], dtype=np.int8), 4)  # q4 min is -8


def test_folded_bias_makes_dequant_a_single_formula() -> None:
    """(q - zp) * scale == (u - (zp + 2**(b-1))) * scale, with u the biased code."""
    bits = 4
    codes = np.array([-8, -1, 0, 7], dtype=np.int32)
    scale = np.float32(0.25)
    zp = 3
    bias = 1 << (bits - 1)
    biased = codes + bias  # what `pack` actually stores
    folded = fold_bias_into_zero_point(np.array(zp), bits)

    from_codes = (codes.astype(np.float32) - np.float32(zp)) * scale
    from_biased = (biased.astype(np.float32) - folded.astype(np.float32)) * scale
    assert np.array_equal(from_codes, from_biased)
    # and the packed payload really is those biased codes
    assert np.array_equal(
        pack(codes.astype(np.int8), bits), pack((biased - bias).astype(np.int8), bits)
    )


# ---------------------------------------------------------------------------
# dataset.csv baseline (the guard that makes every other number traceable)
# ---------------------------------------------------------------------------


def _fp32_row(model: str, mcc: float, dataset: str = EV.BASELINE_DATASET) -> EV.EvalRow:
    """A minimal fp32 row; check_baseline only reads model/precision/dataset/mcc."""
    return EV.EvalRow(
        model=model,
        precision="fp32",
        bits=32,
        dataset=dataset,
        seed=42,
        mcc=mcc,
        accuracy=0.0,
        f1_macro=0.0,
        n_test=1,
        total_mb=0.0,
        weights_mb=0.0,
        compression=1.0,
        max_weight_error=0.0,
        n_params=0,
    )


def test_baseline_covers_all_five_world_models_on_dataset_csv() -> None:
    assert EV.BASELINE_DATASET == "dataset.csv"
    assert set(EV.BASELINE_MCC) == {"lejepa", "t_jepa", "ts_jepa", "cf_jepa", "sigreg"}


def test_check_baseline_accepts_the_recorded_values() -> None:
    """The guard must be satisfiable, or it is just noise."""
    rows = [_fp32_row(model, mcc) for model, mcc in EV.BASELINE_MCC.items()]
    assert EV.check_baseline(rows) == []


def test_check_baseline_trips_when_a_recorded_value_moves() -> None:
    """Perturb one recorded MCC and the guard must fail loudly, naming it."""
    rows = [_fp32_row(model, mcc) for model, mcc in EV.BASELINE_MCC.items()]
    rows[0].mcc += 0.02
    problems = EV.check_baseline(rows)
    assert problems, "guard accepted a moved baseline"
    assert rows[0].model in problems[0]


def test_check_baseline_rejects_an_unrecorded_model() -> None:
    problems = EV.check_baseline([_fp32_row("mamba3_cnn", 0.8567)])
    assert len(problems) == 1
    assert "no recorded fp32 baseline" in problems[0]


def test_check_baseline_does_not_compare_another_dataset() -> None:
    """icaisf rows are not the baseline and must not be silently accepted."""
    icaisf = [_fp32_row(model, 0.999, dataset="dataset.icaisf.csv") for model in EV.BASELINE_MCC]
    assert EV.check_baseline(icaisf) == []


def test_committed_sweep_is_the_dataset_csv_baseline() -> None:
    """results/quant/sweep_dataset.csv: 5 models x 7 precisions, fp32 == baseline."""
    from inmotion.data.paths import REPO_ROOT

    path = REPO_ROOT / "results/quant/sweep_dataset.csv"
    assert path.exists(), f"missing {path}; run `inmotion quantize eval`"

    int_fields = {"bits", "seed", "n_test", "n_params"}
    str_fields = {"model", "precision", "dataset"}

    def cast(key: str, value: str):
        if key in str_fields:
            return value
        return int(value) if key in int_fields else float(value)

    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(EV.EvalRow(**{k: cast(k, v) for k, v in raw.items()}))

    assert len(rows) == len(EV.BASELINE_MCC) * len(EV.DEFAULT_PRECISIONS), len(rows)
    assert {row.dataset for row in rows} == {EV.BASELINE_DATASET}
    assert {row.model for row in rows} == set(EV.BASELINE_MCC)
    assert {row.precision for row in rows} == set(EV.DEFAULT_PRECISIONS)
    assert EV.check_baseline(rows) == []

    fp32 = {row.model: row.mcc for row in rows if row.precision == "fp32"}
    assert fp32 == EV.BASELINE_MCC


def test_cli_eval_defaults_to_dataset_csv(tmp_path) -> None:
    """`inmotion quantize eval` with no --data must measure dataset.csv."""
    from typer.testing import CliRunner

    from inmotion.cli.quantize import app

    out = tmp_path / "sweep.csv"
    result = CliRunner().invoke(
        app, ["eval", "--models", "lejepa", "--formats", "fp32", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["dataset"] for row in rows] == [EV.BASELINE_DATASET]
    assert float(rows[0]["mcc"]) == EV.BASELINE_MCC["lejepa"]
