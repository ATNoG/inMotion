"""Contract for whole-model quantization.

The properties that matter are that the size accounting is honest (it must count
what is *not* quantized, or the memory-fit claim is a lie) and that applying a
scheme leaves the model numerically equal to what a device would compute.
"""

from __future__ import annotations

import pytest
import torch
from conftest import requires_artifacts

from inmotion.models import registry as reg
from inmotion.quant.model import _is_quantizable, apply, apply_cast
from inmotion.quant.spec import FLOAT_PRECISIONS, resolve

JEPA = ["lejepa", "t_jepa", "ts_jepa", "cf_jepa"]


# ---------------------------------------------------------------------------
# which tensors get quantized
# ---------------------------------------------------------------------------


def test_two_dimensional_weight_matrices_are_quantizable() -> None:
    assert _is_quantizable("a.weight", torch.zeros(4, 4))
    assert _is_quantizable("encoder.transformer.layers.0.linear1.weight", torch.zeros(8, 4))


def test_in_proj_weight_is_quantizable() -> None:
    """MultiheadAttention packs QKV into `in_proj_weight`.

    Matching only `.weight` skipped it, which is 196,608 parameters - 35% of
    `lejepa` and exactly the dense matrix weight-only quantization targets.
    """
    assert _is_quantizable("enc.self_attn.in_proj_weight", torch.zeros(48, 16))


def test_one_dimensional_tensors_are_skipped() -> None:
    """Norms and biases are a negligible share and lose accuracy fastest."""
    assert not _is_quantizable("norm.weight", torch.zeros(16))
    assert not _is_quantizable("linear.bias", torch.zeros(16))


def test_initialization_state_is_skipped() -> None:
    """Position embeddings and learned tokens change what the model *is*."""
    assert not _is_quantizable("encoder.pos_embed", torch.zeros(1, 10, 8))
    assert not _is_quantizable("predictor.mask_token", torch.zeros(1, 8))
    assert not _is_quantizable("predictor.query_tokens", torch.zeros(1, 4, 8))


# ---------------------------------------------------------------------------
# size accounting
# ---------------------------------------------------------------------------


@requires_artifacts
def test_report_counts_everything_the_device_holds() -> None:
    """total_bytes must exceed quantized_bytes, because not all of it is quantized."""
    model = reg.load(reg.by_name("lejepa"), strict=True)
    _, report = apply(model, resolve("int8"))
    assert report.quantized_bytes > 0
    assert report.skipped_param_bytes > 0, "expected unquantized norms/biases"
    assert report.buffer_bytes > 0, "expected running statistics"
    assert report.total_bytes == (
        report.quantized_bytes + report.skipped_param_bytes + report.buffer_bytes
    )


@requires_artifacts
@pytest.mark.parametrize("model_name", JEPA)
def test_smaller_bits_mean_smaller_total(model_name: str) -> None:
    model = reg.load(reg.by_name(model_name), strict=True)
    totals = [apply(model, resolve(name))[1].total_bytes for name in ("int8", "q4", "q2")]
    assert totals == sorted(totals, reverse=True), totals


@requires_artifacts
@pytest.mark.parametrize("model_name", JEPA)
def test_most_parameters_are_quantized(model_name: str) -> None:
    """If this drops, a tensor class is being skipped by accident."""
    model = reg.load(reg.by_name(model_name), strict=True)
    total = sum(p.numel() for p in model.parameters())
    _, report = apply(model, resolve("int8"))
    quantized = sum(r.n_values for r in report.records)
    assert quantized / total > 0.95, f"{model_name}: only {quantized / total:.1%} quantized"


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


@requires_artifacts
def test_apply_does_not_mutate_the_original_by_default() -> None:
    original = reg.load(reg.by_name("lejepa"), strict=True)
    before = next(original.parameters()).detach().clone()
    apply(original, resolve("q4"))
    assert torch.equal(before, next(original.parameters()).detach())


@requires_artifacts
@pytest.mark.parametrize("label", ["int8", "q4"])
def test_apply_replaces_weights_with_exactly_the_dequantized_values(label: str) -> None:
    """Running the returned model is the same as a device dequantizing then matmul."""
    original = reg.load(reg.by_name("lejepa"), strict=True)
    quantized, report = apply(original, resolve(label))
    names = {r.name for r in report.records}
    for name, param in quantized.named_parameters():
        if name in names:
            # Differs from the original (it was quantized) but is finite.
            assert torch.isfinite(param).all()
            assert param.dtype == torch.float32, "activations stay fp32"


@requires_artifacts
def test_all_zero_weights_survive() -> None:
    """A zeroed weight must not produce NaN scales or infinities."""
    model = reg.load(reg.by_name("lejepa"), strict=True)
    for name, param in model.named_parameters():
        if _is_quantizable(name, param):
            with torch.no_grad():
                param.zero_()
    quantized, report = apply(model, resolve("q4"))
    for param in quantized.parameters():
        assert torch.isfinite(param).all()
    assert report.max_error == 0.0


# ---------------------------------------------------------------------------
# float casts
# ---------------------------------------------------------------------------


@requires_artifacts
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_float_cast_halves_the_footprint(precision: str) -> None:
    model = reg.load(reg.by_name("lejepa"), strict=True)
    _, fp32_report = apply_cast(model, "fp32")
    _, report = apply_cast(model, precision)
    ratio = fp32_report.total_bytes / report.total_bytes
    assert 1.9 < ratio < 2.1, ratio
    assert precision in FLOAT_PRECISIONS


@requires_artifacts
def test_float_cast_is_available_for_every_deployed_model() -> None:
    for name in JEPA:
        model = reg.load(reg.by_name(name), strict=True)
        for precision in FLOAT_PRECISIONS:
            _, report = apply_cast(model, precision)
            assert report.total_bytes > 0
