"""Checkpoint contract — the definition of "the old models still work".

Each case is a checkpoint documented in ``docs/isac/results/MASTER_RESULTS.md``.
Three things must hold for every one of them:

1. the file exists;
2. it loads with ``strict=True`` — no silently-random weights;
3. the loaded parameter count is exactly the expected integer.

The counts are exact on purpose. MASTER_RESULTS publishes them rounded to three
decimals, and a rounded figure does not catch a drift that lands on the same
rounded value.

``PUBLISHED_MISMATCHES`` records cases where the checkpoint on disk is *not* the
model the paper's table describes. Those are asserted to be documented in
``registry.PROVENANCE_GAPS`` so the discrepancy cannot be quietly erased by
loosening ``strict``.
"""

from __future__ import annotations

import pytest
from conftest import requires_artifacts

from inmotion.models import registry as reg

# Every test here needs the (gitignored) checkpoint store.
pytestmark = requires_artifacts


# model name -> (checkpoint exists, exact module parameter count)
EXPECTED: dict[str, int] = {
    # MASTER_RESULTS.md §2.3
    "lejepa": 565_765,
    "t_jepa": 633_477,
    "ts_jepa": 2_153_349,
    "cf_jepa": 1_232_261,
    "sigreg": 696_900,
    "sigreg_seed3": 696_900,
    "sigreg_seed5": 696_900,
    "deep_stack": 43_575_549,
    "mamba3_cnn": 945_028,
    "mamba3_tcn": 695_812,
    "mamba3_transformer": 561_796,
    "mamba3_multiview": 1_193_349,
    # MASTER_RESULTS.md §2.1/2.2 — pre-HPO and retrained variants
    "lejepa_base": 2_215_813,
    "t_jepa_base": 2_347_653,
    "cf_jepa_base": 2_153_349,
    "sigreg_base": 474_564,
    "sigreg_after_hpo": 317_460,
}

# Stored artifact does not match the parameter count published in the paper.
PUBLISHED_MISMATCHES: set[str] = {
    "mamba3_tcn",
    "mamba3_transformer",
    "mamba3_multiview",
}

TOLERANCE_M = 0.002  # three-decimal rounding in the paper table


def test_every_expected_model_is_registered() -> None:
    missing = set(EXPECTED) - set(reg.spec_names())
    assert not missing, f"registry has no spec for: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_checkpoint_exists(name: str) -> None:
    spec = reg.by_name(name)
    assert spec.exists(), f"{name}: checkpoint missing at {spec.abs_path}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_loads_strict_with_exact_params(name: str) -> None:
    """The load must be strict and the parameter count exact."""
    spec = reg.by_name(name)

    model = reg.load(spec, strict=True)
    n_params = sum(p.numel() for p in model.parameters())

    assert n_params == EXPECTED[name], (
        f"{name}: loaded {n_params:,} params, expected {EXPECTED[name]:,}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_forward_pass_produces_four_logits(name: str) -> None:
    """A loaded checkpoint must actually run and emit one logit per class."""
    import torch

    spec = reg.by_name(name)
    model = reg.load(spec, strict=True)

    in_features = 18 if spec.rich else 4
    x = torch.zeros(2, 10, in_features)
    with torch.no_grad():
        out = model(x)

    assert tuple(out.shape) == (2, 4), f"{name}: unexpected output shape {tuple(out.shape)}"


def test_published_counts_match_except_documented_mismatches() -> None:
    """Anything diverging from the paper's table must be in PROVENANCE_GAPS."""
    undocumented: list[str] = []
    for name in sorted(EXPECTED):
        spec = reg.by_name(name)
        if spec.published_m is None:
            continue
        actual_m = EXPECTED[name] / 1e6
        if abs(actual_m - spec.published_m) <= TOLERANCE_M:
            continue
        if name not in reg.PROVENANCE_GAPS:
            undocumented.append(name)
    assert not undocumented, (
        "these models disagree with MASTER_RESULTS.md but are not documented in "
        f"registry.PROVENANCE_GAPS: {undocumented}"
    )


def test_documented_mismatches_are_still_real() -> None:
    """Guard against someone 'fixing' the gap by editing the wrong side."""
    for name in PUBLISHED_MISMATCHES:
        spec = reg.by_name(name)
        assert name in reg.PROVENANCE_GAPS, f"{name}: missing PROVENANCE_GAPS entry"
        assert spec.published_m is not None
        actual_m = EXPECTED[name] / 1e6
        assert abs(actual_m - spec.published_m) > TOLERANCE_M, (
            f"{name}: no longer diverges from the paper — update PROVENANCE_GAPS if "
            "the correct artifact has been restored"
        )
