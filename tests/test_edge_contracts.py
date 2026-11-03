"""The edge campaign's shared contract must stay internally consistent.

These are cheap structural checks, not a re-test of the campaign: if two
modules disagree about a column name or a device's core count, the failure shows
up here instead of as a silently mis-joined CSV eight hours into a run.
"""

from __future__ import annotations

import pytest

from inmotion.edge import contracts as c


def test_model_set_and_baselines_agree() -> None:
    assert tuple(c.BASELINE_MCC) == c.MODELS
    assert len(set(c.MODELS)) == len(c.MODELS) == 5


def test_precision_sets_are_disjoint_and_ordered() -> None:
    deployable = set(c.PRECISIONS_DEPLOYABLE)
    cast = set(c.PRECISIONS_CAST)
    assert deployable.isdisjoint(cast)
    assert c.PRECISIONS_DEPLOYABLE[0] == "fp32"
    ranks = [c.precision_rank(p) for p in c.PRECISIONS_DEPLOYABLE]
    assert ranks == sorted(ranks)
    assert c.precision_rank("unknown") == len(c.PRECISIONS_DEPLOYABLE) + len(c.PRECISIONS_CAST)


@pytest.mark.parametrize("table", sorted(c.COLUMNS))
def test_schema_columns_are_unique_and_nonempty(table: str) -> None:
    columns = c.COLUMNS[table]
    assert columns
    assert len(set(columns)) == len(columns)
    assert all(name and name.strip() == name for name in columns)


def test_device_identities_are_consistent() -> None:
    assert set(c.DEVICES) == {"wave-router", "wave-router-2"}
    for key, spec in c.DEVICES.items():
        assert spec.key == key
        assert spec.cores == 4
        assert spec.tdp_w > 0
        assert spec.ram_mb > 0
        assert spec.auth in {"key", "password"}
    # One device jumps through the other; the entry device does not.
    assert c.DEVICES["wave-router"].jump is None
    assert c.DEVICES["wave-router-2"].jump is not None


def test_effective_bits_include_metadata_and_validate() -> None:
    # q4 at group 32: 4-bit codes plus a scale and a zero point, so strictly
    # more than 4 bits per parameter once storage is counted.
    bits = c.effective_bits(artifact_bytes=600_000, n_params=1_000_000)
    assert bits == 4.8
    with pytest.raises(ValueError, match="n_params"):
        c.effective_bits(artifact_bytes=10, n_params=0)


def test_power_and_equivalence_constants_are_usable() -> None:
    assert c.POWER_IDLE_FRACTIONS
    assert all(0.0 < f < 1.0 for f in c.POWER_IDLE_FRACTIONS)
    assert list(c.POWER_IDLE_FRACTIONS) == sorted(c.POWER_IDLE_FRACTIONS)
    assert all(value > 0 for value in c.EQUIVALENCE_MARGINS.values())
    assert set(c.STRESS_PHASES) == {"baseline", "inference", "stress", "both"}


def test_figure_stems_are_unique_and_expected() -> None:
    assert len(c.FIGURE_STEMS) == len(set(c.FIGURE_STEMS)) == 8
    assert c.FIGURE_STEMS[0] == "fig1_accuracy_vs_bits"
    assert c.FIGURE_STEMS[-1] == "fig8_stress_robustness"
