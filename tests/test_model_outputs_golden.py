"""Numerical fingerprints for every registered model.

Loading correctly is necessary but not sufficient: a refactor could keep the
checkpoint loading while changing a forward pass (a permute, a normalisation, a
pooling mode). These tests run each model in eval mode on a deterministic input
and compare the output against a frozen vector, so unintended behaviour changes
fail loudly.

The comparison is a tolerance, not a hash. A hash came first and failed for the
wrong reason: the output is accumulated in float32, so changing how the work is
split across threads — let alone rebuilding the environment and getting
different kernels — moves the last bits. Measured on this suite, that noise
reaches 6e-7 and flips the sixth decimal for a value or two, which turned every
model red in a rebuild where no model code had changed. ``rtol=1e-3`` with
``atol=1e-4`` sits two to three orders of magnitude above that noise and well
below what a permute, a normalisation or a pooling change does, which is the
class of defect this file exists to catch. The failure message prints the
largest deviation, so a real change can be told from noise at a glance.

Re-taking the golden vectors is a deliberate act: it means a model's output is
expected to change. Use ``tests/_regen_golden.py`` and say why in the commit
message. Widening the tolerance to silence a failure is not that act.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from conftest import GOLDEN_DIR, requires_artifacts

from inmotion.models import registry as reg

# Every test here needs the (gitignored) checkpoint store.
pytestmark = requires_artifacts


BATCH = 3
SEED = 1234
GOLDEN_FILE = GOLDEN_DIR / "model_outputs.json"

# See the module docstring: noise is ~1e-7, a forward-pass change is ~1e-2.
RTOL = 1e-3
ATOL = 1e-4


def _output(name: str) -> np.ndarray:
    """A model's output on the fixed input, in float64."""
    spec = reg.by_name(name)
    model = reg.load(spec, strict=True)

    in_features = 18 if spec.rich else 4
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(BATCH, 10, in_features, generator=generator)

    with torch.no_grad():
        out = model(x)

    return out.double().flatten().numpy()


@pytest.fixture(scope="module")
def golden() -> dict[str, list[float]]:
    if not GOLDEN_FILE.exists():
        pytest.skip(f"no golden file at {GOLDEN_FILE}; run tests/_regen_golden.py")
    return json.loads(GOLDEN_FILE.read_text())


@pytest.mark.parametrize("name", reg.spec_names())
def test_output_matches_golden(name: str, golden: dict[str, list[float]]) -> None:
    expected = golden.get(name)
    assert isinstance(expected, list), (
        f"{name}: the golden entry is not an output vector; re-take "
        "tests/golden/model_outputs.json with `uv run python tests/_regen_golden.py`"
    )

    actual = _output(name)
    reference = np.asarray(expected, dtype=np.float64)
    assert actual.shape == reference.shape, (
        f"{name}: output shape changed ({actual.shape} now, {reference.shape} in the golden file)"
    )

    deviation = np.abs(actual - reference)
    assert np.allclose(actual, reference, rtol=RTOL, atol=ATOL), (
        f"{name}: model output changed (max |delta| {deviation.max():.3e} at index "
        f"{int(deviation.argmax())}, tolerance rtol={RTOL:g}/atol={ATOL:g}). A change this "
        "size is a forward-pass change, not kernel noise: if it is intended, re-take "
        "tests/golden/model_outputs.json with `uv run python tests/_regen_golden.py` and "
        "say why."
    )
