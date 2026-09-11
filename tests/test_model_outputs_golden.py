"""Numerical fingerprints for every registered model.

Loading correctly is necessary but not sufficient: a refactor could keep the
checkpoint loading while changing a forward pass (a permute, a normalisation, a
pooling mode). These tests run each model in eval mode on a deterministic input
and compare against a frozen digest, so unintended behaviour changes fail loudly.

Regenerating a digest is a deliberate act: it means a model's output is expected
to change. Use ``tests/_regen_golden.py`` and say why in the commit message.

The digest is computed in float64 and rounded before hashing, so it does not
depend on CPU/GPU float32 kernel differences — only on real logic changes.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from conftest import GOLDEN_DIR, requires_artifacts
from inmotion.models import registry as reg

# Every test here needs the (gitignored) checkpoint store.
pytestmark = requires_artifacts



BATCH = 3
SEED = 1234
GOLDEN_FILE = GOLDEN_DIR / "model_outputs.json"


def _fingerprint(name: str) -> str:
    """Deterministic digest of a model's output on a fixed input."""
    spec = reg.by_name(name)
    model = reg.load(spec, strict=True)

    in_features = 18 if spec.rich else 4
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(BATCH, 10, in_features, generator=gen)

    with torch.no_grad():
        out = model(x)

    # Round before hashing: absorbs last-bit float noise across BLAS kernels.
    values = out.double().round(decimals=6).flatten().tolist()
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@pytest.fixture(scope="module")
def golden() -> dict[str, str]:
    if not GOLDEN_FILE.exists():
        pytest.skip(f"no golden file at {GOLDEN_FILE}; run tests/_regen_golden.py")
    return json.loads(GOLDEN_FILE.read_text())


@pytest.mark.parametrize("name", reg.spec_names())
def test_output_matches_golden(name: str, golden: dict[str, str]) -> None:
    expected = golden.get(name)
    assert expected is not None, f"{name}: no golden digest recorded"

    actual = _fingerprint(name)
    assert actual == expected, (
        f"{name}: model output changed (golden {expected}, got {actual}). "
        "If this is intended, regenerate tests/golden/model_outputs.json."
    )
