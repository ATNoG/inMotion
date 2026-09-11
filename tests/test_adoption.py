"""Adoption tests: recovering an architecture from an untagged checkpoint.

These cover the checkpoint families that were trained before sidecars existed.
Each case asserts the recovered spec actually strict-loads, which is the only
claim that matters: an adopted architecture that fits by key set but not by
shape would be worse than no adoption at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import REPO_ROOT, requires_artifacts

from inmotion.models.adopt import describe_pretrain, infer, looks_like_pretrain, read_state
from inmotion.models.factories import build
from inmotion.models.wrappers import normalise_state

pytestmark = requires_artifacts

DL_DIR = REPO_ROOT / "20-aug" / "models" / "dl" / "normal-new-ds"
ISAC_DIR = REPO_ROOT / "20-aug" / "models" / "exotic" / "normal-new-ds"

# Recovered factory per dl/ checkpoint. These encode the shapes the training
# scripts produced, which is how the configs are recovered from the weights.
DL_EXPECTED = {
    "RNN_seed42.pt": "rnn",
    "GRU_seed42.pt": "gru",
    "LSTM_seed42.pt": "lstm",
    "TCN_seed42.pt": "tcn",
    "Mamba_seed42.pt": "mamba",
    "CNN_seed42.pt": "cnn",
    "CNN2DLSTM_seed42.pt": "cnn2d_rnn",
    "MetaFusion_GRU_seed42.pt": "meta_fusion",
    "MetaFusion_LSTM_seed42.pt": "meta_fusion",
    "HPO_GRU_seed42.pt": "gru",
    "HPO_LSTM_seed42.pt": "bilstm",
    "HPO_BILSTM_seed42.pt": "bilstm",
    "HPO_CNN_seed42.pt": "cnn",
    "HPO_MAMBA_seed42.pt": "mamba",
    "HPO_TRANSFORMER_seed42.pt": "transformer",
    "Ensemble_RNN_GRU_seed42.pt": "voting_ensemble",
    "Ensemble_RNN_GRU_LSTM_seed42.pt": "voting_ensemble",
    "Ensemble_GRU_LSTM_seed42.pt": "voting_ensemble",
    "Ensemble_Mamba_LSTM_seed42.pt": "voting_ensemble",
    "Ensemble_LSTM_CNN_seed42.pt": "voting_ensemble",
    "Ensemble_CNN_TCN_seed42.pt": "voting_ensemble",
    "Ensemble_All_seed42.pt": "voting_ensemble",
    "Stacking_All_seed42.pt": "stacking_ensemble",
}


def _strict_loads_from_path(sidecar, path: Path) -> None:
    """The recovered spec must rebuild and strict-load the original file."""
    import torch

    model = build(sidecar.factory, sidecar.arch)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(normalise_state(state, sidecar.prefix), strict=True)


@pytest.mark.parametrize("filename", sorted(DL_EXPECTED))
def test_dl_checkpoint_is_recovered_and_loads_strictly(filename: str) -> None:
    path = DL_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")

    sidecar = infer(path)
    assert sidecar.factory == DL_EXPECTED[filename], (
        f"{filename}: recovered {sidecar.factory}, expected {DL_EXPECTED[filename]}"
    )
    _strict_loads_from_path(sidecar, path)


def test_every_dl_checkpoint_is_recovered() -> None:
    """No checkpoint in the dl/ store should be left unrecoverable."""
    if not DL_DIR.is_dir():
        pytest.skip("dl checkpoint store not present")

    failures: list[str] = []
    for path in sorted(DL_DIR.glob("*.pt")):
        try:
            infer(path)
        except Exception as exc:  # noqa: BLE001 - collected for the report
            failures.append(f"{path.name}: {exc}")
    assert not failures, "unrecoverable dl checkpoints:\n" + "\n".join(failures)


def test_equal_width_ts_and_cf_jepa_report_ambiguity() -> None:
    """Structurally identical variants must be flagged, not silently chosen."""
    path = ISAC_DIR / "cf_jepa_ft_seed42.pt"
    if not path.exists():
        pytest.skip("checkpoint not present")

    sidecar = infer(path, name="cf_jepa_base")
    ambiguous = sidecar.provenance.get("ambiguous_with") or []
    assert ambiguous, "expected the TS-JEPA/CF-JEPA ambiguity to be recorded"


def test_pretrain_bundles_are_recognised_not_speculatively_adopted() -> None:
    """A pretrain bundle must be named as such, not guessed at."""
    path = ISAC_DIR / "lejepa_pretrain_best.pt"
    if not path.exists():
        pytest.skip("pretrain checkpoint not present")

    state = read_state(path)
    assert looks_like_pretrain(state), "pretrain bundle not recognised"
    description = describe_pretrain(state)
    assert "predictor" in description

    with pytest.raises(ValueError, match="pre-training checkpoint"):
        infer(path)
