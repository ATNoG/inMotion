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

from inmotion.models.adopt import (
    describe_pretrain,
    infer,
    looks_like_pretrain,
    read_state,
    resolve_model,
    spec_for_checkpoint,
)
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


# ---------------------------------------------------------------------------
# The HPO checkpoints: they are not in the registry, and the input width they
# need is not visible in the tensor shapes. `t_jepa`'s tokenizer is (182, 128),
# which is 180 flattened features, 18 channels x 10 timesteps; `cf_jepa`'s is a
# Conv1d with in_channels=18; `lejepa` and `sigreg` take 4.
#
# Getting this wrong is the bad kind of failure: the model loads, the features
# are the wrong width, and the numbers come out plausible and wrong rather than
# raising.
# ---------------------------------------------------------------------------

HPO_DIR = ISAC_DIR / "full-after-hpo"

#: filename -> whether the model consumes rich (18-channel) features.
HPO_CHANNELS = {
    "lejepa_ft_seed42.pt": False,
    "sigreg_seed42.pt": False,
    "t_jepa_ft_seed42.pt": True,
    "ts_jepa_ft_seed42.pt": True,
    "cf_jepa_ft_seed42.pt": True,
}


@pytest.mark.parametrize(("filename", "rich"), sorted(HPO_CHANNELS.items()))
def test_adopted_checkpoint_reports_its_channel_requirement(filename: str, rich: bool) -> None:
    """`rich` must come from the matched spec, never from the default."""
    path = HPO_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")

    spec = spec_for_checkpoint(str(path))
    assert spec.rich is rich, f"{filename}: rich={spec.rich}, expected {rich}"
    build(spec.factory, spec.arch)


def test_resolve_model_accepts_a_name_or_a_checkpoint_path() -> None:
    """One entry point, so a command does not have to know which it was given."""
    from inmotion.models.registry import REGISTRY

    by_name = resolve_model("lejepa")
    assert by_name is REGISTRY["lejepa"], "a registered name must come back verbatim"

    path = HPO_DIR / "sigreg_seed42.pt"
    if not path.exists():
        pytest.skip("sigreg_seed42.pt not present")
    by_path = resolve_model(str(path))
    assert by_path.name == "sigreg_seed42"
    assert by_path.source == "adopted"


def test_resolve_model_rejects_a_name_that_is_not_a_checkpoint() -> None:
    with pytest.raises(KeyError, match="neither a registered model"):
        resolve_model("not_a_model_and_not_a_file")
