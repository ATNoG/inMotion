"""SIGReg's architecture must be stated, not inferred.

``_build_sigreg`` used to read the filter count from ``--d-model`` and the block
count from ``--num-layers``. Those names mean width and depth for the
transformer family, so the documented retrain command
(``inmotion train exotic --model sigreg ...``) silently built a 256-filter,
4-block model while the registry expected the registered 192/3 checkpoint — and
wrote it to the checkpoint's own output path.

These tests pin the flags to the architecture at the CLI boundary, which is
where the mismatch was introduced.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from inmotion.cli import train as train_cli
from inmotion.models import registry as reg
from inmotion.pipelines.exotic import RunConfig
from inmotion.pipelines.exotic.builders import MODEL_BUILDERS

#: The registered ``sigreg`` architecture (registry arch + factory defaults).
REGISTERED_FILTERS = 192
REGISTERED_BLOCKS = 3
REGISTERED_LATENT = 128
REGISTERED_PARAMS = 696_900


@pytest.fixture
def capture_cfg(monkeypatch: pytest.MonkeyPatch):
    """Invoke `inmotion train exotic` with the pipeline stubbed out.

    Only the CLI-to-RunConfig step runs, so the test needs no dataset and no
    checkpoint store: exactly the layer where the architecture was mis-stated.
    """

    def _capture(*args: str) -> RunConfig:
        captured: dict[str, RunConfig] = {}

        def fake_run(run, cfg):  # noqa: ANN001 - the pipeline callable
            captured["cfg"] = cfg
            return 0

        monkeypatch.setattr(train_cli, "run_pipeline", fake_run)
        result = CliRunner().invoke(train_cli.app, ["exotic", *args])
        assert result.exit_code == 0, result.output
        return captured["cfg"]

    return _capture


def _params(model) -> int:  # noqa: ANN001 - torch module
    return sum(p.numel() for p in model.parameters())


def test_bare_sigreg_run_builds_the_registered_architecture(capture_cfg) -> None:
    """`--model sigreg` with no flags must be the architecture the registry loads."""
    cfg = capture_cfg("--model", "sigreg")
    assert (cfg.num_filters, cfg.num_blocks, cfg.latent_dim) == (
        REGISTERED_FILTERS,
        REGISTERED_BLOCKS,
        REGISTERED_LATENT,
    )

    model = MODEL_BUILDERS["sigreg"](cfg)
    assert _params(model) == REGISTERED_PARAMS

    # Independent anchor: whatever the registry builds from its own arch is what
    # a bare CLI run builds. No checkpoint file on disk is required. The registry
    # factory wraps the module (`m.` prefix), which carries no parameters, so the
    # prefix is stripped before the comparison.
    spec = reg.by_name("sigreg")
    registered = reg.build(spec.factory, spec.arch)
    registered_state = {k.removeprefix("m."): v for k, v in registered.state_dict().items()}
    assert set(model.state_dict()) == set(registered_state)
    assert all(model.state_dict()[k].shape == registered_state[k].shape for k in registered_state)
    assert _params(model) == _params(registered)


def test_transformer_flags_do_not_touch_sigreg(capture_cfg) -> None:
    """--d-model/--num-layers are for the transformer family and must be inert here."""
    cfg = capture_cfg("--model", "sigreg", "--d-model", "512", "--num-layers", "9")
    assert (cfg.num_filters, cfg.num_blocks) == (REGISTERED_FILTERS, REGISTERED_BLOCKS)
    assert _params(MODEL_BUILDERS["sigreg"](cfg)) == REGISTERED_PARAMS


def test_num_filters_and_num_blocks_are_honoured(capture_cfg) -> None:
    from inmotion.dl.models.sigreg_classifier import SIGRegClassifier

    cfg = capture_cfg("--model", "sigreg", "--num-filters", "128", "--num-blocks", "2")
    assert (cfg.num_filters, cfg.num_blocks) == (128, 2)

    model = MODEL_BUILDERS["sigreg"](cfg)
    reference = SIGRegClassifier(
        in_features=4,
        num_filters=128,
        num_blocks=2,
        latent_dim=128,
        num_classes=4,
        sigreg_lambda=cfg.sigreg_lambda,
    )
    assert _params(model) == _params(reference)
    assert _params(model) != REGISTERED_PARAMS, "the flags changed nothing"


def _jepa(capture_cfg, *flags: str):  # noqa: ANN001, ANN202 - fixture-built
    return MODEL_BUILDERS["lejepa"](capture_cfg("--model", "lejepa", *flags))


def test_transformer_flags_still_drive_the_jepa_family(capture_cfg) -> None:
    """--d-model/--num-layers keep their meaning, and SIGReg's flags are inert here."""
    default = _jepa(capture_cfg)
    small = _jepa(capture_cfg, "--d-model", "64", "--num-layers", "2")
    large = _jepa(capture_cfg, "--d-model", "128", "--num-layers", "2")
    more_layers = _jepa(capture_cfg, "--d-model", "64", "--num-layers", "3")
    sigreg_flags = _jepa(capture_cfg, "--num-filters", "512", "--num-blocks", "9")

    assert len(small.encoder.transformer.layers) == 2
    assert small.encoder.transformer.layers[0].linear1.in_features == 64
    assert len(more_layers.encoder.transformer.layers) == 3
    assert large.encoder.transformer.layers[0].linear1.in_features == 128
    assert _params(large) != _params(small), "d_model no longer changes the model"
    assert _params(more_layers) != _params(small), "num_layers no longer changes the model"
    assert _params(sigreg_flags) == _params(default), "sigreg flags leaked into lejepa"


def test_help_lists_the_sigreg_flags() -> None:
    result = CliRunner().invoke(train_cli.app, ["exotic", "--help"])
    assert result.exit_code == 0, result.output
    assert "--num-filters" in result.output
    assert "--num-blocks" in result.output


def test_hpo_winner_lands_on_the_sigreg_flags() -> None:
    """A search winner must reach the builder through the flags it reads.

    The sigreg search suggests ``num_filters``/``num_blocks``. If it kept using
    ``d_model``/``num_layers``, ``_apply_trial_params`` would update fields the
    builder no longer reads and the fall-through training run would silently
    ignore the winning architecture.
    """
    from inmotion.dl.models.sigreg_classifier import SIGRegClassifier
    from inmotion.pipelines.exotic.hpo import _apply_trial_params

    cfg = RunConfig(model="sigreg")
    _apply_trial_params(cfg, {"num_filters": 256, "num_blocks": 4, "latent_dim": 256})
    assert (cfg.num_filters, cfg.num_blocks, cfg.latent_dim) == (256, 4, 256)

    model = MODEL_BUILDERS["sigreg"](cfg)
    reference = SIGRegClassifier(
        in_features=4,
        num_filters=256,
        num_blocks=4,
        latent_dim=256,
        num_classes=4,
        sigreg_lambda=cfg.sigreg_lambda,
    )
    assert _params(model) == _params(reference)
    assert _params(model) != REGISTERED_PARAMS
