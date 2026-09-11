"""Model registry — the single source of truth for loadable models.

A :class:`ModelSpec` records *everything* needed to rebuild a saved model:

``factory``  which architecture constructor to call (see ``factories.py``)
``arch``     the keyword arguments for it
``prefix``   state-dict wrapping, if the checkpoint was saved from a wrapper
``params``   expected parameter count, used as a checksum

Loading is deliberately strict. The previous code called
``load_state_dict(..., strict=False)``, which turns an architecture mismatch
into a *silently random* subset of weights — the model still runs and still
returns numbers, so the error is invisible. :func:`load` refuses by default.

Adding a model is data entry, not code. Use ``adopt`` (see ``inmotion model
adopt``) to infer a spec from a checkpoint's tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .factories import build
from .wrappers import normalise_state

REPO_ROOT = Path(__file__).resolve().parents[3]

_ISAC = Path("20-aug/models/exotic/normal-new-ds")
_BACKUP = Path("backup")


@dataclass(frozen=True)
class ModelSpec:
    """A rebuildable saved model."""

    name: str
    factory: str
    arch: dict[str, Any]
    path: Path
    params: int
    published_m: float | None = None
    """Parameter count in millions as printed in MASTER_RESULTS.md."""
    prefix: str | None = None
    rich: bool = False
    """True when the model consumes 18-channel rich features, False for 4."""
    note: str = ""
    source: str = "curated"
    """``curated`` for hand-verified entries, ``adopted`` for inferred ones."""

    @property
    def abs_path(self) -> Path:
        return self.path if self.path.is_absolute() else REPO_ROOT / self.path

    def exists(self) -> bool:
        return self.abs_path.exists()


# ---------------------------------------------------------------------------
# Curated specs — every model cited in docs/isac/results/MASTER_RESULTS.md.
# Parameter counts are asserted by tests/test_checkpoint_contract.py.
# ---------------------------------------------------------------------------
CURATED: list[ModelSpec] = [
    # -- MASTER_RESULTS.md §2.3 — ICAISF-evaluated ensemble members.
    # `params` is the *module* parameter count; `published_m` is the figure in
    # the paper table (which counted the checkpoint's tensors, buffers included).
    # Where the two disagree the artifact on disk is NOT the model the paper
    # scored — see `PROVENANCE_GAPS` below.
    ModelSpec(
        "lejepa",
        "lejepa",
        {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 256,
            "pred_num_layers": 3,
            "sigreg_lambda": 0.05,
        },
        _ISAC / "full-after-hpo" / "lejepa_ft_seed42.pt",
        565_765,
        published_m=0.566,
        note="after-HPO Pareto winner",
    ),
    ModelSpec(
        "t_jepa",
        "t_jepa",
        {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 256,
            "pred_dim": 64,
            "pred_num_layers": 3,
        },
        _ISAC / "full-after-hpo" / "t_jepa_ft_seed42.pt",
        633_477,
        rich=True,
        published_m=0.633,
        note="after-HPO Pareto winner",
    ),
    ModelSpec(
        "ts_jepa",
        "ts_jepa",
        {},
        _ISAC / "ts_jepa_ft_seed42.pt",
        2_153_349,
        rich=True,
        published_m=2.153,
        note="base run (18ch, embed 256)",
    ),
    ModelSpec(
        "cf_jepa",
        "cf_jepa",
        {},
        _ISAC / "full-after-hpo" / "cf_jepa_ft_seed42.pt",
        1_232_261,
        rich=True,
        published_m=1.232,
        note="after-HPO frontier point",
    ),
    ModelSpec(
        "sigreg",
        "sigreg",
        {"num_filters": 192},
        _BACKUP / "sigreg_seed42.pt",
        696_900,
        prefix="m.",
        published_m=0.697,
        note="192-filter encoder; paper §2.3 checkpoint",
    ),
    ModelSpec(
        "sigreg_seed3",
        "sigreg",
        {"num_filters": 192},
        _BACKUP / "sigreg_seed3.pt",
        696_900,
        prefix="m.",
        published_m=0.697,
        note="multi-seed diversity member",
    ),
    ModelSpec(
        "sigreg_seed5",
        "sigreg",
        {"num_filters": 192},
        _BACKUP / "sigreg_seed5.pt",
        696_900,
        prefix="m.",
        published_m=0.697,
        note="multi-seed diversity member",
    ),
    ModelSpec(
        "deep_stack",
        "deep_stack",
        {},
        _BACKUP / "DeepStackEnsemble_seed42.pt",
        43_575_549,
        published_m=43.576,
        note="10-base deep stack; MASTER_RESULTS cites this under the stale "
        "path models/dl/augmented/DeepStackEnsemble_seed42.pt",
    ),
    # -- Mamba-3 hybrids. The stored checkpoints are SMALLER than the paper
    # table's parameter counts; see PROVENANCE_GAPS.
    ModelSpec(
        "mamba3_cnn", "mamba3_cnn", {}, _BACKUP / "mamba3_cnn_seed42.pt", 945_028, published_m=0.945
    ),
    ModelSpec(
        "mamba3_tcn",
        "mamba3_tcn",
        {
            "tcn_channels": 128,
            "tcn_depth": 4,
            "d_model": 128,
            "d_state": 16,
            "n_mamba_layers": 2,
            "mimo_rank": 4,
        },
        _BACKUP / "mamba3_tcn_seed42.pt",
        695_812,
        published_m=0.843,
    ),
    ModelSpec(
        "mamba3_transformer",
        "mamba3_transformer",
        {"d_model": 128, "d_state": 16, "nhead": 4, "num_blocks": 2, "mimo_rank": 4},
        _BACKUP / "mamba3_transformer_seed42.pt",
        561_796,
        published_m=0.842,
    ),
    ModelSpec(
        "mamba3_multiview",
        "mamba3_multiview",
        {"d_model": 256, "d_state": 16, "n_mamba_layers": 2, "mimo_rank": 4},
        _BACKUP / "mamba3_multiview_seed42.pt",
        1_193_349,
        published_m=1.701,
    ),
    # -- MASTER_RESULTS.md §2.1/2.2 — pre-HPO base runs, kept for provenance.
    # These need 18-channel input, which is why `rich` is True.
    ModelSpec(
        "lejepa_base",
        "lejepa",
        {
            "in_channels": 18,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "pred_num_layers": 2,
            "sigreg_lambda": 0.1,
        },
        _ISAC / "lejepa_ft_seed42.pt",
        2_215_813,
        rich=True,
        note="pre-HPO base run (2.1)",
    ),
    ModelSpec(
        "t_jepa_base",
        "t_jepa",
        {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "pred_dim": 128,
            "pred_num_layers": 2,
        },
        _ISAC / "t_jepa_ft_seed42.pt",
        2_347_653,
        rich=True,
        note="pre-HPO base run (2.1); nhead/pred_dim do not affect any saved "
        "tensor shape, so they are not recoverable from the checkpoint",
    ),
    ModelSpec(
        "cf_jepa_base",
        "cf_jepa",
        {"num_layers": 4, "dim_feedforward": 512},
        _ISAC / "cf_jepa_ft_seed42.pt",
        2_153_349,
        rich=True,
        note="pre-HPO base run (2.1)",
    ),
    ModelSpec(
        "sigreg_base",
        "sigreg",
        {"num_filters": 192, "num_blocks": 2},
        _ISAC / "sigreg_seed42.pt",
        474_564,
        prefix="m.",
        note="pre-HPO base run (2.1), 2 residual blocks",
    ),
    ModelSpec(
        "sigreg_after_hpo",
        "sigreg",
        {"num_filters": 128},
        _ISAC / "full-after-hpo" / "sigreg_seed42.pt",
        317_460,
        prefix="m.",
        note="128-filter retrain at the HPO winner config (2.2)",
    ),
]


# ---------------------------------------------------------------------------
# Provenance gaps — recorded, not papered over.
#
# Keyed by model name; each entry says what the artifact store actually holds
# versus what MASTER_RESULTS.md §2.3 publishes. `published_m` above is the
# paper figure, `params` is what the stored checkpoint really is.
# ---------------------------------------------------------------------------
PROVENANCE_GAPS: dict[str, str] = {
    "mamba3_tcn": (
        "MASTER_RESULTS.md §2.3 cites 0.843M params, but the stored checkpoint is "
        "695,812 module params (697,868 tensors incl. buffers). The published figure "
        "matches a 3-Mamba-layer build; the stored artifact has 2 layers and is "
        "1,128,349 params smaller. The MCC attributed to it (0.8844) therefore cannot "
        "be reproduced from this file."
    ),
    "mamba3_transformer": (
        "MASTER_RESULTS.md §2.3 cites 0.842M params, but the stored checkpoint is "
        "561,796 (563,076 tensors). The published figure matches a 3-block build; the "
        "stored artifact has 2 blocks. MCC 0.8581 is not reproducible from this file."
    ),
    "mamba3_multiview": (
        "MASTER_RESULTS.md §2.3 cites 1.701M params, but the stored checkpoint is "
        "1,193,349 (1,194,375 tensors), consistent with 2 Mamba layers rather than 3. "
        "MCC 0.8640 is not reproducible from this file."
    ),
    "deep_stack": (
        "MASTER_RESULTS.md §2.3 points at models/dl/augmented/DeepStackEnsemble_seed42.pt, "
        "which does not exist. The equivalent artifact used here is "
        "backup/DeepStackEnsemble_seed42.pt."
    ),
}


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in CURATED}


def spec_names() -> list[str]:
    """Names of every registered model."""
    return sorted(REGISTRY)


def by_name(name: str, path: Path | str | None = None) -> ModelSpec:
    """Look up a spec by name, optionally overriding its checkpoint path.

    Overriding the path is what lets the contract tests point a curated
    architecture at a checkpoint that lives somewhere else on disk.
    """
    if name not in REGISTRY:
        raise KeyError(f"no model named {name!r}; known: {spec_names()}")
    spec = REGISTRY[name]
    if path is None:
        return spec
    from dataclasses import replace

    return replace(spec, path=Path(path))


def all_specs() -> Iterable[ModelSpec]:
    """Every registered spec, including adopted ones."""
    return tuple(REGISTRY.values())


def load(
    spec: ModelSpec | str,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
    map_location: str = "cpu",
) -> nn.Module:
    """Build ``spec`` and load its checkpoint.

    ``strict=True`` (the default) raises on any key mismatch. Passing
    ``strict=False`` is possible but logged as a warning by ``verify`` because
    it silently leaves part of the model randomly initialised.
    """
    if isinstance(spec, str):
        spec = by_name(spec)

    path = spec.abs_path
    if not path.exists():
        raise FileNotFoundError(f"{spec.name}: checkpoint not found at {path}")

    model = build(spec.factory, spec.arch)
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{spec.name}: expected a state_dict, got {type(state).__name__}")
    state = normalise_state(state, spec.prefix)

    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()


def param_count(spec: ModelSpec) -> int:
    """Parameter count of the architecture as built from ``spec.arch``."""
    return sum(p.numel() for p in build(spec.factory, spec.arch).parameters())
