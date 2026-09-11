"""Infer a :class:`~inmotion.models.checkpoint.Sidecar` from an untagged checkpoint.

Existing checkpoints predate the sidecar convention, so their architecture has to
be recovered. Two strategies are tried in order:

1. **Signature match.** Every registered spec's module is built and its state-dict
   key set and shapes are compared with the checkpoint's. An exact match
   identifies the architecture with no guessing.
2. **Shape-driven search.** For a checkpoint nothing in the registry matches, a
   bounded per-family grid is searched until a configuration loads with
   ``strict=True``. Dimensions are read out of the tensor shapes rather than
   hardcoded.

If neither succeeds, the error lists what was tried and how each candidate
differed, so a factory can be added deliberately instead of the model being
loaded with ``strict=False`` and left partly random.

Key normalisation
-----------------
Checkpoints may or may not carry a wrapper prefix (``"c."`` for the JEPA
classifier, ``"m."`` for :class:`~inmotion.models.wrappers.TupleWrap`). Detection
works on *plain* keys; the prefix a factory needs is applied only when checking a
candidate, so a raw SIGReg checkpoint and a re-saved wrapped one are both
recognised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from .checkpoint import Sidecar
from .factories import build
from .registry import REGISTRY
from .wrappers import normalise_state

# Wrapper prefixes a factory might need; tried in order when not already known.
CANDIDATE_PREFIXES: tuple[str | None, ...] = (None, "m.")


def read_state(path: Path | str) -> dict:
    """Load a state dict, accepting a raw dict or a module checkpoint."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise TypeError(f"{path}: cannot interpret {type(obj).__name__} as a state dict")


def plain_state(state: dict) -> dict:
    """Strip any wrapper prefix so detection sees canonical child names."""
    keys = list(state)
    if not keys:
        return {}
    for prefix in ("c.", "m."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in state.items()}
    return dict(state)


def _matches(factory: str, arch: dict, prefix: str | None, plain: dict) -> bool:
    """True when ``arch`` reproduces ``plain`` exactly in keys and shapes."""
    try:
        candidate = build(factory, arch)
    except Exception:
        return False
    target = normalise_state(plain, prefix)
    own = candidate.state_dict()
    if set(own) != set(target):
        return False
    return all(own[k].shape == target[k].shape for k in own)


def _recover_prefix(factory: str, arch: dict, plain: dict) -> str | None:
    """The prefix a factory needs to match ``plain``, or None if none works."""
    for prefix in CANDIDATE_PREFIXES:
        if _matches(factory, arch, prefix, plain):
            return prefix
    return None


def match_all(plain: dict) -> list[tuple[str, dict, str | None, str]]:
    """Every registered architecture that reproduces ``plain`` exactly.

    More than one can match: structurally identical variants (e.g. TS-JEPA and
    CF-JEPA at the same width) are indistinguishable from weights alone, and the
    caller has to disambiguate from provenance rather than shapes.
    """
    out: list[tuple[str, dict, str | None, str]] = []
    seen: set[str] = set()
    for spec in REGISTRY.values():
        signature = f"{spec.factory}:{sorted(spec.arch.items())}"
        if signature in seen:
            continue
        seen.add(signature)
        prefix = _recover_prefix(spec.factory, spec.arch, plain)
        if prefix is not None or _matches(spec.factory, spec.arch, spec.prefix, plain):
            out.append(
                (
                    spec.factory,
                    dict(spec.arch),
                    prefix if prefix is not None else spec.prefix,
                    spec.name,
                )
            )
    return out


def match_registry(plain: dict) -> tuple[str, dict, str | None] | None:
    """First registered architecture that reproduces ``plain`` exactly."""
    matches = match_all(plain)
    return matches[0][:3] if matches else None


# ---------------------------------------------------------------------------
# Shape-driven detection. Each detector reads dimensions out of the checkpoint
# and returns a small candidate grid, so the search stays bounded.
# ---------------------------------------------------------------------------


def _rows(t: torch.Tensor) -> int:
    return int(t.shape[0])


def _heads(d_model: int) -> int:
    """Largest supported head count that divides ``d_model``."""
    for h in (8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


def _layers(keys: Iterable[str], marker: str) -> int:
    """Count sibling modules under a ``...<marker>.<n>.`` path."""
    n = 0
    for k in keys:
        if marker in k:
            tail = k.split(marker, 1)[1]
            head = tail.split(".")[0]
            if head.isdigit():
                n = max(n, int(head) + 1)
    return n


def _sigreg_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    if "encoder.bn.weight" not in plain:
        return []
    return [
        (
            "sigreg",
            {
                "num_filters": _rows(plain["encoder.bn.weight"]),
                "num_blocks": _layers(keys, "encoder.blocks."),
                "latent_dim": _rows(plain["latent_proj.weight"])
                if "latent_proj.weight" in plain
                else 128,
            },
        )
    ]


def _mamba3_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    d_state_key = next((k for k in keys if k.endswith("ssm.dt_proj.weight")), None)
    d_state = int(plain[d_state_key].shape[1]) if d_state_key else 16

    if any(k.startswith("tcn_enc.") for k in keys):
        out.append(
            (
                "mamba3_tcn",
                {
                    "tcn_channels": _rows(plain["tcn_enc.network.0.conv1.conv.weight_v"]),
                    "tcn_depth": _layers(keys, "tcn_enc.network."),
                    "d_model": _rows(plain["mamba_input_proj.weight"]),
                    "d_state": d_state,
                    "n_mamba_layers": _layers(keys, "mamba."),
                },
            )
        )
    elif any(k.startswith("mamba_blocks.") for k in keys):
        conv = next(k for k in keys if k.startswith("local_view.conv.0.weight"))
        out.append(
            (
                "mamba3_multiview",
                {
                    "d_model": _rows(plain[conv]),
                    "d_state": d_state,
                    "n_mamba_layers": _layers(keys, "mamba_blocks."),
                },
            )
        )
    elif "input_proj.weight" in plain and any("attn" in k for k in keys):
        out.append(
            (
                "mamba3_transformer",
                {
                    "d_model": _rows(plain["input_proj.weight"]),
                    "d_state": d_state,
                    "nhead": _heads(_rows(plain["input_proj.weight"])),
                    "num_blocks": _layers(keys, "blocks."),
                },
            )
        )
    return out


def _jepa_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """Candidates for the JEPA / world-model family."""
    tok = next((k for k in keys if k.endswith("tokenizer.weight")), None)
    if tok is None:
        return []
    d_model = _rows(plain[tok])
    in_ch = int(plain[tok].shape[1])
    n_layers = _layers(keys, "encoder.transformer.layers.")
    ff_key = next((k for k in keys if k.endswith("layers.0.linear1.weight")), None)
    dim_ff = _rows(plain[ff_key]) if ff_key else 4 * d_model
    nhead = _heads(d_model)
    common: dict[str, Any] = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": n_layers,
        "dim_feedforward": dim_ff,
    }

    if "encoder.tokenizer.feat_idx_embed.weight" in plain:
        return [("t_jepa", {**common, "n_channels": in_ch})]
    if any("context_encoder" in k for k in keys):
        return [("cf_jepa", {**common, "in_channels": in_ch})]
    if any("patch_embed" in k for k in keys):
        return [("ts_jepa", {**common, "in_channels": in_ch, "embed_dim": d_model})]
    # LeJEPA and T-JEPA both use a tokenizer; the feature-index embedding above
    # distinguishes T-JEPA, so a bare tokenizer with 4 inputs is LeJEPA.
    return [("lejepa", {**common, "in_channels": in_ch})]


def _deep_stack_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    if any(k.startswith("base_models.") for k in keys):
        return [("deep_stack", {})]
    return []


DETECTORS = (_mamba3_grid, _sigreg_grid, _jepa_grid, _deep_stack_grid)


def search_family(plain: dict) -> tuple[str, dict, str | None] | None:
    """Search the per-family grids for a strictly-loadable configuration."""
    keys = set(plain)
    for detector in DETECTORS:
        for factory, arch in detector(keys, plain):
            prefix = _recover_prefix(factory, arch, plain)
            if prefix is not None or _matches(factory, arch, None, plain):
                return factory, arch, prefix
    return None


def _diag(plain: dict, factory: str, arch: dict) -> str:
    """Why a candidate architecture does not fit the checkpoint."""
    try:
        own = build(factory, arch).state_dict()
    except Exception as exc:
        return f"build failed: {type(exc).__name__}: {exc}"
    target = normalise_state(plain, None)
    missing = sorted(set(own) - set(target))
    extra = sorted(set(target) - set(own))
    shaped = sorted(k for k in set(own) & set(target) if own[k].shape != target[k].shape)
    parts = []
    if missing:
        parts.append(f"{len(missing)} absent (e.g. {missing[:2]})")
    if extra:
        parts.append(f"{len(extra)} unexpected (e.g. {extra[:2]})")
    if shaped:
        parts.append(f"{len(shaped)} shape mismatches (e.g. {shaped[:2]})")
    return "; ".join(parts) or "no differences"


def infer(
    path: Path | str,
    *,
    name: str | None = None,
    factory: str | None = None,
) -> Sidecar:
    """Recover a :class:`Sidecar` for an untagged checkpoint.

    ``factory`` pins the architecture when the checkpoint is ambiguous. ``name``
    is used to prefer a registered spec of the same name, which resolves the
    common case of relaunching adoption for a checkpoint already in the registry.

    When several architectures fit, the ambiguity is recorded in the sidecar's
    ``provenance["ambiguous_with"]`` rather than being hidden, because shapes
    cannot decide between structurally identical variants.

    Raises ``ValueError`` when nothing reproduces the checkpoint exactly.
    """
    path = Path(path)
    state = read_state(path)
    if not state:
        raise ValueError(f"{path}: empty state dict")
    plain = plain_state(state)
    model_name = name or path.stem

    registry_hits = match_all(plain)
    if factory is not None:
        registry_hits = [h for h in registry_hits if h[0] == factory]

    if registry_hits:
        # Prefer the spec whose registered name matches the requested name.
        chosen = next((h for h in registry_hits if h[3] == model_name), registry_hits[0])
        other_factories = sorted({h[0] for h in registry_hits} - {chosen[0]})
        others = sorted(h[3] for h in registry_hits if h[3] != chosen[3])
        provenance: dict[str, Any] = {
            "recovered_by": "registry_signature_match",
            "checkpoint": str(path),
            "matched_spec": chosen[3],
        }
        if others:
            provenance["ambiguous_with"] = others
            provenance["ambiguous_factories"] = other_factories
        return Sidecar(
            name=model_name,
            factory=chosen[0],
            arch=chosen[1],
            params=sum(p.numel() for p in build(chosen[0], chosen[1]).parameters()),
            prefix=chosen[2],
            source="adopted",
            provenance=provenance,
        )

    found = search_family(plain)
    if found is not None:
        fac, arch, prefix = found
        if factory is not None and fac != factory:
            found = None
        else:
            return Sidecar(
                name=model_name,
                factory=fac,
                arch=arch,
                params=sum(p.numel() for p in build(fac, arch).parameters()),
                prefix=prefix,
                source="adopted",
                provenance={"recovered_by": "shape_search", "checkpoint": str(path)},
            )

    tried: list[str] = []
    for detector in DETECTORS:
        for fac, arch in detector(set(plain), plain):
            tried.append(f"  {fac}({arch}) -> {_diag(plain, fac, arch)}")
    detail = "\n".join(tried) or "  (no detector recognised this checkpoint)"
    raise ValueError(
        f"{path}: could not reproduce this checkpoint with any known architecture.\n"
        f"Add a factory in inmotion/models/factories.py. Attempts:\n{detail}"
    )


def infer_many(paths: Iterable[Path]) -> dict[str, Sidecar | Exception]:
    """Adopt every checkpoint in ``paths``, recording failures instead of raising."""
    out: dict[str, Sidecar | Exception] = {}
    for path in paths:
        try:
            out[str(path)] = infer(path)
        except Exception as exc:  # noqa: BLE001 - collected for reporting
            out[str(path)] = exc
    return out
