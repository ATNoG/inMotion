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


def _recurrent_dims(plain: dict, prefix: str) -> tuple[int, int, bool] | None:
    """(hidden_size, num_layers, bidirectional) from an RNN/LSTM/GRU state dict."""
    ih0 = plain.get(f"{prefix}.weight_ih_l0")
    hh0 = plain.get(f"{prefix}.weight_hh_l0")
    if ih0 is None or hh0 is None:
        return None
    gates = 3 if prefix == "gru" else 4  # GRU has 3 gates, LSTM/RNN 1..4
    if prefix == "rnn":
        gates = 1
    hidden = int(hh0.shape[1]) if prefix == "rnn" else int(hh0.shape[0]) // gates
    bidirectional = f"{prefix}.weight_ih_l0_reverse" in plain
    n = 0
    while f"{prefix}.weight_ih_l{n}" in plain:
        n += 1
    return hidden, n, bidirectional


def _dl_recurrent_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """RNN / GRU / LSTM / BiLSTM candidates."""
    out: list[tuple[str, dict]] = []
    in_features = (
        int(plain[[k for k in keys if k.endswith("weight_ih_l0")][0]].shape[1])
        if any(k.endswith("weight_ih_l0") for k in keys)
        else None
    )
    if in_features is None:
        return out

    has_attn = "attn.weight" in plain
    for prefix, factories in (("rnn", ["rnn"]), ("gru", ["gru"]), ("lstm", ["bilstm", "lstm"])):
        dims = _recurrent_dims(plain, prefix)
        if dims is None:
            continue
        hidden, layers, bidir = dims
        for fac in factories:
            # BiLSTMClassifier is inherently bidirectional.
            if fac == "bilstm" and not bidir:
                continue
            candidate = {"in_features": in_features, "hidden_size": hidden}
            if fac == "bilstm":
                candidate.update({"num_layers": layers, "use_attention": has_attn})
            else:
                candidate.update({"num_layers": layers, "bidirectional": bidir})
                # Only the GRU/LSTM classifiers have an attention head.
                if fac in ("gru", "lstm"):
                    candidate["use_attention"] = has_attn
            out.append((fac, candidate))
    return out


def _dl_conv_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """TCN and multi-branch CNN candidates."""
    out: list[tuple[str, dict]] = []

    if "network.0.conv1.conv.weight_v" in plain:
        first = plain["network.0.conv1.conv.weight_v"]
        out.append(
            (
                "tcn",
                {
                    "in_features": int(first.shape[1]) if first.dim() == 3 else 4,
                    "num_channels": _rows(plain["network.0.conv1.conv.bias"]),
                    "kernel_size": int(first.shape[2]),
                    "depth": _layers(keys, "network."),
                },
            )
        )

    if any(k.startswith("branches.") for k in keys):
        n_branches = _layers(keys, "branches.")
        kernels = []
        for i in range(n_branches):
            key = f"branches.{i}.0.weight"
            if key not in plain:
                break
            kernels.append(int(plain[key].shape[2]))
        out.append(
            (
                "cnn",
                {
                    # classifier.weight is (num_classes, num_filters), so the width is
                    # the *second* dimension.
                    "num_filters": int(plain["classifier.weight"].shape[1]),
                    "num_blocks": _layers(keys, "res_blocks."),
                    "kernel_sizes": kernels or None,
                    "in_features": int(plain["branches.0.0.weight"].shape[1])
                    if "branches.0.0.weight" in plain
                    else 4,
                },
            )
        )
    return out


def _dl_mamba_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """MambaClassifier candidates.

    ``d_state`` cannot be read directly: ``dt_proj`` is shaped
    ``(d_inner, dt_rank)`` and the state width only appears in ``x_proj``, whose
    output is ``dt_rank + 2 * d_state * mimo_rank``. Both unknowns are enumerated
    over the values the training code uses and resolved by the strict-load check.
    """
    # MambaClassifier.input_proj is a Sequential(Linear, LayerNorm), so its
    # weight lives at "input_proj.0.weight" -- not "input_proj.weight".
    proj_key = "input_proj.0.weight" if "input_proj.0.weight" in plain else "input_proj.weight"
    if "blocks.0.in_proj.weight" not in plain or proj_key not in plain:
        return []
    in_proj = plain["blocks.0.in_proj.weight"]
    d_model = int(in_proj.shape[1])
    d_inner = int(in_proj.shape[0]) // 2
    dt = plain.get("blocks.0.ssm.dt_proj.weight")
    x_proj = plain.get("blocks.0.ssm.x_proj.weight")

    candidates: list[tuple[str, dict]] = []
    if dt is None or dt.dim() != 2:
        return [
            (
                "mamba",
                {
                    "in_features": int(plain[proj_key].shape[1]),
                    "d_model": d_model,
                    "num_layers": _layers(keys, "blocks."),
                },
            )
        ]

    dt_rank = int(dt.shape[1])
    base = {
        "in_features": int(plain[proj_key].shape[1]),
        "d_model": d_model,
        "expand": max(1, d_inner // d_model) if d_model else 2,
        "num_layers": _layers(keys, "blocks."),
    }
    if x_proj is not None and x_proj.dim() == 2:
        remainder = int(x_proj.shape[0]) - dt_rank
        for mimo_rank in (1, 2, 4, 8):
            if remainder <= 0:
                break
            d_state, leftover = divmod(remainder, 2 * mimo_rank)
            if leftover == 0 and d_state > 0:
                candidates.append(("mamba", {**base, "d_state": d_state, "mimo_rank": mimo_rank}))
    if not candidates:
        candidates.append(("mamba", {**base, "d_state": 16}))
    return candidates


def _dl_transformer_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    if "input_proj.weight" not in plain or not any("self_attn" in k for k in keys):
        return []
    if "pos_emb" not in plain:
        return []
    d_model = _rows(plain["input_proj.weight"])
    return [
        (
            "transformer",
            {
                "in_features": int(plain["input_proj.weight"].shape[1]),
                "d_model": d_model,
                "n_heads": _heads(d_model),
                "num_layers": _layers(keys, "encoder.layers."),
                "max_seq_len": int(plain["pos_emb"].shape[1])
                if plain["pos_emb"].dim() == 3
                else 512,
            },
        )
    ]


def _dl_fusion_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """MetaFusion, which carries metadata embeddings."""
    if "noise_embed.weight" not in plain or "rssi_norm.weight" not in plain:
        return []
    # The sub-module is always named `rnn.*`, so rnn_type is not recoverable
    # from key names; enumerate the three the classifier supports.
    dims = _recurrent_dims(plain, "rnn")
    if dims is None:
        return []
    hidden, layers, bidir = dims
    out: list[tuple[str, dict]] = []
    for rnn_type in ("lstm", "gru", "rnn"):
        out.append(
            (
                "meta_fusion",
                {
                    "in_features": int(plain["rnn.weight_ih_l0"].shape[1]),
                    "hidden_size": hidden,
                    "num_layers": layers,
                    "meta_embed_dim": int(plain["noise_embed.weight"].shape[1]),
                    "rnn_type": rnn_type,
                    "bidirectional": bidir,
                },
            )
        )
    return out


def _dl_cnn2d_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    if "cnn.0.block.0.weight" not in plain:
        return []
    rnn_key = "rnn.weight_ih_l0"
    if rnn_key not in plain:
        return []
    dims = _recurrent_dims(plain, "rnn")
    if dims is None:
        return []
    hidden, n_rnn_layers, bidir = dims
    out: list[tuple[str, dict]] = []
    for rnn_type in ("lstm", "gru", "rnn"):
        out.append(
            (
                "cnn2d_rnn",
                {
                    "num_filters": int(plain["cnn.0.block.0.weight"].shape[0]),
                    "cnn_depth": _layers(keys, "cnn."),
                    "hidden_size": hidden,
                    "num_rnn_layers": n_rnn_layers,
                    "rnn_type": rnn_type,
                    "bidirectional": bidir,
                },
            )
        )
    return out


DETECTORS = (
    _mamba3_grid,
    _sigreg_grid,
    _jepa_grid,
    _deep_stack_grid,
    _dl_mamba_grid,
    _dl_transformer_grid,
    _dl_fusion_grid,
    _dl_conv_grid,
    _dl_recurrent_grid,
    _dl_cnn2d_grid,
)


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


def resolve_plain(
    plain: dict, *, factory: str | None = None
) -> tuple[str, dict, str | None] | None:
    """Resolve a *plain* (prefix-stripped) state dict to (factory, arch, prefix)."""
    hits = match_all(plain)
    if factory is not None:
        hits = [h for h in hits if h[0] == factory]
    if hits:
        return hits[0][0], hits[0][1], hits[0][2]
    found = search_family(plain)
    if found is not None and (factory is None or found[0] == factory):
        return found
    return None


def _members(keys: set[str], plain: dict, container: str) -> list[dict[str, Any]] | None:
    """Recursively resolve the members of an ensemble container."""
    indices = set()
    for k in keys:
        if k.startswith(f"{container}."):
            head = k[len(container) + 1 :].split(".")[0]
            if head.isdigit():
                indices.add(int(head))
    if not indices:
        return None

    members: list[dict[str, Any]] = []
    for index in range(max(indices) + 1):
        marker = f"{container}.{index}."
        sub = {k[len(marker) :]: v for k, v in plain.items() if k.startswith(marker)}
        if not sub:
            return None
        found = resolve_plain(plain_state(sub))
        if found is None:
            return None
        factory, arch, _prefix = found
        members.append({"factory": factory, "arch": arch})
    return members


def _composite_grid(keys: set[str], plain: dict) -> list[tuple[str, dict]]:
    """Voting and stacking ensembles, rebuilt from their own members.

    Each member is resolved with the same machinery used for a standalone
    checkpoint, so an ensemble is recoverable exactly when its members are.
    """
    out: list[tuple[str, dict]] = []

    if "weights" in plain and any(k.startswith("models.") for k in keys):
        members = _members(keys, plain, "models")
        if members:
            weights = plain["weights"].flatten().tolist()
            out.append(("voting_ensemble", {"models": members, "weights": weights}))

    if any(k.startswith("base_models.") for k in keys) and any(k.startswith("meta.") for k in keys):
        members = _members(keys, plain, "base_models")
        if members:
            out.append(("stacking_ensemble", {"base_models": members}))

    return out


# ---------------------------------------------------------------------------
# Pre-training checkpoints.
#
# The `*_pretrain_best.pt` files are not state dicts of a single module. They
# hold a bundle: ``predictor`` plus either ``encoder`` (LeJEPA) or
# ``context_encoder`` and ``target_encoder`` (T-JEPA / TS-JEPA / CF-JEPA), along
# with ``epoch`` and ``probe_mcc``.
#
# Rebuilding one means loading the parts into the matching attributes, which is
# what the exotic pipeline already does. Reconstructing it from shapes alone is
# ambiguous --
# the predictor's configuration is not determined by the encoder's -- and a
# wrong guess produces a checkpoint that loads and then quietly resumes with a
# mis-shaped predictor. So these are recognised and reported explicitly instead
# of being adopted speculatively. Resuming training keeps working through
# `inmotion run pipeline exotic -- --checkpoint`, which is unchanged.
# ---------------------------------------------------------------------------

PRETRAIN_MARKERS = ("predictor",)


def looks_like_pretrain(state: dict) -> bool:
    """True when ``state`` is a pre-training bundle rather than a state dict."""
    if not all(marker in state for marker in PRETRAIN_MARKERS):
        return False
    return any(key in state for key in ("encoder", "context_encoder"))


def describe_pretrain(state: dict) -> str:
    """A one-line description of a pre-training bundle."""
    encoder_key = "encoder" if "encoder" in state else "context_encoder"
    encoder = plain_state(state[encoder_key])
    predictor = plain_state(state.get("predictor", {}))
    epoch = state.get("epoch", "?")
    mcc = state.get("probe_mcc")
    mcc_text = f", probe_mcc={mcc:.4f}" if isinstance(mcc, float) else ""
    return (
        f"pre-training bundle: {encoder_key}={len(encoder)} tensors, "
        f"predictor={len(predictor)} tensors, epoch={epoch}{mcc_text}"
    )


DETECTORS = DETECTORS + (_composite_grid,)


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

    if looks_like_pretrain(state):
        raise ValueError(
            f"{path}: {describe_pretrain(state)}.\n"
            "This is a pre-training checkpoint, not a classifier: it carries a "
            "separate predictor and encoder rather than one module's weights, so "
            "it has no single architecture to adopt. Resume training with "
            "`inmotion run pipeline exotic -- --model <name> --checkpoint "
            f"{path}`, which restores the parts explicitly."
        )

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
