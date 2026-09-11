"""Checkpoint I/O with a JSON sidecar.

A ``.pt`` file alone is not a usable artifact: it does not say which
architecture it belongs to, how the inputs were scaled, or what the output
classes mean. :class:`Sidecar` carries that information next to the weights, and
:func:`save` / :func:`load` keep the two in sync.

Layout written by :func:`save`::

    <dir>/model.pt           weights
    <dir>/model.json         Sidecar
    <dir>/preprocessing.json FeaturePipeline metadata
    <dir>/preprocessing.npz  FeaturePipeline statistics

Loading verifies the sidecar against the weights: the architecture must build,
the state dict must load with ``strict=True``, and the parameter count must
match. Anything else raises rather than returning a partly-random model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..data.preprocessing import FeaturePipeline
from .factories import build
from .wrappers import normalise_state

SIDECAR_VERSION = 1
WEIGHTS_NAME = "model.pt"
SIDECAR_NAME = "model.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    """Best-effort current commit, so an artifact can be traced to its code."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class Sidecar:
    """Everything needed to rebuild and interpret a saved model."""

    name: str
    factory: str
    arch: dict[str, Any]
    params: int
    prefix: str | None = None
    rich: bool = False
    classes: list[str] = field(default_factory=list)
    # Free-form provenance: dataset, seed, metrics, notes.
    provenance: dict[str, Any] = field(default_factory=dict)
    # "curated" when the architecture was hand-verified against the paper,
    # "adopted" when it was inferred from the checkpoint's tensor shapes.
    source: str = "adopted"
    schema_version: int = SIDECAR_VERSION
    created_utc: str = field(default_factory=_utc_now)
    git_sha: str | None = field(default_factory=_git_sha)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Sidecar:
        data = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"sidecar has unknown fields: {sorted(unknown)}")
        return cls(**data)


def save(
    model: nn.Module,
    directory: Path | str,
    sidecar: Sidecar,
    pipeline: FeaturePipeline | None = None,
) -> Path:
    """Write weights, sidecar and (optionally) preprocessing state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    sidecar.params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), directory / WEIGHTS_NAME)
    (directory / SIDECAR_NAME).write_text(sidecar.to_json())
    if pipeline is not None:
        pipeline.save(directory)
        sidecar.classes = pipeline.classes or sidecar.classes
        (directory / SIDECAR_NAME).write_text(sidecar.to_json())
    return directory


def read_sidecar(directory: Path | str) -> Sidecar:
    directory = Path(directory)
    path = directory / SIDECAR_NAME
    if not path.exists():
        raise FileNotFoundError(f"no sidecar at {path}")
    return Sidecar.from_json(path.read_text())


def load(
    directory: Path | str,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, Sidecar, FeaturePipeline | None]:
    """Rebuild the model, its sidecar and (if present) its preprocessing state."""
    directory = Path(directory)
    sidecar = read_sidecar(directory)

    weights = directory / WEIGHTS_NAME
    if not weights.exists():
        raise FileNotFoundError(f"no weights at {weights}")

    model = build(sidecar.factory, sidecar.arch)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = normalise_state(state, sidecar.prefix)
    model.load_state_dict(state, strict=strict)

    n_params = sum(p.numel() for p in model.parameters())
    if sidecar.params and n_params != sidecar.params:
        raise ValueError(
            f"{sidecar.name}: sidecar says {sidecar.params:,} params but the "
            f"rebuilt model has {n_params:,}. The sidecar and weights disagree."
        )

    pipeline = None
    if (directory / "preprocessing.json").exists():
        pipeline = FeaturePipeline.load(directory)

    return model.to(device).eval(), sidecar, pipeline


def sidecar_for_spec(spec: Any, *, source: str = "curated") -> Sidecar:
    """Build a sidecar from a registry :class:`ModelSpec`."""
    return Sidecar(
        name=spec.name,
        factory=spec.factory,
        arch=dict(spec.arch),
        params=spec.params,
        prefix=spec.prefix,
        rich=spec.rich,
        source=source,
        provenance={"published_params_m": spec.published_m, "note": spec.note},
    )
