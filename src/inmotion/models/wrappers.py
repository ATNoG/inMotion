"""Thin `nn.Module` adapters and state-dict key helpers.

Some checkpoints were saved from a wrapper module, so their keys carry a prefix
that the bare module does not have (or vice versa). These helpers normalise that
without the callers needing to know which case they are in.
"""

from __future__ import annotations

from torch import Tensor, nn


class TupleWrap(nn.Module):
    """Expose logits of a model that returns ``(logits, latents)``.

    ``SIGRegClassifier`` and the JEPA classifiers return a tuple; the ensemble
    harness and most evaluation code expect logits only.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.m = module

    def forward(self, x: Tensor) -> Tensor:
        return self.m(x)[0]


def strip_prefix(state: dict, prefix: str = "c.") -> dict:
    """Remove ``prefix`` from every key that starts with it."""
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


def add_prefix(state: dict, prefix: str) -> dict:
    """Prepend ``prefix`` to every key."""
    return {f"{prefix}{k}": v for k, v in state.items()}


def normalise_state(state: dict, prefix: str | None) -> dict:
    """Return ``state`` shaped for a module that expects ``prefix`` wrapping.

    ``None`` means the module is unwrapped, so a ``c.``-prefixed checkpoint is
    unwrapped to match.
    """
    if prefix is None:
        return strip_prefix(state, "c.") if any(k.startswith("c.") for k in state) else state
    return add_prefix(state, prefix)
