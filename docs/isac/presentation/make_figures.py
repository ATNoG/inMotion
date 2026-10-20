#!/usr/bin/env python3
"""Presentation copies of the ISAC figures.

Same data and styling as the paper -- the code is imported from
`../paper/make_figures.py`, not copied -- but rendered on a transparent
background so each figure sits on the slide instead of in a white box
(the moloch slide body is #F8F8F8, so a pure-white figure reads as a card).

`docs/isac/paper/images/` is never written to.

Run via `make figures` (or `uv run python make_figures.py`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER_SCRIPT = HERE.parent / "paper" / "make_figures.py"
OUT = HERE / "images"

# the figures the deck embeds; the paper script writes a .png beside each .pdf
FIGURES = ("fig_rssi_trajectory", "fig_transfer", "fig_feature_importance")
RASTERS = ("rssi_trajectory.png", "transfer_mcc.png", "feature_importance_18ch.png")


def load_paper_module():
    """Import the paper's figure script without running its __main__ block."""
    spec = importlib.util.spec_from_file_location("paper_figures", PAPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PAPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paper = load_paper_module()

    # same functions, different destination
    paper.OUT = OUT

    # no background: transparent=True only clears the figure patch, so the
    # axes patch (white in the seaborn style) has to go too
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"

    original_savefig = mpl_figure.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        kwargs["transparent"] = True
        return original_savefig(self, fname, *args, **kwargs)

    mpl_figure.Figure.savefig = savefig

    for name in FIGURES:
        print(f"  {name}")
        getattr(paper, name)()

    # only the vector PDFs are used; drop the rasters this run produced
    for raster in RASTERS:
        path = OUT / raster
        if path.exists():
            path.unlink()

    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
