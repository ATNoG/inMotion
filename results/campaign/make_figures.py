#!/usr/bin/env python3
"""Figures for the extended-quantization campaign.

Reads the measured CSVs and writes publication figures to `figures/`. Nothing
here measures anything: every number is read from a CSV that a command produced,
so a figure can always be traced back to the run behind it.

    uv run python results/campaign/make_figures.py

Inputs, all optional except the first — a missing one is reported and skipped
rather than silently producing an empty panel:

    results/campaign/sweep_all_formats.csv   accuracy per (model, precision)
    results/campaign/perf_device.csv     FLOPS, energy, RAM, CPU% per device
    results/campaign/scaling.csv         threads -> latency, for saturation

Outputs: figures/fig_*.pdf (vector, for LaTeX) and figures/fig_*.png (300 dpi).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIGURES = HERE / "figures"

# --- Publication defaults -------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)

# Okabe-Ito: colourblind-safe, and distinguishable in grayscale by marker.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

FIG_SINGLE = (3.25, 2.6)
FIG_FULL = (6.75, 3.0)

#: Ascending storage cost, so the trade-off panels read left-to-right as
#: "cheaper" -> "more accurate" rather than following the CSV's row order.
PRECISION_ORDER = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q8",
    "int8",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"  skip {path.relative_to(REPO)}: not measured yet")
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path)
        print(f"  wrote {path.relative_to(REPO)}")
    plt.close(fig)


def accuracy_figures(rows: list[dict[str, str]]) -> None:
    """Accuracy against precision, and the memory it costs to get there."""
    models = sorted({r["model"] for r in rows})
    precisions = [p for p in PRECISION_ORDER if any(r["precision"] == p for r in rows)]
    fmt = {p: i for i, p in enumerate(precisions)}

    # --- MCC per precision -------------------------------------------------
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for i, model in enumerate(models):
        series = {r["precision"]: r for r in rows if r["model"] == model}
        xs = [fmt[p] for p in precisions if p in series]
        ys = [float(series[p]["mcc"]) for p in precisions if p in series]
        ax.plot(
            xs,
            ys,
            label=model,
            color=OKABE_ITO[i % len(OKABE_ITO)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=4,
            zorder=3,
        )
    ax.axhline(0.0, color="#B0BEC5", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(
        0.02,
        0.02,
        "chance",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#777",
        va="bottom",
    )
    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels(precisions, rotation=45, ha="right")
    ax.set_xlabel("Precision (ascending storage cost)")
    ax.set_ylabel("Test MCC on dataset.csv")
    ax.set_title("Quantization accuracy by model")
    ax.legend(ncol=2, loc="lower right")
    save(fig, "fig_accuracy_vs_precision")

    # --- MCC against the memory it buys ------------------------------------
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for i, model in enumerate(models):
        series = [r for r in rows if r["model"] == model]
        series.sort(key=lambda r: fmt[r["precision"]])
        ax.plot(
            [float(r["total_mb"]) for r in series],
            [float(r["mcc"]) for r in series],
            label=model,
            color=OKABE_ITO[i % len(OKABE_ITO)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=4,
            zorder=3,
        )
        # Name the precisions along ONE reference curve only. Labelling every
        # curve put five copies of "q1" and five of "fp32" at the same
        # coordinates, which is noise rather than information.
        if i == 0:
            for row in series:
                ax.annotate(
                    row["precision"],
                    (float(row["total_mb"]), float(row["mcc"])),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=6,
                    color="#666",
                )
    ax.set_xscale("log")
    ax.set_xlabel("Artifact size (MB, log scale)")
    ax.set_ylabel("Test MCC on dataset.csv")
    ax.set_title("Accuracy against artifact size")
    ax.legend(ncol=2, loc="lower right")
    save(fig, "fig_size_vs_accuracy")

    # --- the whole matrix at a glance --------------------------------------
    matrix = np.full((len(models), len(precisions)), np.nan)
    for r in rows:
        matrix[models.index(r["model"]), fmt[r["precision"]]] = float(r["mcc"])
    fig, ax = plt.subplots(figsize=FIG_FULL)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels(precisions, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(precisions)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            # Contrast is chosen from the value, not the colormap index, so a
            # label is never darker than the cell it sits on.
            frac = (value - np.nanmin(matrix)) / max(np.nanmax(matrix) - np.nanmin(matrix), 1e-9)
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if frac < 0.6 else "#222",
            )
    ax.set_title("Test MCC by model and precision")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, label="MCC")
    save(fig, "fig_mcc_heatmap")


def main() -> None:
    print("extended-quantization campaign figures")

    # The campaign's own sweep, not results/quant/sweep_dataset.csv: that one is
    # the shipped artifact for the paper figures and stays at its verified
    # precision set.
    sweep = read_csv(HERE / "sweep_all_formats.csv")
    if sweep:
        accuracy_figures(sweep)

    # Measured later by the device runs; skipped until the CSVs exist.
    read_csv(HERE / "perf_device.csv")
    read_csv(HERE / "scaling.csv")

    print("done")


if __name__ == "__main__":
    main()
