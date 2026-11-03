"""The eight edge-campaign figures, drawn from the tables and nothing else.

Every series is read from a CSV written by a producer in this package (or by the
device workstream), the header is asserted against ``contracts.COLUMNS`` before
anything is drawn, and the rows each figure actually plots are written back to
``<out_dir>/data/<stem>.csv`` so a number in the report can be traced to the
figure without re-running a measurement.

Chart choices follow the loaded ``academic-plotting`` and
``scientific-visualization`` skills: serif 8 pt, the Okabe-Ito colourblind-safe
palette, redundant colour+marker+linestyle encoding, uncertainty always drawn,
no dual axes, and no connecting line that implies an ordering the data does not
have. Effective bits — never nominal width — are the x-axis for accuracy, and
the cast precisions are absent from every device-facing figure because the
runtime cannot load them.

Determinism
-----------
``SOURCE_DATE_EPOCH`` is pinned before matplotlib is imported, so the PDFs and
PNGs are byte-identical across consecutive builds; the figure test asserts it.
"""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from inmotion.edge import contracts as c  # noqa: E402

# ── Style ─────────────────────────────────────────────────────────────────

#: Okabe-Ito, the colourblind-safe set the paper already uses.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PINK = "#CC79A7"
GREY = "#999999"

MODEL_COLORS = {
    "lejepa": BLUE,
    "t_jepa": ORANGE,
    "ts_jepa": GREEN,
    "cf_jepa": PINK,
    "sigreg": VERMILLION,
}
MODEL_MARKERS = {"lejepa": "o", "t_jepa": "s", "ts_jepa": "^", "cf_jepa": "D", "sigreg": "X"}
MODEL_LINESTYLES = {
    "lejepa": "-",
    "t_jepa": "--",
    "ts_jepa": "-.",
    "cf_jepa": (0, (5, 1)),
    "sigreg": (0, (1, 1)),
}
DEVICE_COLORS = {"wave-router": BLUE, "wave-router-2": VERMILLION}
DEVICE_MARKERS = {"wave-router": "o", "wave-router-2": "s"}
DEVICE_LINESTYLES = {"wave-router": "-", "wave-router-2": "--"}

#: Wording for the tables, and the CLI command that writes each one. A missing
#: table names both, so the failure says what to run rather than what is absent.
TABLES: dict[str, str] = {
    "artifacts": "artifacts",
    "accuracy": "accuracy",
    "per_class": "accuracy",
    "confusion": "accuracy",
    "replay": "replay",
    "load": "bench",
    "bench": "bench",
    "cores": "bench",
    "power": "power",
    "impact": "analyze",
    "inference_impact": "analyze",
    "stress_raw": "stress",
    "stress_probes": "stress",
}

FIGURE_SIZE = (7.16, 3.0)
FIGURE_SIZE_TALL = (7.16, 5.4)


def _style() -> None:
    """Publication defaults: serif, 8 pt, thin rules, no chart junk."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ececec",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.4,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ── Table access ──────────────────────────────────────────────────────────


def _missing_table(tables_dir: Path, table: str) -> FileNotFoundError:
    command = TABLES.get(table, table)
    return FileNotFoundError(
        f"missing table {table}.csv under {tables_dir}: create it with "
        f"`uv run inmotion edge {command}` (see .dsh/plans/edge-contracts.md)"
    )


def _read(tables_dir: Path, table: str) -> pd.DataFrame:
    """Read one table and assert its header against the shared contract."""
    path = Path(tables_dir) / f"{table}.csv"
    if not path.is_file():
        raise _missing_table(tables_dir, table)
    # round_trip uses Python's correctly-rounded parser: the tables store exact
    # float reprs and the default C parser mis-rounds some of them by one ulp.
    frame = pd.read_csv(path, float_precision="round_trip")
    expected = list(c.COLUMNS[table])
    if list(frame.columns) != expected:
        raise ValueError(
            f"{path} header drifted from contracts.COLUMNS[{table!r}]: "
            f"found {list(frame.columns)}, expected {expected}"
        )
    if frame.empty:
        raise ValueError(
            f"{path} has no rows; `uv run inmotion edge {TABLES[table]}` wrote nothing"
        )
    return frame


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce the named columns to float, keeping NaN for unmeasurable cells."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _deployable(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Rows the runtime can actually load; cast precisions have no artifact."""
    flag = accuracy["runtime_loadable"].astype(str).str.strip().str.lower()
    return accuracy[flag.isin({"true", "1"})].copy()


def _present(values, order: tuple[str, ...]) -> list[str]:
    seen = set(values)
    return [name for name in order if name in seen]


def _device_short(key: str) -> str:
    label = c.DEVICES[key].label if key in c.DEVICES else key
    return label.split("·")[0].strip()


def _ci(value: pd.Series, lo: pd.Series, hi: pd.Series):
    """Asymmetric y-error for an errorbar, with NaN-safe lower bounds."""
    return np.vstack([np.maximum(value - lo, 0.0), np.maximum(hi - value, 0.0)])


#: Friendly tick values on the effective-bits axis: dense enough that the
#: 5–9 bit cluster the metadata pushes the schemes into is still readable.
_BIT_TICKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _bits_axis(ax, values: pd.Series | np.ndarray) -> None:
    """Effective bits on a log axis, ticked inside the data range."""
    finite = np.asarray(
        [value for value in values if np.isfinite(value) and value > 0], dtype=float
    )
    # Start at 4 bits even when the smallest artifact sits a little higher: the
    # campaign's schemes cluster in 5-10 bits and a truncated axis would hide it.
    low = min(float(finite.min()) * 0.92, 4.0) if finite.size else 1.0
    high = max(float(finite.max()) * 1.08, 32.0) if finite.size else 64.0
    ax.set_xscale("log", base=2)
    ticks = [tick for tick in _BIT_TICKS if low <= tick <= high]
    if len(ticks) < 2:
        ticks = [low, high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.set_xlim(low, high)
    ax.set_xlabel("effective bits per parameter (log₂ scale)")


def _venn_legend(fig: Figure, handles: list[Line2D], ncol: int = 5) -> None:
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )


def _note(fig: Figure, text: str) -> None:
    """Caption note, hand-wrapped so the tight bbox keeps the column width."""
    fig.supxlabel(textwrap.fill(text, width=132), fontsize=6.0)


def _model_handle(model: str) -> Line2D:
    return Line2D(
        [],
        [],
        color=MODEL_COLORS.get(model, GREY),
        marker=MODEL_MARKERS.get(model, "o"),
        linestyle=MODEL_LINESTYLES.get(model, "-"),
        label=model,
    )


def _device_handle(device: str) -> Line2D:
    return Line2D(
        [],
        [],
        color=DEVICE_COLORS.get(device, GREY),
        marker=DEVICE_MARKERS.get(device, "o"),
        linestyle=DEVICE_LINESTYLES.get(device, "-"),
        label=_device_short(device),
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Write the plotted rows atomically, so a reader never sees a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def _save(fig: Figure, out_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    """Save one figure in every requested format, atomically."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        tmp = path.with_name(f".{path.name}.tmp")
        fig.savefig(tmp, format=fmt, dpi=dpi, bbox_inches="tight")
        os.replace(tmp, path)
        written.append(path)
    return written


# ── F1: accuracy vs effective bits ────────────────────────────────────────


def _annotate_storage_tie(ax, accuracy: pd.DataFrame) -> None:
    """Call out the q2/q4 tie: equal stored bits, very different accuracy.

    Metadata (a float32 scale and a zero point per group) dominates these small
    models, so q2 and q4 land on the same effective-bits column for every model.
    A dotted bracket joins the two accuracies at that column; the text states the
    cross-model range rather than jittering the points apart, because the storage
    really does put them at the same byte count.
    """
    q2 = accuracy[accuracy["precision"] == "q2"]
    q4 = accuracy[accuracy["precision"] == "q4"]
    if q2.empty or q4.empty:
        return
    pairs = q4.merge(q2, on="model", suffixes=("_4", "_2"))
    if pairs.empty:
        return
    ties = ~np.isclose(pairs["effective_bits_4"], pairs["effective_bits_2"], rtol=0.02)
    if bool(ties.any()):
        return
    for _, row in pairs.iterrows():
        ax.plot(
            [row["effective_bits_4"], row["effective_bits_4"]],
            [row["mcc_2"], row["mcc_4"]],
            color="#888888",
            linewidth=0.7,
            linestyle=":",
            zorder=1.5,
        )
    bits_high = float(pairs["effective_bits_4"].max())
    bits_low = float(pairs["effective_bits_4"].min())
    lejepa = pairs[pairs["model"] == "lejepa"]
    headline_bits = float(lejepa["effective_bits_4"].iloc[0]) if len(lejepa) else bits_high
    ax.annotate(
        f"q2 and q4 store the same {headline_bits:.1f} bits/param\n"
        f"({bits_low:.1f}–{bits_high:.1f} across models); only q4 keeps accuracy\n"
        f"q2 MCC {pairs['mcc_2'].min():.2f}–{pairs['mcc_2'].max():.2f}, "
        f"q4 MCC {pairs['mcc_4'].min():.2f}–{pairs['mcc_4'].max():.2f}",
        xy=(headline_bits, (pairs["mcc_2"].max() + pairs["mcc_4"].min()) / 2.0),
        xytext=(10.5, 0.70),
        textcoords="data",
        fontsize=5.8,
        ha="left",
        va="center",
        color="#444444",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.6, "shrinkB": 3},
    )


def _annotate_smaller_int8(ax, accuracy: pd.DataFrame) -> None:
    """State that int8 costs less than q8 and does not lose accuracy."""
    int8 = accuracy[accuracy["precision"] == "int8"]
    q8 = accuracy[accuracy["precision"] == "q8"]
    if int8.empty or q8.empty:
        return
    bits_int8 = float(int8["effective_bits"].median())
    bits_q8 = float(q8["effective_bits"].median())
    if not bits_int8 < bits_q8:
        return
    if float(int8["mcc"].median()) + 0.005 < float(q8["mcc"].median()):
        return
    ax.annotate(
        f"int8 stores {bits_int8:.1f} bits/param vs q8's {bits_q8:.1f} and loses nothing",
        xy=(bits_int8, float(int8["mcc"].max())),
        xytext=(11.0, 0.965),
        textcoords="data",
        fontsize=5.9,
        ha="left",
        va="center",
        color="#444444",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.6, "shrinkB": 3},
    )


def _fig1(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    accuracy = _numeric(
        _deployable(_read(tables_dir, "accuracy")),
        ("effective_bits", "mcc", "mcc_lo", "mcc_hi", "delta_mcc", "delta_lo", "delta_hi"),
    )
    accuracy = accuracy.dropna(subset=["effective_bits", "mcc"]).sort_values(
        ["model", "effective_bits"]
    )
    models = _present(accuracy["model"], c.MODELS)
    if not models:
        raise ValueError("accuracy.csv has no deployable rows; `uv run inmotion edge accuracy` run")

    fig, (ax, ax_delta) = plt.subplots(1, 2, figsize=FIGURE_SIZE, layout="constrained")
    _annotate_storage_tie(ax, accuracy)
    _annotate_smaller_int8(ax, accuracy)
    for model in models:
        sub = accuracy[accuracy["model"] == model].dropna(
            subset=["mcc", "mcc_lo", "mcc_hi", "delta_mcc", "delta_lo", "delta_hi"]
        )
        style = {
            "color": MODEL_COLORS[model],
            "marker": MODEL_MARKERS[model],
            # No connecting line: q2/q4 and q1/q3 share an effective-bits value,
            # so a line would draw a vertical zig-zag that implies an ordering
            # the storage does not have. Precision is encoded by marker shape.
            "linestyle": "none",
            "markersize": 3.8,
            "capsize": 1.5,
            "elinewidth": 0.6,
        }
        ax.errorbar(
            sub["effective_bits"],
            sub["mcc"],
            yerr=_ci(sub["mcc"], sub["mcc_lo"], sub["mcc_hi"]),
            **style,
        )
        ax_delta.errorbar(
            sub["effective_bits"],
            sub["delta_mcc"],
            yerr=_ci(sub["delta_mcc"], sub["delta_lo"], sub["delta_hi"]),
            **style,
        )

    ax.axhline(0.0, color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=0)
    ax.text(
        0.99,
        0.03,
        "chance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=GREY,
    )
    ax.set_ylabel("test MCC (95 % CI)")
    ax.set_ylim(-0.07, 1.0)
    _bits_axis(ax, accuracy["effective_bits"])
    ax.set_title("(a) accuracy vs effective bits")

    ax_delta.axhspan(-0.01, 0.01, color="#EAF2F8", zorder=0)
    ax_delta.axhline(0.0, color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    ax_delta.set_ylabel("ΔMCC vs fp32 (paired 95 % CI)")
    _bits_axis(ax_delta, accuracy["effective_bits"])
    ax_delta.set_title("(b) paired degradation vs fp32")
    ax_delta.legend(
        handles=[Patch(facecolor="#EAF2F8", edgecolor="none", label="±0.01 tolerance")],
        loc="lower right",
        fontsize=6,
    )

    _venn_legend(fig, [_model_handle(model) for model in models])
    _note(
        fig,
        "n = 703 test rows · dataset.csv (split seed 42) · 95 % percentile bootstrap CI, "
        "2 000 resamples, seed 42 · ΔMCC uses the same resampled rows as fp32 (paired) · "
        "x is effective bits, not nominal width",
    )

    frame = accuracy[
        [
            "model",
            "precision",
            "effective_bits",
            "mcc",
            "mcc_lo",
            "mcc_hi",
            "delta_mcc",
            "delta_lo",
            "delta_hi",
        ]
    ].reset_index(drop=True)
    return fig, frame


# ── F2: failure modes ─────────────────────────────────────────────────────


def _majority_share(confusion: pd.DataFrame) -> pd.DataFrame:
    """Share of predictions falling on the most-predicted class, per row."""
    by_pred = (
        confusion.groupby(["model", "precision", "predicted_class"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "predicted"})
    )
    totals = by_pred.groupby(["model", "precision"])["predicted"].transform("sum")
    by_pred["share"] = by_pred["predicted"] / totals
    idx = by_pred.groupby(["model", "precision"])["share"].idxmax()
    return by_pred.loc[idx].rename(columns={"predicted_class": "majority_class"})


def _fig2(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    confusion = _numeric(_read(tables_dir, "confusion"), ("count",))
    accuracy = _numeric(_deployable(_read(tables_dir, "accuracy")), ("effective_bits", "mcc"))
    models = _present(confusion["model"], c.MODELS)
    target = "lejepa" if "lejepa" in models else models[0] if models else ""
    if not target:
        raise ValueError("confusion.csv has no rows; `uv run inmotion edge accuracy` run")
    wanted = ["fp32", "q4", "q2"]
    sub = confusion[confusion["model"] == target]
    present = [precision for precision in wanted if precision in set(sub["precision"])]
    missing = [precision for precision in wanted if precision not in present]
    if missing:
        raise ValueError(
            f"confusion.csv has no {missing} rows for {target}; "
            f"`uv run inmotion edge accuracy` must export and score those precisions"
        )

    classes = sorted(set(confusion["true_class"]))
    counts = {}
    for precision in present:
        block = sub[sub["precision"] == precision]
        matrix = np.zeros((len(classes), len(classes)))
        for row in block.itertuples(index=False):
            matrix[classes.index(row.true_class), classes.index(row.predicted_class)] = row.count
        counts[precision] = matrix
    vmax = max(float(matrix.max()) for matrix in counts.values())

    fig = plt.figure(figsize=(7.16, 4.6), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
    inner = outer[0].subgridspec(1, len(present), wspace=0.12)
    for position, precision in enumerate(present):
        ax = fig.add_subplot(inner[0, position])
        matrix = counts[precision]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = int(matrix[i, j])
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if matrix[i, j] > 0.62 * vmax else "#333333",
                )
        ax.set_xticks(range(len(classes)), classes, fontsize=6)
        ax.set_yticks(range(len(classes)), classes, fontsize=6)
        prefix = "(a) " if position == 0 else ""
        ax.set_title(f"{prefix}{target} · {precision}", fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=6.5)
        if position == 0:
            ax.set_ylabel("true", fontsize=6.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    share = _majority_share(confusion)
    share = share.merge(
        accuracy[["model", "precision", "effective_bits"]],
        on=["model", "precision"],
        how="inner",
    ).dropna(subset=["effective_bits"])
    if share.empty:
        raise ValueError(
            "confusion.csv and accuracy.csv share no (model, precision) rows; "
            "`uv run inmotion edge accuracy` writes both"
        )
    ax_share = fig.add_subplot(outer[1, 0])
    for model in _present(share["model"], c.MODELS):
        block = share[share["model"] == model].sort_values("effective_bits")
        ax_share.plot(
            block["effective_bits"],
            block["share"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle="none",
            markersize=3.8,
        )
    ax_share.axhline(
        1.0 / len(classes),
        color=GREY,
        linewidth=0.8,
        linestyle=(0, (4, 2)),
        label=f"chance share (1/{len(classes)})",
    )
    ax_share.set_ylim(0.0, 1.05)
    ax_share.legend(loc="upper right", fontsize=6)
    ax_share.set_ylabel("majority-class prediction share")
    _bits_axis(ax_share, share["effective_bits"])
    ax_share.set_title("(b) collapse onto one class")

    _venn_legend(fig, [_model_handle(model) for model in _present(share["model"], c.MODELS)])
    _note(
        fig,
        "n = 703 test rows · dataset.csv · (a) counts are predictions, not accuracies; "
        "(b) share = predictions of the most-predicted class / 703 · no CI: this is a "
        "descriptive count, not an estimate of a population quantity",
    )

    conf_frame = confusion[
        confusion["precision"].isin(present) & (confusion["model"] == target)
    ].copy()
    conf_frame["effective_bits"] = np.nan
    conf_frame["majority_class"] = np.nan
    conf_frame["share"] = np.nan
    conf_frame["panel"] = "confusion"
    share_frame = share[["model", "precision", "effective_bits", "majority_class", "share"]].copy()
    share_frame["panel"] = "majority"
    share_frame["true_class"] = np.nan
    share_frame["predicted_class"] = np.nan
    share_frame["count"] = np.nan
    columns = [
        "panel",
        "model",
        "precision",
        "effective_bits",
        "true_class",
        "predicted_class",
        "count",
        "majority_class",
        "share",
    ]
    frame = pd.concat(
        [conf_frame[columns], share_frame[columns].reset_index(drop=True)], ignore_index=True
    )
    return fig, frame


# ── F3: size-accuracy ────────────────────────────────────────────────────


#: Artifact sizes within this relative distance are one *storage tier*.
#: Group sizes, scales and the metadata index make nominally different schemes
#: land on nearly the same byte count (q1 is 34 bytes smaller than q3 on lejepa),
#: so a byte-exact frontier would promote a collapsed scheme over a working one
#: for a rounding difference. Inside a tier only the best MCC is eligible.
_TIER_RTOL = 1e-3


def _pareto(frame: pd.DataFrame, *, x: str, y: str, rtol: float = _TIER_RTOL) -> set[int]:
    """Indices on the (minimise x, maximise y) Pareto frontier.

    Sizes are bucketed into storage tiers first (``rtol`` relative width); within
    a tier only the best-``y`` point can be non-dominated. The surviving points
    are then filtered by the usual running maximum, so every returned index is
    non-dominated by every other returned index and by the whole frame.
    """
    ordered = frame.sort_values([x, y], ascending=[True, False])
    tier_best: list[tuple[int, float, float]] = []
    tier_start = None
    candidate: tuple[int, float, float] | None = None
    for index, row in ordered.iterrows():
        size, score = float(row[x]), float(row[y])
        if tier_start is None or size > tier_start * (1.0 + rtol):
            if candidate is not None:
                tier_best.append(candidate)
            tier_start = size
            candidate = (index, size, score)
        elif candidate is None or score > candidate[2]:
            candidate = (index, size, score)
    if candidate is not None:
        tier_best.append(candidate)

    best = -np.inf
    chosen: set[int] = set()
    for index, _size, score in tier_best:
        if score > best:
            best = score
            chosen.add(index)
    return chosen


def _fig3_labels(accuracy: pd.DataFrame) -> pd.Series:
    """Which points earn a text label: Pareto, deployed, and the q1/q2 failures.

    Forty-five labels collide; these four groups are the ones a reader has to
    identify (the frontier, the operating points, and where accuracy collapses).
    Every other point is still encoded by colour and marker shape.
    """
    return accuracy["on_pareto"] | accuracy["ringed"] | accuracy["precision"].isin(("q1", "q2"))


def _fig3(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    accuracy = _numeric(
        _deployable(_read(tables_dir, "accuracy")),
        ("artifact_bytes", "effective_bits", "mcc", "mcc_lo", "mcc_hi"),
    ).dropna(subset=["artifact_bytes", "mcc"])
    accuracy = accuracy[accuracy["artifact_bytes"] > 0]
    if accuracy.empty:
        raise ValueError(
            "accuracy.csv has no artifact rows to size; run `uv run inmotion edge accuracy`"
        )
    accuracy = accuracy.reset_index(drop=True)
    accuracy["size_mb"] = accuracy["artifact_bytes"] / 1e6
    on_pareto = _pareto(accuracy, x="size_mb", y="mcc")
    accuracy["on_pareto"] = [index in on_pareto for index in accuracy.index]
    accuracy["ringed"] = accuracy["precision"].isin(("q4", "int8"))

    models = _present(accuracy["model"], c.MODELS)
    labelled = _fig3_labels(accuracy)
    fig, ax = plt.subplots(figsize=(7.16, 3.6), layout="constrained")
    offsets = ((3, 3), (3, -5), (-4, 3), (-4, -5), (0, 6), (0, -7))
    for model in models:
        block = accuracy[accuracy["model"] == model].sort_values("size_mb")
        ax.scatter(
            block["size_mb"],
            block["mcc"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            s=16,
            linewidths=0,
            zorder=3,
            label=model,
        )
        for order, (index, row) in enumerate(block.iterrows()):
            if not bool(labelled.loc[index]):
                continue
            dx, dy = offsets[order % len(offsets)]
            ax.annotate(
                row["precision"],
                (row["size_mb"], row["mcc"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=4.4,
                color=MODEL_COLORS[model],
                ha="left" if dx > 0 else ("center" if dx == 0 else "right"),
                va="bottom" if dy > 0 else "top",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )
        ringed = block[block["ringed"]]
        ax.scatter(
            ringed["size_mb"],
            ringed["mcc"],
            facecolors="none",
            edgecolors="#222222",
            s=48,
            linewidths=0.7,
            zorder=2,
        )

    frontier = accuracy[accuracy["on_pareto"]].sort_values("size_mb")
    ax.step(
        frontier["size_mb"],
        frontier["mcc"],
        where="post",
        color=GREY,
        linewidth=0.9,
        linestyle=(0, (5, 2)),
        zorder=1,
    )
    ax.scatter(
        frontier["size_mb"],
        frontier["mcc"],
        facecolors="none",
        edgecolors="#111111",
        marker="s",
        s=70,
        linewidths=0.9,
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("artifact size (MB, log scale; weights + metadata)")
    ax.set_ylabel("test MCC")
    ax.set_title("size–accuracy trade-off, five models × nine deployable precisions")
    handles = [_model_handle(model) for model in models]
    handles += [
        Line2D(
            [],
            [],
            color="#111111",
            marker="s",
            linestyle="none",
            markerfacecolor="none",
            label="global Pareto set",
        ),
        Line2D(
            [],
            [],
            color="#222222",
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            label="q4 / int8 (deployed)",
        ),
    ]
    _venn_legend(fig, handles, ncol=min(7, len(handles)))
    _note(
        fig,
        "n = 703 test rows · dataset.csv · MCC point estimate (CI omitted here: the "
        "same values are drawn with 95 % bootstrap CI in Fig. 1) · labels are precisions · "
        "Pareto set minimises bytes and maximises MCC",
    )

    frame = accuracy[
        [
            "model",
            "precision",
            "artifact_bytes",
            "size_mb",
            "effective_bits",
            "mcc",
            "mcc_lo",
            "mcc_hi",
            "on_pareto",
            "ringed",
        ]
    ].reset_index(drop=True)
    return fig, frame


# ── F4: device accuracy parity ───────────────────────────────────────────


def _fig4(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    replay = _numeric(
        _read(tables_dir, "replay"),
        ("samples", "agreement", "mcc_python", "mcc_device", "mcc_delta", "max_logit_diff"),
    )
    # `replay.csv` marks an MCC that sklearn cannot compute with -2 (outside the
    # metric's [-1, 1] range); a legitimate negative MCC must stay plottable.
    for column in ("mcc_python", "mcc_device", "mcc_delta"):
        replay[column] = replay[column].mask(replay[column] <= -2.0)
    replay = replay.dropna(subset=["mcc_python", "mcc_device"])
    accuracy = _numeric(_deployable(_read(tables_dir, "accuracy")), ("mcc_lo", "mcc_hi"))
    merged = replay.merge(
        accuracy[["model", "precision", "mcc_lo", "mcc_hi"]],
        on=["model", "precision"],
        how="left",
        validate="many_to_one",
    )
    unjoined = merged[merged["mcc_lo"].isna()][["model", "precision"]].drop_duplicates()
    if not unjoined.empty:
        pairs = [f"{row.model}-{row.precision}" for row in unjoined.itertuples(index=False)]
        raise ValueError(
            f"replay.csv has no matching accuracy.csv row for {pairs}; "
            "`uv run inmotion edge accuracy` must score every replayed artifact"
        )
    joined = merged

    devices = _present(joined["device"], tuple(c.DEVICES))
    if not devices:
        devices = sorted(joined["device"].unique())
    fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="constrained")
    low = float(min(joined["mcc_python"].min(), joined["mcc_device"].min()))
    high = float(max(joined["mcc_python"].max(), joined["mcc_device"].max()))
    pad = max(0.02, 0.06 * (high - low))
    span = np.array([low - pad, high + pad])

    ax.fill_between(
        span,
        span - 0.005,
        span + 0.005,
        color="#EAF2F8",
        linewidth=0,
        zorder=0,
        label="±0.005 replay tolerance",
    )
    ax.plot(
        span, span, color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1, label="identity"
    )

    for device in devices:
        block = joined[joined["device"] == device]
        ax.errorbar(
            block["mcc_python"],
            block["mcc_device"],
            xerr=_ci(block["mcc_python"], block["mcc_lo"], block["mcc_hi"]),
            color=DEVICE_COLORS.get(device, GREY),
            marker=DEVICE_MARKERS.get(device, "o"),
            linestyle="none",
            markersize=4.0,
            capsize=1.5,
            elinewidth=0.6,
            zorder=3,
            label=_device_short(device),
        )

    agreement = float(joined["agreement"].min())
    worst = float(joined["mcc_delta"].abs().max())
    ax.text(
        0.03,
        0.97,
        f"min agreement {agreement:.2%}\nmax |ΔMCC| {worst:.4f}\npoints: {len(joined)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 2.0},
    )
    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_xlabel("Python MCC (artifact weights, 95 % CI)")
    ax.set_ylabel("on-device MCC")
    ax.set_title("artifact parity on the two routers")
    ax.legend(loc="lower right", fontsize=6)
    _venn_legend(fig, [_device_handle(device) for device in devices])
    samples = joined["samples"].min(), joined["samples"].max()
    _note(
        fig,
        f"n = replay rows ({len(joined)}), {int(samples[0])}–{int(samples[1])} samples per row · "
        "dataset.csv · Python CI is the bootstrap interval from accuracy.csv; on-device "
        "MCC is one deterministic measurement per artifact (agreement shown above) · "
        "band = replay tolerance",
    )

    frame = joined[
        [
            "device",
            "model",
            "precision",
            "samples",
            "agreement",
            "mcc_python",
            "mcc_lo",
            "mcc_hi",
            "mcc_device",
            "mcc_delta",
            "max_logit_diff",
            "parity_ok",
        ]
    ].rename(columns={"mcc_lo": "mcc_python_lo", "mcc_hi": "mcc_python_hi"})
    return fig, frame.reset_index(drop=True)


# ── F5: runtime cost ─────────────────────────────────────────────────────


def _aggregate(
    frame: pd.DataFrame, value: str, keys: tuple[str, ...] = ("device", "precision")
) -> pd.DataFrame:
    """Median across models × repetitions, with an IQR band across those cells."""
    numeric = _numeric(frame, (value,))
    numeric = numeric.dropna(subset=[value])
    grouped = numeric.groupby(list(keys))[value]
    out = grouped.agg(
        value="median",
        lo=lambda series: series.quantile(0.25),
        hi=lambda series: series.quantile(0.75),
        n="count",
    ).reset_index()
    return out


def _fig5(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    bench = _read(tables_dir, "bench")
    load = _read(tables_dir, "load")
    power = _read(tables_dir, "power")

    # One client and a warm load: the runtime-cost question is per inference,
    # not the concurrency matrix (that is F6) and not the cold cache.
    concurrency = pd.to_numeric(bench["concurrency"], errors="coerce").fillna(0)
    single = bench[concurrency <= 1]
    bench_single = single if not single.empty else bench
    cold = load["cold"].astype(str).str.strip().str.lower()
    warm = load[~cold.isin({"true", "1"})]
    load_warm = warm if not warm.empty else load

    panels: list[tuple[str, str, pd.DataFrame, float, str]] = [
        ("p50 latency (ms)", "p50", _aggregate(bench_single, "p50_ms"), 1.0, "bench.csv:p50_ms"),
        ("p95 latency (ms)", "p95", _aggregate(bench_single, "p95_ms"), 1.0, "bench.csv:p95_ms"),
        ("load time (ms)", "load", _aggregate(load_warm, "load_ms"), 1.0, "load.csv:load_ms"),
        (
            "peak RSS (MiB)",
            "rss",
            _aggregate(bench_single, "peak_rss_kb"),
            1.0 / 1024.0,
            "bench.csv:peak_rss_kb",
        ),
        (
            "CPU ms per inference",
            "cpu",
            _aggregate(bench_single, "cpu_ms_per_infer"),
            1.0,
            "bench.csv:cpu_ms_per_infer",
        ),
    ]

    energy = _numeric(power, ("mj_per_infer_est", "mj_per_infer_lo", "mj_per_infer_hi"))
    energy = energy.dropna(subset=["mj_per_infer_est"])
    energy = energy[energy["mj_per_infer_est"] >= 0]
    energy_frame = (
        energy.groupby(["device", "precision"])
        .agg(
            value=("mj_per_infer_est", "median"),
            lo=("mj_per_infer_lo", "median"),
            hi=("mj_per_infer_hi", "median"),
            n=("mj_per_infer_est", "count"),
        )
        .reset_index()
    )
    panels.append(
        (
            "energy (mJ per inference)",
            "energy",
            energy_frame,
            1.0,
            "power.csv:mj_per_infer_est (CPU-package model)",
        )
    )

    devices = _present(pd.concat([panel[2]["device"] for panel in panels]), tuple(c.DEVICES))
    if not devices:
        raise ValueError("bench.csv/load.csv/power.csv have no device rows")

    positions: dict[str, int] = {}
    for _title, _key, frame, _scale, _provenance in panels:
        if frame is None or frame.empty:
            raise ValueError(
                f"an empty panel has no rows to plot; `uv run inmotion edge {TABLES['bench']}`"
            )
        for precision in frame["precision"].unique():
            positions.setdefault(precision, c.precision_rank(precision))
    ordered = sorted(positions, key=lambda name: positions[name])
    slot = {precision: index for index, precision in enumerate(ordered)}

    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_TALL, layout="constrained")
    frames: list[pd.DataFrame] = []
    for ax, (title, _key, frame, scale, provenance) in zip(axes.ravel(), panels):
        for device in devices:
            block = frame[frame["device"] == device].copy()
            if block.empty:
                continue
            block["precision_rank"] = block["precision"].map(slot)
            block = block.sort_values("precision_rank")
            xs = block["precision_rank"].to_numpy(dtype=float)
            colour = DEVICE_COLORS.get(device, GREY)
            value = block["value"] * scale
            lo = block["lo"] * scale
            hi = block["hi"] * scale
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(
                xs,
                value,
                color=colour,
                marker=DEVICE_MARKERS.get(device, "o"),
                linestyle=DEVICE_LINESTYLES.get(device, "-"),
                markersize=3.2,
            )
            block["panel"] = title
            block["provenance"] = provenance
            block["value"] = value
            block["lo"] = lo
            block["hi"] = hi
            frames.append(block)
        ax.set_xticks(range(len(ordered)), ordered, fontsize=6.0)
        ax.set_title(title)
        ax.set_xlabel("precision (ordered by effective bits)")

    unit_labels = {
        "p50": "latency (ms)",
        "p95": "latency (ms)",
        "load": "load time (ms)",
        "rss": "peak RSS (MiB)",
        "cpu": "CPU ms / inference",
        "energy": "mJ / inference",
    }
    for ax, (title, key, _frame, _scale, _provenance) in zip(axes.ravel(), panels):
        ax.set_ylabel(unit_labels.get(key, "value"))
    _venn_legend(fig, [_device_handle(device) for device in devices], ncol=len(devices))
    _note(
        fig,
        "line = median across 5 models × repetitions; band = inter-quartile range across "
        "those cells · panels read bench.csv (single client), load.csv (warm loads) and "
        "power.csv · energy is a labelled CPU-package model, not wall power; its band is "
        "the model's lo–hi range · the latency and RAM panels are zoomed to the measured "
        "range, not zero-based; absolute values are in bench.csv/load.csv · n per "
        "precision is in the data CSV",
    )

    frame = pd.concat(frames, ignore_index=True)
    return fig, frame[
        ["panel", "device", "precision", "precision_rank", "value", "lo", "hi", "n", "provenance"]
    ]


# ── F6: core scaling ────────────────────────────────────────────────────


def _fig6(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    cores = _numeric(
        _read(tables_dir, "cores"),
        ("rep", "cores", "concurrency", "p50_ms", "p95_ms", "throughput_s"),
    )
    if cores.empty:
        raise ValueError("cores.csv has no rows; `uv run inmotion edge bench` wrote nothing")
    devices = _present(cores["device"], tuple(c.DEVICES)) or sorted(cores["device"].unique())
    preferred = ("lejepa", "ts_jepa", "sigreg")
    models = [name for name in preferred if name in set(cores["model"])]
    if len(models) < 3:
        models += [name for name in _present(cores["model"], c.MODELS) if name not in models]
    models = [name for name in c.MODELS if name in models][:3]
    if not models:
        raise ValueError("cores.csv has no model rows; `uv run inmotion edge bench`")

    fig, axes = plt.subplots(
        len(devices),
        3,
        figsize=(FIGURE_SIZE[0], 2.2 * len(devices)),
        layout="constrained",
        squeeze=False,
    )
    frames: list[pd.DataFrame] = []
    for row, device in enumerate(devices):
        block = cores[cores["device"] == device]
        pinned = block[(block["concurrency"] == 1) & block["cores"].isin([1, 2, 3, 4])]

        def pinned_agg(column: str) -> pd.DataFrame:
            usable = pinned.dropna(subset=[column])
            grouped = usable.groupby(["model", "cores"])[column]
            return grouped.agg(
                value="median",
                lo=lambda series: series.quantile(0.25),
                hi=lambda series: series.quantile(0.75),
                n="count",
            ).reset_index()

        throughput = pinned_agg("throughput_s")
        latency = pinned_agg("p95_ms")

        concurrent = block[block["concurrency"] > 1].dropna(subset=["throughput_s"])
        # Speed-up must compare like with like: the 1-client run with the SAME
        # affinity string as the concurrent runs, or the pinned-core rows would
        # fold a core-scaling gain into the client-scaling number.
        concurrent_cpus = {str(value) for value in concurrent["cpus"].dropna().unique()}
        if len(concurrent_cpus) == 1:
            baseline = block[
                (block["concurrency"] == 1) & block["cpus"].astype(str).isin(concurrent_cpus)
            ]
        else:
            baseline = block[(block["concurrency"] == 1) & (block["cores"] == block["cores"].max())]
        baseline = baseline[baseline["throughput_s"].notna()]
        base_map = baseline.groupby("model")["throughput_s"].median().to_dict()
        if concurrent.empty or not base_map:
            raise ValueError(
                f"cores.csv has no concurrency rows for {device}; "
                "`uv run inmotion edge bench --profile full`"
            )
        speed = (
            concurrent.groupby(["model", "concurrency"])["throughput_s"]
            .agg(median="median", n="count", low="min", high="max")
            .reset_index()
        )
        speed["value"] = [
            item["median"] / base_map[item["model"]] if base_map.get(item["model"]) else np.nan
            for _, item in speed.iterrows()
        ]
        speed["lo"] = [
            item["low"] / base_map[item["model"]] if base_map.get(item["model"]) else np.nan
            for _, item in speed.iterrows()
        ]
        speed["hi"] = [
            item["high"] / base_map[item["model"]] if base_map.get(item["model"]) else np.nan
            for _, item in speed.iterrows()
        ]
        speed = speed[speed["model"].isin(models)]

        ax_tp, ax_p95, ax_conc = axes[row]
        for model in models:
            colour = MODEL_COLORS[model]
            marker = MODEL_MARKERS[model]
            linestyle = MODEL_LINESTYLES[model]
            for frame, ax in ((throughput, ax_tp), (latency, ax_p95)):
                sub = frame[frame["model"] == model].sort_values("cores")
                if sub.empty:
                    continue
                ax.fill_between(
                    sub["cores"], sub["lo"], sub["hi"], color=colour, alpha=0.13, linewidth=0
                )
                ax.plot(
                    sub["cores"],
                    sub["value"],
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=3.4,
                )
            sub = speed[speed["model"] == model].sort_values("concurrency")
            if not sub.empty:
                ax_conc.fill_between(
                    sub["concurrency"], sub["lo"], sub["hi"], color=colour, alpha=0.13, linewidth=0
                )
                ax_conc.plot(
                    sub["concurrency"],
                    sub["value"],
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=3.4,
                )

        short = _device_short(device)
        if row == 0:
            ax_tp.set_title("(a) throughput vs pinned cores")
            ax_p95.set_title("(b) p95 latency vs pinned cores")
            ax_conc.set_title("(c) concurrency speed-up")
        ax_tp.set_ylabel(f"{short}\nthroughput (inferences / s)")
        ax_p95.set_ylabel("p95 latency (ms)")
        ax_conc.set_ylabel("speed-up vs 1 client")
        ax_tp.set_xticks([1, 2, 3, 4])
        ax_p95.set_xticks([1, 2, 3, 4])
        concurrency = sorted(speed["concurrency"].unique())
        ideal = np.array([1, max(concurrency)]) if concurrency else np.array([1, 2])
        ax_conc.plot(ideal, ideal, color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), label="ideal")
        ax_conc.set_xticks(concurrency)
        if row == 0:
            ax_conc.legend(loc="upper left", fontsize=6)
        if row == len(devices) - 1:
            ax_tp.set_xlabel("pinned cores")
            ax_p95.set_xlabel("pinned cores")
            ax_conc.set_xlabel("concurrent clients")

        tp_rows = throughput.copy()
        tp_rows["panel"] = "throughput"
        tp_rows["concurrency"] = 1
        tp_rows["device"] = device
        p95_rows = latency.copy()
        p95_rows["panel"] = "p95"
        p95_rows["concurrency"] = 1
        p95_rows["device"] = device
        conc_rows = pd.DataFrame(
            {
                "panel": "speedup",
                "device": device,
                "model": speed["model"],
                "cores": 4,
                "concurrency": speed["concurrency"],
                "value": speed["value"],
                "lo": speed["lo"],
                "hi": speed["hi"],
                "n": speed["n"],
            }
        )
        columns = ["panel", "device", "model", "cores", "concurrency", "value", "lo", "hi", "n"]
        frames.append(
            pd.concat([tp_rows[columns], p95_rows[columns], conc_rows], ignore_index=True)
        )

    _venn_legend(fig, [_model_handle(model) for model in models], ncol=3)
    _note(
        fig,
        "rows are devices · line = median, band = inter-quartile range across repetitions · "
        "(c) is throughput divided by the same model's 1-client throughput · n per cell is in "
        "the data CSV",
    )
    return fig, pd.concat(frames, ignore_index=True)


# ── F7: router impact ───────────────────────────────────────────────────


def _rtt_kpi(impact: pd.DataFrame) -> str:
    for name in ("rtt_p95_ms_lan", "rtt_p95_ms_fwd"):
        if name in set(impact["kpi"]):
            return name
    raise ValueError(
        f"impact.csv has neither 'rtt_p95_ms_lan' nor 'rtt_p95_ms_fwd' (found "
        f"{sorted(set(impact['kpi']))}); `uv run inmotion edge analyze` must emit one of them"
    )


def _fig7(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    impact = _numeric(
        _read(tables_dir, "impact"),
        (
            "baseline_median",
            "inference_median",
            "stress_median",
            "both_median",
            "delta_inference",
            "delta_inference_lo",
            "delta_inference_hi",
            "delta_stress",
            "delta_stress_lo",
            "delta_stress_hi",
            "p_value",
            "effect_size",
            "margin",
        ),
    )
    rtt = _rtt_kpi(impact)
    wanted = ["cpu_busy_pct", rtt, "drop_rate_pct"]
    available = set(impact["kpi"])
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(
            f"impact.csv has no {missing} rows (found {sorted(available)}); "
            "`uv run inmotion edge analyze` must emit ROUTER_KPIS for every device"
        )

    titles = {
        "cpu_busy_pct": ("CPU busy", "%"),
        rtt: ("RTT p95" + (" (LAN)" if rtt == "rtt_p95_ms_lan" else " (forwarded)"), "ms"),
        "drop_rate_pct": ("packet drop rate", "%"),
    }
    phases = ("inference", "stress", "both")
    devices = _present(impact["device"], tuple(c.DEVICES)) or sorted(impact["device"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.6), layout="constrained", sharex=True)
    frames: list[pd.DataFrame] = []
    positions: list[float] = []
    labels: list[str] = []
    slot = 0.0
    for device in devices:
        for phase in phases:
            positions.append(slot)
            labels.append(f"{_device_short(device).replace('Device ', 'D')}\n{phase}")
            slot += 1.0
        slot += 0.5

    for ax, kpi in zip(axes, wanted):
        rows = impact[impact["kpi"] == kpi]
        margin_row = rows["margin"].dropna()
        margin = float(margin_row.iloc[0]) if not margin_row.empty else np.nan
        title, unit = titles[kpi]
        if np.isfinite(margin):
            ax.axhspan(-margin, margin, color="#EAF2F8", linewidth=0, zorder=0)
            ax.text(
                0.99,
                0.04,
                f"±{margin:g} {unit} margin",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.8,
                color="#555555",
            )
        ax.axhline(0.0, color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)

        for device_index, device in enumerate(devices):
            for phase_index, phase in enumerate(phases):
                row = rows[rows["device"] == device]
                if row.empty:
                    continue
                record = row.iloc[0]
                x = positions[device_index * len(phases) + phase_index]
                baseline = float(record["baseline_median"])
                if phase == "inference":
                    value = float(record["inference_median"]) - baseline
                    lo = float(record["delta_inference_lo"])
                    hi = float(record["delta_inference_hi"])
                    has_ci = np.isfinite(lo) and np.isfinite(hi)
                elif phase == "stress":
                    value = float(record["stress_median"]) - baseline
                    lo = float(record["delta_stress_lo"])
                    hi = float(record["delta_stress_hi"])
                    has_ci = np.isfinite(lo) and np.isfinite(hi)
                else:
                    value = float(record["both_median"]) - baseline
                    lo = hi = np.nan
                    has_ci = False
                colour = DEVICE_COLORS.get(device, GREY)
                marker = DEVICE_MARKERS.get(device, "o")
                if has_ci:
                    ax.errorbar(
                        x,
                        value,
                        yerr=np.array([[max(value - lo, 0.0)], [max(hi - value, 0.0)]]),
                        color=colour,
                        marker=marker,
                        markersize=3.6,
                        capsize=1.8,
                        elinewidth=0.7,
                        linestyle="none",
                        zorder=3,
                    )
                else:
                    ax.plot(
                        x,
                        value,
                        color=colour,
                        marker=marker,
                        markersize=3.6,
                        markerfacecolor="white",
                        markeredgewidth=0.9,
                        linestyle="none",
                        zorder=3,
                    )
                frames.append(
                    {
                        "kpi": kpi,
                        "unit": unit,
                        "device": device,
                        "phase": phase,
                        "baseline_median": baseline,
                        "phase_median": baseline + value,
                        "delta": value,
                        "lo": lo,
                        "hi": hi,
                        "has_ci": has_ci,
                        "margin": margin,
                        "verdict": record.get("verdict", ""),
                        "p_value": record.get("p_value", np.nan),
                        "effect_size": record.get("effect_size", np.nan),
                    }
                )
        ax.set_ylabel(f"Δ vs baseline ({unit})")
        ax.set_title(f"{title} · margin ±{margin:g} {unit}" if np.isfinite(margin) else title)
        ax.set_xlim(-0.6, max(positions) + 0.6)

    axes[-1].set_xticks(positions, labels, fontsize=6.0)
    _venn_legend(fig, [_device_handle(device) for device in devices], ncol=len(devices))
    _note(
        fig,
        "point = difference of phase medians vs the baseline phase; whisker = 95 % "
        "block-bootstrap CI over the 5 repetitions (seed 42); open marker = 'both' phase, "
        "median only (no CI in impact.csv) · shaded band = pre-declared equivalence margin · "
        "n = 5 repetitions × 4 phases per device",
    )
    frame = pd.DataFrame(frames)
    return fig, frame


# ── F8: stress robustness ────────────────────────────────────────────────


def _fig8(tables_dir: Path) -> tuple[Figure, pd.DataFrame]:
    inference = _numeric(
        _read(tables_dir, "inference_impact"),
        ("unloaded_median", "loaded_median", "delta", "delta_lo", "delta_hi", "margin"),
    )
    missing = [name for name in c.INFERENCE_KPIS if name not in set(inference["kpi"])]
    if missing:
        raise ValueError(
            f"inference_impact.csv has no {missing} rows (found "
            f"{sorted(set(inference['kpi']))}); `uv run inmotion edge analyze` must emit "
            "contracts.INFERENCE_KPIS"
        )
    devices = _present(inference["device"], tuple(c.DEVICES)) or sorted(
        inference["device"].unique()
    )
    titles = {
        "infer_p50_ms": "(a) inference p50",
        "infer_p95_ms": "(b) inference p95",
        "infer_cpu_ms_per_infer": "(c) CPU ms per inference",
    }

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, layout="constrained")
    frames: list[dict[str, object]] = []
    for ax, kpi in zip(axes, c.INFERENCE_KPIS):
        rows = inference[inference["kpi"] == kpi]
        unit = str(rows["unit"].iloc[0]) if not rows.empty else "ms"
        has_margin = not rows.empty and bool(rows["margin"].notna().any())
        margin = float(rows["margin"].dropna().iloc[0]) if has_margin else np.nan
        ymax = 0.0
        ymin = np.inf
        for device_index, device in enumerate(devices):
            block = rows[rows["device"] == device]
            if block.empty:
                continue
            record = block.iloc[0]
            unloaded = float(record["unloaded_median"])
            loaded = float(record["loaded_median"])
            delta = float(record["delta"])
            ci_low_abs = unloaded + float(record["delta_lo"])
            ci_high_abs = unloaded + float(record["delta_hi"])
            colour = DEVICE_COLORS.get(device, GREY)
            marker = DEVICE_MARKERS.get(device, "o")
            x = float(device_index)

            if np.isfinite(margin):
                ax.fill_between(
                    [x - 0.30, x + 0.30],
                    unloaded * (1.0 - margin),
                    unloaded * (1.0 + margin),
                    color=colour,
                    alpha=0.13,
                    linewidth=0,
                    zorder=0,
                )
            ax.plot([x - 0.08, x + 0.08], [unloaded, loaded], color=colour, linewidth=0.9, zorder=2)
            ax.plot(
                x - 0.08,
                unloaded,
                color=colour,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.0,
                markersize=4.2,
                linestyle="none",
                zorder=3,
            )
            ax.errorbar(
                x + 0.08,
                loaded,
                yerr=np.array(
                    [
                        [max(loaded - ci_low_abs, 0.0)],
                        [max(ci_high_abs - loaded, 0.0)],
                    ]
                ),
                color=colour,
                marker=marker,
                markersize=4.2,
                capsize=1.8,
                elinewidth=0.7,
                linestyle="none",
                zorder=3,
            )
            relative = (loaded - unloaded) / unloaded if unloaded else np.nan
            ax.annotate(
                f"{relative:+.1%}",
                (x, max(unloaded, loaded)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=5.8,
                color=colour,
            )
            verdict = str(record.get("verdict", ""))
            if verdict:
                ax.annotate(
                    verdict,
                    (x, max(unloaded, loaded)),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.4,
                    color="#555555",
                )
            ymax = max(ymax, loaded, unloaded, ci_high_abs)
            if np.isfinite(margin):
                ymax = max(ymax, unloaded * (1.0 + margin))
                ymin = min(ymin, unloaded * (1.0 - margin), ci_low_abs)
            frames.append(
                {
                    "kpi": kpi,
                    "unit": unit,
                    "device": device,
                    "unloaded_median": unloaded,
                    "loaded_median": loaded,
                    "delta": delta,
                    "delta_lo": float(record["delta_lo"]),
                    "delta_hi": float(record["delta_hi"]),
                    "ci_low_abs": ci_low_abs,
                    "ci_high_abs": ci_high_abs,
                    "relative_delta": relative,
                    "margin": margin,
                    "verdict": verdict,
                }
            )
        ax.set_xticks(
            range(len(devices)), [_device_short(device) for device in devices], fontsize=6.5
        )
        ax.set_xlim(-0.45, len(devices) - 0.55)
        label = {
            "infer_p50_ms": "p50 latency",
            "infer_p95_ms": "p95 latency",
            "infer_cpu_ms_per_infer": "CPU ms / inference",
        }.get(kpi, kpi)
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(titles[kpi])
        lower = max(0.0, 0.92 * ymin) if np.isfinite(ymin) else 0.0
        if ymax > 0:
            ax.set_ylim(lower, ymax * 1.10)

    _venn_legend(
        fig,
        [_device_handle(device) for device in devices],
        ncol=len(devices),
    )
    _note(
        fig,
        "open marker = unloaded (inference-only phase), filled = loaded (both phase) · "
        "whisker = 95 % CI of the paired Δ (block bootstrap, 5 000 draws, seed 42) · "
        "shaded band = ±30 % equivalence margin around the unloaded median · "
        "n = 5 repetitions per phase per device",
    )
    return fig, pd.DataFrame(frames)


# ── Build ───────────────────────────────────────────────────────────────


_BUILDERS = {
    "fig1_accuracy_vs_bits": _fig1,
    "fig2_failure_modes": _fig2,
    "fig3_size_accuracy": _fig3,
    "fig4_device_accuracy": _fig4,
    "fig5_runtime_cost": _fig5,
    "fig6_core_scaling": _fig6,
    "fig7_router_impact": _fig7,
    "fig8_stress_robustness": _fig8,
}


def build(
    tables_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
    *,
    only: tuple[str, ...] | None = None,
) -> list[Path]:
    """Build the eight figures (or ``only`` a subset) from ``tables_dir``.

    Every figure requires its own tables: a missing one raises
    :class:`FileNotFoundError` naming the table and the CLI command that writes
    it, so the campaign fails loudly instead of drawing a half-empty panel.
    ``only`` is how a run that has accuracy tables but no device tables yet draws
    F1–F3 without disabling that guarantee for the full build.

    Returns the written paths, figures and their ``data/`` CSVs.
    """
    tables_dir = Path(tables_dir)
    out_dir = Path(out_dir)
    stems = tuple(c.FIGURE_STEMS) if only is None else tuple(only)
    unknown = [stem for stem in stems if stem not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown figure stem(s) {unknown}; known: {list(c.FIGURE_STEMS)}")

    _style()
    written: list[Path] = []
    for stem in stems:
        if stem not in c.FIGURE_STEMS:
            raise ValueError(f"{stem} is not in contracts.FIGURE_STEMS")
        figure, frame = _BUILDERS[stem](tables_dir)
        written.append(_write_frame(frame, out_dir / "data" / f"{stem}.csv"))
        written.extend(_save(figure, out_dir, stem, formats, dpi))
        plt.close(figure)
    return written
