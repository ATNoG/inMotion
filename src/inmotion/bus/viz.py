"""Figures that let the extracted routes be checked by eye.

Two routes are drawn per operator, and each figure carries three views of the
same two routes:

* an **OD grid** per route — the matrix restricted to that route, origins down
  the side and destinations across the top, coloured by how many direct trips
  connect the pair. A route read correctly is upper-triangular: nothing runs
  backwards and a stop is never its own destination.
* a **route ladder** — the two routes' stops in travel order, one ladder each,
  positioned at the elapsed time from their own first stop. Every stop the two
  routes share is joined by a connector. Where the extraction is wrong, the
  ladder is jagged and the connectors cross at random; where it is right, the
  two of a line's directions mirror each other and the connectors are parallel.

Which two routes get drawn is not a choice: :func:`similar_pairs` ranks every
pair of routes of an operator by how much of their stop list they share, and the
top pair is plotted. The figure therefore shows the most comparable pair the
data contains, and the console reports the pair and its agreement so the
comparison can be judged rather than trusted.

Nothing is typed in by hand; every number comes from ``data/bus/processed``.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from inmotion.bus.viz_config import PlotConfig  # noqa: E402

# Okabe-Ito, matching the palette the rest of the repository's figures use.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#8a8a8a"
FAINT = "#cfcfcf"


@dataclass(frozen=True)
class Route:
    """One route: a line, one direction, one service."""

    operator: str
    line: str
    direction: str
    service_group: str

    @property
    def label(self) -> str:
        return f"{self.operator} {self.line} {self.direction}"


@dataclass(frozen=True)
class SimilarPair:
    """Two routes and how much they have in common."""

    first: Route
    second: Route
    shared: int
    jaccard: float
    agreement: float
    """Spearman correlation of the shared stops' positions.

    +1 means both routes visit the shared stops in the same order, −1 means the
    second is the first reversed, which is what a line's two directions should
    look like.
    """


def load_processed(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stop times and the per-line OD relation.

    The per-route grid is read from ``od_pairs.csv`` rather than
    ``od_matrix.csv``: the matrix in that file is aggregated over lines, so it
    cannot say which line provides a pair, and a route is what is being checked.

    ``line`` is read as text: AveiroBus numbers its lines and also calls two of
    them ``azul`` and ``verde``, which pandas would otherwise split in two.
    """
    root = Path(directory)
    stop_times = pd.read_csv(root / "stop_times.csv", dtype={"line": str})
    pairs = pd.read_csv(root / "od_pairs.csv", dtype={"line": str})
    return stop_times, pairs


def route_profile(stop_times: pd.DataFrame, route: Route) -> pd.DataFrame:
    """One route's stops in travel order, with their elapsed time.

    ``elapsed_min`` is measured from each trip's own first stop, so trips that
    start at different points along the route are still comparable, and the
    median over the route's trips is what is reported.
    """
    frame = stop_times[
        stop_times["served"]
        & (stop_times["operator"] == route.operator)
        & (stop_times["line"] == route.line)
        & (stop_times["direction"] == route.direction)
        & (stop_times["service_group"] == route.service_group)
    ].copy()
    if frame.empty:
        return frame
    frame["elapsed"] = frame["minutes"] - frame.groupby("trip_id")["minutes"].transform("min")
    grouped = (
        frame.groupby(["stop_id", "stop"], as_index=False)
        .agg(
            stop_seq=("stop_seq", "median"),
            elapsed=("elapsed", "median"),
            calls=("trip_id", "nunique"),
        )
        .sort_values(["stop_seq", "stop"])
        .reset_index(drop=True)
    )
    grouped["position"] = range(1, len(grouped) + 1)
    return grouped


def similar_pairs(
    stop_times: pd.DataFrame,
    operator: str,
    *,
    service_group: str = "weekday",
    minimum_stops: int = 6,
) -> list[SimilarPair]:
    """Every pair of one operator's routes, ranked by shared stops.

    Routes are compared on their stop *sets* first, so a pair worth drawing is
    found without a human picking one, and then on the order those stops appear
    in, which is what says whether the two are the same corridor.
    """
    frame = stop_times[
        stop_times["served"]
        & (stop_times["operator"] == operator)
        & (stop_times["service_group"] == service_group)
    ]
    profiles: dict[Route, pd.DataFrame] = {}
    for (line, direction), group in frame.groupby(["line", "direction"], sort=True):
        route = Route(operator, str(line), str(direction), service_group)
        profile = route_profile(stop_times, route)
        if len(profile) >= minimum_stops:
            profiles[route] = profile

    ranked: list[SimilarPair] = []
    for first, second in itertools.combinations(profiles, 2):
        one, two = profiles[first], profiles[second]
        shared = set(one["stop_id"]) & set(two["stop_id"])
        union = set(one["stop_id"]) | set(two["stop_id"])
        if not shared:
            continue
        positions = pd.DataFrame(
            {
                "first": one.set_index("stop_id").loc[sorted(shared), "position"].to_numpy(),
                "second": two.set_index("stop_id").loc[sorted(shared), "position"].to_numpy(),
            }
        )
        ranked.append(
            SimilarPair(
                first=first,
                second=second,
                shared=len(shared),
                jaccard=len(shared) / len(union),
                agreement=float(positions["first"].corr(positions["second"], method="spearman")),
            )
        )
    return sorted(ranked, key=lambda pair: (-pair.jaccard, -pair.shared, pair.first.label))


def od_grid(pairs: pd.DataFrame, route: Route, order: list[str]) -> pd.DataFrame:
    """One route's OD matrix, as origins by destinations in travel order.

    Summed over departure bands, so a cell is the number of direct trips this
    route provides between the two stops, whenever they leave.
    """
    frame = pairs[
        (pairs["operator"] == route.operator)
        & (pairs["line"] == route.line)
        & (pairs["direction"] == route.direction)
        & (pairs["service_group"] == route.service_group)
    ]
    grid = frame.pivot_table(
        index="origin_id", columns="destination_id", values="trips", aggfunc="sum", fill_value=0
    )
    return grid.reindex(index=order, columns=order, fill_value=0)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#ebebeb",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 110,
        }
    )


def _heatmap(ax, grid: pd.DataFrame, profile: pd.DataFrame, route: Route, colour: str) -> None:
    """One route's OD grid, in the route's own travel order."""
    labels = list(profile["stop"])
    image = ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4.6)
    ax.set_yticklabels(labels, fontsize=4.6)
    ax.set_xlabel("destination")
    ax.set_ylabel("origin")
    served = int((grid.to_numpy() > 0).sum())
    ax.set_title(
        f"{route.label}\n{len(labels)} stops · {served} of {len(labels) ** 2} cells served",
        color=colour,
    )
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.042, pad=0.02)
    bar.set_label("direct trips", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)


def _ladder(ax, profiles: list[pd.DataFrame], pair: SimilarPair, label_stops: bool) -> None:
    """Both routes' stops in travel order, connected wherever they are shared."""
    colours = (BLUE, ORANGE)
    shared = set(profiles[0]["stop_id"]) & set(profiles[1]["stop_id"])
    labels_by_side: dict[int, list[tuple[str, float, float]]] = {}
    for index, profile in enumerate(profiles):
        x = float(index)
        shared_mask = profile["stop_id"].isin(shared).to_numpy()
        ax.plot(
            [x] * len(profile),
            profile["elapsed"],
            "-",
            color=colours[index],
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            [x] * int(shared_mask.sum()),
            profile["elapsed"][shared_mask],
            s=24,
            facecolor=colours[index],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=f"{profile.attrs.get('label', index)} · shared",
        )
        unique = profile[~shared_mask]
        ax.scatter(
            [x] * len(unique),
            unique["elapsed"],
            s=12,
            facecolor="white",
            edgecolor=colours[index],
            linewidth=1.0,
            zorder=4,
            label="this route only",
        )
        if label_stops:
            # A label is printed only when it clears the previous one on that
            # side by a real number of *points*, measured after the axes are
            # laid out — a fraction of the time span is not the same thing, and
            # on the return direction several stops fall within a minute.
            for row in sorted(profile.itertuples(index=False), key=lambda r: r.elapsed):
                labels_by_side.setdefault(index, []).append((row.stop, row.elapsed, x))

    first_by_stop = profiles[0].set_index("stop_id")
    second_by_stop = profiles[1].set_index("stop_id")
    for stop_id in sorted(shared):
        ax.plot(
            [0, 1],
            [first_by_stop.at[stop_id, "elapsed"], second_by_stop.at[stop_id, "elapsed"]],
            "-",
            color=FAINT,
            linewidth=0.8,
            zorder=2,
        )
    if label_stops:
        # Measured after the axes are laid out, and in points: several of the
        # return direction's stops fall inside the same minute, so a fraction of
        # the time span is not a spacing.
        ax.figure.canvas.draw()
        for side, entries in labels_by_side.items():
            placed: list[float] = []
            for stop, elapsed, x in entries:
                y_display = float(ax.transData.transform((x, elapsed))[1])
                if any(abs(y_display - taken) < 8.0 for taken in placed):
                    continue
                placed.append(y_display)
                ax.annotate(
                    stop,
                    (x, elapsed),
                    xytext=(-6 if side == 0 else 6, 0),
                    textcoords="offset points",
                    ha="right" if side == 0 else "left",
                    va="center",
                    fontsize=5.0,
                    color="#333333",
                )

    if pair.agreement < -0.5:
        crossing = "mirror order"
    elif pair.agreement > 0.5:
        crossing = "same order"
    else:
        crossing = "mixed order"
    union = len(set(profiles[0]["stop_id"]) | set(profiles[1]["stop_id"]))
    ax.set_title(
        f"{pair.shared} shared stops of {union} (Jaccard {pair.jaccard:.2f}) · "
        f"position agreement {pair.agreement:+.2f} → {crossing}"
    )
    ax.set_xlim(-1.05, 2.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"{pair.first.label}\n{len(profiles[0])} stops",
            f"{pair.second.label}\n{len(profiles[1])} stops",
        ]
    )
    thinning = " · stop names thinned where they bunch" if label_stops else ""
    ax.set_ylabel("minutes from that route's own first stop" + thinning)
    ax.invert_yaxis()
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=6.0, frameon=False)


def plot_route_pair(
    stop_times: pd.DataFrame,
    pairs: pd.DataFrame,
    pair: SimilarPair,
    out_path: Path,
    *,
    label_stops: bool = True,
) -> Path:
    """Render one comparison figure and return its path."""
    first = route_profile(stop_times, pair.first)
    second = route_profile(stop_times, pair.second)
    first.attrs["label"] = pair.first.label
    second.attrs["label"] = pair.second.label

    figure = plt.figure(figsize=(13.0, 9.5), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    _heatmap(
        figure.add_subplot(outer[0, 0]),
        od_grid(pairs, pair.first, list(first["stop_id"])),
        first,
        pair.first,
        BLUE,
    )
    _heatmap(
        figure.add_subplot(outer[0, 1]),
        od_grid(pairs, pair.second, list(second["stop_id"])),
        second,
        pair.second,
        ORANGE,
    )
    _ladder(figure.add_subplot(outer[1, :]), [first, second], pair, label_stops)
    figure.suptitle(
        f"Two similar {pair.first.operator} routes, {pair.first.service_group} service",
        fontsize=11,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def run(config: PlotConfig) -> int:
    """Draw the most similar pair of routes of each operator. Returns an exit code."""
    # Fixed timestamp: matplotlib stamps PDF CreationDate from the clock, so two
    # runs seconds apart would differ byte-wise for no informational gain.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1577836800")
    _style()

    stop_times, pairs_table = load_processed(config.processed_dir)
    operators = sorted(stop_times["operator"].unique())
    written: list[Path] = []
    for operator in operators:
        pairs = similar_pairs(
            stop_times,
            operator,
            service_group=config.service_group,
            minimum_stops=config.minimum_stops,
        )
        if not pairs:
            print(f"{operator}: no comparable route pair")
            continue
        pair = pairs[0]
        path = Path(config.out_dir) / f"route_pair_{operator}.{config.formats[0]}"
        plot_route_pair(
            stop_times,
            pairs_table,
            pair,
            path,
            label_stops=config.label_stops,
        )
        written.append(path)
        print(
            f"{operator}: {pair.first.label} vs {pair.second.label} · "
            f"{pair.shared} shared stops · Jaccard {pair.jaccard:.2f} · "
            f"agreement {pair.agreement:+.2f}"
        )
        for runner_up in pairs[1:3]:
            print(
                f"    next: {runner_up.first.label} vs {runner_up.second.label} · "
                f"Jaccard {runner_up.jaccard:.2f} · agreement {runner_up.agreement:+.2f}"
            )
    for path in written:
        print(f"wrote {path}")
    return 0
