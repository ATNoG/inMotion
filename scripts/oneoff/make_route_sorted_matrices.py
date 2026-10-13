#!/usr/bin/env python3
"""Route-ordered version of the four wide OD matrices.

The shipped matrices order origins and destinations by stop_id, which is
alphabetical and therefore arbitrary with respect to the network, so the route
structure is invisible (48-50% of rides land above the diagonal, i.e. no signal
at all). This writes the same numbers with both axes in network order.

A single greedy chain over the whole operator does not work: the network is 110
lines over 13 municipalities, several of them circular, so no global order can
triangularise it. Instead the stops are laid out in *blocks*: grouped by the line
that serves them most, blocks ordered by that line's total service, and stops
ordered along their line's stop sequence. Each line's block is internally
triangular, so the matrix becomes block-triangular.

Rows and columns are the full stop union of the operator, so the sorted matrices
are genuinely square (the shipped ones are not: 180x183, 1640x1635, ...).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROC = Path("data/bus/processed")
OUT = Path("plots/bus")
OUT.mkdir(parents=True, exist_ok=True)

od = pd.read_csv(PROC / "od_matrix.csv", low_memory=False)
od = od[od.trips > 0]
pairs = pd.read_csv(PROC / "od_pairs.csv", low_memory=False)

st = pd.read_csv(PROC / "stop_times.csv", low_memory=False)
st = st[st.served].copy()
st["line"] = st["line"].astype(str)

# Dominant line per stop, and the position along that line.
_served = (
    st.groupby(["stop_id", "line"]).size().rename("n").reset_index()
    .sort_values(["stop_id", "n", "line"], ascending=[True, False, True])
    .drop_duplicates("stop_id")
)
dom = dict(zip(_served.stop_id, _served.line))
_pos = st.groupby(["stop_id", "line"]).stop_seq.mean().rename("p").reset_index()
_pos = _pos[_pos.line == _pos.stop_id.map(dom)]
meanpos = dict(zip(_pos.stop_id, _pos.p))


def block_order(op: str, universe: list[str]) -> list[str]:
    """Stops grouped by dominant line, ordered along their line."""
    line_trips = pairs[pairs.operator == op].groupby("line").trips.sum()
    return sorted(
        universe,
        key=lambda s: (0, -line_trips.get(dom.get(s), 0), dom.get(s) or "~", meanpos.get(s, 1e9), s),
    )


def wide(sub: pd.DataFrame, rows: list[str], cols: list[str]) -> pd.DataFrame:
    return (
        sub.pivot_table(index="origin_id", columns="destination_id", values="trips", aggfunc="sum")
        .reindex(index=rows, columns=cols)
        .fillna(0.0)
    )


rows_spec = [(op, g) for op in ["aveirobus", "busway"] for g in ["weekday", "weekend"]]
fig, axes = plt.subplots(len(rows_spec), 2, figsize=(19, 8.5 * len(rows_spec)), dpi=100)
stats, orders = [], {}

for r, (op, group) in enumerate(rows_spec):
    g = od[od.operator == op]
    if op not in orders:
        universe = sorted(set(g.origin_id) | set(g.destination_id))
        order = block_order(op, universe)
        pd.DataFrame({"position": range(len(order)), "stop_id": order,
                      "dominant_line": [dom.get(s, "") for s in order]}).to_csv(
            PROC / f"od_stop_order_{op}.csv", index=False
        )
        orders[op] = (universe, order)
    universe, order = orders[op]

    ma = wide(g[g.service_group == group], universe, universe)
    ms = ma.loc[order, order]
    ms.to_csv(PROC / f"od_matrix_wide_{op}_{group}_sorted.csv", index_label="origin_id")

    above = np.triu(ma.values, 1).sum()
    below = np.tril(ma.values, -1).sum()
    tot = above + below
    above_s = np.triu(ms.values, 1).sum()
    pct_a, pct_s = above / tot * 100, above_s / tot * 100
    stats.append((op, group, len(universe), int(tot), pct_a, pct_s))
    pos = np.percentile(ms.values[ms.values > 0], 98) if (ms.values > 0).any() else 1

    for c, (m, label) in enumerate(
        [(ma.values, f"stop_id order - {pct_a:.0f}% above diagonal"),
         (ms.values, f"network block order - {pct_s:.0f}% above diagonal")]
    ):
        ax = axes[r, c]
        ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=pos, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{op} / {group} - {label}", fontsize=13, fontweight="bold",
                     color=("#d62728", "#2ca02c")[c])
        ax.set_xlabel("destination (" + ("stop_id" if c == 0 else "network") + " order)")
        if c == 0:
            ax.set_ylabel("origin (stop_id order)")
        else:
            ax.set_ylabel("origin (network order)")

fig.suptitle(
    "OD matrices sorted along the network instead of alphabetically\n"
    "identical trip counts and identical square stop universe - only the axis order changes",
    fontsize=20, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.975])
p = OUT / "C_od_matrices_route_sorted.png"
fig.savefig(p, dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

s = pd.DataFrame(stats, columns=["operator", "service_group", "stops", "trips",
                                "pct_above_stop_id", "pct_above_network"])
s.to_csv(OUT / "C_route_sort_gain.csv", index=False)
print(f"  saved {p} ({p.stat().st_size/1024:.0f} KB)")
print(s.round(1).to_string(index=False))
print("\nwrote od_matrix_wide_<op>_<group>_sorted.csv + od_stop_order_<op>.csv")
