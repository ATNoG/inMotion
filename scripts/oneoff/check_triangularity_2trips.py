#!/usr/bin/env python3
"""Smallest case that settles the triangularity question: two trips, Z -> A -> M.

Route order (Z, A, M) and stop_id order (A, M, Z) disagree on purpose.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from inmotion.bus import od
from inmotion.bus.records import Call, Trip

OUT = Path("plots/bus")

TRIPS = [
    Trip(
        trip_id=tid, operator="busway", line="16", direction="ida", service="weekday",
        season="A", trip_index=1, source="demo",
        calls=[
            Call(stop_seq=i + 1, stop=s, time=f"{m // 60:02d}:{m % 60:02d}", minutes=m)
            for i, (s, m) in enumerate(calls)
        ],
    )
    for tid, calls in [("t1", [("Z", 360), ("A", 370), ("M", 380)]),
                      ("t2", [("Z", 420), ("A", 430), ("M", 440)])]
]

matrix = od.od_matrix(od.od_pairs(od.stop_times_frame(TRIPS)))
per_pair = matrix.groupby(["origin_id", "destination_id"])["trips"].sum()


def build(order):
    idx = {s: i for i, s in enumerate(order)}
    m = np.zeros((len(order), len(order)))
    for (o, d), t in per_pair.items():
        m[idx[o], idx[d]] = t
    return m


ALPHA = ["busway:a", "busway:m", "busway:z"]
ROUTE = ["busway:z", "busway:a", "busway:m"]
Ma, Mr = build(ALPHA), build(ROUTE)


def verdict(m, label):
    up, lo, dg = np.triu(m, 1).sum(), np.tril(m, -1).sum(), np.diag(m).sum()
    ok = "strictly upper triangular" if lo == 0 else "NOT triangular"
    print(f"  {label:<12} above={up:.0f}  below={lo:.0f}  diagonal={dg:.0f}  -> {ok}")
    print(f"               {m.astype(int).tolist()}")
    return ok


print("2 trips, line 16 ida, weekday:")
for t in TRIPS:
    print("  " + " -> ".join(f"{c.stop}@{c.time}" for c in t.calls))
print(f"\nod_pairs: {len(per_pair)} distinct pairs, {per_pair.sum():.0f} rides")
print(f"matrix rows: {len(matrix)}  (bands: {sorted(matrix.departure_band.unique())})\n")
print("Triangularity check:")
va = verdict(Ma, "stop_id order")
vr = verdict(Mr, "route order")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=130)

ax = axes[0]
xs = [0, 1, 2]
for row, (tid, off) in enumerate([("t1", 0.28), ("t2", -0.28)]):
    for x1, x2 in zip(xs, xs[1:]):
        ax.add_patch(FancyArrowPatch((x1, 1 + off), (x2, 1 + off), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.8, color="#1f77b4"))
    ax.plot(xs, [1 + off] * 3, "o", color="#1f77b4", ms=7)
    ax.annotate(tid, (0, 1 + off), xytext=(-0.28, 1 + off), fontsize=10, va="center")
ax.set_xticks(xs); ax.set_xticklabels(["Z", "A", "M"])
ax.set_yticks([]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.3, 1.8)
ax.set_title("the 2 trips: Z -> A -> M", fontweight="bold")
ax.set_xlabel("stop, in route order")
ax.grid(axis="x", alpha=0.2)

for ax, m, order, title, col in [
    (axes[1], Ma, ALPHA, "stop_id order (A, M, Z)", "#d62728"),
    (axes[2], Mr, ROUTE, "route order (Z, A, M)", "#2ca02c"),
]:
    ax.imshow(m, cmap="YlGn", vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks(range(3)); ax.set_xticklabels([s.split(":")[1] for s in order])
    ax.set_yticks(range(3)); ax.set_yticklabels([s.split(":")[1] for s in order])
    ax.set_xlabel("destination"); ax.set_ylabel("origin")
    for i in range(3):
        for j in range(3):
            v = int(m[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontweight="bold")
    ax.plot([-0.5, 2.5], [-0.5, 2.5], color=col, ls="--", lw=2)
    up, lo = np.triu(m, 1).sum(), np.tril(m, -1).sum()
    ax.set_title(f"{title}\nabove={up:.0f} below={lo:.0f}", fontweight="bold", color=col)

fig.suptitle(
    "Triangularity is a property of the ordering, not of the data\n"
    "identical 3x3 matrix, two orderings - only route order is triangular",
    fontsize=14, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
p = OUT / "triangularity_2trips_demo.png"
fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n  {va} / {vr}")
print(f"  saved {p} ({p.stat().st_size/1024:.0f} KB)")
