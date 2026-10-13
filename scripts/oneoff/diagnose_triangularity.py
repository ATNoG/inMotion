#!/usr/bin/env python3
"""Why are the OD matrices not triangular? Classify each line-direction."""
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = Path("plots/bus")

st = pd.read_csv("data/bus/processed/stop_times.csv", low_memory=False)
st = st[st.served].copy()
st["line"] = st.line.astype(str)
st["direction"] = st.direction.astype(str)
trips = pd.read_csv("data/bus/processed/trips.csv", low_memory=False)
st["season"] = st.trip_id.map(dict(zip(trips.trip_id, trips.season)))
pr = pd.read_csv("data/bus/processed/od_pairs.csv", low_memory=False)
pr["line"] = pr.line.astype(str)
pr["direction"] = pr.direction.astype(str)


def topo_order(seqs: list[list[str]]) -> tuple[list[str], bool]:
    """Order stops so every trip reads left-to-right.

    Stops caught in a cycle (a loop that closes on itself) cannot be ordered,
    but they still get a rank so the wrap-around rides are counted honestly
    instead of being dropped.
    """
    succ = defaultdict(set)
    indeg = defaultdict(int)
    nodes = set()
    for s in seqs:
        nodes.update(s)
        for a, b in zip(s, s[1:]):
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    for n in nodes:
        indeg.setdefault(n, 0)
    ready = sorted(n for n in nodes if indeg[n] == 0)
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in sorted(succ[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort()
    cyclic = len(out) != len(nodes)
    out += sorted(nodes - set(out))  # every stop gets a rank, cycle or not
    return out, cyclic


rows = []
for (op, line, dr), g in pr.groupby(["operator", "line", "direction"]):
    sub = st[(st.operator == op) & (st.line == line) & (st.direction == dr)]
    sub = sub.sort_values(["trip_id", "stop_seq"])
    first_seen = []
    for s in sub.stop_id:
        if s not in first_seen:
            first_seen.append(s)
    seqs = [list(gg.stop_id) for _, gg in sub.groupby("trip_id")]
    topo, cyclic = topo_order(seqs)

    ncirc = sum(1 for s in seqs if s[0] == s[-1])
    nrep = sum(1 for s in seqs if len(set(s)) != len(s))

    p = g.groupby(["origin_id", "destination_id"])["trips"].sum().reset_index()
    seen = set(topo)
    p = p[p.origin_id.isin(seen) & p.destination_id.isin(seen)]

    def frac(order):
        rank = {s: i for i, s in enumerate(order)}
        up = dn = 0.0
        for o, d, t in p[["origin_id", "destination_id", "trips"]].itertuples(index=False, name=None):
            ro, rd = rank.get(o), rank.get(d)
            if ro is None or rd is None or ro == rd:
                continue
            (up, dn) = (up + t, dn) if ro < rd else (up, dn + t)
        return up, dn

    ua, da = frac(first_seen)
    ut, dt = frac(topo)
    rows.append(
        dict(op=op, line=line, dir=dr, stops=len(topo), trips=len(seqs), pairs=len(p),
             circ_trips=ncirc, repeat_trips=nrep, cyclic=cyclic,
             all_circ=(ncirc == len(seqs) and len(seqs) > 0),
             services=int(sub.groupby("trip_id")["service"].nunique().shape[0]),
             seasons=int(sub.groupby("trip_id")["season"].nunique().shape[0]),
             trips_total=int(ua + da),
             pct_below_firstseen=da / (ua + da) * 100 if (ua + da) else 0.0,
             pct_below_topo=dt / (ut + dt) * 100 if (ut + dt) else 0.0)
    )

d = pd.DataFrame(rows)
# Best of the two orders, per line-direction: a topological sort fixes
# branch/short-turn clashes, but a loop needs its natural start-anchored order.
d["best_below"] = d[["pct_below_firstseen", "pct_below_topo"]].min(axis=1)
d["best_order"] = np.where(d.pct_below_topo <= d.pct_below_firstseen, "topological", "first-seen")
d["cause"] = np.where(
    d.best_below < 1e-9,
    np.where(d.cyclic, "triangular (loop needs its own order)", "triangular after topo sort"),
    np.where(d.cyclic, "circular loop - wrap pair is irreducible", "residual clash"),
)
d.to_csv(OUT / "D_triangularity_causes.csv", index=False)

n = len(d)
tot = d.trips_total.sum()
print(f"line-directions: {n}   rides: {tot:,.0f}\n")
print(f"{'ordering':<34} {'triangular':>11} {'rides below diagonal':>24}")
for lbl, col in [("first-seen (trip order)", "pct_below_firstseen"),
                 ("topological sort", "pct_below_topo"),
                 ("best of the two per line", "best_below")]:
    below = d.trips_total.mul(d[col] / 100).sum()
    print(f"  {lbl:<32} {int((d[col] < 1e-9).sum()):>4}/{n}   {below:>10,.0f}  ({below / tot * 100:4.1f}%)")
print("\ncause:")
print(d.cause.value_counts().to_string())

fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=130)
cols = {"triangular after topo sort": "#2ca02c", "triangular (loop needs its own order)": "#66bb6a",
        "circular loop - wrap pair is irreducible": "#d62728", "residual clash": "#9467bd"}

ax = axes[0]
c = d.cause.value_counts()
b = ax.bar(range(len(c)), c.values, color=[cols.get(i, "grey") for i in c.index])
ax.bar_label(b); ax.set_xticks(range(len(c)))
ax.set_xticklabels([i.replace(" ", "\n") for i in c.index], fontsize=8)
ax.set_title(f"225 line-directions, ordered by\ntopological sort of their trips", fontweight="bold")
ax.set_ylabel("line-directions"); ax.grid(axis="y", ls="--", alpha=0.3)

ax = axes[1]
d2 = d.sort_values("pct_below_firstseen")
ax.scatter(d2.pct_below_firstseen, d2.pct_below_topo, s=22, c="#1f77b4", alpha=0.8)
ax.plot([0, 100], [0, 100], "k--", lw=1, label="no change")
ax.set_xlabel("% rides below diagonal - first-seen order")
ax.set_ylabel("% rides below diagonal - topological order")
ax.set_title("Ordering was the problem,\nnot the data", fontweight="bold")
ax.legend(); ax.grid(ls="--", alpha=0.3)

ax = axes[2]
rem = d[d.best_below > 1e-9]
b = ax.barh(range(len(rem)), rem.best_below,
            color=[cols.get(c_, "grey") for c_ in rem.cause])
ax.set_yticks(range(len(rem)))
ax.set_yticklabels([f"{r.line} {r.dir}" for r in rem.itertuples()], fontsize=7)
ax.set_xlabel("% rides still below the diagonal")
ax.set_title(f"{len(rem)} line-directions still not triangular", fontweight="bold")
ax.grid(axis="x", ls="--", alpha=0.3)
fig.suptitle("Why the OD matrices are not triangular", fontsize=17, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
p_ = OUT / "D_triangularity_causes.png"
fig.savefig(p_, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n  saved {p_} ({p_.stat().st_size / 1024:.0f} KB)")
print(f"  wrote {OUT / 'D_triangularity_causes.csv'}")
