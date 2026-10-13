#!/usr/bin/env python3
"""Tier B: triangularity audit, next-hop Markov matrix, per-line small multiples, reachability."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = Path("plots/bus")
OUT.mkdir(parents=True, exist_ok=True)

st = pd.read_csv("data/bus/processed/stop_times.csv", low_memory=False)
pairs = pd.read_csv("data/bus/processed/od_pairs.csv", low_memory=False)
served = st[st.served].sort_values(["trip_id", "stop_seq"])
# line ids are mixed str/int across the two feeds; compare on str
served["line"] = served["line"].astype(str)
pairs["line"] = pairs["line"].astype(str)
served["direction"] = served["direction"].astype(str)
pairs["direction"] = pairs["direction"].astype(str)


def route_order(op, line, dr):
    """Stops in first-seen order along the route, which is the order that makes
    an OD matrix upper-triangular."""
    sub = served[(served.operator == op) & (served.line == line) & (served.direction == dr)]
    order = []
    for s in sub.stop_id:
        if s not in order:
            order.append(s)
    return order


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p} ({p.stat().st_size/1024:.0f} KB)")


# ------------------------------------------------------------------ B0
# Triangularity audit: is the OD structure upper-triangular once stops are
# ordered along the route instead of alphabetically?
rows = []
for (op, line, dr), g in pairs.groupby(["operator", "line", "direction"]):
    p = g.groupby(["origin_id", "destination_id"]).trips.sum()
    ids = sorted(set(g.origin_id) | set(g.destination_id))
    if len(ids) < 2:
        continue
    order = route_order(op, line, dr)
    order += [x for x in ids if x not in order]
    idx = {x: i for i, x in enumerate(order)}
    M = np.zeros((len(order), len(order)))
    for (a, b), t in p.items():
        M[idx[a], idx[b]] = t
    up, lo = np.triu(M, 1).sum(), np.tril(M, -1).sum()
    sub = served[(served.operator == op) & (served.line == line) & (served.direction == dr)]
    f = sub.groupby("trip_id").stop_id.first()
    l = sub.groupby("trip_id").stop_id.last()
    circular = bool((f == l).all())
    rows.append(
        {
            "operator": op, "line": line, "direction": dr, "stops": len(ids),
            "trips": float(p.sum()), "above": float(up), "below": float(lo),
            "pct_below": lo / (up + lo) * 100 if (up + lo) else 0.0, "circular": circular,
        }
    )
tri = pd.DataFrame(rows)
tri.to_csv(OUT / "B0_triangularity_audit.csv", index=False)

lin = tri[~tri.circular]
circ = tri[tri.circular]
fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=130)

ax = axes[0]
g = pairs.groupby(["operator", "line", "direction"])
alpha_share = []
for (op, line, dr), gg in g:
    p = gg.groupby(["origin_id", "destination_id"]).trips.sum()
    ids = sorted(set(gg.origin_id) | set(gg.destination_id))
    idx = {x: i for i, x in enumerate(ids)}
    M = np.zeros((len(ids), len(ids)))
    for (a, b), t in p.items():
        M[idx[a], idx[b]] = t
    tot = np.triu(M, 1).sum() + np.tril(M, -1).sum()
    alpha_share.append(np.triu(M, 1).sum() / tot * 100 if tot else np.nan)
alpha_share = np.array(alpha_share, dtype=float)
ax.hist(alpha_share, bins=np.arange(0, 105, 5), color="#7f7f7f", edgecolor="white")
ax.axvline(100, color="crimson", ls="--", label="fully triangular (100%)")
ax.set_title(f"Alphabetical stop order\nmean {alpha_share.mean():.0f}% above diagonal", fontweight="bold")
ax.set_xlabel("% of trips above the diagonal"); ax.set_ylabel("line-directions")
ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.3)

ax = axes[1]
ax.hist(lin.pct_below, bins=np.arange(0, 105, 5), color="#2ca02c", edgecolor="white")
n_exact = int((lin.pct_below < 1e-9).sum())
ax.set_title(f"Route-sequence order (linear lines)\n{n_exact}/{len(lin)} perfectly triangular", fontweight="bold")
ax.set_xlabel("% of trips below the diagonal"); ax.grid(ls="--", alpha=0.3)

ax = axes[2]
top = tri.nlargest(12, "trips")
cols = ["#d62728" if c else "#2ca02c" for c in top.circular]
ax.barh(range(len(top)), top.pct_below, color=cols)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{r.line} {r.direction}" for r in top.itertuples()], fontsize=8)
ax.set_xlabel("% below diagonal"); ax.grid(axis="x", ls="--", alpha=0.3)
ax.set_title("Busiest line-directions\ngreen = triangular, red = circular loop", fontweight="bold")
fig.suptitle(
    "Triangularity audit: alphabetical order is arbitrary, route order is not",
    fontsize=16, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "B0_triangularity_audit.png")
print(f"  linear line-directions: {n_exact}/{len(lin)} exactly triangular; circular: {len(circ)}")

# ------------------------------------------------------------------ B1
# Next-hop Markov matrix from consecutive served stops
nxt = served.groupby("trip_id").stop_id.shift(-1)
tr = served.assign(nxt=nxt).dropna(subset=["nxt"])
cnt = tr.groupby(["stop_id", "nxt"]).size().rename("n").reset_index()
print(f"  next-hop transitions: {len(tr):,} | distinct directed pairs: {len(cnt):,}")

nodes = cnt.groupby("stop_id").n.sum().sort_values(ascending=False)
top = nodes.head(40).index.tolist()
C = cnt.pivot(index="stop_id", columns="nxt", values="n").reindex(index=top, columns=top).fillna(0.0)
P = C.div(C.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=130)
im = axes[0].imshow(C.values, cmap="viridis", interpolation="nearest")
axes[0].set_title(f"Next-hop trip counts (top {len(top)} stops)", fontweight="bold")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
im = axes[1].imshow(P.values, cmap="magma", vmin=0, interpolation="nearest")
axes[1].set_title(f"Transition probability P(next=j | from=i)\nrow-normalised, top {len(top)} stops", fontweight="bold")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="probability")
for ax in axes:
    ax.set_xlabel("next stop"); ax.set_ylabel("from stop")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(
    "Next-Hop Markov Matrix (adjacent stops, from stop_times)\n"
    "this is the true network adjacency - the OD matrix only records trip endpoints",
    fontsize=16, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "B1_next_hop_markov_matrix.png")

odeg = cnt.groupby("stop_id").nxt.nunique()
indeg = cnt.groupby("nxt").stop_id.nunique()
fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
bins = np.arange(1, 42)
ax.hist(indeg.reindex(nodes.index).fillna(0), bins=bins, alpha=0.6, label="in-degree (distinct predecessors)", color="#1f77b4")
ax.hist(odeg.reindex(nodes.index).fillna(0), bins=bins, alpha=0.6, label="out-degree (distinct successors)", color="#ff7f0e")
ax.set_yscale("log"); ax.set_xlabel("distinct neighbours"); ax.set_ylabel("stops (log)")
ax.set_title("Adjacency degree distribution", fontweight="bold")
ax.legend(); ax.grid(ls="--", alpha=0.3)
fig.tight_layout()
save(fig, "B1b_degree_distribution.png")

# ------------------------------------------------------------------ B2
# Per-line OD matrices as small multiples, in route-sequence order
grp = pairs.groupby(["operator", "line", "direction"])
items = [(k, v) for k, v in grp if len(set(v.origin_id) | set(v.destination_id)) >= 4]
items.sort(key=lambda kv: -kv[1].trips.sum())
sel = items[:120]
cols = int(np.ceil(np.sqrt(len(sel))))
rows_n = int(np.ceil(len(sel) / cols))
VMAX = float(max(v.trips.sum() for _, v in sel)) if sel else 1

fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.5, rows_n * 1.6), dpi=110)
axes = np.atleast_2d(axes).ravel()
for ax, ((op, line, dr), v) in zip(axes, sel):
    p = v.groupby(["origin_id", "destination_id"]).trips.sum()
    order = route_order(op, line, dr)
    order += [x for x in sorted(set(p.index.get_level_values(0)) | set(p.index.get_level_values(1))) if x not in order]
    idx = {x: i for i, x in enumerate(order)}
    M = np.zeros((len(order), len(order)))
    for (a, b), t in p.items():
        M[idx[a], idx[b]] = t
    ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=VMAX, interpolation="nearest", aspect="auto")
    ax.set_title(f"{line} {dr}  {int(p.sum())}t", fontsize=5.5)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(sel):]:
    ax.axis("off")
fig.suptitle(
    f"Per-line-direction OD matrices, stops in route-sequence order\n"
    f"{len(sel)} busiest of {len(items)} line-directions - the upper-triangle structure is the route order",
    fontsize=15, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "B2_per_line_matrices.png")

# ------------------------------------------------------------------ B3
# Reachability within k hops on the next-hop graph, and trip concentration
adj = {}
for a, b in cnt[["stop_id", "nxt"]].itertuples(index=False, name=None):
    adj.setdefault(a, []).append(b)

pair_trips = pairs.groupby(["operator", "service_group", "origin_id", "destination_id"]).trips.sum()

def reachable(a, k):
    seen, frontier, depth = {a}, [a], 0
    while depth < k:
        nxt_lvl = []
        for x in frontier:
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y); nxt_lvl.append(y)
        frontier = nxt_lvl; depth += 1
        if not frontier:
            break
    return seen

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), dpi=130)
rows_r = []
for op in ["aveirobus", "busway"]:
    s2 = pair_trips.xs(op, level="operator")
    by_origin = {a: g for a, g in s2.groupby(level="origin_id")}
    for k in (1, 2, 3):
        vals = []
        for a, g in by_origin.items():
            r = reachable(a, k)
            vals.append(float(g[g.index.get_level_values("destination_id").isin(r)].sum()))
        rows_r.append((op, k, float(np.mean(vals)), float(np.median(vals))))
r = pd.DataFrame(rows_r, columns=["operator", "k_hops", "mean_trips", "median_trips"])
for op in ["aveirobus", "busway"]:
    s = r[r.operator == op]
    axes[0].plot(s.k_hops, s.mean_trips, "o-", label=op, linewidth=2)
axes[0].set_xlabel("hops from origin"); axes[0].set_ylabel("mean trips reachable per origin")
axes[0].set_title("Trip reachability within k hops", fontweight="bold")
axes[0].legend(); axes[0].grid(ls="--", alpha=0.3)

for op in ["aveirobus", "busway"]:
    for sg in ["weekday", "weekend"]:
        s = pair_trips.xs((op, sg), level=["operator", "service_group"])
        v = np.sort(s.values)[::-1]
        cum = np.cumsum(v) / v.sum() * 100
        x = np.arange(1, len(v) + 1) / len(v) * 100
        axes[1].plot(x, cum, linewidth=1.6, label=f"{op} {sg}")
axes[1].plot([0, 100], [0, 100], "k--", lw=1, label="perfect equality")
axes[1].set_xlabel("% of OD pairs (ranked by trips)"); axes[1].set_ylabel("% of all trips")
axes[1].set_title("Trip concentration (Lorenz)", fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(ls="--", alpha=0.3)
fig.suptitle("Network reachability and concentration", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "B3_reachability_lorenz.png")
r.to_csv(OUT / "B3_reachability.csv", index=False)
print(r.round(1).to_string(index=False))
print("Tier B done.")
