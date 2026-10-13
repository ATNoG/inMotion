#!/usr/bin/env python3
"""Tier A: departure-band slices, weekday-weekend delta, inflow/outflow balance, headway quality.

Reads only files already produced by the bus pipeline. Writes PNG directly.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = Path("plots/bus")
OUT.mkdir(parents=True, exist_ok=True)
OD = pd.read_csv("data/bus/processed/od_matrix.csv", low_memory=False)
OD = OD[OD.trips > 0].copy()

BANDS = ["early", "am_peak", "midday", "evening", "pm_peak", "night"]
OPERS = ["aveirobus", "busway"]
N_HUB = 30


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p} ({p.stat().st_size/1024:.0f} KB)")


def pivot(sub, hubs):
    m = sub.pivot_table(index="origin_id", columns="destination_id", values="trips", aggfunc="sum")
    return m.reindex(index=hubs, columns=hubs).fillna(0.0).values


# Hub universe per operator: top stops by total trips (origin + destination)
hubs = {}
for op in OPERS:
    g = OD[OD.operator == op]
    tot = pd.concat([g.groupby("origin_id").trips.sum(), g.groupby("destination_id").trips.sum()])
    hubs[op] = tot.groupby(level=0).sum().sort_values(ascending=False).head(N_HUB).index.tolist()

# ---------------------------------------------------------------- A1
# 20 slices: operator x service_group (rows) x departure band (cols)
VMAX = max(OD.trips.max(), 1)
fig, axes = plt.subplots(len(OPERS) * 2, len(BANDS), figsize=(22, 19), dpi=130)
rowlabels = [(o, s) for o in OPERS for s in ["weekday", "weekend"]]

for r, (op, sg) in enumerate(rowlabels):
    for c, band in enumerate(BANDS):
        ax = axes[r, c]
        sub = OD[(OD.operator == op) & (OD.service_group == sg) & (OD.departure_band == band)]
        m = pivot(sub, hubs[op])
        ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=VMAX, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(band.replace("_", " ").upper(), fontsize=13, fontweight="bold")
        if c == 0:
            ax.set_ylabel(f"{op}\n{sg}", fontsize=11, fontweight="bold")
        ax.text(
            0.5, -0.06, f"trips={int(m.sum()):,}  nz={int((m>0).sum())}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444",
        )

fig.suptitle(
    f"OD Slices by Company x Service Type x Departure Band\n(top {N_HUB} stops per operator, trip counts, shared scale)",
    fontsize=19, fontweight="bold", y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.965])
save(fig, "A1_departure_band_20_slices.png")

# ---------------------------------------------------------------- A1b
# Supply profile: share of trips per band
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=130, sharey=True)
for ax, op in zip(axes, OPERS):
    g = OD[OD.operator == op]
    piv = g.pivot_table(index="departure_band", columns="service_group", values="trips", aggfunc="sum")
    piv = piv.reindex(BANDS).fillna(0)
    share = piv.div(piv.sum(axis=0), axis=1) * 100
    x = np.arange(len(BANDS)); w = 0.38
    for i, sg in enumerate(["weekday", "weekend"]):
        b = ax.bar(x + (i - 0.5) * w, share[sg], w, label=sg, color=["#d62728", "#ff9896"][i])
        ax.bar_label(b, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("_", " ") for b in BANDS], rotation=15)
    ax.set_title(f"{op}  (total {int(piv.sum().sum()):,} trips)", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.3); ax.legend(title="type")
    ax.set_ylabel("share of operator trips (%)")
fig.suptitle("Timetable Supply by Departure Band", fontsize=17, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "A1b_supply_profile_by_band.png")

# ---------------------------------------------------------------- A2
# Weekday - weekend delta, diverging
fig, axes = plt.subplots(1, 2, figsize=(17, 8), dpi=130)
for ax, op in zip(axes, OPERS):
    g = OD[OD.operator == op]
    wd = pivot(g[g.service_group == "weekday"], hubs[op])
    we = pivot(g[g.service_group == "weekend"], hubs[op])
    d = wd - we
    lim = np.percentile(np.abs(d[d != 0]), 98) if (d != 0).any() else 1
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{op}\nweekday minus weekend\nwd={int(wd.sum()):,} trips  we={int(we.sum()):,} trips  ratio={wd.sum()/max(we.sum(),1):.1f}x",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="trips (weekday - weekend)")
fig.suptitle(
    f"Weekday vs Weekend OD Delta (diverging, red = weekday-heavy)\ntop {N_HUB} stops per operator, clipped at p98",
    fontsize=16, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "A2_weekday_weekend_delta.png")

# ---------------------------------------------------------------- A3
# Inflow vs outflow balance per stop
fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=130)
for r, op in enumerate(OPERS):
    for c, sg in enumerate(["weekday", "weekend"]):
        ax = axes[r, c]
        g = OD[(OD.operator == op) & (OD.service_group == sg)]
        out_ = g.groupby("origin_id").trips.sum()
        in_ = g.groupby("destination_id").trips.sum()
        allstops = out_.index.union(in_.index)
        b = pd.DataFrame(
            {"out": out_.reindex(allstops).fillna(0), "in": in_.reindex(allstops).fillna(0)}
        )
        b["net"] = b["in"] - b["out"]
        pick = pd.concat([b.nsmallest(15, "net"), b.nlargest(15, "net")]).sort_values("net")
        cols = ["#d62728" if v < 0 else "#2ca02c" for v in pick["net"]]
        ax.barh(range(len(pick)), pick["net"], color=cols)
        ax.set_yticks(range(len(pick)))
        ax.set_yticklabels([s.split(":", 1)[-1] for s in pick.index], fontsize=7)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("net trips (arrivals - departures)")
        ax.set_title(f"{op} / {sg}   net range {b['net'].min():.0f} .. {b['net'].max():.0f}", fontweight="bold")
        ax.grid(axis="x", ls="--", alpha=0.3)
fig.suptitle(
    "Inflow vs Outflow Balance per Stop\nred = net sender, green = net receiver (top 15 each)",
    fontsize=17, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "A3_inflow_outflow_balance.png")

# ---------------------------------------------------------------- A4
# Implied headway quality.
# Span must be measured WITHIN a departure band: aggregating min(first)/max(last)
# across bands would stretch the span over the whole service day and invent
# multi-hour "gaps" between two independent peak services.
agg = (
    OD.groupby(["operator", "service_group", "departure_band", "origin_id", "destination_id"])
    .agg(trips=("trips", "sum"), first=("first_minutes", "min"), last=("last_minutes", "max"))
    .reset_index()
)
agg["span"] = agg["last"] - agg["first"]
agg["headway"] = np.where(agg.trips > 0, agg.span / agg.trips.replace(0, np.nan), np.nan)
agg = agg[agg.span > 0]

fig, axes = plt.subplots(2, 2, figsize=(17, 11), dpi=130, sharex=True)
stats = []
for r, op in enumerate(OPERS):
    for c, sg in enumerate(["weekday", "weekend"]):
        ax = axes[r, c]
        g = agg[(agg.operator == op) & (agg.service_group == sg)]
        bins = np.arange(0, 95, 5)
        for i, band in enumerate(BANDS):
            s = g[g.departure_band == band].headway.dropna()
            ax.hist(s, bins=bins, alpha=0.5, label=f"{band} (n={len(s):,})", edgecolor="white", linewidth=0.3)
            if len(s):
                stats.append((op, sg, band, len(s), float(s.median()), float((s > 15).mean() * 100)))
        ax.axvline(15, color="crimson", ls="--", lw=1.5)
        ax.set_yscale("log")
        ax.set_title(f"{op} / {sg}", fontweight="bold", fontsize=12)
        ax.set_xlabel("implied headway (min) = band span / trips")
        ax.set_ylabel("pair x band cells (log)")
        ax.legend(fontsize=7); ax.grid(ls="--", alpha=0.3)
fig.suptitle(
    "Implied Headway per OD Pair, measured inside each departure band\n"
    "low = well served, high = long gaps between departures (dashed = 15 min)",
    fontsize=16, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "A4_headway_quality.png")

sdf = pd.DataFrame(stats, columns=["operator", "service_group", "departure_band", "n_cells", "median_headway_min", "pct_gt_15min"])
sdf.to_csv(OUT / "A4_headway_quality.csv", index=False)
print(sdf.pivot_table(index="departure_band", columns=["operator", "service_group"], values="pct_gt_15min").round(1).to_string())
print("Tier A done.")
