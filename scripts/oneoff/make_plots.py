#!/usr/bin/env python3
"""Generate two big plots for bus OD matrices and routes."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUT_DIR = Path("plots/bus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Giant plot: all wide matrices by company + type
# ------------------------------------------------------------------
files = {
    "Aveirobus – Weekday": "data/bus/processed/od_matrix_wide_aveirobus_weekday.csv",
    "Aveirobus – Weekend": "data/bus/processed/od_matrix_wide_aveirobus_weekend.csv",
    "Busway – Weekday":    "data/bus/processed/od_matrix_wide_busway_weekday.csv",
    "Busway – Weekend":    "data/bus/processed/od_matrix_wide_busway_weekend.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(28, 24), dpi=150)
fig.suptitle("OD Matrices by Company and Type (Trip Counts)", fontsize=22, fontweight="bold", y=0.98)

# Fixed color scale across all (max observed ~109)
V_MAX = 110
cmap = plt.colormaps["YlOrRd"]

for ax, (title, path) in zip(axes.flatten(), files.items()):
    df = pd.read_csv(path, index_col=0)
    # Drop human-readable origin name column -> keep numeric destinations only
    if "origin" in df.columns:
        df = df.drop(columns=["origin"])
    df_num = df.select_dtypes(include=[np.number])
    arr = df_num.values
    # For very large matrices we can downsample purely for display speed,
    # but keep original density for accuracy. 1661x1635 is okay for imshow.
    # Per-matrix vmax so sparse low counts are visible (clip extreme outliers)
    pos = arr[arr > 0]
    v_local = int(np.percentile(pos, 95)) if len(pos) else 1
    v_local = max(v_local, 1)
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=min(v_local, V_MAX), aspect="auto", interpolation="nearest")
    ax.set_title(f"{title}\nshape={arr.shape[0]}×{arr.shape[1]}  max={int(arr.max())}  v95={v_local}  non-zero={int((arr>0).sum())}", fontsize=11)
    ax.set_xlabel("Destination stop index")
    ax.set_ylabel("Origin stop index")
    # Hide ticks because too dense
    ax.set_xticks([])
    ax.set_yticks([])
    # Add colorbar per subplot (small)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = OUT_DIR / "giant_od_matrices_by_company_and_type.png"
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Plot 1 saved:", fig_path)

# ------------------------------------------------------------------
# 2) Big plot: all routes (scatter of all OD pairs)
# ------------------------------------------------------------------
od = pd.read_csv("data/bus/processed/od_matrix.csv")
# Drop zero-trip rows for clarity
od = od[od["trips"] > 0].copy()

fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

# Color by operator, shape by service_group
colors = {"aveirobus": "#d62728", "busway": "#1f77b4"}
markers = {"weekday": "o", "weekend": "s"}
labels_done = set()

for op in od["operator"].unique():
    for sg in od["service_group"].unique():
        sub = od[(od["operator"] == op) & (od["service_group"] == sg)]
        label = f"{op} – {sg}"
        if label in labels_done:
            label = None
        else:
            labels_done.add(label)
        ax.scatter(
            sub["trips"],
            sub["median_run_min"],
            s=sub["trips"] * 3 + 10,
            c=colors.get(op, "gray"),
            marker=markers.get(sg, "o"),
            alpha=0.35,
            edgecolors="none",
            label=label,
        )

ax.set_xlabel("Total trips (sum across departure bands)", fontsize=14)
ax.set_ylabel("Median run time (min)", fontsize=14)
ax.set_title("All Bus Routes – OD Pairs (Size = number of lines, Alpha = density)", fontsize=18, fontweight="bold")
ax.set_xscale("log")
ax.set_yscale("linear")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(title="Operator – Type", loc="upper left", fontsize=9, markerscale=0.8)

# Add a small text box with count
count_total = len(od)
ax.annotate(f"Points shown: {count_total:,}\nZero-trip pairs excluded",
            xy=(0.98, 0.02), xycoords="axes fraction",
            fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

fig.tight_layout()
fig_path2 = OUT_DIR / "all_routes_big_plot.png"
fig.savefig(fig_path2, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Plot 2 saved:", fig_path2)

# Verify outputs exist and are PNG (not webp)
for p in [fig_path, fig_path2]:
    import os
    sz = os.path.getsize(p)
    print("  ", p.name, "size=", sz, "bytes", "format=", p.suffix)
