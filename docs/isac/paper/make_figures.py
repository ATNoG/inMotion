#!/usr/bin/env python3
"""Generate publication-quality, spacious, and uncluttered figures for the ISAC 2026 paper.

Adheres strictly to scientific-visualization, matplotlib, and academic-plotting skills:
  - Clean seaborn-v0_8-whitegrid layout with generous margins and padding ("airosa")
  - Colorblind-safe palettes (Okabe-Ito / ColorBrewer) with high contrast
  - Built-in legend titles (ax.legend(title=..., ...)) ensuring Matplotlib's internal
    geometry engine manages the title-legend stack with 100% ZERO collision
  - Custom non-overlapping callouts with leader lines
  - High resolution (600 DPI vector PDF + 300 DPI raster PNG)
  - Designed for full column width (width=\\columnwidth) in IEEE two-column format
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ── Global Style Configuration ──────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 600

# Colorblind-safe palette (Okabe-Ito / ColorBrewer)
C_AA = "#1b9e77"  # Vibrant Teal (AA: Inside cabin)
C_AB = "#d95f02"  # Warm Terracotta (AB: Alighting)
C_BA = "#7570b3"  # Slate Purple (BA: Boarding)
C_BB = "#e6ab02"  # Amber Gold (BB: Platform stop)

C_SSL = "#1b9e77"  # Vibrant Teal (SSL World Models)
C_SSM = "#7570b3"  # Slate Purple (Mamba-3 SSM)
C_SUP = "#d95f02"  # Warm Terracotta (Supervised Models)
C_STACK = "#e7298a"  # Magenta (DeepStack)
C_PRIOR = "#8da0cb"  # Soft Blue-Gray (Earlier collection)
C_MUTED = "#7f8c8d"  # Neutral Slate Gray

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "savefig.dpi": DPI,
        # "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Mean RSSI Trajectory per Mobility Class (+-1 Std Dev)
# ═══════════════════════════════════════════════════════════════════════════
def fig_rssi_trajectory():
    data_path = Path(__file__).resolve().parents[2] / "dataset.csv"
    if not data_path.exists():
        data_path = Path("dataset.csv")
    df = pd.read_csv(data_path)

    features = [str(i) for i in range(1, 11)]
    timesteps = np.arange(1, 11)

    class_configs = [
        ("AA", "AA (Cabin Transit)", C_AA, "o", "-"),
        ("AB", "AB (Alighting)", C_AB, "s", "-"),
        ("BA", "BA (Boarding)", C_BA, "^", "-"),
        ("BB", "BB (Platform Waiting)", C_BB, "D", "-"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for cls_key, label_str, col, mark, ls in class_configs:
        sub = df[df["label"] == cls_key][features].values
        mean_vals = sub.mean(axis=0)
        std_vals = sub.std(axis=0)

        ax.plot(
            timesteps,
            mean_vals,
            marker=mark,
            markersize=3.6,
            label=label_str,
            color=col,
            linewidth=1.3,
            linestyle=ls,
            alpha=0.95,
            zorder=4,
        )
        ax.fill_between(
            timesteps, mean_vals - std_vals, mean_vals + std_vals, color=col, alpha=0.15, zorder=2
        )

    ax.set_xlabel("Observation Timestep $t$ (seconds)", fontsize=8)
    ax.set_ylabel("Mean RSSI (dBm)", fontsize=8)
    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(timesteps)
    ax.set_ylim(-65, -31)
    ax.set_yticks([-60, -50, -40])

    # Built-in legend title: Matplotlib places title strictly above legend items
    leg = ax.legend(
        title=r"Mean RSSI Trajectory per Class ($\pm 1\,\sigma$)",
        title_fontsize=9.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handletextpad=0.3,
        columnspacing=0.8,
        alignment="center",
    )
    plt.setp(leg.get_title(), fontweight="bold")

    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.15, right=0.96)
    fig.savefig(OUT / "rssi_trajectory.pdf")
    fig.savefig(OUT / "rssi_trajectory.png", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Best single model per family (extended collection)
# ═══════════════════════════════════════════════════════════════════════════
def fig_single_model():
    models = [
        "TS-JEPA",
        "LeJEPA",
        "CF-JEPA",
        "T-JEPA",
        "Mamba-3 MV",
        "DeepStack",
        "Mamba-3 TCN",
        "Mamba-3 CNN",
        "HPO LSTM",
        "CNN",
        "HPO GRU",
        "SIGReg",
    ]
    mcc = [
        0.9194,
        0.9143,
        0.9083,
        0.8982,
        0.7853,
        0.7811,
        0.7816,
        0.7703,
        0.6256,
        0.6201,
        0.5746,
        0.4872,
    ]
    fam = ["SSL", "SSL", "SSL", "SSL", "SSM", "Stack", "SSM", "SSM", "Sup", "Sup", "Sup", "Sup"]
    color_dict = {"SSL": C_SSL, "SSM": C_SSM, "Stack": C_STACK, "Sup": C_SUP}
    colors = [color_dict[f] for f in fam]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    y = np.arange(len(models))

    bars = ax.barh(y, mcc, color=colors, height=0.64, edgecolor="white", linewidth=0.6, alpha=0.92)

    for bar, val in zip(bars, mcc):
        ax.text(
            val + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=6.6,
            color="#222222",
            fontweight="normal",
        )

    ax.axvline(0.756, color="#d95f02", linestyle=":", linewidth=1.1, alpha=0.85)
    ax.text(
        0.758,
        len(models) - 0.5,
        "Prior Classical ML (0.756)",
        color="#d95f02",
        fontsize=6.2,
        va="bottom",
        ha="left",
        style="italic",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.2)
    ax.set_xlabel("Matthews Correlation Coefficient (MCC)", fontsize=8)
    ax.set_xlim(0.40, 1.08)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    legend_elements = [
        Patch(facecolor=C_SSL, edgecolor="white", label="SSL World Model"),
        Patch(facecolor=C_SSM, edgecolor="white", label="Mamba-3 SSM"),
        Patch(facecolor=C_SUP, edgecolor="white", label="Supervised Baseline"),
    ]
    leg = ax.legend(
        handles=legend_elements,
        title="Single-Model Performance (New 3,511 Samples)",
        title_fontsize=9.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handletextpad=0.3,
        columnspacing=0.6,
        alignment="center",
    )
    plt.setp(leg.get_title(), fontweight="bold")

    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.22, right=0.96)
    fig.savefig(OUT / "single_model_mcc.pdf")
    fig.savefig(OUT / "single_model_mcc.png", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-collection generalisation (SSL vs Supervised)
# ═══════════════════════════════════════════════════════════════════════════
def fig_transfer():
    names = ["TS-JEPA", "LeJEPA", "CF-JEPA", "T-JEPA", "Mamba-3 TCN", "DeepStack", "SIGReg"]
    prior = [0.8785, 0.8850, 0.8661, 0.8694, 0.8844, 0.8810, 0.8858]
    extended = [0.9194, 0.9143, 0.9083, 0.8982, 0.7816, 0.7811, 0.4872]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(
        x - width / 2,
        prior,
        width,
        label="Prior (2,251 samples)",
        color=C_PRIOR,
        edgecolor="white",
        linewidth=0.6,
    )
    bars2 = ax.bar(
        x + width / 2,
        extended,
        width,
        label="New (3,511 samples)",
        color=C_SSL,
        edgecolor="white",
        linewidth=0.6,
    )

    for i in range(4, len(names)):
        bars2[i].set_color(C_SUP)

    for i, (p, e) in enumerate(zip(prior, extended)):
        diff = e - p
        delta_str = f"{diff:+.2f}"
        ax.text(
            i + width / 2,
            e + 0.02,
            delta_str,
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            color=C_SSL if diff > 0 else C_SUP,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=24, ha="right", fontsize=7.0)
    ax.set_ylabel("Test MCC", fontsize=8)
    ax.set_ylim(0.0, 1.22)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    leg = ax.legend(
        title="Cross-Collection Generalization",
        title_fontsize=9.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.6,
        handletextpad=0.3,
        columnspacing=0.8,
        alignment="center",
    )
    plt.setp(leg.get_title(), fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.14, right=0.96)
    fig.savefig(OUT / "transfer_mcc.pdf")
    fig.savefig(OUT / "transfer_mcc.png", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Parameter efficiency (MCC vs size, log x) + leader pointer callouts
# ═══════════════════════════════════════════════════════════════════════════
def fig_param_efficiency():
    members = [
        ("GRU", 0.039, 0.3360, "Sup", (8, -3), "left"),
        ("LSTM", 0.052, 0.5547, "Sup", (-18, 14), "right"),
        ("HPO CNN", 0.081, 0.5785, "Sup", (-12, -18), "right"),
        ("BiLSTM", 0.136, 0.5622, "Sup", (10, 12), "left"),
        ("HPO Mamba", 0.278, 0.6049, "Sup", (-8, 14), "center"),
        ("LeJEPA", 0.566, 0.8850, "SSL", (-35, 18), "right"),
        ("T-JEPA", 0.633, 0.8694, "SSL", (-35, -16), "right"),
        ("SIGReg", 0.697, 0.8858, "Sup", (-8, 22), "center"),
        ("Mamba-3 Tf", 0.842, 0.8581, "SSM", (-15, -24), "center"),
        ("Mamba-3 TCN", 0.843, 0.8844, "SSM", (10, 24), "left"),
        ("Mamba-3 CNN", 0.945, 0.8567, "SSM", (10, -22), "left"),
        ("CF-JEPA", 1.232, 0.8661, "SSL", (28, -12), "left"),
        ("Mamba-3 MV", 1.701, 0.8640, "SSM", (25, 4), "left"),
        ("TS-JEPA", 2.153, 0.8785, "SSL", (15, 18), "left"),
        ("TCN", 2.180, 0.5545, "Sup", (-14, -16), "right"),
        ("HPO GRU", 4.008, 0.5306, "Sup", (12, -14), "left"),
        ("HPO LSTM", 13.110, 0.6092, "Sup", (10, 10), "left"),
        ("DeepStack", 43.576, 0.8810, "Stack", (0, 10), "center"),
    ]

    col_map = {"SSL": C_SSL, "SSM": C_SSM, "Stack": C_STACK, "Sup": C_MUTED}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for name, p, m, cat, offset, ha in members:
        ax.scatter(p, m, color=col_map[cat], s=30, edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(
            name,
            (p, m),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            fontsize=5.8,
            color="#222222",
            fontweight="normal",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "lw": 0.6,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )

    # Frontier path
    fx = [0.035, 0.566, 1.263]
    fy = [0.5305, 0.8694, 0.8904]
    ax.plot(
        fx,
        fy,
        color=C_SUP,
        linewidth=1.4,
        linestyle="-",
        marker="o",
        markersize=4.2,
        zorder=5,
        label="Three-member frontier",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.02, 85)
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Parameters (M, log scale)", fontsize=8)
    ax.set_ylabel("Test MCC (Prior Collection)", fontsize=8)

    ax.set_title("Parameter Efficiency & Model Scaling", fontsize=9.0, fontweight="bold", pad=8)

    # Legend placed in EMPTY LOWER-RIGHT
    legend_items = [
        Line2D([0], [0], color=C_SUP, marker="o", lw=1.3, label="Size Frontier (0.8904)"),
        Patch(facecolor=C_SSL, edgecolor="white", label="SSL World Model"),
        Patch(facecolor=C_SSM, edgecolor="white", label="Mamba-3 SSM"),
        Patch(facecolor=C_STACK, edgecolor="white", label="DeepStack (43.6M)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d5d5d5",
        fontsize=5.8,
        handletextpad=0.3,
    )

    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.14, right=0.96)
    fig.savefig(OUT / "param_efficiency.pdf")
    fig.savefig(OUT / "param_efficiency.png", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Game-Theoretic SHAP Feature Channel Attribution (18 channels)
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_importance():
    # Game-theoretic Shapley Additive exPlanations (SHAP) across 18 feature channels
    # evaluated on TS-JEPA context representations (dataset.csv test split).
    features = [
        ("Peak Location", 0.00604, "Shape"),
        ("STFT Band 0", 0.00597, "Spectral"),
        ("Rolling FFT DC", 0.00440, "Spectral"),
        ("Rolling FFT AC", 0.00399, "Spectral"),
        ("Sign Change Count", 0.00385, "Shape"),
        ("Raw RSSI", 0.00321, "Base"),
        ("Rolling Range", 0.00312, "Stats"),
        ("Slope Sign", 0.00275, "Shape"),
        (r"Acceleration ($\Delta^2$)", 0.00253, "Base"),
        ("Rolling Skew", 0.00243, "Stats"),
        ("Rolling Mean", 0.00125, "Stats"),
        ("Rolling Kurtosis", 0.00122, "Stats"),
        ("STFT Band 3", 0.00106, "Spectral"),
        ("STFT Band 2", 0.00105, "Spectral"),
        ("Rolling Std", 0.00095, "Stats"),
        ("STFT Band 1", 0.00087, "Spectral"),
        (r"Velocity ($\Delta$)", 0.00068, "Base"),
        ("Window Dev.", 0.00056, "Base"),
    ]

    names = [f[0] for f in features][::-1]
    drops = [f[1] * 1000 for f in features][::-1]  # Scale x10^-3 for clean readability
    families = [f[2] for f in features][::-1]

    fam_colors = {
        "Base": C_SSL,  # Vibrant Teal
        "Spectral": C_SSM,  # Slate Purple
        "Stats": C_PRIOR,  # Soft Blue-Gray
        "Shape": C_SUP,  # Warm Terracotta
    }
    colors = [fam_colors[f] for f in families]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    y = np.arange(len(names))
    bars = ax.barh(
        y, drops, color=colors, height=0.66, edgecolor="white", linewidth=0.5, alpha=0.92
    )

    for bar, val in zip(bars, drops):
        ax.text(
            val + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=6.2,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel(r"Mean Absolute SHAP Value ($\times 10^{-3}$)", fontsize=8)
    ax.set_xlim(0.0, 7.2)
    ax.set_xticks([0.0, 2.0, 4.0, 6.0])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    # Legend + Title centered over the ENTIRE FIGURE canvas
    legend_items = [
        Patch(facecolor=C_SSL, edgecolor="white", label="Base Kinematics"),
        Patch(facecolor=C_SSM, edgecolor="white", label="Spectral (FFT/STFT)"),
        Patch(facecolor=C_PRIOR, edgecolor="white", label="Rolling Stats"),
        Patch(facecolor=C_SUP, edgecolor="white", label="Shape Descriptors"),
    ]
    leg = fig.legend(
        handles=legend_items,
        title=r"SHAP Feature Channel Attribution",
        title_fontsize=9.0,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.855),
        ncol=2,
        frameon=False,
        fontsize=6.0,
        handletextpad=0.3,
        columnspacing=0.8,
        alignment="center",
    )
    plt.setp(leg.get_title(), fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.30, right=0.96)
    fig.savefig(OUT / "feature_importance_18ch.pdf")
    fig.savefig(OUT / "feature_importance_18ch.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating ISAC paper figures (built-in legend titles, zero overlap)...")
    fig_rssi_trajectory()
    fig_single_model()
    fig_transfer()
    fig_param_efficiency()
    fig_feature_importance()
    print("Done: All figures written to", OUT)
