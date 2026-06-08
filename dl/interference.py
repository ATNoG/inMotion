"""Cross-route interference analysis — how concurrent paths affect RSSI signatures.

Analyses the `concurrent_noise_path` column to measure how strongly each interfering
route distorts the RSSI readings of the primary route. Generates publication-quality
PDF plots at 600 DPI with large fonts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROUTES = ["AA", "AB", "BA", "BB"]
FEATURE_COLS = [str(i) for i in range(1, 11)]
TIME_STEPS = list(range(1, 11))


class InterferenceAnalyzer:
    """Analyze how concurrent noise from different routes affects RSSI readings."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.DPI = 600
        self.FONT_SIZE = 18
        self.TITLE_SIZE = 20
        self.LABEL_SIZE = 18
        self.TICK_SIZE = 16
        self.LEGEND_SIZE = 16
        self.FONT_SCALE = 2.0

        self._setup_style()

    def _setup_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": self.FONT_SIZE,
            "axes.titlesize": self.TITLE_SIZE,
            "axes.labelsize": self.LABEL_SIZE,
            "xtick.labelsize": self.TICK_SIZE,
            "ytick.labelsize": self.TICK_SIZE,
            "legend.fontsize": self.LEGEND_SIZE,
            "figure.titlesize": self.TITLE_SIZE + 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
        sns.set_context("paper", font_scale=self.FONT_SCALE)

    def _save(self, name: str) -> Path:
        return self.output_dir / f"{name}.pdf"

    # ── Data preparation ─────────────────────────────────────────────────────

    def _build_interference_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean RSSI shift per (label, concurrent_path) relative to pure baseline.

        Returns (shift_matrix, count_matrix, pure_means) where:
          - shift_matrix[label_idx, cpath_idx] = mean RSSI difference vs pure baseline
          - count_matrix[label_idx, cpath_idx] = number of samples
          - pure_means[label_idx] = mean RSSI when no noise
        """
        shift = np.empty((4, 4))
        counts = np.empty((4, 4), dtype=int)
        pure_means = np.empty(4)

        pure_df = df[df["noise"] == False]
        noise_df = df[df["noise"] == True]

        for i, label in enumerate(ROUTES):
            pure = pure_df[pure_df["label"] == label]
            pure_rssi = pure[FEATURE_COLS].values.flatten()
            pure_mean = float(np.mean(pure_rssi)) if len(pure_rssi) > 0 else 0.0
            pure_means[i] = pure_mean

            for j, cpath in enumerate(ROUTES):
                sub = noise_df[
                    (noise_df["label"] == label)
                    & (noise_df["concurrent_noise_path"] == cpath)
                ]
                sub_rssi = sub[FEATURE_COLS].values.flatten()
                sub_mean = float(np.mean(sub_rssi)) if len(sub_rssi) > 0 else 0.0
                shift[i, j] = sub_mean - pure_mean
                counts[i, j] = len(sub)

        return shift, counts, pure_means

    # ── Plot 1: Interference heatmap ────────────────────────────────────────

    def plot_interference_heatmap(self, df: pd.DataFrame) -> None:
        """4×4 heatmap: (row=label, col=interfering path) → mean RSSI shift (dBm)."""
        shift, counts, _ = self._build_interference_matrix(df)

        annot = np.empty((4, 4), dtype=object)
        for i in range(4):
            for j in range(4):
                annot[i, j] = f"{shift[i, j]:+.1f}\nn={counts[i, j]}"

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            shift, annot=annot, fmt="", cmap="RdBu_r", center=0,
            xticklabels=ROUTES, yticklabels=ROUTES, ax=ax,
            cbar_kws={"label": "Δ RSSI (dBm)"},
            linewidths=0.5,
        )
        ax.set_xlabel("Concurrent Noise Path")
        ax.set_ylabel("Primary Route")
        ax.set_title("Interference Matrix — Mean RSSI Shift vs Pure Baseline")
        plt.tight_layout()
        plt.savefig(self._save("interference_heatmap"), dpi=self.DPI, bbox_inches="tight")
        plt.close()

    # ── Plot 2: Per-route RSSI trajectories under interference ──────────────

    def plot_interference_trajectories(self, df: pd.DataFrame) -> None:
        """For each route, plot mean RSSI ± std per time step under each concurrent path."""
        colors = sns.color_palette("husl", 5)  # 4 concurrent + pure

        for label in ROUTES:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Pure baseline
            pure = df[(df["label"] == label) & (df["noise"] == False)]
            if len(pure) > 0:
                pure_mean = pure[FEATURE_COLS].mean().values
                pure_std = pure[FEATURE_COLS].std().values
                ax.plot(TIME_STEPS, pure_mean, "k-", linewidth=3, label="No interference (pure)")
                ax.fill_between(TIME_STEPS, pure_mean - pure_std, pure_mean + pure_std,
                                alpha=0.15, color="black")

            for k, cpath in enumerate(ROUTES):
                sub = df[
                    (df["label"] == label)
                    & (df["noise"] == True)
                    & (df["concurrent_noise_path"] == cpath)
                ]
                if len(sub) == 0:
                    continue
                sub_mean = sub[FEATURE_COLS].mean().values
                sub_std = sub[FEATURE_COLS].std().values
                ax.plot(TIME_STEPS, sub_mean, marker="o", linewidth=2,
                        color=colors[k], label=f"Interferer: {cpath}")
                ax.fill_between(TIME_STEPS, sub_mean - sub_std, sub_mean + sub_std,
                                alpha=0.12, color=colors[k])

            ax.set_xlabel("Time Step")
            ax.set_ylabel("RSSI (dBm)")
            ax.set_title(f"RSSI Trajectory — Route {label} Under Interference")
            ax.legend(loc="best")
            ax.set_xticks(TIME_STEPS)
            plt.tight_layout()
            plt.savefig(self._save(f"interference_trajectory_{label}"), dpi=self.DPI, bbox_inches="tight")
            plt.close()

    # ── Plot 3: Interference delta bar chart ────────────────────────────────

    def plot_interference_delta_bars(self, df: pd.DataFrame) -> None:
        """Per-route grouped bar chart: Δ RSSI per interfering path."""
        shift, _, _ = self._build_interference_matrix(df)
        colors = sns.color_palette("husl", 4)

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(4)
        width = 0.2

        for j, cpath in enumerate(ROUTES):
            offset = (j - 1.5) * width
            ax.bar(x + offset, shift[:, j], width, label=f"Interferer: {cpath}", color=colors[j])

        ax.set_xlabel("Primary Route")
        ax.set_ylabel("Δ RSSI (dBm) from Pure Baseline")
        ax.set_title("Interference Impact — RSSI Shift per Interfering Route")
        ax.set_xticks(x)
        ax.set_xticklabels(ROUTES)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(self._save("interference_delta_bars"), dpi=self.DPI, bbox_inches="tight")
        plt.close()

    # ── Plot 4: Per-route violin plot ───────────────────────────────────────

    def plot_interference_violins(self, df: pd.DataFrame) -> None:
        """Per-route: violin plots of RSSI distribution under each interference condition."""
        for label in ROUTES:
            fig, ax = plt.subplots(figsize=(12, 8))

            data_groups = []
            tick_labels = []

            pure = df[(df["label"] == label) & (df["noise"] == False)]
            pure_vals = pure[FEATURE_COLS].values.flatten()
            if len(pure_vals) > 0:
                data_groups.append(pure_vals)
                tick_labels.append("pure")

            for cpath in ROUTES:
                sub = df[
                    (df["label"] == label)
                    & (df["noise"] == True)
                    & (df["concurrent_noise_path"] == cpath)
                ]
                vals = sub[FEATURE_COLS].values.flatten()
                if len(vals) > 0:
                    data_groups.append(vals)
                    tick_labels.append(cpath)

            if not data_groups:
                plt.close()
                continue

            positions = np.arange(len(data_groups))
            vp = ax.violinplot(data_groups, positions=positions, showmeans=True, showextrema=True)

            for body in vp["bodies"]:
                body.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Interference Condition")
            ax.set_ylabel("RSSI (dBm)")
            ax.set_title(f"RSSI Distribution Under Interference — Route {label}")
            plt.tight_layout()
            plt.savefig(self._save(f"interference_violin_{label}"), dpi=self.DPI, bbox_inches="tight")
            plt.close()

    # ── Plot 5: Per-timestep interference heatmap ───────────────────────────

    def plot_timestep_interference(self, df: pd.DataFrame) -> None:
        """4 × 10 heatmap: timestep × interfering path, showing mean RSSI shift per timestep."""
        pure_df = df[df["noise"] == False]
        noise_df = df[df["noise"] == True]

        for label in ROUTES:
            pure = pure_df[pure_df["label"] == label]
            pure_per_ts = pure[FEATURE_COLS].mean().values  # (10,)

            shift_ts = np.empty((4, 10))
            for k, cpath in enumerate(ROUTES):
                sub = noise_df[
                    (noise_df["label"] == label)
                    & (noise_df["concurrent_noise_path"] == cpath)
                ]
                if len(sub) > 0:
                    sub_per_ts = sub[FEATURE_COLS].mean().values
                    shift_ts[k, :] = sub_per_ts - pure_per_ts
                else:
                    shift_ts[k, :] = np.nan

            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(
                shift_ts, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                xticklabels=TIME_STEPS, yticklabels=ROUTES, ax=ax,
                cbar_kws={"label": "Δ RSSI (dBm)"},
                linewidths=0.5,
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Interfering Route")
            ax.set_title(f"Per-Timestep RSSI Shift — Route {label}")
            plt.tight_layout()
            plt.savefig(self._save(f"interference_timestep_{label}"), dpi=self.DPI, bbox_inches="tight")
            plt.close()

    # ── Plot 6: PCA colored by interference condition ───────────────────────

    def plot_pca_by_interference(self, df: pd.DataFrame) -> None:
        """PCA scatter colored by interference condition, one plot per route."""
        X_all = df[FEATURE_COLS].values.astype(np.float32)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_all)

        for label in ROUTES:
            fig, ax = plt.subplots(figsize=(12, 10))
            colors = sns.color_palette("husl", 5)

            # Pure baseline
            mask_pure = (df["label"] == label) & (df["noise"] == False)
            ax.scatter(
                X_pca[mask_pure, 0], X_pca[mask_pure, 1],
                c=[colors[0]], label="No interference", alpha=0.6, s=60, edgecolors="black",
            )

            for k, cpath in enumerate(ROUTES):
                mask = (
                    (df["label"] == label)
                    & (df["noise"] == True)
                    & (df["concurrent_noise_path"] == cpath)
                )
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    c=[colors[k + 1]], label=f"Interferer: {cpath}", alpha=0.6, s=60,
                )

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title(f"PCA — Route {label} Under Interference")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self._save(f"interference_pca_{label}"), dpi=self.DPI, bbox_inches="tight")
            plt.close()

    # ── Orchestrator ────────────────────────────────────────────────────────

    def run_full_analysis(self, df: pd.DataFrame) -> list[Path]:
        """Generate all interference analysis plots. Returns list of saved paths."""
        self.plot_interference_heatmap(df)
        self.plot_interference_trajectories(df)
        self.plot_interference_delta_bars(df)
        self.plot_interference_violins(df)
        self.plot_timestep_interference(df)
        self.plot_pca_by_interference(df)

        plots = sorted(self.output_dir.glob("*.pdf"))
        return plots
