"""Visualization module for DL classification results — paper-quality PDFs at 600 DPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DLConfig


class DLVisualizer:
    """Publication-quality visualizations for DL model results."""

    def __init__(self, config: DLConfig, output_dir: Path | None = None) -> None:
        self.config = config
        self.results_plots_dir = output_dir or (config.plots_dir / "results")
        self.results_plots_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = sns.color_palette("husl", 20)

        # Hardcoded large font sizes for paper — per taste preferences
        self.PLOT_DPI = 600
        self.FONT_SIZE = 18
        self.TITLE_SIZE = 20
        self.LABEL_SIZE = 18
        self.TICK_SIZE = 16
        self.LEGEND_SIZE = 16
        self.FONT_SCALE = 2.0

        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        plt.rcParams.update(
            {
                "font.size": self.FONT_SIZE,
                "axes.titlesize": self.TITLE_SIZE,
                "axes.labelsize": self.LABEL_SIZE,
                "xtick.labelsize": self.TICK_SIZE,
                "ytick.labelsize": self.TICK_SIZE,
                "legend.fontsize": self.LEGEND_SIZE,
                "figure.titlesize": self.TITLE_SIZE + 2,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        sns.set_context("paper", font_scale=self.FONT_SCALE)

    def _save(self, base_name: str) -> Path:
        return self.results_plots_dir / f"{base_name}.pdf"

    # ── Confusion Matrices ───────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str],
        model_name: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        plt.savefig(self._save(f"confusion_matrix_{model_name}"), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    def plot_all_confusion_matrices(
        self,
        confusion_matrices: dict[str, np.ndarray],
        class_names: list[str],
        top_n: int = 10,
    ) -> None:
        for name, cm in list(confusion_matrices.items())[:top_n]:
            self.plot_confusion_matrix(cm, class_names, name)

    # ── Classifier Comparisons ───────────────────────────────────────────────

    def plot_comparison_bars(
        self,
        results_df: pd.DataFrame,
        metric: str = "test_mcc",
        top_n: int = 15,
    ) -> None:
        col = metric if metric in results_df.columns else f"test_{metric.lower()}"
        if col not in results_df.columns:
            return
        df = results_df.nlargest(top_n, col).copy()
        display_metric = metric.replace("test_", "").replace("_", " ").upper()

        fig, ax = plt.subplots(figsize=(14, 10))
        colors = sns.color_palette("viridis", len(df))
        ax.barh(range(len(df)), df[col], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"])
        ax.set_xlabel(display_metric)
        ax.set_title(f"Top {top_n} DL Models by {display_metric}")
        ax.invert_yaxis()
        for i, val in enumerate(df[col]):
            ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontsize=self.TICK_SIZE)
        plt.tight_layout()
        fname = f"model_comparison_{col}"
        plt.savefig(self._save(fname), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    # ── Multi-metric Comparison ──────────────────────────────────────────────

    def plot_multi_metric_comparison(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10,
    ) -> None:
        metric_cols = []
        for m in ["f1_macro", "f1_weighted", "test_acc", "test_mcc", "precision_macro", "recall_macro"]:
            if m in results_df.columns:
                metric_cols.append(m)
        if not metric_cols:
            return

        df = results_df.nlargest(top_n, "test_mcc" if "test_mcc" in results_df.columns else metric_cols[0]).copy()

        fig, ax = plt.subplots(figsize=(16, 10))
        x = np.arange(len(df))
        width = 0.14

        for i, metric in enumerate(metric_cols):
            offset = (i - len(metric_cols) / 2 + 0.5) * width
            ax.bar(x + offset, df[metric], width, label=metric, color=self.colors[i % len(self.colors)])

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"Top {top_n} DL Models — Multi-Metric Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend(loc="lower right", fontsize=self.LEGEND_SIZE - 2)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self._save("multi_metric_comparison"), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    # ── Per-Class Performance ────────────────────────────────────────────────

    def plot_class_performance(
        self,
        per_class_reports: dict[str, dict[str, dict[str, float]]],
        class_names: list[str],
        top_n: int = 5,
    ) -> None:
        metrics = ["precision", "recall", "f1-score"]
        models = list(per_class_reports.keys())[:top_n]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            data = []
            for model_name in models:
                report = per_class_reports[model_name]
                values = [report.get(cls, {}).get(metric, 0) for cls in class_names]
                data.append(values)
            data = np.array(data)

            x = np.arange(len(class_names))
            width = 0.8 / len(models)

            for i, model_name in enumerate(models):
                offset = (i - len(models) / 2 + 0.5) * width
                ax.bar(x + offset, data[i], width, label=model_name, color=self.colors[i % len(self.colors)])

            ax.set_xlabel("Class")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Per-Class {metric.capitalize()} — Top {top_n} DL Models")
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend(loc="lower right", fontsize=self.LEGEND_SIZE)
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(self._save(f"class_performance_{metric}"), dpi=self.PLOT_DPI, bbox_inches="tight")
            plt.close()

    # ── Metric Heatmap ───────────────────────────────────────────────────────

    def plot_metric_heatmap(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
    ) -> None:
        metric_cols = []
        for m in ["test_mcc", "test_acc", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
            if m in results_df.columns:
                metric_cols.append(m)
        if not metric_cols:
            return

        df = results_df.nlargest(top_n, "test_mcc" if "test_mcc" in results_df.columns else metric_cols[0])
        heatmap_df = df[["model"] + metric_cols].set_index("model")

        fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.5)))
        sns.heatmap(
            heatmap_df.values,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=[c.replace("test_", "").replace("_", " ") for c in metric_cols],
            yticklabels=heatmap_df.index,
            ax=ax,
            cbar_kws={"label": "Score"},
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Performance Heatmap — Top {top_n} DL Models")
        plt.tight_layout()
        plt.savefig(self._save("metric_heatmap"), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    # ── Metric vs Time Scatter ───────────────────────────────────────────────

    def plot_metric_vs_time(
        self,
        results_df: pd.DataFrame,
        metric: str = "test_mcc",
    ) -> None:
        if "train_time_s" not in results_df.columns or metric not in results_df.columns:
            return

        df = results_df.dropna(subset=["train_time_s", metric])
        if df.empty:
            return
        if df["train_time_s"].max() <= 0:
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(
            df["train_time_s"],
            df[metric],
            c=df.get("best_val_mcc", df[metric]),
            cmap="viridis",
            s=120,
            alpha=0.7,
        )
        for _, row in df.iterrows():
            ax.annotate(
                str(row["model"]),
                (row["train_time_s"], row[metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=self.TICK_SIZE - 2,
                alpha=0.8,
            )
        ax.set_xlabel("Training Time (seconds, log scale)")
        display_metric = metric.replace("test_", "").replace("_", " ").upper()
        ax.set_ylabel(display_metric)
        ax.set_title(f"{display_metric} vs Training Time")
        ax.set_xscale("log")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Best Val MCC")
        plt.tight_layout()
        plt.savefig(self._save(f"{metric}_vs_time"), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    # ── Training Times ───────────────────────────────────────────────────────

    def plot_training_times(
        self,
        results_df: pd.DataFrame,
    ) -> None:
        if "train_time_s" not in results_df.columns:
            return
        df = results_df.dropna(subset=["train_time_s"]).sort_values("train_time_s", ascending=True)
        if df["train_time_s"].max() <= 0:
            return

        fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))
        colors = ["green" if t < 100 else "orange" if t < 400 else "red" for t in df["train_time_s"]]
        ax.barh(range(len(df)), df["train_time_s"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"])
        ax.set_xlabel("Training Time (seconds)")
        ax.set_title("DL Model Training Times")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(self._save("training_times"), dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close()

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def create_all_plots(
        self,
        results_df: pd.DataFrame,
        confusion_matrices: dict[str, np.ndarray],
        class_names: list[str],
        per_class_reports: dict[str, dict[str, dict[str, float]]] | None = None,
        top_n_models: int = 15,
    ) -> None:
        self.plot_comparison_bars(results_df, "test_mcc", top_n_models)
        self.plot_comparison_bars(results_df, "test_acc", top_n_models)
        if "f1_macro" in results_df.columns:
            self.plot_comparison_bars(results_df, "f1_macro", top_n_models)
        self.plot_multi_metric_comparison(results_df, min(top_n_models, 10))
        self.plot_metric_heatmap(results_df, top_n_models)
        self.plot_metric_vs_time(results_df, "test_mcc")
        self.plot_metric_vs_time(results_df, "test_acc")
        if "f1_macro" in results_df.columns:
            self.plot_metric_vs_time(results_df, "f1_macro")
        self.plot_training_times(results_df)

        if confusion_matrices:
            self.plot_all_confusion_matrices(confusion_matrices, class_names, top_n=top_n_models)

        if per_class_reports:
            self.plot_class_performance(per_class_reports, class_names, top_n=min(5, len(per_class_reports)))
