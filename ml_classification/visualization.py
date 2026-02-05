"""Visualization module for ML classification results."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config


def load_results_from_csv(
    results_csv: Path | str,
    detailed_csv: Path | str | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    pd.DataFrame,
    list[str],
    dict[str, dict[str, Any]],
]:
    """Load classification results from CSV files for plot regeneration.

    Args:
        results_csv: Path to classification_results.csv
        detailed_csv: Path to detailed_classifier_results.csv (optional, will infer if not provided)

    Returns:
        Tuple of (results_df, confusion_matrices, feature_importances,
                  feature_importance_summary, class_names, classification_reports)
    """
    results_csv = Path(results_csv)
    results_df = pd.read_csv(results_csv)

    # Infer detailed CSV path if not provided
    if detailed_csv is None:
        detailed_csv = results_csv.parent / "detailed_classifier_results.csv"
    detailed_csv = Path(detailed_csv)

    if not detailed_csv.exists():
        raise FileNotFoundError(f"Detailed results not found: {detailed_csv}")

    detailed_df = pd.read_csv(detailed_csv)

    # Extract class names from column patterns (e.g., AA_precision, BB_recall)
    class_names = []
    for col in detailed_df.columns:
        if col.endswith("_precision") and not col.startswith("FeatImp"):
            class_name = col.replace("_precision", "")
            if class_name not in class_names:
                class_names.append(class_name)

    # Extract confusion matrices
    confusion_matrices: dict[str, np.ndarray] = {}
    for _, row in detailed_df.iterrows():
        clf_name = row["Classifier"]
        if pd.notna(row.get("Confusion_Matrix")):
            try:
                cm_list = json.loads(row["Confusion_Matrix"])
                confusion_matrices[clf_name] = np.array(cm_list)
            except (json.JSONDecodeError, TypeError):
                pass

    # Extract feature importances
    feature_importances: dict[str, np.ndarray] = {}
    feat_imp_cols = [c for c in detailed_df.columns if c.startswith("FeatImp_")]

    for _, row in detailed_df.iterrows():
        clf_name = row["Classifier"]
        importances = []
        has_importance = False
        for col in feat_imp_cols:
            val = row.get(col)
            if pd.notna(val):
                importances.append(float(val))
                has_importance = True
            else:
                importances.append(0.0)
        if has_importance and any(imp > 0 for imp in importances):
            feature_importances[clf_name] = np.array(importances)

    # Compute feature importance summary
    feature_importance_summary = pd.DataFrame()
    if feature_importances:
        feature_names = [c.replace("FeatImp_", "") for c in feat_imp_cols]

        # Normalize each classifier's importances to [0, 1] before aggregating
        normalized_importances = []
        for importances in feature_importances.values():
            # Min-max normalization
            min_val = np.min(importances)
            max_val = np.max(importances)
            if max_val > min_val:
                normalized = (importances - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(importances)
            normalized_importances.append(normalized)

        all_importances = np.array(normalized_importances)
        mean_importances = np.mean(all_importances, axis=0)
        std_importances = np.std(all_importances, axis=0)
        feature_importance_summary = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Importance": mean_importances,
                "Std_Importance": std_importances,
            }
        )

    # Reconstruct classification reports
    classification_reports: dict[str, dict[str, Any]] = {}
    for _, row in detailed_df.iterrows():
        clf_name = row["Classifier"]
        report: dict[str, Any] = {}
        for cls_name in class_names:
            precision_col = f"{cls_name}_precision"
            recall_col = f"{cls_name}_recall"
            f1_col = f"{cls_name}_f1"
            support_col = f"{cls_name}_support"

            if precision_col in row and pd.notna(row.get(precision_col)):
                report[cls_name] = {
                    "precision": row.get(precision_col, 0),
                    "recall": row.get(recall_col, 0),
                    "f1-score": row.get(f1_col, 0),
                    "support": int(row.get(support_col, 0))
                    if pd.notna(row.get(support_col))
                    else 0,
                }
        if report:
            classification_reports[clf_name] = report

    return (
        results_df,
        confusion_matrices,
        feature_importances,
        feature_importance_summary,
        class_names,
        classification_reports,
    )


class Visualizer:
    """Visualization utilities for ML classification results."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_plots_dir = config.plots_dir / "results"
        self.results_plots_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = sns.color_palette("husl", 20)

        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        """Setup matplotlib style with publication-ready fonts."""
        plt.rcParams.update(
            {
                "font.size": self.config.plot_font_size,
                "axes.titlesize": self.config.plot_title_size,
                "axes.labelsize": self.config.plot_label_size,
                "xtick.labelsize": self.config.plot_tick_size,
                "ytick.labelsize": self.config.plot_tick_size,
                "legend.fontsize": self.config.plot_legend_size,
                "figure.titlesize": self.config.plot_title_size + 2,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        sns.set_context("paper", font_scale=self.config.plot_font_scale)

    def _get_save_path(self, base_name: str, save_path: Path | None = None) -> Path:
        """Get the save path with the configured format."""
        if save_path:
            return save_path.with_suffix(f".{self.config.plot_format}")
        return self.results_plots_dir / f"{base_name}.{self.config.plot_format}"

    def plot_classifier_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "Accuracy",
        top_n: int = 15,
        save_path: Path | None = None,
    ) -> None:
        """Plot comparison of classifiers by a metric."""
        df = results_df.nlargest(top_n, metric)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = sns.color_palette("viridis", len(df))
        bars = ax.barh(range(len(df)), df[metric], color=colors)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Classifier"])
        ax.set_xlabel(metric)
        ax.set_title(f"Top {top_n} Classifiers by {metric}")
        ax.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontsize=self.config.plot_tick_size)

        plt.tight_layout()
        save_path = self._get_save_path(f"classifier_comparison_{metric.lower()}", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_multi_metric_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: list[str] | None = None,
        top_n: int = 10,
        save_path: Path | None = None,
    ) -> None:
        """Plot comparison of classifiers across multiple metrics."""
        if metrics is None:
            metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "MCC", "CV_Mean"]

        available_metrics = [m for m in metrics if m in results_df.columns]
        df = results_df.nlargest(top_n, "Accuracy")

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(df))
        width = 0.12

        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, df[metric], width, label=metric, color=self.colors[i])

        ax.set_xlabel("Classifier")
        ax.set_ylabel("Score")
        ax.set_title(f"Top {top_n} Classifiers - Multi-Metric Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Classifier"], rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        save_path = self._get_save_path("multi_metric_comparison", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_cv_scores(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Path | None = None,
    ) -> None:
        """Plot cross-validation scores with error bars."""
        df = results_df.nlargest(top_n, "CV_Mean")

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = range(len(df))
        ax.barh(
            y_pos,
            df["CV_Mean"],
            xerr=df["CV_Std"],
            color=self.colors[: len(df)],
            capsize=3,
            alpha=0.8,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Classifier"])
        ax.set_xlabel("Cross-Validation Score")
        ax.set_title(f"Top {top_n} Classifiers - CV Scores with Standard Deviation")
        ax.invert_yaxis()

        for i, (mean, std) in enumerate(zip(df["CV_Mean"], df["CV_Std"])):
            ax.text(
                mean + std + 0.01,
                i,
                f"{mean:.3f}±{std:.3f}",
                va="center",
                fontsize=self.config.plot_tick_size - 1,
            )

        plt.tight_layout()
        save_path = self._get_save_path("cv_scores", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_training_times(
        self,
        results_df: pd.DataFrame,
        save_path: Path | None = None,
    ) -> None:
        """Plot training times for all classifiers."""
        df = results_df.sort_values("Train_Time_s", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = ["green" if t < 1 else "orange" if t < 10 else "red" for t in df["Train_Time_s"]]
        ax.barh(range(len(df)), df["Train_Time_s"], color=colors)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Classifier"])
        ax.set_xlabel("Training Time (seconds)")
        ax.set_title("Classifier Training Times")
        ax.set_xscale("log")

        plt.tight_layout()
        save_path = self._get_save_path("training_times", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str],
        classifier_name: str,
        save_path: Path | None = None,
    ) -> None:
        """Plot confusion matrix for a classifier."""
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
        ax.set_title(f"Confusion Matrix - {classifier_name}")

        plt.tight_layout()
        save_path = self._get_save_path(f"confusion_matrix_{classifier_name}", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_all_confusion_matrices(
        self,
        confusion_matrices: dict[str, np.ndarray],
        class_names: list[str],
        top_n: int = 6,
    ) -> None:
        """Plot confusion matrices for top classifiers (saved as separate files)."""
        n_classifiers = min(top_n, len(confusion_matrices))

        for i, (name, cm) in enumerate(list(confusion_matrices.items())[:n_classifiers]):
            fig, ax = plt.subplots(figsize=(8, 6))
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
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix - {name}")
            plt.tight_layout()
            save_path = self._get_save_path(f"confusion_matrix_{name}")
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
            plt.close()

    def plot_feature_importance(
        self,
        feature_importances: dict[str, np.ndarray],
        feature_names: list[str],
        top_n_classifiers: int = 6,
    ) -> None:
        """Plot feature importance for multiple classifiers (saved as separate files)."""
        n_classifiers = min(top_n_classifiers, len(feature_importances))

        for i, (name, importances) in enumerate(list(feature_importances.items())[:n_classifiers]):
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(importances)
            ax.barh(
                range(len(importances)),
                importances[sorted_idx],
                color=self.colors[i % len(self.colors)],
            )
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels([feature_names[j] for j in sorted_idx])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance - {name}")
            plt.tight_layout()
            save_path = self._get_save_path(f"feature_importance_{name}")
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
            plt.close()

    def plot_mean_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        save_path: Path | None = None,
    ) -> None:
        """Plot mean feature importance across all classifiers."""
        fig, ax = plt.subplots(figsize=(10, 6))

        df = feature_importance_df.sort_values("Mean_Importance", ascending=True)
        print(df)
        ax.barh(
            range(len(df)),
            df["Mean_Importance"],
            xerr=df["Std_Importance"],
            color="steelblue",
            capsize=3,
            alpha=0.8,
        )
        
        for i, (mean, std) in enumerate(zip(df["Mean_Importance"], df["Std_Importance"])):
            ax.text(
            mean + std + 0.005,
            i,
            f"{mean:.3f}±{std:.3f}",
            va="center",
            fontsize=self.config.plot_tick_size - 1,
            )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Feature"])
        ax.set_xlabel("Mean Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Average Feature Importance Across Classifiers")

        plt.tight_layout()
        save_path = self._get_save_path("mean_feature_importance", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_accuracy_vs_time(
        self,
        results_df: pd.DataFrame,
        save_path: Path | None = None,
    ) -> None:
        """Plot accuracy vs training time trade-off."""
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            results_df["Train_Time_s"],
            results_df["Accuracy"],
            c=results_df["CV_Mean"],
            cmap="viridis",
            s=100,
            alpha=0.7,
        )

        for _, row in results_df.iterrows():
            ax.annotate(
                row["Classifier"],
                (row["Train_Time_s"], row["Accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=self.config.plot_tick_size - 2,
                alpha=0.8,
            )

        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Training Time Trade-off")
        ax.set_xscale("log")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("CV Mean Score")

        plt.tight_layout()
        save_path = self._get_save_path("accuracy_vs_time", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_metric_heatmap(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Path | None = None,
    ) -> None:
        """Plot heatmap of metrics for top classifiers."""
        all_metrics = [
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "MCC",
            "CV_Mean",
        ]
        metrics = [m for m in all_metrics if m in results_df.columns]
        df = results_df.nlargest(top_n, "Accuracy")[["Classifier"] + metrics]

        fig, ax = plt.subplots(figsize=(12, 10))

        data = df[metrics].values

        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=metrics,
            yticklabels=df["Classifier"],
            ax=ax,
            cbar_kws={"label": "Score"},
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"Performance Heatmap - Top {top_n} Classifiers")

        plt.tight_layout()
        save_path = self._get_save_path("metric_heatmap", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def plot_class_performance(
        self,
        classification_reports: dict[str, dict[str, Any]],
        class_names: list[str],
        top_n_classifiers: int = 5,
    ) -> None:
        """Plot per-class performance for top classifiers (saved as separate files)."""
        metrics = ["precision", "recall", "f1-score"]
        classifiers = list(classification_reports.keys())[:top_n_classifiers]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = []
            for clf_name in classifiers:
                report = classification_reports[clf_name]
                values = [report.get(cls, {}).get(metric, 0) for cls in class_names]
                data.append(values)

            data = np.array(data)

            x = np.arange(len(class_names))
            width = 0.8 / len(classifiers)

            for i, clf_name in enumerate(classifiers):
                offset = (i - len(classifiers) / 2 + 0.5) * width
                ax.bar(x + offset, data[i], width, label=clf_name, color=self.colors[i])

            ax.set_xlabel("Class")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Per-Class {metric.capitalize()} Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend(loc="lower right", fontsize=self.config.plot_legend_size)
            ax.set_ylim(0, 1.1)

            plt.tight_layout()
            save_path = self._get_save_path(f"class_performance_{metric}")
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
            plt.close()

    def create_all_plots(
        self,
        results_df: pd.DataFrame,
        confusion_matrices: dict[str, np.ndarray],
        feature_importances: dict[str, np.ndarray],
        feature_importance_summary: pd.DataFrame,
        class_names: list[str],
        classification_reports: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Generate all visualization plots."""
        self.plot_classifier_comparison(results_df, "Accuracy")
        self.plot_classifier_comparison(results_df, "F1_Score")
        if "MCC" in results_df.columns:
            self.plot_classifier_comparison(results_df, "MCC")
        self.plot_multi_metric_comparison(results_df)
        self.plot_cv_scores(results_df)
        self.plot_training_times(results_df)
        self.plot_accuracy_vs_time(results_df)
        self.plot_metric_vs_time(results_df, "F1_Score")
        if "MCC" in results_df.columns:
            self.plot_metric_vs_time(results_df, "MCC")
        self.plot_metric_heatmap(results_df)

        if confusion_matrices:
            self.plot_all_confusion_matrices(confusion_matrices, class_names)

        if feature_importances:
            feature_names = self.config.feature_columns
            self.plot_feature_importance(feature_importances, feature_names)

        if not feature_importance_summary.empty:
            self.plot_mean_feature_importance(feature_importance_summary)

        if classification_reports:
            self.plot_class_performance(classification_reports, class_names)

    def plot_metric_vs_time(
        self,
        results_df: pd.DataFrame,
        metric: str = "F1_Score",
        save_path: Path | None = None,
    ) -> None:
        """Plot metric vs training time trade-off."""
        if metric not in results_df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            results_df["Train_Time_s"],
            results_df[metric],
            c=results_df["CV_Mean"],
            cmap="viridis",
            s=100,
            alpha=0.7,
        )

        for _, row in results_df.iterrows():
            ax.annotate(
                row["Classifier"],
                (row["Train_Time_s"], row[metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=self.config.plot_tick_size - 2,
                alpha=0.8,
            )

        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} vs Training Time Trade-off")
        ax.set_xscale("log")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("CV Mean Score")

        plt.tight_layout()
        save_path = self._get_save_path(f"{metric.lower()}_vs_time", save_path)
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()
