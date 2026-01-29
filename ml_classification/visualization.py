"""Visualization module for ML classification results."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .config import Config


class Visualizer:
    """Visualization utilities for ML classification results."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_plots_dir = config.plots_dir / "results"
        self.results_plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = sns.color_palette("husl", 20)
    
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
            ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontsize=9)
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / f"classifier_comparison_{metric.lower()}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
            metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "CV_Mean"]
        
        df = results_df.nlargest(top_n, "Accuracy")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, df[metric], width, label=metric, color=self.colors[i])
        
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Score")
        ax.set_title(f"Top {top_n} Classifiers - Multi-Metric Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Classifier"], rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "multi_metric_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
        ax.barh(y_pos, df["CV_Mean"], xerr=df["CV_Std"], color=self.colors[:len(df)],
                capsize=3, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Classifier"])
        ax.set_xlabel("Cross-Validation Score")
        ax.set_title(f"Top {top_n} Classifiers - CV Scores with Standard Deviation")
        ax.invert_yaxis()
        
        for i, (mean, std) in enumerate(zip(df["CV_Mean"], df["CV_Std"])):
            ax.text(mean + std + 0.01, i, f"{mean:.3f}±{std:.3f}", va="center", fontsize=8)
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "cv_scores.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
        save_path = save_path or self.results_plots_dir / "training_times.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={"label": "Count"}
        )
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {classifier_name}")
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / f"confusion_matrix_{classifier_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_all_confusion_matrices(
        self,
        confusion_matrices: dict[str, np.ndarray],
        class_names: list[str],
        top_n: int = 6,
        save_path: Path | None = None,
    ) -> None:
        """Plot confusion matrices for top classifiers in a grid."""
        n_classifiers = min(top_n, len(confusion_matrices))
        n_cols = 3
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_classifiers > 1 else [axes]
        
        for i, (name, cm) in enumerate(list(confusion_matrices.items())[:n_classifiers]):
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[i]
            )
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
            axes[i].set_title(name)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle("Confusion Matrices - Top Classifiers", fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "all_confusion_matrices.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_importances: dict[str, np.ndarray],
        feature_names: list[str],
        top_n_classifiers: int = 6,
        save_path: Path | None = None,
    ) -> None:
        """Plot feature importance for multiple classifiers."""
        n_classifiers = min(top_n_classifiers, len(feature_importances))
        n_cols = 3
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_classifiers > 1 else [axes]
        
        for i, (name, importances) in enumerate(list(feature_importances.items())[:n_classifiers]):
            sorted_idx = np.argsort(importances)
            axes[i].barh(range(len(importances)), importances[sorted_idx], color=self.colors[i % len(self.colors)])
            axes[i].set_yticks(range(len(importances)))
            axes[i].set_yticklabels([feature_names[j] for j in sorted_idx])
            axes[i].set_xlabel("Importance")
            axes[i].set_title(name)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle("Feature Importance by Classifier", fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_mean_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        save_path: Path | None = None,
    ) -> None:
        """Plot mean feature importance across all classifiers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = feature_importance_df.sort_values("Mean_Importance", ascending=True)
        
        ax.barh(range(len(df)), df["Mean_Importance"], color="steelblue")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Feature"])
        ax.set_xlabel("Mean Importance")
        ax.set_title("Average Feature Importance Across Classifiers")
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "mean_feature_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
                fontsize=7,
                alpha=0.8,
            )
        
        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Training Time Trade-off")
        ax.set_xscale("log")
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("CV Mean Score")
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "accuracy_vs_time.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_metric_heatmap(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Path | None = None,
    ) -> None:
        """Plot heatmap of metrics for top classifiers."""
        metrics = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "CV_Mean"]
        df = results_df.nlargest(top_n, "Accuracy")[["Classifier"] + metrics]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        data = df[metrics].values
        
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=metrics, yticklabels=df["Classifier"],
            ax=ax, cbar_kws={"label": "Score"},
            vmin=0, vmax=1,
        )
        
        ax.set_title(f"Performance Heatmap - Top {top_n} Classifiers")
        
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "metric_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_class_performance(
        self,
        classification_reports: dict[str, dict[str, Any]],
        class_names: list[str],
        top_n_classifiers: int = 5,
        save_path: Path | None = None,
    ) -> None:
        """Plot per-class performance for top classifiers."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ["precision", "recall", "f1-score"]
        
        classifiers = list(classification_reports.keys())[:top_n_classifiers]
        
        for ax, metric in zip(axes, metrics):
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
            ax.set_title(f"Per-Class {metric.capitalize()}")
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_ylim(0, 1.1)
        
        plt.suptitle("Per-Class Performance Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = save_path or self.results_plots_dir / "class_performance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
        self.plot_multi_metric_comparison(results_df)
        self.plot_cv_scores(results_df)
        self.plot_training_times(results_df)
        self.plot_accuracy_vs_time(results_df)
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
