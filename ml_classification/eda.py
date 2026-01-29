"""Exploratory Data Analysis module for comprehensive dataset analysis."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import Config


class ExploratoryDataAnalysis:
    """Comprehensive exploratory data analysis for WiFi fingerprinting dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.analysis_results: dict[str, Any] = {}

    def basic_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute basic statistics of the dataset."""
        stats = {
            "n_samples": len(df),
            "n_features": len(self.config.feature_columns),
            "classes": df[self.config.target_column].unique().tolist(),
            "n_classes": df[self.config.target_column].nunique(),
            "class_distribution": df[self.config.target_column].value_counts().to_dict(),
            "n_unique_macs": df[self.config.mac_column].nunique(),
            "noise_distribution": df[self.config.noise_column].value_counts().to_dict(),
            "feature_stats": df[self.config.feature_columns].describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }
        self.analysis_results["basic_statistics"] = stats
        return stats

    def feature_analysis(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature distributions and correlations."""
        features = df[self.config.feature_columns]

        analysis = {
            "mean_rssi_per_position": features.mean(axis=1).values,
            "std_rssi_per_position": features.std(axis=1).values,
            "range_rssi_per_position": (features.max(axis=1) - features.min(axis=1)).values,
            "correlation_matrix": features.corr().values,
            "feature_means": features.mean().to_dict(),
            "feature_stds": features.std().to_dict(),
            "temporal_gradient": np.gradient(features.values, axis=1),
        }
        self.analysis_results["feature_analysis"] = analysis
        return analysis

    def class_separability_analysis(
        self, X: np.ndarray, y: np.ndarray, class_names: list[str]
    ) -> dict[str, Any]:
        """Analyze class separability using various metrics."""
        unique_classes = np.unique(y)
        class_means = {}
        class_stds = {}

        for cls in unique_classes:
            mask = y == cls
            class_means[class_names[cls]] = X[mask].mean(axis=0)
            class_stds[class_names[cls]] = X[mask].std(axis=0)

        pca = PCA(n_components=2, random_state=self.config.random_seed)
        X_pca = pca.fit_transform(X)

        analysis = {
            "class_means": class_means,
            "class_stds": class_stds,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "pca_components": X_pca,
        }
        self.analysis_results["class_separability"] = analysis
        return analysis

    def plot_class_distribution(self, df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        class_counts = df[self.config.target_column].value_counts()
        colors = sns.color_palette("husl", len(class_counts))
        axes[0].bar(class_counts.index, class_counts.values, color=colors)
        axes[0].set_xlabel("Class (Route)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Class Distribution")
        for i, (cls, count) in enumerate(class_counts.items()):
            axes[0].annotate(str(count), xy=(i, count), ha="center", va="bottom")

        noise_counts = df.groupby([self.config.target_column, self.config.noise_column]).size()
        noise_df = noise_counts.unstack(fill_value=0)
        noise_df.plot(kind="bar", ax=axes[1], color=["#ff7f0e", "#2ca02c"])
        axes[1].set_xlabel("Class (Route)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Class Distribution by Noise Condition")
        axes[1].legend(["No Noise", "With Noise"], title="Noise")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_rssi_distributions(self, df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Plot RSSI value distributions across time steps and classes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        features = df[self.config.feature_columns]
        axes[0, 0].boxplot([features[col].values for col in self.config.feature_columns])
        axes[0, 0].set_xticklabels(self.config.feature_columns)
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("RSSI (dBm)")
        axes[0, 0].set_title("RSSI Distribution per Time Step")

        for cls in df[self.config.target_column].unique():
            mask = df[self.config.target_column] == cls
            mean_rssi = df.loc[mask, self.config.feature_columns].mean()
            axes[0, 1].plot(range(1, 11), mean_rssi.values, marker="o", label=cls, linewidth=2)
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Mean RSSI (dBm)")
        axes[0, 1].set_title("Mean RSSI Trajectory per Class")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        for i, cls in enumerate(df[self.config.target_column].unique()):
            mask = df[self.config.target_column] == cls
            class_features = df.loc[mask, self.config.feature_columns].values
            axes[1, 0].violinplot(
                class_features, positions=np.arange(10) + i * 12, showmeans=True, showextrema=True
            )
        axes[1, 0].set_xlabel("Time Steps (grouped by class)")
        axes[1, 0].set_ylabel("RSSI (dBm)")
        axes[1, 0].set_title("RSSI Distribution per Class (Violin Plot)")

        all_rssi = df[self.config.feature_columns].values.flatten()
        axes[1, 1].hist(all_rssi, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
        axes[1, 1].set_xlabel("RSSI (dBm)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Overall RSSI Distribution")
        axes[1, 1].axvline(
            np.mean(all_rssi), color="red", linestyle="--", label=f"Mean: {np.mean(all_rssi):.1f}"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Plot correlation heatmap of features."""
        fig, ax = plt.subplots(figsize=(10, 8))

        corr_matrix = df[self.config.feature_columns].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Correlation"},
        )
        ax.set_title("Feature Correlation Heatmap (RSSI Time Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Time Step")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_pca_visualization(
        self, X: np.ndarray, y: np.ndarray, class_names: list[str], save_path: Path | None = None
    ) -> None:
        """Plot PCA visualization of the data."""
        pca = PCA(n_components=2, random_state=self.config.random_seed)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = sns.color_palette("husl", len(np.unique(y)))
        for i, cls in enumerate(np.unique(y)):
            mask = y == cls
            axes[0].scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[colors[i]],
                label=class_names[cls],
                alpha=0.7,
                s=50,
            )
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        axes[0].set_title("PCA Visualization")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        pca_full = PCA(random_state=self.config.random_seed)
        pca_full.fit(X)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        axes[1].bar(
            range(1, len(cumsum) + 1),
            pca_full.explained_variance_ratio_,
            alpha=0.7,
            label="Individual",
        )
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, "r-o", label="Cumulative")
        axes[1].axhline(y=0.95, color="g", linestyle="--", label="95% threshold")
        axes[1].set_xlabel("Principal Component")
        axes[1].set_ylabel("Explained Variance Ratio")
        axes[1].set_title("PCA Explained Variance")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_tsne_visualization(
        self, X: np.ndarray, y: np.ndarray, class_names: list[str], save_path: Path | None = None
    ) -> None:
        """Plot t-SNE visualization of the data."""
        perplexity = min(30, len(X) - 1)
        tsne = TSNE(n_components=2, random_state=self.config.random_seed, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("husl", len(np.unique(y)))

        for i, cls in enumerate(np.unique(y)):
            mask = y == cls
            ax.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[colors[i]],
                label=class_names[cls],
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE Visualization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_class_trajectories(self, df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Plot RSSI trajectories for each class with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        classes = df[self.config.target_column].unique()
        time_steps = list(range(1, 11))

        for idx, cls in enumerate(classes):
            ax = axes[idx // 2, idx % 2]
            mask = df[self.config.target_column] == cls
            class_data = df.loc[mask, self.config.feature_columns].values

            mean = class_data.mean(axis=0)
            std = class_data.std(axis=0)

            ax.plot(time_steps, mean, "b-", linewidth=2, label="Mean")
            ax.fill_between(
                time_steps, mean - std, mean + std, alpha=0.3, color="blue", label="±1 Std"
            )
            ax.fill_between(
                time_steps, mean - 2 * std, mean + 2 * std, alpha=0.15, color="blue", label="±2 Std"
            )

            for i in range(min(10, len(class_data))):
                ax.plot(time_steps, class_data[i], "gray", alpha=0.2, linewidth=0.5)

            ax.set_xlabel("Time Step")
            ax.set_ylabel("RSSI (dBm)")
            ax.set_title(f"Class: {cls} (n={len(class_data)})")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(time_steps)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def run_full_analysis(
        self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, class_names: list[str]
    ) -> dict[str, Any]:
        """Run complete EDA and save all plots."""
        plots_dir = self.config.plots_dir / "eda"
        plots_dir.mkdir(parents=True, exist_ok=True)

        self.basic_statistics(df)
        self.feature_analysis(df)
        self.class_separability_analysis(X, y, class_names)

        self.plot_class_distribution(df, plots_dir / "class_distribution.png")
        self.plot_rssi_distributions(df, plots_dir / "rssi_distributions.png")
        self.plot_correlation_heatmap(df, plots_dir / "correlation_heatmap.png")
        self.plot_pca_visualization(X, y, class_names, plots_dir / "pca_visualization.png")
        self.plot_tsne_visualization(X, y, class_names, plots_dir / "tsne_visualization.png")
        self.plot_class_trajectories(df, plots_dir / "class_trajectories.png")

        return self.analysis_results

    def generate_report(self) -> str:
        """Generate a text report of the analysis."""
        if not self.analysis_results:
            return "No analysis has been run yet."

        stats = self.analysis_results.get("basic_statistics", {})

        report = [
            "=" * 60,
            "EXPLORATORY DATA ANALYSIS REPORT",
            "=" * 60,
            "",
            "Dataset Overview:",
            f"  - Total samples: {stats.get('n_samples', 'N/A')}",
            f"  - Number of features: {stats.get('n_features', 'N/A')}",
            f"  - Number of classes: {stats.get('n_classes', 'N/A')}",
            f"  - Classes: {stats.get('classes', 'N/A')}",
            f"  - Unique MAC addresses: {stats.get('n_unique_macs', 'N/A')}",
            "",
            "Class Distribution:",
        ]

        for cls, count in stats.get("class_distribution", {}).items():
            pct = count / stats.get("n_samples", 1) * 100
            report.append(f"  - {cls}: {count} ({pct:.1f}%)")

        report.extend(
            [
                "",
                "Noise Condition Distribution:",
            ]
        )
        for noise, count in stats.get("noise_distribution", {}).items():
            report.append(f"  - {noise}: {count}")

        report.extend(
            [
                "",
                "=" * 60,
            ]
        )

        return "\n".join(report)
