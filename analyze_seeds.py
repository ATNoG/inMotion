#!/usr/bin/env python3
"""Analyze and combine results from multi-seed experiments.

This script reads results from experiments with different seeds,
computes statistics, and generates comparison plots.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEEDS = [3, 5, 42]


def setup_plot_style() -> None:
    """Setup publication-ready plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    sns.set_context("paper", font_scale=1.2)
    plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-seed experiment results")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS, help="Seeds used in experiments"
    )
    parser.add_argument(
        "--output-dir", type=str, default="seeds_results_combined", 
        help="Output directory for combined results"
    )
    return parser.parse_args()


def load_seed_results(seeds: list[int]) -> dict[int, pd.DataFrame]:
    """Load classification results for each seed."""
    results = {}
    for seed in seeds:
        results_path = Path(f"results_{seed}") / "classification_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            df["Seed"] = seed
            results[seed] = df
            print(f"  Loaded results for seed {seed}: {len(df)} classifiers")
        else:
            print(f"  ⚠️ Results not found for seed {seed}")
    return results


def compute_statistics(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute mean and std statistics across seeds."""
    combined = pd.concat(results.values(), ignore_index=True)
    
    metrics = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "MCC", "CV_Mean"]
    available_metrics = [m for m in metrics if m in combined.columns]
    
    stats = combined.groupby("Classifier")[available_metrics].agg(["mean", "std"]).reset_index()
    stats.columns = ["Classifier"] + [f"{m}_{s}" for m in available_metrics for s in ["Mean", "Std"]]
    
    stats = stats.sort_values("Accuracy_Mean", ascending=False).reset_index(drop=True)
    return stats


def plot_metric_comparison_across_seeds(
    results: dict[int, pd.DataFrame],
    metric: str,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot metric comparison for each seed."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    seeds = sorted(results.keys())
    all_classifiers = set()
    for df in results.values():
        all_classifiers.update(df.nlargest(top_n, metric)["Classifier"].tolist())
    
    combined = pd.concat(results.values(), ignore_index=True)
    classifier_means = combined.groupby("Classifier")[metric].mean()
    top_classifiers = classifier_means.nlargest(top_n).index.tolist()
    
    x = np.arange(len(top_classifiers))
    width = 0.25
    colors = sns.color_palette("husl", len(seeds))
    
    for i, seed in enumerate(seeds):
        df = results[seed]
        values = []
        for clf in top_classifiers:
            if clf in df["Classifier"].values:
                val = df[df["Classifier"] == clf][metric].values[0]
            else:
                val = 0
            values.append(val)
        
        offset = (i - len(seeds) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f"Seed {seed}", color=colors[i])
    
    ax.set_xlabel("Classifier")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Top {top_n} Classifiers - {metric.replace('_', ' ')} Comparison Across Seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(top_classifiers, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric.lower()}_seed_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_variability(
    results: dict[int, pd.DataFrame],
    metric: str,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot metric variability across seeds with error bars."""
    combined = pd.concat(results.values(), ignore_index=True)
    
    stats = combined.groupby("Classifier")[metric].agg(["mean", "std"]).reset_index()
    stats.columns = ["Classifier", "Mean", "Std"]
    stats = stats.nlargest(top_n, "Mean")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = range(len(stats))
    colors = sns.color_palette("viridis", len(stats))
    
    ax.barh(y_pos, stats["Mean"], xerr=stats["Std"], color=colors, capsize=3, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats["Classifier"])
    ax.set_xlabel(f"{metric.replace('_', ' ')} (Mean ± Std across seeds)")
    ax.set_title(f"Top {top_n} Classifiers - {metric.replace('_', ' ')} Variability")
    ax.invert_yaxis()
    
    for i, (mean, std) in enumerate(zip(stats["Mean"], stats["Std"])):
        ax.text(mean + std + 0.01, i, f"{mean:.3f}±{std:.3f}", va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric.lower()}_variability.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_seed_boxplot(
    results: dict[int, pd.DataFrame],
    metric: str,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Create boxplot showing metric distribution across seeds."""
    combined = pd.concat(results.values(), ignore_index=True)
    
    classifier_means = combined.groupby("Classifier")[metric].mean()
    top_classifiers = classifier_means.nlargest(top_n).index.tolist()
    
    filtered = combined[combined["Classifier"].isin(top_classifiers)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.boxplot(
        data=filtered, 
        x="Classifier", 
        y=metric, 
        hue="Classifier",
        order=top_classifiers,
        palette="viridis",
        legend=False,
        ax=ax
    )
    sns.stripplot(
        data=filtered, 
        x="Classifier", 
        y=metric, 
        order=top_classifiers,
        color="black",
        size=8,
        alpha=0.7,
        ax=ax
    )
    
    ax.set_xlabel("Classifier")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Top {top_n} Classifiers - {metric.replace('_', ' ')} Distribution Across Seeds")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric.lower()}_boxplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_multi_metric_heatmap(
    stats: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot heatmap of mean metrics across seeds."""
    mean_cols = [col for col in stats.columns if col.endswith("_Mean") and col != "CV_Mean_Mean"]
    if not mean_cols:
        return
    
    df = stats.head(top_n).copy()
    
    metric_data = df[mean_cols].values
    metric_labels = [col.replace("_Mean", "") for col in mean_cols]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        metric_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=metric_labels,
        yticklabels=df["Classifier"],
        ax=ax,
        cbar_kws={"label": "Mean Score"},
        vmin=0,
        vmax=1,
    )
    
    ax.set_title(f"Performance Heatmap - Top {top_n} Classifiers (Mean Across Seeds)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_metric_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_seed_stability_ranking(
    results: dict[int, pd.DataFrame],
    output_dir: Path,
    metric: str = "Accuracy",
    top_n: int = 15,
) -> None:
    """Plot how classifier rankings change across seeds."""
    combined = pd.concat(results.values(), ignore_index=True)
    classifier_means = combined.groupby("Classifier")[metric].mean()
    top_classifiers = classifier_means.nlargest(top_n).index.tolist()
    
    rankings = {}
    for seed, df in results.items():
        df_sorted = df.sort_values(metric, ascending=False).reset_index(drop=True)
        df_sorted["Rank"] = df_sorted.index + 1
        rankings[seed] = df_sorted.set_index("Classifier")["Rank"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    seeds = sorted(results.keys())
    colors = sns.color_palette("husl", len(top_classifiers))
    
    for i, clf in enumerate(top_classifiers):
        ranks = [rankings[seed].get(clf, np.nan) for seed in seeds]
        ax.plot(seeds, ranks, marker="o", label=clf, color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel("Seed")
    ax.set_ylabel("Rank")
    ax.set_title(f"Classifier Ranking Stability Across Seeds (by {metric})")
    ax.set_xticks(seeds)
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric.lower()}_ranking_stability.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(
    stats: pd.DataFrame,
    results: dict[int, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a comprehensive summary report."""
    report_lines = [
        "=" * 80,
        "MULTI-SEED EXPERIMENT ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Seeds analyzed: {sorted(results.keys())}",
        f"Total classifiers: {len(stats)}",
        "",
        "-" * 80,
        "TOP 10 CLASSIFIERS (BY MEAN ACCURACY ACROSS SEEDS)",
        "-" * 80,
    ]
    
    for i, row in stats.head(10).iterrows():
        acc_mean = row.get("Accuracy_Mean", 0)
        acc_std = row.get("Accuracy_Std", 0)
        f1_mean = row.get("F1_Score_Mean", 0)
        f1_std = row.get("F1_Score_Std", 0)
        mcc_mean = row.get("MCC_Mean", 0)
        mcc_std = row.get("MCC_Std", 0)
        
        report_lines.append(
            f"{i + 1}. {row['Classifier']}: "
            f"Acc={acc_mean:.4f}±{acc_std:.4f}, "
            f"F1={f1_mean:.4f}±{f1_std:.4f}, "
            f"MCC={mcc_mean:.4f}±{mcc_std:.4f}"
        )
    
    best = stats.iloc[0]
    report_lines.extend([
        "",
        "-" * 80,
        f"BEST CLASSIFIER: {best['Classifier']}",
        "-" * 80,
        "",
    ])
    
    mean_cols = [col for col in stats.columns if col.endswith("_Mean")]
    for col in mean_cols:
        metric_name = col.replace("_Mean", "")
        std_col = f"{metric_name}_Std"
        mean_val = best.get(col, 0)
        std_val = best.get(std_col, 0) if std_col in best else 0
        report_lines.append(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    combined = pd.concat(results.values(), ignore_index=True)
    std_metrics = combined.groupby("Classifier")[["Accuracy", "F1_Score", "MCC"]].std().mean()
    
    report_lines.extend([
        "",
        "-" * 80,
        "AVERAGE VARIABILITY ACROSS SEEDS",
        "-" * 80,
    ])
    
    for metric, std in std_metrics.items():
        report_lines.append(f"  {metric} Std: {std:.4f}")
    
    report_lines.extend([
        "",
        "-" * 80,
        "MOST STABLE CLASSIFIERS (LOWEST ACCURACY VARIANCE)",
        "-" * 80,
    ])
    
    if "Accuracy_Std" in stats.columns:
        stable = stats.nsmallest(5, "Accuracy_Std")
        for _, row in stable.iterrows():
            report_lines.append(
                f"  {row['Classifier']}: Acc={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}"
            )
    
    report_lines.extend(["", "=" * 80])
    
    report_text = "\n".join(report_lines)
    
    report_path = output_dir / "seeds_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def main() -> int:
    args = parse_args()
    setup_plot_style()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-SEED EXPERIMENT ANALYSIS")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("Loading results...")
    results = load_seed_results(args.seeds)
    
    if not results:
        print("⚠️ No results found!")
        return 1
    
    print("\nComputing statistics across seeds...")
    stats = compute_statistics(results)
    
    stats_path = output_dir / "combined_statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"  Statistics saved to: {stats_path}")
    
    combined = pd.concat(results.values(), ignore_index=True)
    combined_path = output_dir / "all_seeds_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Combined results saved to: {combined_path}")
    
    print("\nGenerating comparison plots...")
    
    metrics = ["Accuracy", "F1_Score", "MCC"]
    available_metrics = [m for m in metrics if m in combined.columns]
    
    for metric in available_metrics:
        print(f"  Plotting {metric} comparisons...")
        plot_metric_comparison_across_seeds(results, metric, output_dir)
        plot_metric_variability(results, metric, output_dir)
        plot_seed_boxplot(results, metric, output_dir)
        plot_seed_stability_ranking(results, output_dir, metric)
    
    print("  Plotting multi-metric heatmap...")
    plot_multi_metric_heatmap(stats, output_dir)
    
    print("\nGenerating summary report...")
    generate_summary_report(stats, results, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}/")
    print("  - combined_statistics.csv")
    print("  - all_seeds_combined.csv")
    print("  - seeds_analysis_report.txt")
    for metric in available_metrics:
        print(f"  - {metric.lower()}_seed_comparison.pdf")
        print(f"  - {metric.lower()}_variability.pdf")
        print(f"  - {metric.lower()}_boxplot.pdf")
        print(f"  - {metric.lower()}_ranking_stability.pdf")
    print("  - multi_metric_heatmap.pdf")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
