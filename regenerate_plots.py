#!/usr/bin/env python3
"""Regenerate plots from saved CSV results.

This script loads classification results from CSV files and regenerates
all visualization plots without needing to retrain models.

Usage:
    uv run python regenerate_plots.py --results-dir results_3
    uv run python regenerate_plots.py --csv results_3/classification_results.csv --plots-dir plots_regenerated
    uv run python regenerate_plots.py --results-dir results_3 --eda --data dataset.csv
"""

import argparse
from pathlib import Path

from ml_classification import (
    Config,
    DataLoader,
    ExploratoryDataAnalysis,
    Visualizer,
    load_results_from_csv,
    set_random_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plots from saved classification results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to results directory containing CSV files (e.g., results_3)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to classification_results.csv (alternative to --results-dir)",
    )
    parser.add_argument(
        "--detailed-csv",
        type=str,
        help="Path to detailed_classifier_results.csv (optional, will be inferred from --csv)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        help="Output directory for plots (default: plots_{suffix} based on results dir)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top classifiers to show in plots (default: 15)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Plot output format (default: pdf)",
    )
    # EDA options
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Also generate EDA plots (requires --data)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.csv",
        help="Path to the dataset CSV file for EDA plots (default: dataset.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility in EDA (default: 42)",
    )
    parser.add_argument(
        "--skip-classification-plots",
        action="store_true",
        help="Skip classification result plots (only generate EDA plots if --eda is set)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine paths
    if args.results_dir:
        results_dir = Path(args.results_dir)
        results_csv = results_dir / "classification_results.csv"
        detailed_csv = results_dir / "detailed_classifier_results.csv"
        # Infer plots directory from results directory name
        suffix = results_dir.name.replace("results", "")
        plots_dir = Path(f"plots{suffix}") if not args.plots_dir else Path(args.plots_dir)
    elif args.csv:
        results_csv = Path(args.csv)
        detailed_csv = Path(args.detailed_csv) if args.detailed_csv else None
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path("plots_regenerated")
    elif args.eda:
        # EDA-only mode, no results needed
        results_csv = None
        detailed_csv = None
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path("plots_eda")
    else:
        print("Error: Either --results-dir, --csv, or --eda must be provided")
        return 1

    if results_csv and not results_csv.exists() and not args.skip_classification_plots:
        print(f"Error: Results file not found: {results_csv}")
        return 1

    print("=" * 70)
    print("Regenerating Plots from Saved Results")
    print("=" * 70)
    if results_csv:
        print(f"Results CSV: {results_csv}")
    print(f"Output directory: {plots_dir}")
    print(f"Format: {args.format}")
    if args.eda:
        print(f"EDA enabled: Yes (dataset: {args.data})")
    print("=" * 70)
    print()

    # Create config with appropriate settings
    config = Config(
        data_path=Path(args.data),
        results_dir=results_csv.parent if results_csv else plots_dir,
        plots_dir=plots_dir,
        plot_format=args.format,
        random_seed=args.seed,
    )

    # Generate EDA plots if requested
    if args.eda:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Dataset file not found: {data_path}")
            return 1

        print("Generating EDA plots...")
        set_random_seeds(args.seed)

        # Load and preprocess data
        loader = DataLoader(config)
        df = loader.load_data()
        X, y, class_names_eda = loader.preprocess(df, scale_features=True, encode_labels=True)

        print(f"  Loaded {len(df)} samples, {X.shape[1]} features")
        print(f"  Classes: {class_names_eda}")

        # Run EDA
        eda = ExploratoryDataAnalysis(config)
        eda.run_full_analysis(df, X, y, class_names_eda)
        print(eda.generate_report())
        print(f"  EDA plots saved to: {plots_dir / 'eda'}")
        print()

    # Generate classification result plots
    if not args.skip_classification_plots and results_csv and results_csv.exists():
        print("Loading classification results from CSV files...")
        (
            results_df,
            confusion_matrices,
            feature_importances,
            feature_importance_summary,
            class_names,
            classification_reports,
        ) = load_results_from_csv(results_csv, detailed_csv)

        print(f"  Loaded {len(results_df)} classifiers")
        print(f"  Classes: {class_names}")
        print(f"  Confusion matrices: {len(confusion_matrices)}")
        print(f"  Feature importances: {len(feature_importances)}")
        print()

        print("Generating classification plots...")
        visualizer = Visualizer(config)

        visualizer.create_all_plots(
            results_df,
            confusion_matrices,
            feature_importances,
            feature_importance_summary,
            class_names,
            classification_reports,
        )
        print(f"  Classification plots saved to: {plots_dir / 'results'}")
        print()

    print(f"All plots saved to: {plots_dir}")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())


if __name__ == "__main__":
    exit(main())
