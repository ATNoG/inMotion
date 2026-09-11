#!/usr/bin/env python3
"""Regenerate plots from saved CSV results.

This script loads classification results from CSV files and regenerates
all visualization plots without needing to retrain models.

Usage:
    uv run inmotion regenerate-plots --results-dir results_3
    inmotion regenerate-plots \
        --csv results_3/classification_results.csv \
        --plots-dir plots_regenerated
    inmotion regenerate-plots \
        --results-dir results_3 --eda --data data/processed/dataset.csv
"""

from pathlib import Path

from inmotion.classification import (
    Config,
    DataLoader,
    ExploratoryDataAnalysis,
    Visualizer,
    load_results_from_csv,
    set_random_seeds,
)
from inmotion.viz.regenerate_config import RunConfig


def run(args: RunConfig) -> int:
    """Execute the pipeline.

    ``args`` keeps the field names the argparse namespace used, so the body
    is unchanged; the CLI builds it from typed options.
    """

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
