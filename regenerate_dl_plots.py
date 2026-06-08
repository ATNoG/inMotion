#!/usr/bin/env python3
"""Regenerate DL-specific paper plots from extended results CSV.

Usage:
    uv run python regenerate_dl_plots.py --results-csv results/dl/dl_detailed_seed42.csv --output-dir plots/dl/42
    uv run python regenerate_dl_plots.py --results-csv results/dl/dl_detailed_seed42.csv --eda --data dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dl.config import DLConfig
from dl.results import load_extended_results
from dl.visualization import DLVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate DL result plots")
    p.add_argument("--results-csv", type=Path, required=True,
                   help="Path to extended DL results CSV (e.g., dl_detailed_seed42.csv)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for plots (default: inferred from CSV path)")
    p.add_argument("--top-n", type=int, default=15,
                   help="Number of top models in ranking plots")
    p.add_argument("--eda", action="store_true",
                   help="Also generate EDA plots")
    p.add_argument("--data", type=Path, default=Path("dataset.csv"),
                   help="Dataset CSV for EDA (only with --eda)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_csv = args.results_csv
    if not results_csv.exists():
        print(f"Results CSV not found: {results_csv}")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        seed = results_csv.stem.replace("dl_detailed_seed", "").replace("dl_results_seed", "")
        try:
            seed_int = int(seed)
        except ValueError:
            seed_int = args.seed
        output_dir = Path("plots") / "dl" / str(seed_int)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DLConfig(
        data_path=args.data,
        seed=args.seed,
    )
    config.plots_dir = output_dir

    # Load results
    results_df, confusion_matrices, class_names, per_class_reports = load_extended_results(results_csv)
    print(f"Loaded {len(results_df)} DL model results")
    print(f"  Class names: {class_names}")
    print(f"  Confusion matrices: {len(confusion_matrices)}")
    print(f"  Per-class reports: {len(per_class_reports)}")

    # Generate plots
    reader_output = output_dir / "results"
    reader_output.mkdir(parents=True, exist_ok=True)
    viz = DLVisualizer(config, output_dir=reader_output)
    viz.create_all_plots(
        results_df, confusion_matrices, class_names, per_class_reports,
        top_n_models=args.top_n,
    )
    print(f"Plots saved → {reader_output}")

    # ── EDA (optional) ───────────────────────────────────────────────────────
    if args.eda:
        print("\nRunning EDA...")
        from dl.eda import run_eda_for_dl
        run_eda_for_dl(config)


if __name__ == "__main__":
    main()
