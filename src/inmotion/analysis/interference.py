#!/usr/bin/env python3
"""Analyze cross-route interference using the `concurrent_noise_path` column.

Usage:
    inmotion analyze-interference \
        --data data/processed/dataset.csv --output-dir plots/interference
"""

from __future__ import annotations

import pandas as pd

from inmotion.analysis.interference_config import RunConfig
from inmotion.dl.interference import InterferenceAnalyzer


def run(args: RunConfig) -> int:
    """Execute the pipeline.

    ``args`` keeps the field names the argparse namespace used, so the body
    is unchanged; the CLI builds it from typed options.
    """

    if not args.data.exists():
        print(f"ERROR: Dataset not found: {args.data}")
        return

    df = pd.read_csv(args.data)

    if "concurrent_noise_path" not in df.columns:
        print("ERROR: dataset missing 'concurrent_noise_path' column")
        return

    print(f"Data: {len(df)} samples, labels: {sorted(df['label'].unique())}")
    print(f"Concurrent paths: {sorted(df['concurrent_noise_path'].dropna().unique())}")
    print(f"Noise={df['noise'].value_counts().to_dict()}")
    print()

    analyzer = InterferenceAnalyzer(args.output_dir)
    plots = analyzer.run_full_analysis(df)

    print(f"Generated {len(plots)} plots:")
    for p in plots:
        print(f"  {p.relative_to(args.output_dir)}")
