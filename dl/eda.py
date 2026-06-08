"""EDA wrapper for DL pipeline — delegates to ml_classification.eda.ExploratoryDataAnalysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DLConfig


def run_eda_for_dl(config: DLConfig, data_path: Path | None = None) -> dict:
    """Run full EDA analysis on the dataset, saving plots to config.plots_dir / 'eda'."""
    from ml_classification.config import Config
    from ml_classification.eda import ExploratoryDataAnalysis
    from sklearn.preprocessing import LabelEncoder

    data_file = data_path or config.data_path

    df = pd.read_csv(data_file)

    noise_col = "noise" if "noise" in df.columns else "noise_label"
    label_col = config.label_col if config.label_col in df.columns else "label"

    ml_config = Config(
        data_path=data_file,
        plots_dir=config.plots_dir,
        random_seed=config.seed,
        target_column=label_col,
        noise_column=noise_col,
    )

    eda_output_dir = config.plots_dir / "eda"
    eda_output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [str(i) for i in range(1, 11)]
    X = df[feature_cols].values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = list(le.classes_)

    eda = ExploratoryDataAnalysis(ml_config)
    results = eda.run_full_analysis(df, X, y, class_names)
    print(eda.generate_report())

    return results
