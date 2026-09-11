#!/usr/bin/env python3
"""Main script for WiFi Fingerprinting Classification Analysis.

This script runs a comprehensive analysis of various ML classifiers for
predicting location classes based on WiFi RSSI fingerprints.
"""

import os
import time
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from inmotion.pipelines.classification_config import RunConfig

# Suppress XGBoost verbosity before importing it
os.environ["XGBOOST_VERBOSITY"] = "0"

# Suppress sklearn and other warnings
warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", message=".*Falling back to prediction.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning)

# The imports below are intentionally not at the top of the file:
# XGBOOST_VERBOSITY and the warning filters must be in place before these
# # libraries are imported, so the imports cannot move above the setup.
from inmotion.classification import (  # noqa: E402
    Config,
    DataLoader,
    ExploratoryDataAnalysis,
    OptunaOptimizer,
    TrainingPipeline,
    Visualizer,
    set_random_seeds,
)


def save_detailed_classifier_reports(
    pipeline: TrainingPipeline, config: Config, class_names: list[str]
) -> Path:
    """Save detailed per-classifier reports to a CSV file."""
    import json

    detailed_data = []
    for result in pipeline.results:
        row = {
            "Classifier": result.name,
            "Accuracy": result.accuracy,
            "Balanced_Accuracy": result.balanced_accuracy,
            "Precision": result.precision,
            "Recall": result.recall,
            "F1_Score": result.f1_score,
            "MCC": result.mcc,
            "CV_Mean": result.cv_mean,
            "CV_Std": result.cv_std,
            "Train_Time_s": result.train_time,
        }

        if result.classification_report:
            for cls_name in class_names:
                if cls_name in result.classification_report:
                    cls_metrics = result.classification_report[cls_name]
                    row[f"{cls_name}_precision"] = cls_metrics.get("precision", 0)
                    row[f"{cls_name}_recall"] = cls_metrics.get("recall", 0)
                    row[f"{cls_name}_f1"] = cls_metrics.get("f1-score", 0)
                    row[f"{cls_name}_support"] = cls_metrics.get("support", 0)

        if result.confusion_matrix is not None:
            cm = result.confusion_matrix
            row["Confusion_Matrix"] = json.dumps(cm.tolist())
            for i, cls_i in enumerate(class_names):
                for j, cls_j in enumerate(class_names):
                    row[f"CM_{cls_i}_pred_{cls_j}"] = int(cm[i, j])

        if result.feature_importances is not None:
            for idx, feat in enumerate(config.feature_columns):
                if idx < len(result.feature_importances):
                    row[f"FeatImp_{feat}"] = result.feature_importances[idx]

        detailed_data.append(row)

    import pandas as pd

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df = detailed_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    detailed_path = config.results_dir / "detailed_classifier_results.csv"
    detailed_df.to_csv(detailed_path, index=False)

    return detailed_path


def run(args: RunConfig) -> int:
    """Execute the pipeline.

    ``args`` keeps the field names the argparse namespace used, so the body
    is unchanged; the CLI builds it from typed options.
    """

    suffix = "" if args.no_seed_suffix else f"_{args.seed}"

    config = Config(
        data_path=Path(args.data),
        random_seed=args.seed,
        n_cv_folds=args.cv_folds,
        n_optuna_trials=args.n_trials,
        n_jobs=args.n_jobs,
        optuna_storage=args.storage if args.optimize else "",
        results_dir=Path(f"results{suffix}{args.suffix_after_seed}"),
        plots_dir=Path(f"plots{suffix}{args.suffix_after_seed}"),
        models_dir=Path(f"models{suffix}{args.suffix_after_seed}"),
    )

    print("=" * 70)
    print("WiFi Fingerprinting Classification Analysis")
    print("=" * 70)
    print(f"Dataset: {config.data_path}")
    print(f"Random Seed: {config.random_seed}")
    print(f"CV Folds: {config.n_cv_folds}")
    print(f"Parallel Jobs: {config.n_jobs}")
    print("=" * 70)
    print()

    set_random_seeds(config.random_seed)

    print("[1/6] Loading and preprocessing data...")
    loader = DataLoader(config)
    df = loader.load_data()
    X, y, class_names = loader.preprocess(df, scale_features=True, encode_labels=True)
    X_train, X_test, y_train, y_test = loader.split_data(X, y)

    print(f"  Total samples: {len(df)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {class_names}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()

    if not args.skip_eda:
        print("[2/6] Running Exploratory Data Analysis...")
        eda = ExploratoryDataAnalysis(config)
        eda.run_full_analysis(df, X, y, class_names)
        print(eda.generate_report())
        print()

    print("[3/6] Training classifiers...")
    start_time = time.time()

    pipeline = TrainingPipeline(config)
    results_df = pipeline.train_all_classifiers(
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        use_quick=args.quick,
        include_ensembles=not args.no_ensembles,
        verbose=True,
    )

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print()

    if args.optimize:
        print("[4/6] Running hyperparameter optimization with Optuna...")
        print(f"  Storage: {config.optuna_storage}")
        optimizer = OptunaOptimizer(config)

        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier

        optimization_classifiers = {
            "RandomForest": RandomForestClassifier,
            "ExtraTrees": ExtraTreesClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "XGBoost": XGBClassifier,
            "SVC_RBF": SVC,
            "KNN": KNeighborsClassifier,
            "MLP": MLPClassifier,
            "LogisticRegression": LogisticRegression,
        }

        for name, clf_class in optimization_classifiers.items():
            print(f"  Optimizing {name}...")
            best_params, best_score = optimizer.optimize_classifier(
                clf_class, name, X_train, y_train
            )
            print(f"    Best CV Score: {best_score:.4f}")
            print(f"    Best Params: {best_params}")

        if args.dashboard:
            print("\n  Launching Optuna Dashboard at http://localhost:8080")
            print("  Press Ctrl+C to stop the dashboard")
            import subprocess

            subprocess.run(["optuna-dashboard", config.optuna_storage, "--port", "8080"])
        print()
    else:
        print("[4/6] Skipping hyperparameter optimization (use --optimize to enable)")
        print()

    print("[5/6] Generating visualizations...")
    visualizer = Visualizer(config)

    confusion_matrices = pipeline.get_confusion_matrices()
    feature_importance_summary = pipeline.get_feature_importance_summary()

    classification_reports = {
        r.name: r.classification_report
        for r in pipeline.results
        if r.classification_report is not None
    }

    visualizer.create_all_plots(
        results_df,
        confusion_matrices,
        pipeline.feature_importances,
        feature_importance_summary,
        class_names,
        classification_reports,
    )
    print(f"  Plots saved to: {config.plots_dir}")
    print()

    print("[6/6] Saving results...")
    results_path = pipeline.save_results()
    print(f"  Results saved to: {results_path}")

    detailed_path = save_detailed_classifier_reports(pipeline, config, class_names)
    print(f"  Detailed results saved to: {detailed_path}")

    report = pipeline.generate_report(class_names)
    report_path = config.results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    print()

    print(report)

    best_name, best_model = pipeline.get_best_classifier("accuracy")
    print(f"\nBest model saved at: {config.models_dir / best_name}.joblib")

    return 0
