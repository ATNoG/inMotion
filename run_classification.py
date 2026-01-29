#!/usr/bin/env python3
"""
Main script for WiFi Fingerprinting Classification Analysis.

This script runs a comprehensive analysis of various ML classifiers for
predicting location classes based on WiFi RSSI fingerprints.
"""

import argparse
import sys
import time
from pathlib import Path

from ml_classification import (
    Config,
    DataLoader,
    ExploratoryDataAnalysis,
    ClassifierFactory,
    OptunaOptimizer,
    TrainingPipeline,
    Visualizer,
    set_random_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiFi Fingerprinting Classification Analysis"
    )
    parser.add_argument(
        "--data", type=str, default="dataset.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick mode with fewer classifiers"
    )
    parser.add_argument(
        "--no-ensembles", action="store_true",
        help="Skip ensemble classifiers"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials per classifier"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip exploratory data analysis"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    config = Config(
        data_path=Path(args.data),
        random_seed=args.seed,
        n_cv_folds=args.cv_folds,
        n_optuna_trials=args.n_trials,
        n_jobs=args.n_jobs,
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
        X_train, y_train, X_test, y_test, class_names,
        use_quick=args.quick,
        include_ensembles=not args.no_ensembles,
        verbose=True,
    )
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print()
    
    if args.optimize:
        print("[4/6] Running hyperparameter optimization with Optuna...")
        optimizer = OptunaOptimizer(config)
        
        optimization_classifiers = [
            ("RandomForest", "sklearn.ensemble.RandomForestClassifier"),
            ("XGBoost", "xgboost.XGBClassifier"),
            ("LightGBM", "lightgbm.LGBMClassifier"),
            ("SVC_RBF", "sklearn.svm.SVC"),
        ]
        
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.svm import SVC
        
        classes = {
            "RandomForest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "SVC_RBF": SVC,
        }
        
        for name, _ in optimization_classifiers:
            print(f"  Optimizing {name}...")
            best_params, best_score = optimizer.optimize_classifier(
                classes[name], name, X_train, y_train
            )
            print(f"    Best CV Score: {best_score:.4f}")
            print(f"    Best Params: {best_params}")
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


if __name__ == "__main__":
    sys.exit(main())
