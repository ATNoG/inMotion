#!/usr/bin/env python3
"""
Advanced Optuna Optimization with Dashboard Integration.

This script performs comprehensive hyperparameter optimization using Optuna
with SQLite storage and dashboard visualization support.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml_classification import Config, DataLoader, set_random_seeds


CLASSIFIERS_TO_OPTIMIZE = {
    "RandomForest": {
        "class": RandomForestClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": seed,
            "n_jobs": n_jobs,
        },
    },
    "ExtraTrees": {
        "class": ExtraTreesClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": seed,
            "n_jobs": n_jobs,
        },
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": seed,
        },
    },
    "XGBoost": {
        "class": XGBClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "eval_metric": "mlogloss",
            "n_jobs": n_jobs,
        },
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": n_jobs,
        },
    },
    "CatBoost": {
        "class": CatBoostClassifier,
        "space": lambda trial, seed, n_jobs: {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_state": seed,
            "verbose": False,
        },
    },
    "SVC": {
        "class": SVC,
        "space": lambda trial, seed, n_jobs: {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "probability": True,
            "random_state": seed,
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "space": lambda trial, seed, n_jobs: {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            "p": trial.suggest_int("p", 1, 5),
            "n_jobs": n_jobs,
        },
    },
    "MLP": {
        "class": MLPClassifier,
        "space": lambda trial, seed, n_jobs: {
            "hidden_layer_sizes": tuple([
                trial.suggest_int(f"n_units_l{i}", 32, 256)
                for i in range(trial.suggest_int("n_layers", 1, 3))
            ]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.1, log=True),
            "max_iter": 500,
            "early_stopping": True,
            "random_state": seed,
        },
    },
}


def create_objective(
    classifier_class: type,
    param_space_fn,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    seed: int,
    n_jobs: int,
):
    """Create an Optuna objective function for a classifier."""
    
    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial, seed, n_jobs)
        
        try:
            model = classifier_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
            return scores.mean()
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    return objective


def run_optimization(
    classifier_name: str,
    X: np.ndarray,
    y: np.ndarray,
    storage: str,
    n_trials: int,
    seed: int,
    n_jobs: int,
    cv_folds: int,
) -> tuple[dict, float]:
    """Run Optuna optimization for a single classifier."""
    clf_config = CLASSIFIERS_TO_OPTIMIZE[classifier_name]
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=f"wifi_fingerprint_{classifier_name}",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    objective = create_objective(
        clf_config["class"],
        clf_config["space"],
        X, y, cv, seed, n_jobs,
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization with Dashboard"
    )
    parser.add_argument(
        "--data", type=str, default="dataset.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--classifiers", type=str, nargs="+",
        default=list(CLASSIFIERS_TO_OPTIMIZE.keys()),
        help="Classifiers to optimize"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of trials per classifier"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--storage", type=str, default="sqlite:///optuna_studies.db",
        help="Optuna storage URL"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch Optuna dashboard after optimization"
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8080,
        help="Port for Optuna dashboard"
    )
    args = parser.parse_args()
    
    set_random_seeds(args.seed)
    
    config = Config(
        data_path=Path(args.data),
        random_seed=args.seed,
        n_cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )
    
    print("=" * 70)
    print("Optuna Hyperparameter Optimization")
    print("=" * 70)
    print(f"Dataset: {args.data}")
    print(f"Classifiers: {args.classifiers}")
    print(f"Trials per classifier: {args.n_trials}")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Storage: {args.storage}")
    print("=" * 70)
    print()
    
    print("Loading data...")
    loader = DataLoader(config)
    df = loader.load_data()
    X, y, class_names = loader.preprocess(df, scale_features=True, encode_labels=True)
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Classes: {class_names}")
    print()
    
    results = {}
    total_start = time.time()
    
    for i, clf_name in enumerate(args.classifiers):
        if clf_name not in CLASSIFIERS_TO_OPTIMIZE:
            print(f"Warning: Unknown classifier '{clf_name}', skipping...")
            continue
        
        print(f"[{i+1}/{len(args.classifiers)}] Optimizing {clf_name}...")
        start = time.time()
        
        best_params, best_score = run_optimization(
            clf_name, X, y, args.storage, args.n_trials,
            args.seed, args.n_jobs, args.cv_folds,
        )
        
        elapsed = time.time() - start
        results[clf_name] = {"params": best_params, "score": best_score, "time": elapsed}
        
        print(f"  Best CV Score: {best_score:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Best Params: {best_params}")
        print()
    
    total_time = time.time() - total_start
    
    print("=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: {res['score']:.4f} (optimized in {res['time']:.1f}s)")
    
    print(f"\nTotal optimization time: {total_time:.1f}s")
    
    results_path = config.results_dir / "optuna_optimization_results.txt"
    with open(results_path, "w") as f:
        f.write("Optuna Hyperparameter Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        for name, res in sorted_results:
            f.write(f"{name}:\n")
            f.write(f"  Best CV Score: {res['score']:.4f}\n")
            f.write(f"  Optimization Time: {res['time']:.1f}s\n")
            f.write(f"  Best Parameters:\n")
            for param, value in res["params"].items():
                f.write(f"    {param}: {value}\n")
            f.write("\n")
    print(f"\nResults saved to: {results_path}")
    
    if args.dashboard:
        print(f"\nLaunching Optuna Dashboard at http://localhost:{args.dashboard_port}")
        print("Press Ctrl+C to stop the dashboard")
        import subprocess
        subprocess.run([
            "optuna-dashboard", args.storage,
            "--port", str(args.dashboard_port),
        ])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
