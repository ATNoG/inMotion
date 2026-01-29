"""Optuna hyperparameter optimization module."""

from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import Config


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.studies: dict[str, optuna.Study] = {}
        self.best_params: dict[str, dict[str, Any]] = {}
        self.best_scores: dict[str, float] = {}

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        storage: str | None = None,
    ) -> optuna.Study:
        """Create an Optuna study for a classifier."""
        sampler = TPESampler(seed=self.config.random_seed)
        storage = storage or self.config.optuna_storage

        study = optuna.create_study(
            study_name=f"{self.config.optuna_study_name}_{study_name}",
            direction=direction,
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
        self.studies[study_name] = study
        return study

    def get_search_space(self, classifier_name: str) -> Callable[[optuna.Trial], dict[str, Any]]:
        """Get the hyperparameter search space for a classifier."""
        search_spaces = {
            "DecisionTree": self._decision_tree_space,
            "RandomForest": self._random_forest_space,
            "ExtraTrees": self._extra_trees_space,
            "GradientBoosting": self._gradient_boosting_space,
            "HistGradientBoosting": self._hist_gradient_boosting_space,
            "XGBoost": self._xgboost_space,
            "LightGBM": self._lightgbm_space,
            "CatBoost": self._catboost_space,
            "SVC_RBF": self._svc_rbf_space,
            "SVC_Linear": self._svc_linear_space,
            "KNN": self._knn_space,
            "LogisticRegression": self._logistic_regression_space,
            "MLP": self._mlp_space,
            "AdaBoost": self._adaboost_space,
        }
        return search_spaces.get(classifier_name, self._default_space)

    def _decision_tree_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "random_state": self.config.random_seed,
        }

    def _random_forest_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": self.config.random_seed,
            "n_jobs": self.config.n_jobs,
        }

    def _extra_trees_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": self.config.random_seed,
            "n_jobs": self.config.n_jobs,
        }

    def _gradient_boosting_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": self.config.random_seed,
        }

    def _hist_gradient_boosting_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "random_state": self.config.random_seed,
        }

    def _xgboost_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.config.random_seed,
            "eval_metric": "mlogloss",
            "n_jobs": self.config.n_jobs,
        }

    def _lightgbm_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.config.random_seed,
            "verbose": -1,
            "n_jobs": self.config.n_jobs,
        }

    def _catboost_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_state": self.config.random_seed,
            "verbose": False,
        }

    def _svc_rbf_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": "rbf",
            "probability": True,
            "random_state": self.config.random_seed,
        }

    def _svc_linear_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": "linear",
            "probability": True,
            "random_state": self.config.random_seed,
        }

    def _knn_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            "p": trial.suggest_int("p", 1, 5),
            "n_jobs": self.config.n_jobs,
        }

    def _logistic_regression_space(self, trial: optuna.Trial) -> dict[str, Any]:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": penalty,
            "max_iter": 1000,
            "random_state": self.config.random_seed,
            "n_jobs": self.config.n_jobs,
        }

        if penalty in ["l1", "elasticnet"]:
            params["solver"] = "saga"
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        return params

    def _mlp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256))

        return {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.1, log=True),
            "max_iter": 500,
            "early_stopping": True,
            "random_state": self.config.random_seed,
        }

    def _adaboost_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            "random_state": self.config.random_seed,
        }

    def _default_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {}

    def optimize_classifier(
        self,
        classifier_class: type,
        classifier_name: str,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Optimize hyperparameters for a classifier using Optuna."""
        n_trials = n_trials or self.config.n_optuna_trials
        timeout = timeout or self.config.optuna_timeout

        study = self.create_study(classifier_name)
        search_space_fn = self.get_search_space(classifier_name)

        cv = StratifiedKFold(
            n_splits=self.config.n_cv_folds, shuffle=True, random_state=self.config.random_seed
        )

        def objective(trial: optuna.Trial) -> float:
            params = search_space_fn(trial)

            try:
                model = classifier_class(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring="accuracy", n_jobs=self.config.n_jobs
                )
                return scores.mean()
            except Exception as e:
                print(f"Trial failed with error: {e}")
                return 0.0

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        self.best_params[classifier_name] = study.best_params
        self.best_scores[classifier_name] = study.best_value

        return study.best_params, study.best_value

    def get_optimized_classifier(
        self,
        classifier_class: type,
        classifier_name: str,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int | None = None,
    ) -> ClassifierMixin:
        """Get an optimized classifier instance."""
        best_params, _ = self.optimize_classifier(classifier_class, classifier_name, X, y, n_trials)

        search_space_fn = self.get_search_space(classifier_name)

        class MockTrial:
            def __init__(self, params: dict[str, Any]) -> None:
                self.params = params

            def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
                return self.params.get(name, args[0] if args else 0)

            def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
                return self.params.get(name, args[0] if args else 0.0)

            def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
                return self.params.get(name, choices[0])

        full_params = search_space_fn(MockTrial(best_params))
        return classifier_class(**full_params)

    def launch_dashboard(self, port: int = 8080) -> None:
        """Launch the Optuna dashboard for visualization."""
        if self.config.optuna_storage:
            import subprocess

            subprocess.Popen(
                ["optuna-dashboard", self.config.optuna_storage, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Optuna dashboard launched at http://localhost:{port}")
