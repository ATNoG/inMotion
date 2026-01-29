"""Classifier factory with all ML classification algorithms."""

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier


class ClassifierFactory:
    """Factory class to create and manage all classifiers."""

    def __init__(self, random_seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu

    def get_all_classifiers(self) -> dict[str, ClassifierMixin | CatBoostClassifier]:
        """Get all available classifiers with default parameters."""
        return {
            **self.get_tree_classifiers(),
            **self.get_ensemble_classifiers(),
            **self.get_linear_classifiers(),
            **self.get_svm_classifiers(),
            **self.get_neighbors_classifiers(),
            **self.get_naive_bayes_classifiers(),
            **self.get_neural_network_classifiers(),
            **self.get_discriminant_classifiers(),
            **self.get_boosting_classifiers(),
            **self.get_other_classifiers(),
        }

    def get_tree_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get tree-based classifiers."""
        return {
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_seed),
            "ExtraTree": ExtraTreeClassifier(random_state=self.random_seed),
        }

    def get_ensemble_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get ensemble classifiers."""
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=100, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "Bagging": BaggingClassifier(
                n_estimators=50, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=self.random_seed),
        }

    def get_linear_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get linear classifiers with regularization."""
        return {
            "LogisticRegression_L2": LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
            ),
            "LogisticRegression_L1": LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="saga",
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
            ),
            "LogisticRegression_ElasticNet": LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                solver="saga",
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
            ),
            "RidgeClassifier": RidgeClassifier(alpha=1.0, random_state=self.random_seed),
            "SGDClassifier": SGDClassifier(
                loss="hinge",
                penalty="l2",
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
            ),
            "Perceptron": Perceptron(
                penalty="l2", max_iter=1000, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "PassiveAggressive": PassiveAggressiveClassifier(
                max_iter=1000, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
        }

    def get_svm_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get SVM classifiers."""
        return {
            "SVC_RBF": SVC(
                kernel="rbf", C=1.0, gamma="scale", random_state=self.random_seed, probability=True
            ),
            "SVC_Linear": SVC(
                kernel="linear", C=1.0, random_state=self.random_seed, probability=True
            ),
            "SVC_Poly": SVC(
                kernel="poly", degree=3, C=1.0, random_state=self.random_seed, probability=True
            ),
            "LinearSVC": CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=self.random_seed)
            ),
            "NuSVC": NuSVC(
                kernel="rbf", nu=0.5, gamma="scale", random_state=self.random_seed, probability=True
            ),
        }

    def get_neighbors_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get neighbors-based classifiers."""
        return {
            "KNN_3": KNeighborsClassifier(n_neighbors=3, n_jobs=self.n_jobs),
            "KNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=self.n_jobs),
            "KNN_7": KNeighborsClassifier(n_neighbors=7, n_jobs=self.n_jobs),
            "NearestCentroid": NearestCentroid(),
        }

    def get_naive_bayes_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get Naive Bayes classifiers."""
        return {
            "GaussianNB": GaussianNB(),
        }

    def get_neural_network_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get neural network classifiers."""
        return {
            "MLP_Small": MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
            ),
            "MLP_Medium": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
            ),
            "MLP_Large": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=self.random_seed,
                early_stopping=True,
            ),
            "MLP_L2": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                alpha=0.01,
                random_state=self.random_seed,
                early_stopping=True,
            ),
        }

    def get_discriminant_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get discriminant analysis classifiers."""
        return {
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
        }

    def get_boosting_classifiers(self) -> dict[str, ClassifierMixin | CatBoostClassifier]:
        """Get gradient boosting classifiers with optional GPU support."""
        # XGBoost GPU config
        xgb_params = {
            "n_estimators": 100,
            "random_state": self.random_seed,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        if self.use_gpu:
            xgb_params["device"] = "cuda"
            xgb_params["tree_method"] = "hist"  # GPU-accelerated histogram
        else:
            xgb_params["n_jobs"] = self.n_jobs

        # LightGBM GPU config
        lgbm_params = {
            "n_estimators": 100,
            "random_state": self.random_seed,
            "verbose": -1,
        }
        if self.use_gpu:
            lgbm_params["device"] = "gpu"
            lgbm_params["gpu_platform_id"] = 0
            lgbm_params["gpu_device_id"] = 0
        else:
            lgbm_params["n_jobs"] = self.n_jobs

        # CatBoost GPU config
        catboost_params = {
            "iterations": 100,
            "random_state": self.random_seed,
            "verbose": False,
        }
        if self.use_gpu:
            catboost_params["task_type"] = "GPU"
            catboost_params["devices"] = "0"
        else:
            catboost_params["thread_count"] = self.n_jobs if self.n_jobs > 0 else -1

        return {
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_seed
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=100, random_state=self.random_seed
            ),
            "XGBoost": XGBClassifier(**xgb_params),
            "LightGBM": LGBMClassifier(**lgbm_params),
            "CatBoost": CatBoostClassifier(**catboost_params),
        }

    def get_other_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get other classifiers."""
        return {
            "GaussianProcess": GaussianProcessClassifier(
                kernel=1.0 * RBF(1.0), random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
            "DummyStratified": DummyClassifier(
                strategy="stratified", random_state=self.random_seed
            ),
        }

    def get_voting_ensemble(
        self, base_classifiers: dict[str, ClassifierMixin] | None = None
    ) -> VotingClassifier:
        """Create a voting ensemble from selected classifiers."""
        if base_classifiers is None:
            # XGBoost config for ensemble
            xgb_params = {
                "n_estimators": 100,
                "random_state": self.random_seed,
                "eval_metric": "mlogloss",
            }
            if self.use_gpu:
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"
            else:
                xgb_params["n_jobs"] = self.n_jobs

            # LightGBM config for ensemble
            lgbm_params = {
                "n_estimators": 100,
                "random_state": self.random_seed,
                "verbose": -1,
            }
            if self.use_gpu:
                lgbm_params["device"] = "gpu"
            else:
                lgbm_params["n_jobs"] = self.n_jobs

            base_classifiers = {
                "rf": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_seed, n_jobs=self.n_jobs
                ),
                "xgb": XGBClassifier(**xgb_params),
                "svc": SVC(probability=True, random_state=self.random_seed),
                "lgbm": LGBMClassifier(**lgbm_params),
            }

        estimators = list(base_classifiers.items())
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=self.n_jobs)

    def get_stacking_ensemble(
        self,
        base_classifiers: dict[str, ClassifierMixin] | None = None,
        final_estimator: ClassifierMixin | None = None,
    ) -> StackingClassifier:
        """Create a stacking ensemble from selected classifiers."""
        if base_classifiers is None:
            # XGBoost config for stacking
            xgb_params = {
                "n_estimators": 50,
                "random_state": self.random_seed,
                "eval_metric": "mlogloss",
            }
            if self.use_gpu:
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"
            else:
                xgb_params["n_jobs"] = self.n_jobs

            base_classifiers = {
                "rf": RandomForestClassifier(
                    n_estimators=50, random_state=self.random_seed, n_jobs=self.n_jobs
                ),
                "xgb": XGBClassifier(**xgb_params),
                "svc": SVC(probability=True, random_state=self.random_seed),
            }

        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000, random_state=self.random_seed)

        estimators = list(base_classifiers.items())
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=self.n_jobs,
        )

    def get_classifier_by_name(self, name: str, **kwargs: Any) -> ClassifierMixin:
        """Get a specific classifier by name with custom parameters."""
        all_classifiers = self.get_all_classifiers()

        if name not in all_classifiers:
            raise ValueError(
                f"Unknown classifier: {name}. Available: {list(all_classifiers.keys())}"
            )

        classifier_class = type(all_classifiers[name])
        return classifier_class(**kwargs)

    def get_quick_classifiers(self) -> dict[str, ClassifierMixin]:
        """Get a subset of fast classifiers for quick testing."""
        # XGBoost config
        xgb_params = {
            "n_estimators": 50,
            "random_state": self.random_seed,
            "eval_metric": "mlogloss",
        }
        if self.use_gpu:
            xgb_params["device"] = "cuda"
            xgb_params["tree_method"] = "hist"
        else:
            xgb_params["n_jobs"] = self.n_jobs

        return {
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_seed),
            "RandomForest": RandomForestClassifier(
                n_estimators=50, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "LogisticRegression_L2": LogisticRegression(
                max_iter=1000, random_state=self.random_seed, n_jobs=self.n_jobs
            ),
            "KNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=self.n_jobs),
            "GaussianNB": GaussianNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "XGBoost": XGBClassifier(**xgb_params),
        }
