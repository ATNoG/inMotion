"""Training pipeline for ML classification."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from .classifiers import ClassifierFactory
from .config import Config
from .utils import format_time, save_model


@dataclass
class ClassifierResult:
    """Results from training a classifier."""

    name: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_mean: float
    cv_std: float
    train_time: float
    confusion_matrix: np.ndarray | None = None
    classification_report: dict[str, Any] | None = None
    feature_importances: np.ndarray | None = None
    model_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "Classifier": self.name,
            "Accuracy": self.accuracy,
            "Balanced_Accuracy": self.balanced_accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1_Score": self.f1_score,
            "CV_Mean": self.cv_mean,
            "CV_Std": self.cv_std,
            "Train_Time_s": self.train_time,
        }


class TrainingPipeline:
    """Complete training pipeline for all classifiers."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.factory = ClassifierFactory(random_seed=config.random_seed, n_jobs=config.n_jobs)
        self.results: list[ClassifierResult] = []
        self.trained_models: dict[str, ClassifierMixin] = {}
        self.feature_importances: dict[str, np.ndarray] = {}

    def train_single_classifier(
        self,
        classifier: ClassifierMixin,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: list[str],
        save_models: bool = True,
    ) -> ClassifierResult:
        """Train and evaluate a single classifier."""
        cv = StratifiedKFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )

        start_time = time.time()

        try:
            cv_scores = cross_val_score(
                clone(classifier),
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=self.config.n_jobs,
            )
        except Exception as e:
            print(f"CV failed for {name}: {e}")
            cv_scores = np.array([0.0])

        try:
            model = clone(classifier)
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            return ClassifierResult(
                name=name,
                accuracy=0.0,
                balanced_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                cv_mean=0.0,
                cv_std=0.0,
                train_time=0.0,
            )

        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            self.feature_importances[name] = feature_importances
        elif hasattr(model, "coef_"):
            feature_importances = (
                np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            )
            self.feature_importances[name] = feature_importances

        model_path = None
        if save_models:
            model_path = save_model(model, self.config.models_dir, name)
            self.trained_models[name] = model

        result = ClassifierResult(
            name=name,
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            train_time=train_time,
            confusion_matrix=cm,
            classification_report=report,
            feature_importances=feature_importances,
            model_path=model_path,
        )

        self.results.append(result)
        return result

    def train_all_classifiers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: list[str],
        use_quick: bool = False,
        include_ensembles: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Train all classifiers and return results DataFrame."""
        if use_quick:
            classifiers = self.factory.get_quick_classifiers()
        else:
            classifiers = self.factory.get_all_classifiers()

        if include_ensembles:
            classifiers["VotingEnsemble"] = self.factory.get_voting_ensemble()
            classifiers["StackingEnsemble"] = self.factory.get_stacking_ensemble()

        total = len(classifiers)

        for i, (name, classifier) in enumerate(classifiers.items()):
            if verbose:
                print(f"[{i + 1}/{total}] Training {name}...", end=" ")

            result = self.train_single_classifier(
                classifier, name, X_train, y_train, X_test, y_test, class_names
            )

            if verbose:
                print(
                    f"Accuracy: {result.accuracy:.4f}, CV: {result.cv_mean:.4f}±{result.cv_std:.4f}, Time: {format_time(result.train_time)}"
                )

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        return df

    def save_results(self, filename: str = "classification_results.csv") -> Path:
        """Save results to CSV file."""
        df = self.get_results_dataframe()
        path = self.config.results_dir / filename
        df.to_csv(path, index=False)
        return path

    def get_best_classifier(self, metric: str = "accuracy") -> tuple[str, ClassifierMixin]:
        """Get the best classifier based on a metric."""
        if not self.results:
            raise ValueError("No classifiers have been trained yet.")

        metric_map = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1_score",
            "precision": "precision",
            "recall": "recall",
            "cv_mean": "cv_mean",
        }

        attr = metric_map.get(metric, "accuracy")
        best_result = max(self.results, key=lambda r: getattr(r, attr))

        return best_result.name, self.trained_models.get(best_result.name)

    def get_feature_importance_summary(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Get feature importance summary across all classifiers."""
        if not self.feature_importances:
            return pd.DataFrame()

        if feature_names is None:
            feature_names = self.config.feature_columns

        data = []
        for name, importances in self.feature_importances.items():
            for i, imp in enumerate(importances):
                data.append(
                    {
                        "Classifier": name,
                        "Feature": feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                        "Importance": imp,
                    }
                )

        df = pd.DataFrame(data)

        mean_importance = df.groupby("Feature")["Importance"].mean().reset_index()
        mean_importance.columns = ["Feature", "Mean_Importance"]
        mean_importance = mean_importance.sort_values("Mean_Importance", ascending=False)

        return mean_importance

    def get_confusion_matrices(self) -> dict[str, np.ndarray]:
        """Get confusion matrices for all trained classifiers."""
        return {r.name: r.confusion_matrix for r in self.results if r.confusion_matrix is not None}

    def cross_validate_predictions(
        self,
        classifier: ClassifierMixin,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Get cross-validated predictions for a classifier."""
        cv = StratifiedKFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        return cross_val_predict(classifier, X, y, cv=cv, n_jobs=self.config.n_jobs)

    def generate_report(self, class_names: list[str]) -> str:
        """Generate a comprehensive text report of all results."""
        df = self.get_results_dataframe()

        report_lines = [
            "=" * 70,
            "CLASSIFICATION RESULTS REPORT",
            "=" * 70,
            "",
            f"Total classifiers evaluated: {len(self.results)}",
            f"Cross-validation folds: {self.config.n_cv_folds}",
            f"Test set size: {self.config.test_size * 100:.0f}%",
            "",
            "-" * 70,
            "TOP 10 CLASSIFIERS BY ACCURACY",
            "-" * 70,
        ]

        for i, row in df.head(10).iterrows():
            report_lines.append(
                f"{i + 1}. {row['Classifier']}: "
                f"Acc={row['Accuracy']:.4f}, "
                f"F1={row['F1_Score']:.4f}, "
                f"CV={row['CV_Mean']:.4f}±{row['CV_Std']:.4f}"
            )

        best_name, _ = self.get_best_classifier("accuracy")
        best_result = next(r for r in self.results if r.name == best_name)

        report_lines.extend(
            [
                "",
                "-" * 70,
                f"BEST CLASSIFIER: {best_name}",
                "-" * 70,
                "",
                f"Accuracy: {best_result.accuracy:.4f}",
                f"Balanced Accuracy: {best_result.balanced_accuracy:.4f}",
                f"Precision (weighted): {best_result.precision:.4f}",
                f"Recall (weighted): {best_result.recall:.4f}",
                f"F1 Score (weighted): {best_result.f1_score:.4f}",
                f"CV Score: {best_result.cv_mean:.4f} ± {best_result.cv_std:.4f}",
                f"Training Time: {format_time(best_result.train_time)}",
                "",
                "Classification Report:",
            ]
        )

        if best_result.classification_report:
            for cls, metrics in best_result.classification_report.items():
                if isinstance(metrics, dict):
                    report_lines.append(
                        f"  {cls}: precision={metrics.get('precision', 0):.3f}, "
                        f"recall={metrics.get('recall', 0):.3f}, "
                        f"f1={metrics.get('f1-score', 0):.3f}"
                    )

        feature_importance = self.get_feature_importance_summary()
        if not feature_importance.empty:
            report_lines.extend(
                [
                    "",
                    "-" * 70,
                    "FEATURE IMPORTANCE (AVERAGED ACROSS CLASSIFIERS)",
                    "-" * 70,
                ]
            )
            for _, row in feature_importance.iterrows():
                report_lines.append(f"  {row['Feature']}: {row['Mean_Importance']:.4f}")

        report_lines.extend(["", "=" * 70])

        return "\n".join(report_lines)
