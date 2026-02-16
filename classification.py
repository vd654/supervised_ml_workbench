# classification.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, List[Any]]


def ensure_artifacts_dir(path: str = "artifacts") -> None:
    os.makedirs(path, exist_ok=True)


def get_classification_specs(random_state: int = 42) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # Logistic Regression
    specs.append(
        ModelSpec(
            name="logreg",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, random_state=random_state)),
                ]
            ),
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        )
    )

    # kNN
    specs.append(
        ModelSpec(
            name="knn",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            param_grid={
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # Manhattan vs Euclidean
            },
        )
    )

    # SVM (RBF)
    specs.append(
        ModelSpec(
            name="svm_rbf",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, random_state=random_state)),
                ]
            ),
            param_grid={
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__kernel": ["rbf"],
            },
        )
    )

    # Random Forest (no scaler needed, but keeping pipeline consistent is fine)
    specs.append(
        ModelSpec(
            name="rf",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(random_state=random_state)),
                ]
            ),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        )
    )

    # Gradient Boosting
    specs.append(
        ModelSpec(
            name="gb",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=random_state)),
                ]
            ),
            param_grid={
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [1.0, 0.8],
            },
        )
    )

    return specs


def fit_gridsearch(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: StratifiedKFold,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate_classifier(
    estimator: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_prefix: str = "",
) -> Dict[str, float]:
    # Train preds
    yhat_train = estimator.predict(X_train)
    yproba_train = estimator.predict_proba(X_train)[:, 1]

    # Test preds (only once in your workflow)
    yhat_test = estimator.predict(X_test)
    yproba_test = estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, yhat_train),
        "test_accuracy": accuracy_score(y_test, yhat_test),
        "test_precision": precision_score(y_test, yhat_test),
        "test_recall": recall_score(y_test, yhat_test),
        "test_f1": f1_score(y_test, yhat_test),
        "train_roc_auc": roc_auc_score(y_train, yproba_train),
        "test_roc_auc": roc_auc_score(y_test, yproba_test),
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, yhat_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, yproba_test)
    plt.title(f"{title_prefix} ROC Curve (Test)")
    plt.tight_layout()
    plt.show()

    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_test, yproba_test)
    plt.title(f"{title_prefix} Precision-Recall Curve (Test)")
    plt.tight_layout()
    plt.show()

    return metrics


def interpret_model(
    best_estimator: Pipeline,
    feature_names: List[str],
    top_k: int = 10,
) -> None:
    model = best_estimator.named_steps["model"]

    print("\n=== Interpretation ===")

    # Logistic Regression coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        idx = np.argsort(np.abs(coefs))[::-1][:top_k]
        print(f"\nTop {top_k} |coef| (LogReg):")
        for i in idx:
            print(f"{feature_names[i]:>25s}  coef={coefs[i]: .4f}")

    # Tree-based feature importances
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        idx = np.argsort(imps)[::-1][:top_k]
        print(f"\nTop {top_k} importances (Tree/Boosting):")
        for i in idx:
            print(f"{feature_names[i]:>25s}  imp={imps[i]: .4f}")


def main() -> None:
    RANDOM_STATE = 42
    ensure_artifacts_dir("artifacts")

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specs = get_classification_specs(random_state=RANDOM_STATE)

    results: List[Tuple[str, float, GridSearchCV]] = []

    print("=== Model comparison (CV ROC-AUC) ===")
    for spec in specs:
        grid = fit_gridsearch(spec, X_train, y_train, cv)
        best_cv = grid.best_score_
        print(f"{spec.name:>8s}  best_cv_roc_auc={best_cv:.4f}  best_params={grid.best_params_}")
        results.append((spec.name, best_cv, grid))

    # Select best by CV
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_cv, best_grid = results[0]
    best_estimator = best_grid.best_estimator_

    print("\n=== Selected best model ===")
    print(f"best_model={best_name}  cv_roc_auc={best_cv:.4f}")
    print("best_params:", best_grid.best_params_)

    # Final test evaluation (only once)
    metrics = evaluate_classifier(
        best_estimator,
        X_train, y_train,
        X_test, y_test,
        title_prefix=f"{best_name}",
    )

    print("\n=== Final metrics ===")
    for k, v in metrics.items():
        print(f"{k:>16s}: {v:.4f}")

    # Interpretation
    interpret_model(best_estimator, feature_names, top_k=10)

    # Save artifact
    out_path = "artifacts/best_clf.joblib"
    joblib.dump(best_estimator, out_path)
    print(f"\nSaved best model to: {out_path}")


if __name__ == "__main__":
    main()
