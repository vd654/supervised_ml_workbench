# regression.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, List[Any]]


def ensure_artifacts_dir(path: str = "artifacts") -> None:
    os.makedirs(path, exist_ok=True)


def get_regression_specs(random_state: int = 42) -> List[RegModelSpec]:
    specs: List[RegModelSpec] = []

    # Linear Regression (no hyperparams)
    specs.append(
        RegModelSpec(
            name="linreg",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            param_grid={},  # GridSearchCV works; it will just fit once per split
        )
    )

    # Ridge
    specs.append(
        RegModelSpec(
            name="ridge",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=random_state)),
                ]
            ),
            param_grid={
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        )
    )

    # Lasso
    specs.append(
        RegModelSpec(
            name="lasso",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=10000, random_state=random_state)),
                ]
            ),
            param_grid={
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
        )
    )

    # kNN Regressor
    specs.append(
        RegModelSpec(
            name="knn",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor()),
                ]
            ),
            param_grid={
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        )
    )

    # SVR (RBF)
    specs.append(
        RegModelSpec(
            name="svr",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            param_grid={
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__epsilon": [0.05, 0.1, 0.2],
            },
        )
    )

    # Random Forest Regressor
    specs.append(
        RegModelSpec(
            name="rf",
            pipeline=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=random_state)),
                ]
            ),
            param_grid={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        )
    )

    return specs


def fit_gridsearch(
    spec: RegModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    return grid


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "r2": float(r2)}


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.35)
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.35)
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    RANDOM_STATE = 42
    ensure_artifacts_dir("artifacts")

    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specs = get_regression_specs(random_state=RANDOM_STATE)

    results: List[Tuple[str, float, GridSearchCV]] = []

    print("=== Model comparison (CV neg_MSE, higher is better) ===")
    for spec in specs:
        grid = fit_gridsearch(spec, X_train, y_train, cv)
        best_cv = grid.best_score_  # neg MSE
        print(f"{spec.name:>6s}  best_cv_neg_mse={best_cv:.4f}  best_params={grid.best_params_}")
        results.append((spec.name, best_cv, grid))

    # Select best by CV (max neg_mse)
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_cv, best_grid = results[0]
    best_estimator = best_grid.best_estimator_

    print("\n=== Selected best model ===")
    print(f"best_model={best_name}  cv_neg_mse={best_cv:.4f}")
    print("best_params:", best_grid.best_params_)

    # Train vs Test evaluation
    yhat_train = best_estimator.predict(X_train)
    yhat_test = best_estimator.predict(X_test)

    train_m = regression_metrics(y_train, yhat_train)
    test_m = regression_metrics(y_test, yhat_test)

    print("\n=== Final metrics ===")
    print("Train:", {k: round(v, 4) for k, v in train_m.items()})
    print(" Test:", {k: round(v, 4) for k, v in test_m.items()})

    # Plots (test)
    plot_true_vs_pred(y_test, yhat_test, title=f"{best_name} True vs Pred (Test)")
    plot_residuals(y_test, yhat_test, title=f"{best_name} Residual Plot (Test)")

    # Save artifact
    out_path = "artifacts/best_reg.joblib"
    joblib.dump(best_estimator, out_path)
    print(f"\nSaved best model to: {out_path}")


if __name__ == "__main__":
    main()
