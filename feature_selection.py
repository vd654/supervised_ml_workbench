# feature_selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class FSRun:
    name: str
    grid: GridSearchCV


def run_feature_selection_experiments(random_state: int = 42) -> List[Tuple[str, float, Dict[str, Any]]]:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Base estimator for classification (must support predict_proba for ROC-AUC scoring stability)
    base_logreg = LogisticRegression(max_iter=5000, random_state=random_state)

    runs: List[FSRun] = []

    # 1) No Feature Selection (baseline)
    pipe_base = Pipeline([
        ("scaler", StandardScaler()),
        ("model", base_logreg),
    ])
    grid_base = GridSearchCV(
        pipe_base,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    runs.append(FSRun("baseline_no_fs", grid_base))

    # 2) SelectKBest f_classif
    pipe_kbest_f = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", SelectKBest(score_func=f_classif)),
        ("model", base_logreg),
    ])
    grid_kbest_f = GridSearchCV(
        pipe_kbest_f,
        param_grid={
            "fs__k": [5, 10, 15, 20, 25, 30],
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    runs.append(FSRun("selectkbest_f_classif", grid_kbest_f))

    # 3) SelectKBest mutual_info
    pipe_kbest_mi = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", SelectKBest(score_func=mutual_info_classif)),
        ("model", base_logreg),
    ])
    grid_kbest_mi = GridSearchCV(
        pipe_kbest_mi,
        param_grid={
            "fs__k": [5, 10, 15, 20, 25, 30],
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    runs.append(FSRun("selectkbest_mutual_info", grid_kbest_mi))

    # 4) RFE with Logistic Regression
    rfe_est = LogisticRegression(max_iter=5000, random_state=random_state)
    pipe_rfe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", RFE(estimator=rfe_est)),
        ("model", base_logreg),
    ])
    grid_rfe_lr = GridSearchCV(
        pipe_rfe_lr,
        param_grid={
            "fs__n_features_to_select": [5, 10, 15, 20, 25, 30],
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    runs.append(FSRun("rfe_logreg", grid_rfe_lr))

    # 5) RFE with RandomForest (selector)
    rf_selector = RandomForestClassifier(n_estimators=300, random_state=random_state)
    pipe_rfe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", RFE(estimator=rf_selector)),
        ("model", base_logreg),
    ])
    grid_rfe_rf = GridSearchCV(
        pipe_rfe_rf,
        param_grid={
            "fs__n_features_to_select": [5, 10, 15, 20, 25, 30],
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    runs.append(FSRun("rfe_random_forest_selector", grid_rfe_rf))

    results: List[Tuple[str, float, Dict[str, Any]]] = []
    print("=== Feature Selection Comparison (CV ROC-AUC) ===")

    for run in runs:
        run.grid.fit(X_train, y_train)
        results.append((run.name, run.grid.best_score_, run.grid.best_params_))
        print(f"{run.name:>28s}  cv_roc_auc={run.grid.best_score_:.4f}  best={run.grid.best_params_}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\nBest FS setup:", results[0][0], "score=", f"{results[0][1]:.4f}")
    return results


if __name__ == "__main__":
    run_feature_selection_experiments()