# evaluation.py
from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, RegressorMixin
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
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(
    estimator: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_prefix: str = "",
    plot: bool = True,
) -> Dict[str, float]:
    """Assumes estimator supports predict_proba for ROC/PR curves."""
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    # Probabilities for ROC-AUC/PR
    yproba_train = estimator.predict_proba(X_train)[:, 1]
    yproba_test = estimator.predict_proba(X_test)[:, 1]

    out = {
        "train_accuracy": accuracy_score(y_train, yhat_train),
        "test_accuracy": accuracy_score(y_test, yhat_test),
        "test_precision": precision_score(y_test, yhat_test),
        "test_recall": recall_score(y_test, yhat_test),
        "test_f1": f1_score(y_test, yhat_test),
        "train_roc_auc": roc_auc_score(y_train, yproba_train),
        "test_roc_auc": roc_auc_score(y_test, yproba_test),
    }

    if plot:
        cm = confusion_matrix(y_test, yhat_test)
        ConfusionMatrixDisplay(cm).plot(values_format="d")
        plt.title(f"{title_prefix} Confusion Matrix")
        plt.tight_layout()
        plt.show()

        RocCurveDisplay.from_predictions(y_test, yproba_test)
        plt.title(f"{title_prefix} ROC Curve (Test)")
        plt.tight_layout()
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_test, yproba_test)
        plt.title(f"{title_prefix} PR Curve (Test)")
        plt.tight_layout()
        plt.show()

    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "r2": float(r2)}


def evaluate_regressor(
    estimator: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_prefix: str = "",
    plot: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    train_m = regression_metrics(y_train, yhat_train)
    test_m = regression_metrics(y_test, yhat_test)

    if plot:
        # True vs Pred (test)
        plt.figure()
        plt.scatter(y_test, yhat_test, alpha=0.35)
        mn = float(min(y_test.min(), yhat_test.min()))
        mx = float(max(y_test.max(), yhat_test.max()))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{title_prefix} True vs Pred (Test)")
        plt.tight_layout()
        plt.show()

        # Residual plot (test)
        residuals = y_test - yhat_test
        plt.figure()
        plt.scatter(yhat_test, residuals, alpha=0.35)
        plt.axhline(0.0)
        plt.xlabel("Predicted")
        plt.ylabel("Residual (True - Pred)")
        plt.title(f"{title_prefix} Residual Plot (Test)")
        plt.tight_layout()
        plt.show()

    return train_m, test_m