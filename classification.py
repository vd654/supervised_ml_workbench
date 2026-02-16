import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    PrecisionRecallDisplay
)

RANDOM_STATE = 42

# Daten laden
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Train- Test- Split (mit stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=RANDOM_STATE, # damit man immer den selben Split bekommt 
    stratify=y # damit bei Splitten die Klassenverteilung erhalten bleibt (z.B.: 90% Klasse 0, 10% Klasse 1)
)

# Cross-Validation Setup
# Trainingsdaten in 5 Folds aufteilen
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Modelle + Parameter definieren
models_and_grids = {
    "LogisticRegression":(
        LogisticRegression(max_iter=5000, solver="lbfgs"),
        {"model__C":[0.01, 0.1, 1, 10]} # Grid Search -> testet verschiedene Werte für C (probiert die 4 Werte aus und wählt den besten Wert aus)
    ),
    "kNN":(
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 11, 21], # GridSearch sieht sich alle Werte an und entscheidet sich für die k-Größe
         "model__weights": ["uniform", "distance"]} #uniform -> jeder Nachbar zählt gleich viel
    ),
    "SVM (RBF)":(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {"model__C": [0.1, 1, 10], 
         "model__gamma": ["scale", "auto"]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 500],
         "model__max_depth": [None, 5, 10]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimator": [100,200],
         "model__learning_rate": [0.05, 0.1]}
    ),
}

# Run GreadSearch für jedes Modell
best_name = None
best_score = -np.inf
best_model = None

print("\n CV Vergleich (ROC-AUC):")

for name, (model, param_grid) in models_and_grids.items():

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc", # Metrik nach der entschieden werden soll
        n_jobs=-1,
        refit=True # Wenn besten Parameter gefunden -> Modell noch einmal neu auf allen Trainingsdaten trainieren
    )

    grid.fit(X_train, y_train)

    print(f"{name:>18}: best CV ROC-AUC = {grid.best_score_:.4f} | params = {grid.best_params_}")

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_name = name
        best_model = grid.best_estimator_

# Bestes Modell einmal auf Test evaluieren
print("\nBestes Modell: ")
print("Model:", best_name)
print("Best CV ROC-AUC:", round(best_score, 4))

y_pred = best_model.predict(X_test) # gibt konkrete Klasse aus 0 oder 1
y_score = best_model.predict_proba(X_test)[:, 1] # gibt Wahrscheinlichkeit für Klasse 1 zurück

# Metriken berechnen
print("\n Test Evaluation: ") # nur 1x mal
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1       : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_score):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plots (Confussion, ROC, PR)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix (Test)")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_score)
plt.title("ROC Curve (Test)")
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_score)
plt.title("Precision-Recall Curve (Test)")
plt.show()

# Interpretation
print("\n Interpretation (Top Features)")
model_inside = best_model.named_steps["model"]

if isinstance(model_inside, LogisticRegression): # Koeffizienten zeigen Richtung und Stärke 
    coefs = model_inside.coef_.ravel() # ravel() -> 1D Array
    top_idx = np.argsort(np.abs(coefs))[::-1][:10] # nimmt die 10 größten Werte heraus
    for i in top_idx:
        print(f"{feature_names[i]}: {coefs[i]:.4f}")

elif hasattr(model_inside, "feature_importance_"):
    imps = model_inside.feature_importances_
    top_idx = np.argsort(imps)[::-1][:10] 
    for i in top_idx:
        print(f"{feature_names[i]}: {imps[i]:.4f}")

else:
    print ("Keine simple Feature-Importance für dieses Modell.")