# Supervised ML Workbench
A clean, leakage-free supervised ML baseline for tabular data (classification + regression).

## Classification (Breast Cancer)
- Train/Test split (80/20, stratified, random_state=42)
- Pipelines only (StandardScaler inside Pipeline)
- Models: Logistic Regression, kNN, SVM-RBF, Random Forest, Gradient Boosting
- Hyperparameter tuning via GridSearchCV + StratifiedKFold(5), scoring=ROC-AUC
- Final test evaluation (run once): Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix + ROC/PR curves
- Basic interpretation: LogReg coefficients, tree/boosting feature importances

## Artifacts
Best models are stored in `artifacts/` (ignored by git): `best_clf.joblib`, `best_reg.joblib`.