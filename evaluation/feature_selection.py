import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

# -------------------------------
# Load train/test splits
# -------------------------------
X_train = pd.read_csv("train_X.csv")
y_train = pd.read_csv("train_y.csv").squeeze("columns")

X_test = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv").squeeze("columns")

# -------------------------------
# Utility function to evaluate F1
# -------------------------------
def evaluate_features(model, features, model_name):
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])
    pipeline.fit(X_train[features], y_train)
    y_pred = pipeline.predict(X_test[features])
    score = f1_score(y_test, y_pred)
    print(f"{model_name} - F1 Score: {score:.4f}")
    return score

# -------------------------------
# 1 LASSO Feature Selection (Logistic Regression)
# -------------------------------
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
pipeline_lasso = Pipeline([("smote", SMOTE(random_state=42)), ("model", lasso_model)])
pipeline_lasso.fit(X_train, y_train)
lasso_coefs = pipeline_lasso.named_steps['model'].coef_[0]
lasso_features = X_train.columns[lasso_coefs != 0].tolist()
print("\n LASSO selected features:")
print(lasso_features)

# -------------------------------
# 2 Recursive Feature Elimination (Multiple Models)
# -------------------------------
rfe_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

rfe_selected = {}
for name, model in rfe_models.items():
    rfe = RFE(estimator=model, n_features_to_select=10)
    pipeline_rfe = Pipeline([("smote", SMOTE(random_state=42)), ("model", rfe)])
    pipeline_rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    rfe_selected[name] = selected_features
    print(f"\nRFE selected features for {name}:")
    print(selected_features)

# -------------------------------
# 3 Correlation Analysis
# -------------------------------
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]

X_train_corr = X_train.drop(columns=high_corr_features)
X_test_corr = X_test.drop(columns=high_corr_features)
print("\n Dropped highly correlated features (>0.8):")
print(high_corr_features)

# -------------------------------
# 4 Evaluate performance for each feature set
# -------------------------------
print("\n Evaluating feature sets:")
feature_scores = {}

# Full features
feature_scores["All Features"] = evaluate_features(LogisticRegression(max_iter=1000, random_state=42), X_train.columns, "All Features (LR)")

# LASSO features
feature_scores["LASSO Features"] = evaluate_features(LogisticRegression(max_iter=1000, random_state=42), lasso_features, "LASSO Features (LR)")

# RFE features for each model
for model_name, features in rfe_selected.items():
    # use the same model for evaluation
    model = rfe_models[model_name]
    feature_scores[f"RFE Features ({model_name})"] = evaluate_features(model, features, f"RFE Features ({model_name})")

# Correlation filtered features
feature_scores["Correlation-Filtered"] = evaluate_features(LogisticRegression(max_iter=1000, random_state=42), X_train_corr.columns, "Correlation-Filtered Features (LR)")

# -------------------------------
# 5 Save all selected features
# -------------------------------
joblib.dump({
    "lasso": lasso_features,
    "rfe": rfe_selected,
    "correlation_filtered": X_train_corr.columns.tolist()
}, "models/selected_features.pkl")

print("\n Feature selection completed and saved!")
