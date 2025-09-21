import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils.metrics import evaluate_model  # your existing function
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# -------------------------------
# Load train/test splits
# -------------------------------
X_train = pd.read_csv("train_X.csv")
y_train = pd.read_csv("train_y.csv").squeeze("columns")

X_test = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv").squeeze("columns")

# -------------------------------
# Load best estimators from hyperparameter tuning
# -------------------------------
best_estimators = joblib.load("models/best_estimators.pkl")

# -------------------------------
# Load selected features
# -------------------------------
selected_features_dict = joblib.load("models/selected_features.pkl")

# Use correlation-filtered features as default
selected_features = selected_features_dict.get("correlation_filtered", X_train.columns.tolist())

# -------------------------------
# Evaluate each model individually
# -------------------------------
print("\nEvaluating individual models on correlation-filtered features:\n")

for model_name, model_pipeline in best_estimators.items():
    # Ensure we are only using selected features
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model_pipeline.named_steps["model"] if "model" in model_pipeline.named_steps else model_pipeline)
    ])
    
    pipeline.fit(X_train[selected_features], y_train)
    evaluate_model(pipeline, X_test[selected_features], y_test, model_name=model_name)

# -------------------------------
# Optional: Evaluate ensemble of best models
# -------------------------------
print("\nEvaluating Ensemble of Best Models:")

# Create VotingClassifier using best estimators
voting_estimators = []
for name, pipeline in best_estimators.items():
    model = pipeline.named_steps["model"] if "model" in pipeline.named_steps else pipeline
    voting_estimators.append((name, model))

ensemble = VotingClassifier(
    estimators=voting_estimators,
    voting="soft"
)

# Fit ensemble on training data using selected features
ensemble_pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("ensemble", ensemble)
])
ensemble_pipeline.fit(X_train[selected_features], y_train)

evaluate_model(ensemble_pipeline, X_test[selected_features], y_test, model_name="Ensemble (Best Models)")

print("\nFinal evaluation completed!")
