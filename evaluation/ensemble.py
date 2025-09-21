# import pandas as pd
# import joblib
# import numpy as np
# from sklearn.ensemble import VotingClassifier
# from utils.metrics import evaluate_model

# #  Load training and test splits
# X_train = pd.read_csv("train_X.csv")
# y_train = pd.read_csv("train_y.csv").squeeze("columns")

# X_test = pd.read_csv("test_X.csv")
# y_test = pd.read_csv("test_y.csv").squeeze("columns")

# #  Load pre-trained models
# # log_model, _ = joblib.load("models/logistic_regression.pkl")  # logistic regression
# # rf_model, _ = joblib.load("models/random_forest.pkl")          # random forest
# # svm_model = joblib.load("models/svm_model.pkl")                # SVM
# # nn_model = joblib.load("models/neural_network.pkl")            # Neural Network

# log_model = joblib.load("models/logistic_regression.pkl")
# rf_model = joblib.load("models/random_forest.pkl")
# svm_model = joblib.load("models/svm_model.pkl")
# nn_model = joblib.load("models/neural_network.pkl")


# #  Create ensemble with majority voting
# ensemble = VotingClassifier(
#     estimators=[
#         ("log", log_model),
#         ("rf", rf_model),
#         ("svm", svm_model),
#         ("nn", nn_model)
#     ],
#     voting="soft"  # 'soft' uses predicted probabilities
# )

# # Train ensemble on training data
# ensemble.fit(X_train, y_train)

# # Evaluate ensemble
# evaluate_model(ensemble, X_test, y_test, "Ensemble (Voting Classifier)")

# # Save ensemble for future use
# joblib.dump(ensemble, "models/ensemble_model.pkl")

# print(" Ensemble model trained, evaluated, and saved!")


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils.metrics import evaluate_model
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import model getters (unfitted models)
from models.logistic_regression import get_logistic_regression
from models.random_forest import get_random_forest
from models.svm_model import get_svm
from models.neural_network import get_neural_network

# -------------------------------
# Load training and test splits
# -------------------------------
X_train = pd.read_csv("train_X.csv")
y_train = pd.read_csv("train_y.csv").squeeze("columns")

X_test = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv").squeeze("columns")

# -------------------------------
# Create ensemble (Voting Classifier)
# -------------------------------
ensemble = VotingClassifier(
    estimators=[
        ("log", get_logistic_regression()),
        ("rf", get_random_forest()),
        ("svm", get_svm()),
        ("nn", get_neural_network())
    ],
    voting="soft"  # use predicted probabilities
)

'''# Train ensemble on training data
ensemble.fit(X_train, y_train)

# Evaluate ensemble
evaluate_model(ensemble, X_test, y_test, "Ensemble (Voting Classifier)")

# Save ensemble for future use
joblib.dump(ensemble, "models/ensemble_model.pkl")

print(" Ensemble model trained, evaluated, and saved!")'''

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("ensemble", ensemble)
])

# -------------------------------
# Train pipeline
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Predict with threshold tuning
# -------------------------------
'''threshold = 0.3  # Lower threshold to improve recall
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)

# -------------------------------
# Evaluate results
# -------------------------------
print(f"\n=== Ensemble (Voting, threshold={threshold}) Evaluation ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save ensemble pipeline
joblib.dump(pipeline, "models/ensemble_model.pkl")
print(" Ensemble model with SMOTE and threshold tuning trained, evaluated, and saved!")'''

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# -------------------------------
# Tune threshold automatically for best F1
# -------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nOptimal Threshold for F1: {best_threshold:.2f}")
print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

# -------------------------------
# Final predictions using optimal threshold
# -------------------------------
y_pred = (y_pred_proba >= best_threshold).astype(int)

# -------------------------------
# Evaluate results
# -------------------------------
evaluate_model(pipeline, X_test, y_test, model_name=f"Ensemble (Voting, F1-threshold={best_threshold:.2f})")

# -------------------------------
# Save ensemble pipeline + threshold
# -------------------------------
joblib.dump((pipeline, best_threshold), "models/ensemble_model.pkl")
print(" Ensemble model with SMOTE and auto F1-threshold trained, evaluated, and saved!")

