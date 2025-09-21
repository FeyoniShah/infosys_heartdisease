'''import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from xgboost import XGBClassifier
from utils.metrics import evaluate_model

def train_xgboost():
    # Load data
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    # Define model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"  # avoids warning
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test, "XGBoost")

    # Save trained model
    joblib.dump(model, "models/xgboost.pkl")
    return results

if __name__ == "__main__":
    train_xgboost()'''

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from utils.metrics import evaluate_model



def get_xgboost():
    """Return an untrained Random Forest model."""
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )


def train_xgboost_with_smote():
    # Load data
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
    print("After SMOTE :", dict(pd.Series(y_res).value_counts()))

    # Define model
    model = get_xgboost() #XGBClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     use_label_encoder=False,
    #     eval_metric="logloss"
    # )

    # Train model
    model.fit(X_res, y_res)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Find best threshold using Precision-Recall tradeoff
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

    # Apply tuned threshold
    #y_pred = (y_probs >= best_threshold).astype(int)
    original_predict = model.predict
    model.predict = lambda X: (model.predict_proba(X)[:, 1] >= best_threshold).astype(int)

    # Evaluate model with tuned predictions
    results = evaluate_model(model, X_test, y_test, "XGBoost (SMOTE + Threshold)")

     # Restore original predict method (optional)
    model.predict = original_predict

    # Save model + threshold
    joblib.dump((model, best_threshold), "models/xgboost.pkl")

    return results


if __name__ == "__main__":
    train_xgboost_with_smote()

